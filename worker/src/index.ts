/**
 * CountGD Worker — API orchestrator (Phase 4-5)
 * --------------------------------------------------
 * The PWA runs the fast on-device counting layer entirely in the browser.
 * This Worker adds the optional "verify" layer:
 *   - stores frames + results in R2
 *   - records jobs / exemplars / corrections in D1 (feedback loop)
 *   - proxies the heavy count to a CountGD GPU endpoint (HF Space / Replicate / custom)
 *
 * Cloudflare Workers cannot run the GPU model itself, so heavy inference is
 * delegated to an external GPU service. Set CONFIG.API_BASE in the PWA to this
 * Worker's URL to switch the app from on-device to server-verified counting.
 */
import { Hono } from 'hono';
import { cors } from 'hono/cors';

type Bindings = {
  DB: D1Database;
  BUCKET: R2Bucket;
  COUNTGD_PROVIDER: string;
  COUNTGD_HF_SPACE: string;
  HF_TOKEN?: string;
  REPLICATE_TOKEN?: string;
  COUNTGD_CUSTOM_URL?: string;
};

const app = new Hono<{ Bindings: Bindings }>();
app.use('*', cors());

const now = () => Date.now();
const id = (p: string) => p + Math.random().toString(36).slice(2, 10) + now().toString(36);

app.get('/api/health', (c) => c.json({ ok: true, provider: c.env.COUNTGD_PROVIDER }));

/* ---------- count: store + verify on GPU ---------- */
app.post('/api/count', async (c) => {
  const form = await c.req.formData();
  const image = form.get('image') as File | null;
  const exemplars = String(form.get('exemplars') || '[]');
  const text = String(form.get('text') || '');
  const partTypeId = String(form.get('part_type_id') || '') || null;
  if (!image) return c.json({ error: 'no image' }, 400);

  const bytes = new Uint8Array(await image.arrayBuffer());

  // 1) store original frame in R2
  const imageKey = `frames/${id('f')}.jpg`;
  await c.env.BUCKET.put(imageKey, bytes, { httpMetadata: { contentType: 'image/jpeg' } });

  // 2) run CountGD on GPU
  let result: { count: number; points: number[][]; confidence: number };
  try {
    result = await runCountGD(c.env, bytes, JSON.parse(exemplars), text);
  } catch (e: any) {
    return c.json({ error: 'inference failed', detail: String(e?.message || e), image_key: imageKey }, 502);
  }

  // 3) record job in D1
  const jobId = id('j');
  await c.env.DB.prepare(
    `INSERT INTO count_jobs (id, part_type_id, image_key, predicted_count, confidence, exemplar_boxes, text_prompt, source, created_at)
     VALUES (?, ?, ?, ?, ?, ?, ?, 'countgd', ?)`
  ).bind(jobId, partTypeId, imageKey, result.count, result.confidence, exemplars, text, now()).run();

  return c.json({ job_id: jobId, image_key: imageKey, ...result });
});

/* ---------- feedback loop: save user-corrected count ---------- */
app.post('/api/jobs/:id/correct', async (c) => {
  const jobId = c.req.param('id');
  const { corrected_count } = await c.req.json();
  await c.env.DB.prepare(`UPDATE count_jobs SET corrected_count = ? WHERE id = ?`)
    .bind(corrected_count, jobId).run();

  // auto-calibrate threshold for this part type from accumulated corrections
  const job = await c.env.DB.prepare(`SELECT part_type_id FROM count_jobs WHERE id = ?`).bind(jobId).first<any>();
  if (job?.part_type_id) await recalibrate(c.env, job.part_type_id);
  return c.json({ ok: true });
});

/* ---------- reference library (part types) ---------- */
app.get('/api/part-types', async (c) => {
  const { results } = await c.env.DB.prepare(`SELECT * FROM part_types ORDER BY created_at DESC`).all();
  return c.json(results);
});

app.post('/api/part-types', async (c) => {
  const form = await c.req.formData();
  const name = String(form.get('name') || 'ชิ้นงาน');
  const description = String(form.get('description') || '');
  const exemplars = String(form.get('exemplars') || '[]');
  const ref = form.get('reference_image') as File | null;
  let refKey: string | null = null;
  if (ref) {
    refKey = `refs/${id('r')}.jpg`;
    await c.env.BUCKET.put(refKey, new Uint8Array(await ref.arrayBuffer()), { httpMetadata: { contentType: 'image/jpeg' } });
  }
  const ptId = id('p');
  await c.env.DB.prepare(
    `INSERT INTO part_types (id, name, description, reference_image_key, exemplar_boxes, created_at)
     VALUES (?, ?, ?, ?, ?, ?)`
  ).bind(ptId, name, description, refKey, exemplars, now()).run();
  return c.json({ id: ptId });
});

/* ---------- recent jobs ---------- */
app.get('/api/jobs', async (c) => {
  const { results } = await c.env.DB.prepare(`SELECT * FROM count_jobs ORDER BY created_at DESC LIMIT 50`).all();
  return c.json(results);
});

/* ---------- serve stored images ---------- */
app.get('/api/image/*', async (c) => {
  const key = c.req.path.replace('/api/image/', '');
  const obj = await c.env.BUCKET.get(key);
  if (!obj) return c.notFound();
  return new Response(obj.body, { headers: { 'content-type': obj.httpMetadata?.contentType || 'image/jpeg' } });
});

/* ================================================================
   GPU inference adapters
   ================================================================ */
async function runCountGD(env: Bindings, bytes: Uint8Array, exemplars: any[], text: string) {
  switch (env.COUNTGD_PROVIDER) {
    case 'replicate': return countViaReplicate(env, bytes, exemplars, text);
    case 'custom':    return countViaCustom(env, bytes, exemplars, text);
    case 'hf_space':
    default:          return countViaHfSpace(env, bytes, exemplars, text);
  }
}

// Hugging Face Space (Gradio) — public demo: huggingface.co/spaces/nikigoli/countgd
// Gradio exposes a queue API; this posts to /call/<fn> then polls the event stream.
async function countViaHfSpace(env: Bindings, bytes: Uint8Array, exemplars: any[], text: string) {
  const base = `https://${env.COUNTGD_HF_SPACE.replace('/', '-')}.hf.space`;
  const headers: Record<string, string> = { 'content-type': 'application/json' };
  if (env.HF_TOKEN) headers['authorization'] = `Bearer ${env.HF_TOKEN}`;

  // upload the image to the Space first
  const fd = new FormData();
  fd.append('files', new Blob([bytes], { type: 'image/jpeg' }), 'frame.jpg');
  const up = await fetch(`${base}/upload`, { method: 'POST', body: fd, headers: env.HF_TOKEN ? { authorization: headers.authorization } : undefined });
  if (!up.ok) throw new Error(`HF upload ${up.status}`);
  const [filePath] = await up.json<string[]>();

  // call the predict fn. NOTE: the exact fn name + arg order depend on the Space's
  // current Gradio interface — adjust `fnIndex`/payload to match countgd's app.py.
  const payload = {
    data: [
      { path: filePath, meta: { _type: 'gradio.FileData' } },
      text,                                   // text prompt
      JSON.stringify(exemplars),              // visual exemplar boxes
    ],
  };
  const call = await fetch(`${base}/call/predict`, { method: 'POST', headers, body: JSON.stringify(payload) });
  if (!call.ok) throw new Error(`HF call ${call.status}`);
  const { event_id } = await call.json<{ event_id: string }>();

  // poll the SSE result
  const res = await fetch(`${base}/call/predict/${event_id}`, { headers });
  const txt = await res.text();
  return parseGradioCount(txt);
}

function parseGradioCount(sse: string) {
  // Gradio streams `event: ...` / `data: [...]` lines. Pull the final data array.
  const dataLines = sse.split('\n').filter((l) => l.startsWith('data:')).map((l) => l.slice(5).trim());
  for (let i = dataLines.length - 1; i >= 0; i--) {
    try {
      const arr = JSON.parse(dataLines[i]);
      // CountGD demo returns something like [annotated_image, "Detected N objects", ...]
      const flat = JSON.stringify(arr);
      const m = flat.match(/(\d+)\s*(objects|items|count)/i) || flat.match(/"count"\s*:\s*(\d+)/);
      if (m) return { count: +m[1], points: [], confidence: 0.9 };
    } catch { /* keep scanning */ }
  }
  throw new Error('could not parse CountGD response');
}

// Replicate — deploy CountGD as a custom model, then set REPLICATE_TOKEN + model id.
async function countViaReplicate(env: Bindings, bytes: Uint8Array, exemplars: any[], text: string) {
  const dataUrl = `data:image/jpeg;base64,${btoa(String.fromCharCode(...bytes))}`;
  const create = await fetch('https://api.replicate.com/v1/predictions', {
    method: 'POST',
    headers: { authorization: `Bearer ${env.REPLICATE_TOKEN}`, 'content-type': 'application/json', Prefer: 'wait' },
    body: JSON.stringify({ input: { image: dataUrl, text, exemplars: JSON.stringify(exemplars) } }),
  });
  if (!create.ok) throw new Error(`Replicate ${create.status}`);
  const j = await create.json<any>();
  const out = j.output || {};
  return { count: out.count ?? 0, points: out.points ?? [], confidence: out.confidence ?? 0.9 };
}

// Custom GPU endpoint (RunPod serverless / your own service).
async function countViaCustom(env: Bindings, bytes: Uint8Array, exemplars: any[], text: string) {
  const fd = new FormData();
  fd.append('image', new Blob([bytes], { type: 'image/jpeg' }), 'frame.jpg');
  fd.append('exemplars', JSON.stringify(exemplars));
  fd.append('text', text);
  const res = await fetch(env.COUNTGD_CUSTOM_URL!, { method: 'POST', body: fd });
  if (!res.ok) throw new Error(`custom ${res.status}`);
  return res.json();
}

/* ---------- calibration: avg(predicted/corrected) → threshold nudge ---------- */
async function recalibrate(env: Bindings, partTypeId: string) {
  const rows = await env.DB.prepare(
    `SELECT predicted_count, corrected_count FROM count_jobs
     WHERE part_type_id = ? AND corrected_count IS NOT NULL ORDER BY created_at DESC LIMIT 20`
  ).bind(partTypeId).all<any>();
  if (!rows.results.length) return;
  let ratios = rows.results.map((r) => (r.predicted_count ? r.corrected_count / r.predicted_count : 1));
  const avg = ratios.reduce((s, x) => s + x, 0) / ratios.length;
  // if we tend to over-count (avg<1) raise threshold; under-count → lower it
  const pt = await env.DB.prepare(`SELECT calibrated_threshold FROM part_types WHERE id = ?`).bind(partTypeId).first<any>();
  let thr = pt?.calibrated_threshold ?? 0.3;
  thr = Math.max(0.1, Math.min(0.7, thr * (1 + (1 - avg) * 0.2)));
  await env.DB.prepare(`UPDATE part_types SET calibrated_threshold = ? WHERE id = ?`).bind(thr, partTypeId).run();
}

export default app;
