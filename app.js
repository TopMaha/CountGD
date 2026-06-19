/* ============================================================
   CountGD — on-device exemplar counting PWA
   Phase 1-3: live camera + snapshot/exemplar counting (opencv.js)
   Phase 5 hook: optional CountGD GPU verify via CONFIG.API_BASE
   ============================================================ */

const CONFIG = {
  // Leave empty for fully on-device mode (no GPU cost, works offline).
  // Set to your Cloudflare Worker URL to enable CountGD server verify.
  API_BASE: '',
  LIVE_PROC_WIDTH: 480,     // downscale frames for fast live processing
  SMOOTH_FRAMES: 5,         // temporal median window
  MOTION_THRESH: 8,         // mean abs frame-diff below this = "still"
  STILL_MS: 1500,           // auto-verify after image is still this long
};

const $ = (id) => document.getElementById(id);

const App = (() => {
  let cvReady = false;
  let stream = null;
  let facing = 'environment';
  let rafId = null;
  let liveColorMode = false;     // false = grayscale contrast, true = sample color
  let sampleColor = null;        // {h,s,v} when user taps in color mode

  // live smoothing / motion
  let countBuf = [];
  let prevGray = null;
  let stillSince = 0;
  let lastVerifyAt = 0;

  // snapshot / annotate state
  let annImage = null;           // HTMLImageElement
  let exemplars = [];            // [{x,y,w,h}] in image pixel coords
  let tool = 'draw';
  let drawing = null;            // current box being drawn

  // results state
  let resDetections = [];        // [{x,y,r}] centroids in image coords (image space)
  let resImage = null;
  let resThreshold = 0.35;

  /* ---------------- boot ---------------- */
  function init() {
    bindSliders();
    bindAnnotateCanvas();
    bindResultCanvas();
    $('fileInput').addEventListener('change', onFilePicked);
    if ('serviceWorker' in navigator) {
      navigator.serviceWorker.register('sw.js').catch(()=>{});
    }
    renderLibrary();
    loadOpenCV();   // load the 10MB runtime in the background — shell renders instantly
  }

  // dynamically inject opencv.js so the app shell never waits on the heavy wasm download
  function loadOpenCV() {
    if (window.cv || document.getElementById('cvScript')) return;
    const s = document.createElement('script');
    s.id = 'cvScript';
    s.async = true;
    s.src = 'https://docs.opencv.org/4.9.0/opencv.js';
    s.onload = onCvLoad;
    s.onerror = onCvError;
    document.body.appendChild(s);
  }

  function onCvLoad() {
    // opencv.js may be ready immediately, or signal via onRuntimeInitialized
    if (window.cv && cv.Mat) { cvReady = true; }
    else if (window.cv) { cv['onRuntimeInitialized'] = () => { cvReady = true; }; }
  }
  function onCvError() { toast('โหลด OpenCV ไม่สำเร็จ — ตรวจอินเทอร์เน็ต'); }

  /* ---------------- navigation ---------------- */
  function show(id) {
    document.querySelectorAll('.screen').forEach(s => s.classList.remove('active'));
    $('screen-' + id).classList.add('active');
  }
  function backToHome() { stopLive(); show('home'); $('modeTag').textContent = 'นับชิ้นงาน · on-device'; }
  function exitLive() { backToHome(); }

  /* ================================================================
     LIVE CAMERA  (Phase 1-2)
     ================================================================ */
  async function openLive() {
    show('live');
    $('modeTag').textContent = 'กล้องสด · realtime';
    await startCamera();
  }

  async function startCamera() {
    stopLive();
    try {
      stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: { ideal: facing }, width: { ideal: 1280 }, height: { ideal: 720 } },
        audio: false
      });
      const v = $('liveVideo');
      v.srcObject = stream;
      await v.play();
      $('liveStatus').textContent = cvReady ? 'กำลังนับ…' : 'กำลังโหลดเครื่องนับ…';
      countBuf = []; prevGray = null; stillSince = 0;
      loopLive();
    } catch (e) {
      $('liveStatus').textContent = 'เปิดกล้องไม่ได้';
      toast('เปิดกล้องไม่ได้: ' + (e.message || e.name));
    }
  }

  function stopLive() {
    if (rafId) cancelAnimationFrame(rafId), rafId = null;
    if (stream) { stream.getTracks().forEach(t => t.stop()); stream = null; }
    if (prevGray) { prevGray.delete(); prevGray = null; }
  }

  async function flipCamera() {
    facing = facing === 'environment' ? 'user' : 'environment';
    await startCamera();
  }

  // offscreen processing canvas
  const proc = document.createElement('canvas');
  const procCtx = proc.getContext('2d', { willReadFrequently: true });

  function loopLive() {
    const v = $('liveVideo');
    const overlay = $('liveOverlay');
    if (!v.videoWidth) { rafId = requestAnimationFrame(loopLive); return; }

    // size overlay to displayed video
    const dispW = v.clientWidth, dispH = v.clientHeight;
    if (overlay.width !== dispW || overlay.height !== dispH) {
      overlay.width = dispW; overlay.height = dispH;
    }

    if (cvReady) {
      // downscale frame for speed
      const pw = CONFIG.LIVE_PROC_WIDTH;
      const ph = Math.round(pw * v.videoHeight / v.videoWidth);
      proc.width = pw; proc.height = ph;
      procCtx.drawImage(v, 0, 0, pw, ph);

      let detections = [];
      try {
        detections = detectBlobs(procCtx, pw, ph, liveParams());
      } catch (e) { /* opencv can throw transiently */ }

      // motion detection
      const motion = computeMotion(procCtx, pw, ph);
      const still = motion >= 0 && motion < CONFIG.MOTION_THRESH;
      updateStatus(still);

      // temporal smoothing of count
      pushCount(detections.length);
      const smooth = medianCount();
      $('liveCount').textContent = smooth;

      drawLiveOverlay(overlay, detections, v.videoWidth, v.videoHeight, dispW, dispH, still);

      // auto-verify when steady
      if (still) {
        if (!stillSince) stillSince = performance.now();
        else if (performance.now() - stillSince > CONFIG.STILL_MS &&
                 performance.now() - lastVerifyAt > 4000 && CONFIG.API_BASE) {
          lastVerifyAt = performance.now();
          verifyLive();
        }
      } else { stillSince = 0; }
    }
    rafId = requestAnimationFrame(loopLive);
  }

  function liveParams() {
    return {
      thresh: +$('sThresh').value,
      minArea: +$('sMin').value,
      maxArea: +$('sMax').value,
      colorMode: liveColorMode,
      sampleColor,
    };
  }

  function pushCount(n) { countBuf.push(n); if (countBuf.length > CONFIG.SMOOTH_FRAMES) countBuf.shift(); }
  function medianCount() {
    if (!countBuf.length) return 0;
    const s = [...countBuf].sort((a, b) => a - b);
    return s[Math.floor(s.length / 2)];
  }

  function computeMotion(ctx, w, h) {
    let src = cv.matFromImageData(ctx.getImageData(0, 0, w, h));
    let gray = new cv.Mat();
    cv.cvtColor(src, gray, cv.COLOR_RGBA2GRAY);
    cv.resize(gray, gray, new cv.Size(80, 60));
    let m = -1;
    if (prevGray) {
      let diff = new cv.Mat();
      cv.absdiff(gray, prevGray, diff);
      m = cv.mean(diff)[0];
      diff.delete();
      prevGray.delete();
    }
    prevGray = gray;
    src.delete();
    return m;
  }

  function updateStatus(still) {
    const dot = $('liveDot'), st = $('liveStatus');
    if (!cvReady) { dot.className = 'dot'; st.textContent = 'กำลังโหลด…'; return; }
    if (still) { dot.className = 'dot ok'; st.textContent = 'ภาพนิ่ง · พร้อมยืนยัน'; }
    else { dot.className = 'dot warn'; st.textContent = 'ขยับอยู่…'; }
  }

  function drawLiveOverlay(canvas, dets, vw, vh, dw, dh, still) {
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, dw, dh);
    // map proc-space coords (pw x ph) → video space → cover-fit display
    const pw = proc.width, ph = proc.height;
    // video is object-fit:cover → compute scale + crop offset
    const scale = Math.max(dw / vw, dh / vh);
    const offX = (dw - vw * scale) / 2, offY = (dh - vh * scale) / 2;
    const sx = vw / pw, sy = vh / ph;
    ctx.lineWidth = 2;
    ctx.strokeStyle = still ? '#22c55e' : '#22d3ee';
    ctx.fillStyle = still ? 'rgba(34,197,94,.18)' : 'rgba(34,211,238,.15)';
    for (const d of dets) {
      const cx = (d.x * sx) * scale + offX;
      const cy = (d.y * sy) * scale + offY;
      const r = Math.max(4, (d.r || 6) * sx * scale);
      ctx.beginPath(); ctx.arc(cx, cy, r, 0, 7); ctx.fill(); ctx.stroke();
    }
  }

  // freeze current frame at full resolution → go to results (verify)
  async function verifyLive() {
    const v = $('liveVideo');
    if (!v.videoWidth) return;
    const c = document.createElement('canvas');
    c.width = v.videoWidth; c.height = v.videoHeight;
    c.getContext('2d').drawImage(v, 0, 0);
    const dataUrl = c.toDataURL('image/jpeg', 0.92);
    stopLive();
    annImage = await loadImg(dataUrl);
    exemplars = [];                       // live verify: no manual exemplars
    if (sampleColor) {/* color seed reused inside count */}
    await countNow({ from: 'live' });
  }

  function toggleLiveSliders() { $('liveSliders').classList.toggle('hidden'); }
  function toggleLiveColor() {
    liveColorMode = !liveColorMode;
    $('colorIcon').textContent = liveColorMode ? '🎨' : '⚫';
    $('colorLbl').textContent = liveColorMode ? 'แตะเลือกสี' : 'โหมดสี';
    toast(liveColorMode ? 'แตะที่ชิ้นงานในจอเพื่อเลือกสีที่จะนับ' : 'กลับสู่โหมดความต่างของแสง');
    if (liveColorMode) {
      $('liveOverlay').style.pointerEvents = 'auto';
      $('liveOverlay').onclick = onLiveTapColor;
    } else {
      sampleColor = null;
      $('liveOverlay').style.pointerEvents = 'none';
      $('liveOverlay').onclick = null;
    }
  }
  function onLiveTapColor(e) {
    const v = $('liveVideo'), o = $('liveOverlay');
    const rect = o.getBoundingClientRect();
    const dx = e.clientX - rect.left, dy = e.clientY - rect.top;
    // map display → proc space
    const dw = rect.width, dh = rect.height, vw = v.videoWidth, vh = v.videoHeight;
    const scale = Math.max(dw / vw, dh / vh);
    const offX = (dw - vw * scale) / 2, offY = (dh - vh * scale) / 2;
    const vx = (dx - offX) / scale, vy = (dy - offY) / scale;
    const px = Math.round(vx * proc.width / vw), py = Math.round(vy * proc.height / vh);
    const d = procCtx.getImageData(Math.max(0,px-1), Math.max(0,py-1), 3, 3).data;
    sampleColor = rgbToHsv(d[0], d[1], d[2]);
    toast('เลือกสีแล้ว — กำลังนับเฉพาะชิ้นสีนี้');
  }

  /* ================================================================
     SNAPSHOT + ANNOTATE  (Phase 3)
     ================================================================ */
  function openSnapshot() { $('fileInput').click(); }

  async function onFilePicked(e) {
    const file = e.target.files[0];
    e.target.value = '';
    if (!file) return;
    showLoading('กำลังโหลดรูป…');
    const url = URL.createObjectURL(file);
    annImage = await loadImg(url);
    hideLoading();
    openAnnotate();
  }

  function openAnnotate() {
    show('annotate');
    $('modeTag').textContent = 'วาดกรอบตัวอย่าง';
    exemplars = [];
    $('partDesc').value = '';
    $('annImg').src = annImage.src;
    setTool('draw');
    requestAnimationFrame(sizeAnnCanvas);
    updateExUI();
  }

  function sizeAnnCanvas() {
    const img = $('annImg'), c = $('annCanvas');
    const r = img.getBoundingClientRect();
    c.width = r.width; c.height = r.height;
    c.style.width = r.width + 'px'; c.style.height = r.height + 'px';
    drawExemplars();
  }
  window.addEventListener('resize', () => {
    if ($('screen-annotate').classList.contains('active')) sizeAnnCanvas();
  });

  function setTool(t) {
    tool = t;
    $('toolDraw').classList.toggle('on', t === 'draw');
    $('toolErase').classList.toggle('on', t === 'erase');
  }

  // map display canvas coords → image pixel coords
  function dispToImg(px, py) {
    const img = $('annImg');
    const r = img.getBoundingClientRect();
    // image uses object-fit:contain
    const scale = Math.min(r.width / annImage.width, r.height / annImage.height);
    const dw = annImage.width * scale, dh = annImage.height * scale;
    const offX = (r.width - dw) / 2, offY = (r.height - dh) / 2;
    return { x: (px - offX) / scale, y: (py - offY) / scale, scale, offX, offY };
  }

  function bindAnnotateCanvas() {
    const c = $('annCanvas');
    const getXY = (ev) => {
      const r = c.getBoundingClientRect();
      const t = ev.touches ? ev.touches[0] : ev;
      return { x: t.clientX - r.left, y: t.clientY - r.top };
    };
    let start = null;
    const down = (ev) => {
      ev.preventDefault();
      const p = getXY(ev);
      if (tool === 'erase') { eraseAt(p.x, p.y); return; }
      start = p; drawing = { x0: p.x, y0: p.y, x1: p.x, y1: p.y };
    };
    const move = (ev) => {
      if (!drawing) return; ev.preventDefault();
      const p = getXY(ev); drawing.x1 = p.x; drawing.y1 = p.y; drawExemplars();
    };
    const up = () => {
      if (!drawing) return;
      const i0 = dispToImg(Math.min(drawing.x0, drawing.x1), Math.min(drawing.y0, drawing.y1));
      const i1 = dispToImg(Math.max(drawing.x0, drawing.x1), Math.max(drawing.y0, drawing.y1));
      const w = i1.x - i0.x, h = i1.y - i0.y;
      if (w > 6 && h > 6 && exemplars.length < 5) {
        exemplars.push({ x: i0.x, y: i0.y, w, h });
      }
      drawing = null; updateExUI(); drawExemplars();
    };
    c.addEventListener('mousedown', down); c.addEventListener('mousemove', move);
    window.addEventListener('mouseup', up);
    c.addEventListener('touchstart', down, { passive: false });
    c.addEventListener('touchmove', move, { passive: false });
    c.addEventListener('touchend', up);
  }

  function eraseAt(dx, dy) {
    const p = dispToImg(dx, dy);
    for (let i = exemplars.length - 1; i >= 0; i--) {
      const e = exemplars[i];
      if (p.x >= e.x && p.x <= e.x + e.w && p.y >= e.y && p.y <= e.y + e.h) {
        exemplars.splice(i, 1); break;
      }
    }
    updateExUI(); drawExemplars();
  }

  function drawExemplars() {
    const c = $('annCanvas'), ctx = c.getContext('2d');
    ctx.clearRect(0, 0, c.width, c.height);
    const ref = dispToImg(0, 0);
    ctx.lineWidth = 2;
    exemplars.forEach((e, i) => {
      const x = e.x * ref.scale + ref.offX, y = e.y * ref.scale + ref.offY;
      const w = e.w * ref.scale, h = e.h * ref.scale;
      ctx.strokeStyle = '#22d3ee'; ctx.fillStyle = 'rgba(34,211,238,.15)';
      ctx.fillRect(x, y, w, h); ctx.strokeRect(x, y, w, h);
      ctx.fillStyle = '#22d3ee'; ctx.font = 'bold 13px sans-serif';
      ctx.fillText(i + 1, x + 4, y + 15);
    });
    if (drawing) {
      ctx.strokeStyle = '#fff'; ctx.setLineDash([5, 4]);
      ctx.strokeRect(Math.min(drawing.x0, drawing.x1), Math.min(drawing.y0, drawing.y1),
        Math.abs(drawing.x1 - drawing.x0), Math.abs(drawing.y1 - drawing.y0));
      ctx.setLineDash([]);
    }
  }

  function updateExUI() {
    $('exCount').textContent = exemplars.length;
    $('btnCount').disabled = false;   // allow count even with 0 (auto blob mode)
  }
  function clearExemplars() { exemplars = []; updateExUI(); drawExemplars(); }

  async function runCount() {
    await countNow({ from: 'snapshot' });
  }

  /* ================================================================
     COUNTING ENGINE
     ================================================================ */
  async function countNow({ from }) {
    if (!cvReady) { toast('เครื่องนับยังโหลดไม่เสร็จ รอสักครู่'); return; }
    showLoading('กำลังนับ…');
    await new Promise(r => setTimeout(r, 30));

    let result;
    if (CONFIG.API_BASE) {
      try { result = await verifyWithServer(); }
      catch (e) { toast('server verify ล้มเหลว ใช้ on-device แทน'); result = countOnDevice(); }
    } else {
      result = countOnDevice();
    }

    resDetections = result.detections;
    resImage = annImage;
    resThreshold = 0.35;
    hideLoading();
    showResults(result, from);
  }

  // exemplar-guided on-device counting
  function countOnDevice() {
    const src = cv.imread(canvasFromImg(annImage));
    let params = deriveParams(src);
    const detections = detectFromMat(src, params);
    src.delete();
    return {
      detections,
      count: detections.length,
      confidence: estimateConfidence(detections, params),
      source: exemplars.length ? 'exemplar (on-device)' : 'blob (on-device)',
      params,
    };
  }

  // derive size band + color range from drawn exemplars (or defaults)
  function deriveParams(src) {
    if (!exemplars.length) {
      // automatic: use sliders defaults scaled to image, no color filter
      const px = src.cols * src.rows;
      return { minArea: px * 0.00003, maxArea: px * 0.02, color: null, useColor: false };
    }
    let hsv = new cv.Mat();
    cv.cvtColor(src, hsv, cv.COLOR_RGBA2RGB);
    cv.cvtColor(hsv, hsv, cv.COLOR_RGB2HSV);
    let areas = [], hs = [], ss = [], vs = [];
    for (const e of exemplars) {
      const x = clamp(Math.round(e.x), 0, src.cols - 1);
      const y = clamp(Math.round(e.y), 0, src.rows - 1);
      const w = clamp(Math.round(e.w), 1, src.cols - x);
      const h = clamp(Math.round(e.h), 1, src.rows - y);
      areas.push(w * h);
      // sample center region color
      const cx = x + w / 2 | 0, cy = y + h / 2 | 0;
      const roi = hsv.roi(new cv.Rect(clamp(cx - w/4|0,0,src.cols-1), clamp(cy - h/4|0,0,src.rows-1),
        Math.max(1, w/2|0), Math.max(1, h/2|0)));
      const m = cv.mean(roi);
      hs.push(m[0]); ss.push(m[1]); vs.push(m[2]);
      roi.delete();
    }
    hsv.delete();
    const medA = median(areas);
    const mh = median(hs), ms = median(ss), mv = median(vs);
    return {
      minArea: medA * 0.25,
      maxArea: medA * 2.5,
      color: { h: mh, s: ms, v: mv },
      useColor: true,
      hsv: { lo: [Math.max(0, mh - 18), Math.max(20, ms - 70), Math.max(20, mv - 80)],
             hi: [Math.min(179, mh + 18), 255, 255] },
    };
  }

  function detectFromMat(src, params) {
    let mask = new cv.Mat();
    if (params.useColor && params.hsv) {
      let hsv = new cv.Mat();
      cv.cvtColor(src, hsv, cv.COLOR_RGBA2RGB);
      cv.cvtColor(hsv, hsv, cv.COLOR_RGB2HSV);
      let lo = cv.matFromArray(1, 3, cv.CV_64F, params.hsv.lo);
      let hi = cv.matFromArray(1, 3, cv.CV_64F, params.hsv.hi);
      let low = new cv.Mat(hsv.rows, hsv.cols, hsv.type(), [params.hsv.lo[0], params.hsv.lo[1], params.hsv.lo[2], 0]);
      let high = new cv.Mat(hsv.rows, hsv.cols, hsv.type(), [params.hsv.hi[0], params.hsv.hi[1], params.hsv.hi[2], 255]);
      cv.inRange(hsv, low, high, mask);
      hsv.delete(); lo.delete(); hi.delete(); low.delete(); high.delete();
    } else {
      let gray = new cv.Mat();
      cv.cvtColor(src, gray, cv.COLOR_RGBA2GRAY);
      cv.GaussianBlur(gray, gray, new cv.Size(5, 5), 0);
      cv.threshold(gray, mask, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU);
      gray.delete();
    }
    // morphology cleanup
    let k = cv.getStructuringElement(cv.MORPH_ELLIPSE, new cv.Size(3, 3));
    cv.morphologyEx(mask, mask, cv.MORPH_OPEN, k);
    cv.morphologyEx(mask, mask, cv.MORPH_CLOSE, k);
    k.delete();

    let contours = new cv.MatVector(), hier = new cv.Mat();
    cv.findContours(mask, contours, hier, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE);
    const dets = [];
    for (let i = 0; i < contours.size(); i++) {
      const cnt = contours.get(i);
      const a = cv.contourArea(cnt);
      if (a >= params.minArea && a <= params.maxArea) {
        const m = cv.moments(cnt);
        if (m.m00 > 0) {
          const cx = m.m10 / m.m00, cy = m.m01 / m.m00;
          const r = Math.sqrt(a / Math.PI);
          dets.push({ x: cx, y: cy, r, area: a, score: 1 });
        }
      }
      cnt.delete();
    }
    mask.delete(); contours.delete(); hier.delete();
    return dedup(dets);
  }

  // live-mode blob detection (from procCtx ImageData)
  function detectBlobs(ctx, w, h, p) {
    let src = cv.matFromImageData(ctx.getImageData(0, 0, w, h));
    let mask = new cv.Mat();
    if (p.colorMode && p.sampleColor) {
      let hsv = new cv.Mat();
      cv.cvtColor(src, hsv, cv.COLOR_RGBA2RGB);
      cv.cvtColor(hsv, hsv, cv.COLOR_RGB2HSV);
      const c = p.sampleColor;
      let low = new cv.Mat(hsv.rows, hsv.cols, hsv.type(),
        [Math.max(0, c.h - 18), Math.max(40, c.s - 80), Math.max(40, c.v - 90), 0]);
      let high = new cv.Mat(hsv.rows, hsv.cols, hsv.type(),
        [Math.min(179, c.h + 18), 255, 255, 255]);
      cv.inRange(hsv, low, high, mask);
      hsv.delete(); low.delete(); high.delete();
    } else {
      let gray = new cv.Mat();
      cv.cvtColor(src, gray, cv.COLOR_RGBA2GRAY);
      cv.GaussianBlur(gray, gray, new cv.Size(5, 5), 0);
      cv.adaptiveThreshold(gray, mask, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv.THRESH_BINARY_INV, 25, Math.max(2, (p.thresh / 12) | 0));
      gray.delete();
    }
    let k = cv.getStructuringElement(cv.MORPH_ELLIPSE, new cv.Size(3, 3));
    cv.morphologyEx(mask, mask, cv.MORPH_OPEN, k);
    k.delete();
    let contours = new cv.MatVector(), hier = new cv.Mat();
    cv.findContours(mask, contours, hier, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE);
    const dets = [];
    for (let i = 0; i < contours.size(); i++) {
      const cnt = contours.get(i);
      const a = cv.contourArea(cnt);
      if (a >= p.minArea && a <= p.maxArea) {
        const m = cv.moments(cnt);
        if (m.m00 > 0) dets.push({ x: m.m10 / m.m00, y: m.m01 / m.m00, r: Math.sqrt(a / Math.PI), area: a });
      }
      cnt.delete();
    }
    src.delete(); mask.delete(); contours.delete(); hier.delete();
    return dedup(dets);
  }

  // simple distance-based de-dup (NMS-ish)
  function dedup(dets) {
    const out = [];
    dets.sort((a, b) => (b.area || 0) - (a.area || 0));
    for (const d of dets) {
      let dup = false;
      for (const o of out) {
        const dist = Math.hypot(d.x - o.x, d.y - o.y);
        if (dist < Math.max(o.r, 4) * 0.8) { dup = true; break; }
      }
      if (!dup) out.push(d);
    }
    return out;
  }

  function estimateConfidence(dets, params) {
    if (!dets.length) return 0.3;
    // confidence from area consistency (low variance = high confidence)
    const areas = dets.map(d => d.area);
    const m = mean(areas), sd = Math.sqrt(mean(areas.map(a => (a - m) ** 2)));
    const cv_ = m ? sd / m : 1;
    let conf = 0.95 - Math.min(0.6, cv_ * 0.5);
    if (exemplars.length >= 3) conf += 0.03;
    return clamp(conf, 0.3, 0.98);
  }

  /* -------- server verify (CountGD) hook (Phase 5) -------- */
  async function verifyWithServer() {
    const blob = await (await fetch(annImage.src)).blob();
    const fd = new FormData();
    fd.append('image', blob, 'frame.jpg');
    fd.append('exemplars', JSON.stringify(exemplars));
    fd.append('text', $('partDesc') ? $('partDesc').value : '');
    const res = await fetch(CONFIG.API_BASE + '/api/count', { method: 'POST', body: fd });
    if (!res.ok) throw new Error('http ' + res.status);
    const j = await res.json();
    return {
      detections: (j.points || []).map(p => ({ x: p[0], y: p[1], r: p.r || 6, area: 100, score: p.score || 1 })),
      count: j.count, confidence: j.confidence || 0.9, source: 'CountGD (server)',
      params: { minArea: 0, maxArea: 1e9 },
    };
  }

  /* ================================================================
     RESULTS  (Phase 6 — manual edit + threshold + export)
     ================================================================ */
  function showResults(result, from) {
    show('result');
    $('modeTag').textContent = 'ผลการนับ';
    $('resCount').textContent = result.count;
    const conf = Math.round((result.confidence || 0.8) * 100);
    $('confBar').style.width = conf + '%';
    $('confVal').textContent = conf + '%';
    $('resMeta').textContent = `${result.source} · ${exemplars.length} กรอบตัวอย่าง` +
      ($('partDesc') && $('partDesc').value ? ` · ${$('partDesc').value}` : '');
    requestAnimationFrame(() => { sizeResCanvas(); drawResults(); });
  }

  function sizeResCanvas() {
    const c = $('resCanvas');
    const r = c.parentElement.getBoundingClientRect();
    c.width = r.width; c.height = r.height;
  }

  function drawResults() {
    const c = $('resCanvas'), ctx = c.getContext('2d');
    if (!resImage) return;
    const scale = Math.min(c.width / resImage.width, c.height / resImage.height);
    const dw = resImage.width * scale, dh = resImage.height * scale;
    const offX = (c.width - dw) / 2, offY = (c.height - dh) / 2;
    ctx.clearRect(0, 0, c.width, c.height);
    ctx.drawImage(resImage, offX, offY, dw, dh);
    const visible = resDetections.filter(d => (d.score ?? 1) >= resThreshold);
    ctx.lineWidth = 2;
    visible.forEach((d, i) => {
      const x = d.x * scale + offX, y = d.y * scale + offY;
      const r = Math.max(5, d.r * scale);
      ctx.strokeStyle = d.manual ? '#f59e0b' : '#22d3ee';
      ctx.fillStyle = d.manual ? 'rgba(245,158,11,.2)' : 'rgba(34,211,238,.18)';
      ctx.beginPath(); ctx.arc(x, y, r, 0, 7); ctx.fill(); ctx.stroke();
    });
    $('resCount').textContent = visible.length;
    c._geom = { scale, offX, offY };
  }

  function bindResultCanvas() {
    $('resCanvas').addEventListener('click', (ev) => {
      const c = $('resCanvas'); if (!c._geom) return;
      const r = c.getBoundingClientRect();
      const px = ev.clientX - r.left, py = ev.clientY - r.top;
      const { scale, offX, offY } = c._geom;
      const ix = (px - offX) / scale, iy = (py - offY) / scale;
      // delete if near existing
      for (let i = 0; i < resDetections.length; i++) {
        const d = resDetections[i];
        if (Math.hypot(d.x - ix, d.y - iy) < Math.max(d.r, 8) * 1.3) {
          resDetections.splice(i, 1); drawResults(); return;
        }
      }
      resDetections.push({ x: ix, y: iy, r: 8, area: 100, score: 1, manual: true });
      drawResults();
    });
    $('resThresh').addEventListener('input', (e) => {
      resThreshold = +e.target.value / 100;
      $('resThreshV').textContent = resThreshold.toFixed(2);
      drawResults();
    });
  }

  function exportResult() {
    drawResults();
    const c = $('resCanvas');
    // PNG
    const a = document.createElement('a');
    a.href = c.toDataURL('image/png');
    a.download = `count_${Date.now()}.png`;
    a.click();
    // CSV
    const count = resDetections.filter(d => (d.score ?? 1) >= resThreshold).length;
    const csv = 'index,x,y\n' + resDetections.map((d, i) => `${i + 1},${Math.round(d.x)},${Math.round(d.y)}`).join('\n');
    const cb = new Blob([`count,${count}\n` + csv], { type: 'text/csv' });
    const a2 = document.createElement('a');
    a2.href = URL.createObjectURL(cb); a2.download = `count_${Date.now()}.csv`; a2.click();
    toast('Export PNG + CSV แล้ว');
  }

  /* ================================================================
     REFERENCE LIBRARY (Phase 8 — localStorage)
     ================================================================ */
  function lib() { try { return JSON.parse(localStorage.getItem('countgd.lib') || '[]'); } catch { return []; } }
  function setLib(v) { localStorage.setItem('countgd.lib', JSON.stringify(v)); }

  function saveToLibrary() {
    const name = ($('partDesc') && $('partDesc').value) || prompt('ชื่อชนิดชิ้นงาน:') || 'ชิ้นงาน';
    // thumbnail
    const t = document.createElement('canvas'); t.width = 120; t.height = 120;
    const tc = t.getContext('2d');
    if (resImage) {
      const s = Math.min(120 / resImage.width, 120 / resImage.height);
      tc.drawImage(resImage, (120 - resImage.width * s) / 2, (120 - resImage.height * s) / 2,
        resImage.width * s, resImage.height * s);
    }
    const items = lib();
    items.unshift({
      id: 'p' + Date.now(), name,
      thumb: t.toDataURL('image/jpeg', 0.6),
      exemplars: exemplars,
      count: resDetections.length,
      created: Date.now(),
    });
    setLib(items.slice(0, 50));
    toast('บันทึกลงคลังแล้ว: ' + name);
    renderLibrary();
  }

  function openLibrary() { show('library'); $('modeTag').textContent = 'คลังชิ้นงาน'; renderLibrary(); }

  function renderLibrary() {
    const list = $('libList'); if (!list) return;
    const items = lib();
    if (!items.length) { list.innerHTML = '<div class="empty">ยังไม่มีชิ้นงานบันทึกไว้<br>นับแล้วกด 💾 บันทึก เพื่อใช้ซ้ำครั้งหน้า</div>'; return; }
    list.innerHTML = items.map(it => `
      <div class="lib-item">
        <img src="${it.thumb}" alt="">
        <div class="info">
          <h4>${escapeHtml(it.name)}</h4>
          <p>${it.exemplars.length} กรอบตัวอย่าง · นับล่าสุด ${it.count} ชิ้น</p>
          <p>${new Date(it.created).toLocaleDateString('th-TH')}</p>
        </div>
        <button class="del" onclick="App.deleteLib('${it.id}')">🗑️</button>
      </div>`).join('');
  }
  function deleteLib(id) { setLib(lib().filter(x => x.id !== id)); renderLibrary(); }

  /* ================================================================
     UTILITIES
     ================================================================ */
  function bindSliders() {
    const fmt = (n) => n >= 1000 ? (n / 1000).toFixed(n >= 10000 ? 0 : 1) + 'k' : n;
    const wire = (s, v, f) => { const el = $(s); el && el.addEventListener('input', () => $(v).textContent = (f || (x=>x))(+el.value)); };
    wire('sThresh', 'vThresh'); wire('sMin', 'vMin'); wire('sMax', 'vMax', fmt);
  }
  function bindResultCanvas_noop() {}

  function loadImg(src) {
    return new Promise((res, rej) => {
      const i = new Image(); i.onload = () => res(i); i.onerror = rej; i.src = src;
    });
  }
  function canvasFromImg(img) {
    const c = document.createElement('canvas'); c.width = img.width; c.height = img.height;
    c.getContext('2d').drawImage(img, 0, 0); return c;
  }
  function rgbToHsv(r, g, b) {
    r /= 255; g /= 255; b /= 255;
    const mx = Math.max(r, g, b), mn = Math.min(r, g, b), d = mx - mn;
    let h = 0;
    if (d) {
      if (mx === r) h = ((g - b) / d) % 6;
      else if (mx === g) h = (b - r) / d + 2;
      else h = (r - g) / d + 4;
      h *= 60; if (h < 0) h += 360;
    }
    return { h: h / 2, s: (mx ? d / mx : 0) * 255, v: mx * 255 }; // opencv H is 0-179
  }
  const clamp = (v, a, b) => Math.max(a, Math.min(b, v));
  const mean = (a) => a.reduce((s, x) => s + x, 0) / (a.length || 1);
  const median = (a) => { if (!a.length) return 0; const s = [...a].sort((x, y) => x - y); return s[Math.floor(s.length / 2)]; };
  function escapeHtml(s) { return (s || '').replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c])); }

  let toastTimer;
  function toast(msg) {
    const t = $('toast'); t.textContent = msg; t.classList.add('show');
    clearTimeout(toastTimer); toastTimer = setTimeout(() => t.classList.remove('show'), 2600);
  }
  function showLoading(txt) { $('loadingTxt').textContent = txt || 'กำลังนับ…'; $('loading').classList.add('show'); }
  function hideLoading() { $('loading').classList.remove('show'); }

  window.addEventListener('DOMContentLoaded', init);

  return {
    onCvLoad, onCvError, openLive, exitLive, flipCamera, toggleLiveSliders, toggleLiveColor,
    verifyLive, openSnapshot, backToHome, setTool, clearExemplars, runCount,
    exportResult, saveToLibrary, openLibrary, deleteLib,
  };
})();
