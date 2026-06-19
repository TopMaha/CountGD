# CountGD — นับชิ้นงานจากกล้องสด + รูปถ่าย

PWA นับจำนวนชิ้นงานในโรงงาน **ไม่ต้องเทรนโมเดลใหม่** — แค่วาดกรอบตัวอย่าง (visual exemplar) ชิ้นที่ต้องการ ระบบจะหาชิ้นที่เหมือนกันทั้งภาพแล้วนับให้

แนวคิด: **Open-World / Few-shot Exemplar Counting** แทน YOLO ธรรมดา (ที่ต้องเก็บ dataset + เทรนใหม่ทุกครั้งที่ชิ้นงานเปลี่ยน)

## สถาปัตยกรรม 2 ชั้น (Hybrid)

| ชั้น | ทำงานที่ไหน | ความเร็ว | ใช้เมื่อไหร่ |
|------|------------|----------|-------------|
| 🟢 **Live Count** | ในเบราว์เซอร์ (opencv.js) | ~15–30 fps | เล็งกล้อง → เห็นเลขทันที ไม่กินค่า GPU |
| 🔵 **Verify Count** | GPU endpoint (CountGD) | ~1–3 วิ/เฟรม | กดยืนยัน / ภาพนิ่ง → ได้เลขแม่นสูงสุด |

> **"สด + แม่น + นับอะไรก็ได้" พร้อมกันในโมเดลเดียวทำไม่ได้** — โมเดลแม่นสุด (CountGD) รันบน GPU ช้าเกินสำหรับวิดีโอสด จึงแยกเป็นชั้นสดเร็วบนเครื่อง + ชั้นยืนยันแม่นบน server

## สถานะ (เฟสที่ทำเสร็จ)

- ✅ **เฟส 1–2** PWA shell + กล้องสด: เปิดกล้อง → เห็นเลขนับ realtime + temporal smoothing (median 5 เฟรม) + ไฟสถานะนิ่ง/ขยับ
- ✅ **เฟส 3** โหมดถ่ายรูป/อัปโหลด + วาดกรอบ exemplar (touch + mouse)
- ✅ **เครื่องนับ on-device** exemplar-guided: วัดขนาด+สีจากกรอบที่วาด → หาชิ้นที่เหมือน (opencv.js contour/blob + NMS de-dup)
- ✅ **เฟส 6** หน้าผลลัพธ์ + แก้มือ (+1/−1) + slider เกณฑ์ความมั่นใจ + Export PNG/CSV
- ✅ **เฟส 8** คลังชิ้นงาน (Reference Library) เก็บใน localStorage ใช้ซ้ำได้
- 🟡 **เฟส 4–5** Worker + R2 + D1 + CountGD GPU verify — โค้ดพร้อมใน `worker/` ยังไม่ deploy (ต้องตั้ง GPU endpoint)

แอป **ใช้งานได้จริงทันทีแบบ on-device** โดยไม่ต้อง deploy server — ชั้น verify เป็นทางเลือกเสริมเพื่อความแม่นสูงสุด

## รันแอป (frontend)

ทุกอย่างเป็น static files — เปิดด้วย web server ใดก็ได้:

```bash
npx http-server . -p 5500
# เปิด http://localhost:5500 (ต้องใช้ HTTPS หรือ localhost เพื่อให้กล้องทำงาน)
```

Deploy ขึ้น **GitHub Pages** / **Cloudflare Pages** ได้เลย (เป็น static)

### การใช้งาน
1. **กล้องสด** — เล็งชิ้นงาน เห็นเลขทันที, ปรับ slider ความไว/ขนาด, แตะ "โหมดสี" แล้วแตะชิ้นงานเพื่อนับเฉพาะสีนั้น, กดชัตเตอร์/ยืนยันเพื่อ freeze + นับละเอียด
2. **ถ่ายรูป** — เลือก/ถ่ายรูป → ลากกรอบรอบชิ้นตัวอย่าง 1–5 ชิ้น (เลือกขนาด/มุมต่างกัน) → กดนับ
3. **ผลลัพธ์** — แตะรูปเพื่อเพิ่ม/ลบจุด, เลื่อน slider เกณฑ์, Export, บันทึกลงคลัง

## Verify layer (เฟส 4–5) — เปิดเมื่อต้องการความแม่นสูงสุด

โค้ด Worker อยู่ใน [`worker/`](worker/) (Hono + D1 + R2). ขั้นตอน deploy:

```bash
cd worker
npm install
wrangler d1 create countgd          # ใส่ database_id ลง wrangler.toml
wrangler r2 bucket create countgd-images
npm run db:init:remote              # สร้างตาราง
wrangler secret put HF_TOKEN        # (ถ้าใช้ HF Space แบบ private)
npm run deploy
```

จากนั้นในไฟล์ [`app.js`](app.js) ตั้ง `CONFIG.API_BASE` เป็น URL ของ Worker → แอปจะส่งภาพไป verify ด้วย **CountGD** บน GPU อัตโนมัติเมื่อกดยืนยัน/ภาพนิ่ง

GPU endpoint รองรับ 3 แบบ (ตั้งใน `wrangler.toml` → `COUNTGD_PROVIDER`):
- `hf_space` — Hugging Face Space (default: `nikigoli/countgd`, rate-limited; deploy เองสำหรับ production)
- `replicate` — deploy CountGD เป็น custom model
- `custom` — RunPod serverless / GPU ของตัวเอง

> CountGD: *Multi-Modal Open-World Counting* (NeurIPS 2024, Oxford VGG) — รับทั้ง text + visual exemplar = SOTA บน FSC-147 / CARPK
> Repo: https://github.com/niki-amini-naieni/CountGD · Demo: https://huggingface.co/spaces/nikigoli/countgd

## เทคนิคความแม่นยำที่ใส่ไว้แล้ว
- Temporal smoothing (median) กันเลขกระพริบในโหมดสด
- Frame-diff motion detection → freeze เฉพาะตอนภาพนิ่ง + auto-verify
- NMS / distance de-dup กันนับซ้ำ
- Exemplar-guided size + HSV color band (นับเฉพาะชิ้นที่เหมือนตัวอย่าง)
- Full-resolution capture ตอนยืนยัน (ชิ้นเล็กยังนับได้)
- Confidence จากความสม่ำเสมอของขนาดชิ้น
- Calibration loop ฝั่ง server: เก็บผลแก้มือ → ปรับ threshold อัตโนมัติ

## โครงสร้าง
```
index.html      # PWA shell + UI (ภาษาไทย)
app.js          # counting engine + camera + annotate + results + library
sw.js           # service worker (offline shell + cache opencv.js)
manifest.json   # PWA manifest
icon.svg / icon-*.png
worker/         # Cloudflare Worker (verify layer) — Hono + D1 + R2 + CountGD proxy
```

## โรดแมปต่อ
- เฟส 7: Tiling สำหรับภาพหนาแน่น (>100 ชิ้น) แบ่งกริด + de-dup รอยต่อ
- ONNX Runtime Web (YOLO11n) เป็นทางเลือกชั้น 1 สำหรับงานที่ blob ไม่พอ
- Claude Vision cross-check ตอนจำนวนน้อย
- Reference Library sync ขึ้น D1/R2 (ตอนนี้อยู่ localStorage)
