-- CountGD D1 schema
-- ชนิดชิ้นงาน + reference exemplar ที่บันทึกไว้ (reuse ครั้งถัดไป)
CREATE TABLE IF NOT EXISTS part_types (
  id TEXT PRIMARY KEY,
  name TEXT NOT NULL,
  description TEXT,
  reference_image_key TEXT,          -- key ใน R2
  exemplar_boxes TEXT,               -- JSON: [{x,y,w,h}, ...]
  calibrated_threshold REAL DEFAULT 0.3,
  created_at INTEGER
);

-- แต่ละครั้งที่นับ
CREATE TABLE IF NOT EXISTS count_jobs (
  id TEXT PRIMARY KEY,
  part_type_id TEXT,
  image_key TEXT,                    -- รูปต้นฉบับใน R2
  result_image_key TEXT,            -- รูป annotate ใน R2
  predicted_count INTEGER,
  corrected_count INTEGER,          -- หลังผู้ใช้แก้มือ (feedback loop)
  confidence REAL,
  exemplar_boxes TEXT,              -- JSON
  text_prompt TEXT,
  source TEXT,                      -- 'countgd' | 'on-device' | 'live'
  created_at INTEGER,
  FOREIGN KEY (part_type_id) REFERENCES part_types(id)
);

CREATE INDEX IF NOT EXISTS idx_jobs_part ON count_jobs(part_type_id);
CREATE INDEX IF NOT EXISTS idx_jobs_created ON count_jobs(created_at);
