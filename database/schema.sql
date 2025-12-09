CREATE TABLE IF NOT EXISTS financial_reports (
    id SERIAL PRIMARY KEY,
    stock_code VARCHAR(10) NOT NULL,
    year INT NOT NULL,
    period VARCHAR(20) NOT NULL, -- 'Quý', '6 tháng', 'Cả năm'
    quarter INT,
    consolidation_status VARCHAR(20), -- 'Hợp nhất', 'Công ty mẹ'
    is_audited BOOLEAN DEFAULT FALSE,
    report_url TEXT,
    local_path TEXT,
    ocr_provider VARCHAR(50), -- 'marker', 'docling', ...
    ocr_status VARCHAR(20) DEFAULT 'PENDING', -- 'PENDING', 'PROCESSING', 'COMPLETED', 'FAILED'
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS financial_items (
    id SERIAL PRIMARY KEY,
    report_id INT REFERENCES financial_reports(id) ON DELETE CASCADE,
    item_code VARCHAR(20), -- Mã số
    item_name TEXT NOT NULL, -- Chỉ tiêu
    value DECIMAL(20, 2), -- Giá trị
    notes_ref VARCHAR(50), -- Thuyết minh
    section VARCHAR(50), -- 'BS' (Balance Sheet), 'PL' (Profit Loss), 'CF' (Cash Flow)
    page_number INT -- For "Click-to-Source"
);

CREATE TABLE IF NOT EXISTS financial_notes (
    id SERIAL PRIMARY KEY,
    report_id INT REFERENCES financial_reports(id) ON DELETE CASCADE,
    note_number VARCHAR(50),
    note_title TEXT,
    content TEXT,
    embedding VECTOR(1536)
);

-- Index
CREATE INDEX idx_reports_stock_year ON financial_reports(stock_code, year);
