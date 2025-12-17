
CREATE TABLE IF NOT EXISTS well (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255),
    depth FLOAT,
    value FLOAT
);

CREATE TABLE IF NOT EXISTS tvt_predict (
    name VARCHAR(255) PRIMARY KEY,
    x FLOAT,
    y FLOAT,
    h_kol FLOAT

);

CREATE TABLE IF NOT EXISTS tvt_fact (
    name VARCHAR(255) PRIMARY KEY,
    x FLOAT,
    y FLOAT,
    h_kol FLOAT

);

CREATE INDEX IF NOT EXISTS idx_well_name ON well(name);
CREATE INDEX IF NOT EXISTS idx_tvt_predict_xy ON tvt_predict(x, y);
CREATE INDEX IF NOT EXISTS idx_tvt_predict_xy ON tvt_fact(x, y);