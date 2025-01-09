CREATE TABLE IF NOT EXISTS workflow_runs (
    id SERIAL PRIMARY KEY,
    tstp TIMESTAMP NOT NULL,
    step_name VARCHAR(255) NOT NULL,
    script_path VARCHAR(255) NOT NULL,
    status VARCHAR(50) NOT NULL,  -- 'success' or 'error'
    error_message TEXT
); 