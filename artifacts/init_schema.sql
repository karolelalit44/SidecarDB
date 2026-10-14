-- ============================================================================
-- Generic NL-to-SQL Search Engine — Domain Schema DDL (PostgreSQL 16 + pgvector)
-- Domain: Investigative & Case Management
-- ============================================================================

-- 1. Enable Required Extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";
CREATE EXTENSION IF NOT EXISTS "vector";

-- 2. Drop Existing Tables (Cascade for clean rebuilds)
DROP TABLE IF EXISTS case_activity_logs CASCADE;
DROP TABLE IF EXISTS evidence_records CASCADE;
DROP TABLE IF EXISTS case_parties CASCADE;
DROP TABLE IF EXISTS parties CASCADE;
DROP TABLE IF EXISTS case_assignments CASCADE;
DROP TABLE IF EXISTS cases CASCADE;
DROP TABLE IF EXISTS officers CASCADE;
DROP TABLE IF EXISTS police_stations CASCADE;

-- ============================================================================
-- Table 1: POLICE_STATIONS
-- ============================================================================
CREATE TABLE police_stations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    station_code VARCHAR(30) NOT NULL UNIQUE,
    name VARCHAR(150) NOT NULL,
    jurisdiction_zone VARCHAR(100) NOT NULL,
    district VARCHAR(100) NOT NULL,
    contact_phone VARCHAR(25),
    is_active BOOLEAN NOT NULL DEFAULT true,
    created_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp()
);

COMMENT ON TABLE police_stations IS 'Police stations, jurisdiction branches, and headquarters.';
COMMENT ON COLUMN police_stations.station_code IS 'Unique alphanumeric station code (e.g., PS-CRIME-01, PS-CYBER-02).';
COMMENT ON COLUMN police_stations.jurisdiction_zone IS 'Administrative police zone or region.';

-- ============================================================================
-- Table 2: OFFICERS
-- ============================================================================
CREATE TABLE officers (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    badge_number VARCHAR(30) NOT NULL UNIQUE,
    full_name VARCHAR(150) NOT NULL,
    rank VARCHAR(50) NOT NULL,
    contact_number VARCHAR(25),
    email VARCHAR(120),
    station_id UUID NOT NULL REFERENCES police_stations(id) ON DELETE RESTRICT,
    is_active BOOLEAN NOT NULL DEFAULT true,
    created_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp()
);

COMMENT ON TABLE officers IS 'Law enforcement officers, investigators, and supervisory personnel.';
COMMENT ON COLUMN officers.badge_number IS 'Unique police badge identification number (e.g., IND-8842, IND-4109).';
COMMENT ON COLUMN officers.rank IS 'Official rank (e.g., Inspector, Deputy Superintendent of Police (DSP), Sub-Inspector, Constable).';
COMMENT ON COLUMN officers.full_name IS 'Full legal name of the officer.';

-- ============================================================================
-- Table 3: CASES
-- ============================================================================
CREATE TABLE cases (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    case_number VARCHAR(50) NOT NULL UNIQUE,
    fir_number VARCHAR(50) NOT NULL UNIQUE,
    title VARCHAR(255) NOT NULL,
    description TEXT NOT NULL,
    category VARCHAR(100) NOT NULL,
    status VARCHAR(50) NOT NULL CHECK (status IN ('Under Investigation', 'Charge Sheet Filed', 'Closed', 'Cold Case', 'Reopened')),
    priority VARCHAR(20) NOT NULL CHECK (priority IN ('Critical', 'High', 'Medium', 'Low')),
    estimated_fraud_amount NUMERIC(15,2) NOT NULL DEFAULT 0.00 CHECK (estimated_fraud_amount >= 0),
    incident_date TIMESTAMPTZ,
    registered_date TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
    embedding vector(1536)
);

COMMENT ON TABLE cases IS 'Criminal, fraud, and cyber investigation case records.';
COMMENT ON COLUMN cases.case_number IS 'Unique internal police case file number (e.g., CASE-2025-0211).';
COMMENT ON COLUMN cases.fir_number IS 'First Information Report (FIR) formal police registration number (e.g., FIR-2025-891).';
COMMENT ON COLUMN cases.status IS 'Current procedural stage of the case.';
COMMENT ON COLUMN cases.estimated_fraud_amount IS 'Disputed or defrauded currency amount in financial crimes.';
COMMENT ON COLUMN cases.embedding IS '1536-dimensional semantic vector embedding of case title and narrative for hybrid similarity search.';

-- ============================================================================
-- Table 4: CASE_ASSIGNMENTS
-- ============================================================================
CREATE TABLE case_assignments (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    case_id UUID NOT NULL REFERENCES cases(id) ON DELETE CASCADE,
    officer_id UUID NOT NULL REFERENCES officers(id) ON DELETE RESTRICT,
    role VARCHAR(60) NOT NULL CHECK (role IN ('Investigating Officer (IO)', 'Assisting Officer', 'Supervisory Officer', 'Forensic Specialist')),
    assigned_date TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
    relieved_date TIMESTAMPTZ,
    is_current BOOLEAN NOT NULL DEFAULT true,
    assignment_notes TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
    CONSTRAINT chk_tenure CHECK (relieved_date IS NULL OR relieved_date >= assigned_date)
);

COMMENT ON TABLE case_assignments IS 'Mapping of police officers assigned to investigations with roles and dates.';
COMMENT ON COLUMN case_assignments.role IS 'Assigned functional role on case. Investigating Officer (IO) is the primary lead.';
COMMENT ON COLUMN case_assignments.is_current IS 'Boolean flag: true if currently actively assigned; false if relieved.';

-- ============================================================================
-- Table 5: PARTIES
-- ============================================================================
CREATE TABLE parties (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    party_type VARCHAR(50) NOT NULL CHECK (party_type IN ('Individual', 'Corporate', 'Financial Institution', 'Non-Profit Entity')),
    full_name VARCHAR(150) NOT NULL,
    alias_name VARCHAR(150),
    national_id_number VARCHAR(50),
    phone_number VARCHAR(30),
    email VARCHAR(120),
    residential_address TEXT,
    is_active BOOLEAN NOT NULL DEFAULT true,
    created_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp()
);

COMMENT ON TABLE parties IS 'Persons and corporate entities associated with legal cases.';
COMMENT ON COLUMN parties.party_type IS 'Entity classification: Individual, Corporate, or Financial Institution.';
COMMENT ON COLUMN parties.national_id_number IS 'Government tax or identity number (Protected PII).';

-- ============================================================================
-- Table 6: CASE_PARTIES
-- ============================================================================
CREATE TABLE case_parties (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    case_id UUID NOT NULL REFERENCES cases(id) ON DELETE CASCADE,
    party_id UUID NOT NULL REFERENCES parties(id) ON DELETE RESTRICT,
    role_in_case VARCHAR(50) NOT NULL CHECK (role_in_case IN ('Accused', 'Complainant', 'Key Witness', 'Victim', 'Person of Interest')),
    legal_sections TEXT,
    custody_status VARCHAR(50) NOT NULL DEFAULT 'Not Applicable' CHECK (custody_status IN ('At Large', 'Arrested', 'Out on Bail', 'Judicial Custody', 'Not Applicable')),
    tagged_date TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
    notes TEXT,
    CONSTRAINT uq_case_party_role UNIQUE (case_id, party_id, role_in_case)
);

COMMENT ON TABLE case_parties IS 'Associates parties to cases specifying legal status, sections, and custody.';
COMMENT ON COLUMN case_parties.role_in_case IS 'Legal standing: Accused (suspect), Complainant (reporter), Key Witness, or Victim.';
COMMENT ON COLUMN case_parties.legal_sections IS 'Statutory penal sections applied (e.g., IPC 420 Cheating, IPC 120B Conspiracy).';
COMMENT ON COLUMN case_parties.custody_status IS 'Custody status of accused: At Large (absconding), Arrested, Out on Bail, or Judicial Custody.';

-- ============================================================================
-- Table 7: EVIDENCE_RECORDS
-- ============================================================================
CREATE TABLE evidence_records (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    case_id UUID NOT NULL REFERENCES cases(id) ON DELETE CASCADE,
    evidence_number VARCHAR(50) NOT NULL UNIQUE,
    evidence_type VARCHAR(60) NOT NULL CHECK (evidence_type IN ('Digital Forensic', 'Hard Drive', 'Bank Statement', 'Contract', 'Mobile Device', 'Physical Object', 'Audio/Video Recording')),
    title VARCHAR(200) NOT NULL,
    description TEXT NOT NULL,
    chain_of_custody_officer_id UUID REFERENCES officers(id) ON DELETE SET NULL,
    storage_location VARCHAR(120) NOT NULL,
    collected_date TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
    status VARCHAR(50) NOT NULL DEFAULT 'In Custody' CHECK (status IN ('In Custody', 'Sent to Lab', 'Presented in Court', 'Released', 'Disposed')),
    embedding vector(1536),
    created_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp()
);

COMMENT ON TABLE evidence_records IS 'Physical, digital, and financial evidence items seized or gathered.';
COMMENT ON COLUMN evidence_records.evidence_number IS 'Unique chain-of-custody tracking number (e.g., EVD-2025-0045).';
COMMENT ON COLUMN evidence_records.chain_of_custody_officer_id IS 'Officer currently legally accountable for this evidence item.';
COMMENT ON COLUMN evidence_records.embedding IS '1536-dimensional semantic vector embedding of evidence forensic notes.';

-- ============================================================================
-- Table 8: CASE_ACTIVITY_LOGS
-- ============================================================================
CREATE TABLE case_activity_logs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    case_id UUID NOT NULL REFERENCES cases(id) ON DELETE CASCADE,
    logged_by_officer_id UUID REFERENCES officers(id) ON DELETE SET NULL,
    activity_type VARCHAR(60) NOT NULL CHECK (activity_type IN ('Raid', 'Arrest', 'Interrogation', 'Subpoena', 'Lab Report Received', 'Search Warrant Executed', 'Charge Sheet Filed', 'Case Note')),
    summary TEXT NOT NULL,
    activity_timestamp TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
    created_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp()
);

COMMENT ON TABLE case_activity_logs IS 'Chronological audit trail of investigative activities and operational events.';
COMMENT ON COLUMN case_activity_logs.activity_type IS 'Action type: Raid, Arrest, Interrogation, Subpoena, etc.';

-- ============================================================================
-- Indexes for High Performance & Production Readiness
-- ============================================================================

-- 1. Foreign Key Performance Indexes (prevents sequential scans on multi-table JOINs)
CREATE INDEX idx_officers_station_id ON officers(station_id);
CREATE INDEX idx_case_assignments_case_id ON case_assignments(case_id);
CREATE INDEX idx_case_assignments_officer_id ON case_assignments(officer_id);
CREATE INDEX idx_case_parties_case_id ON case_parties(case_id);
CREATE INDEX idx_case_parties_party_id ON case_parties(party_id);
CREATE INDEX idx_evidence_records_case_id ON evidence_records(case_id);
CREATE INDEX idx_evidence_records_custody_officer ON evidence_records(chain_of_custody_officer_id);
CREATE INDEX idx_case_activity_logs_case_id ON case_activity_logs(case_id);
CREATE INDEX idx_case_activity_logs_officer_id ON case_activity_logs(logged_by_officer_id);

-- 2. Query Optimization & Filter B-Trees
CREATE INDEX idx_cases_status_priority ON cases(status, priority);
CREATE INDEX idx_cases_category ON cases(category);
CREATE INDEX idx_case_assignments_current_role ON case_assignments(is_current, role);
CREATE INDEX idx_case_parties_role_custody ON case_parties(role_in_case, custody_status);

-- 3. Trigram (GIN) Indexes for Fuzzy/Substring Search
CREATE INDEX idx_officers_name_trgm ON officers USING gin (full_name gin_trgm_ops);
CREATE INDEX idx_parties_name_trgm ON parties USING gin (full_name gin_trgm_ops);
CREATE INDEX idx_cases_title_trgm ON cases USING gin (title gin_trgm_ops);

-- 4. HNSW Vector Indexes for pgvector ANN Search
CREATE INDEX idx_cases_embedding_hnsw ON cases USING hnsw (embedding vector_cosine_ops);
CREATE INDEX idx_evidence_embedding_hnsw ON evidence_records USING hnsw (embedding vector_cosine_ops);
