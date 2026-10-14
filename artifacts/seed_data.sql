-- ============================================================================
-- Generic NL-to-SQL Search Engine — Domain Seed Data
-- Realistic Investigative & Law Enforcement Dataset
-- ============================================================================

-- Helper function to generate deterministic test vectors (1536 dimensions)
CREATE OR REPLACE FUNCTION generate_test_vector(seed_multiplier real) 
RETURNS vector AS $$
    SELECT array_agg(sin(i * seed_multiplier)::real)::vector(1536) 
    FROM generate_series(1, 1536) i;
$$ LANGUAGE sql IMMUTABLE;

-- ----------------------------------------------------------------------------
-- 1. POLICE_STATIONS
-- ----------------------------------------------------------------------------
INSERT INTO police_stations (id, station_code, name, jurisdiction_zone, district, contact_phone, is_active) VALUES
(
    '11111111-1111-1111-1111-111111111101'::uuid,
    'PS-CRIME-01',
    'Central Cyber & Economic Offences Police Station',
    'Zone 1 - North Metro',
    'Central Metropolis',
    '+91-22-2651-9001',
    true
),
(
    '11111111-1111-1111-1111-111111111102'::uuid,
    'PS-CRIME-02',
    'Financial Intelligence & Anti-Corruption Branch',
    'Zone 2 - South Metro',
    'South District',
    '+91-22-2651-9002',
    true
),
(
    '11111111-1111-1111-1111-111111111103'::uuid,
    'PS-CRIME-03',
    'Special Crime Investigation Bureau (East)',
    'Zone 3 - East Division',
    'East Metropolis',
    '+91-22-2651-9003',
    true
)
ON CONFLICT (id) DO NOTHING;

-- ----------------------------------------------------------------------------
-- 2. OFFICERS
-- (Includes 3 officers named "Sharma" for Grounded Multi-Result Ambiguity testing)
-- ----------------------------------------------------------------------------
INSERT INTO officers (id, badge_number, full_name, rank, contact_number, email, station_id, is_active) VALUES
(
    '22222222-2222-2222-2222-222222222201'::uuid,
    'IND-8842',
    'Rajesh Sharma',
    'Inspector',
    '+91-98200-11842',
    'rajesh.sharma@police.gov.in',
    '11111111-1111-1111-1111-111111111101'::uuid,
    true
),
(
    '22222222-2222-2222-2222-222222222202'::uuid,
    'IND-4109',
    'Amit Sharma',
    'Sub-Inspector',
    '+91-98200-24109',
    'amit.sharma@police.gov.in',
    '11111111-1111-1111-1111-111111111102'::uuid,
    true
),
(
    '22222222-2222-2222-2222-222222222203'::uuid,
    'IND-7721',
    'Priya Sharma',
    'Constable',
    '+91-98200-37721',
    'priya.sharma@police.gov.in',
    '11111111-1111-1111-1111-111111111103'::uuid,
    true
),
(
    '22222222-2222-2222-2222-222222222204'::uuid,
    'IND-1001',
    'Vikramaditya Rao',
    'Deputy Superintendent of Police (DSP)',
    '+91-98200-11001',
    'vikram.rao@police.gov.in',
    '11111111-1111-1111-1111-111111111101'::uuid,
    true
),
(
    '22222222-2222-2222-2222-222222222205'::uuid,
    'IND-2045',
    'Sunita Verma',
    'Inspector',
    '+91-98200-42045',
    'sunita.verma@police.gov.in',
    '11111111-1111-1111-1111-111111111102'::uuid,
    true
),
(
    '22222222-2222-2222-2222-222222222206'::uuid,
    'IND-3388',
    'Kabir Khan',
    'Sub-Inspector',
    '+91-98200-53388',
    'kabir.khan@police.gov.in',
    '11111111-1111-1111-1111-111111111101'::uuid,
    true
),
(
    '22222222-2222-2222-2222-222222222207'::uuid,
    'IND-5590',
    'Neha Deshmukh',
    'Inspector',
    '+91-98200-65590',
    'neha.deshmukh@police.gov.in',
    '11111111-1111-1111-1111-111111111103'::uuid,
    true
)
ON CONFLICT (id) DO NOTHING;

-- ----------------------------------------------------------------------------
-- 3. CASES
-- (Includes benchmark cases CASE-2025-0211, CASE-2025-0219, CASE-2024-0104, etc.)
-- ----------------------------------------------------------------------------
INSERT INTO cases (id, case_number, fir_number, title, description, category, status, priority, estimated_fraud_amount, incident_date, registered_date, embedding) VALUES
(
    '33333333-3333-3333-3333-333333333301'::uuid,
    'CASE-2025-0211',
    'FIR-2025-891',
    'Corporate Embezzlement at Apex Logistics',
    'Systematic diversion of corporate funds totaling 4.5 million through fake vendor invoicing and shell holding entities.',
    'Financial Fraud',
    'Under Investigation',
    'High',
    4500000.00,
    '2025-01-15 09:30:00+00',
    '2025-02-10 10:00:00+00',
    generate_test_vector(0.01)
),
(
    '33333333-3333-3333-3333-333333333302'::uuid,
    'CASE-2025-0219',
    'FIR-2025-912',
    'Cryptocurrency Ponzi Scheme Investigation (BitVault)',
    'Unregistered decentralized crypto asset pool promising 30% monthly APY, defrauding retail investors across multiple states.',
    'Cyber Crime',
    'Under Investigation',
    'Critical',
    18500000.00,
    '2025-01-20 14:00:00+00',
    '2025-02-18 11:30:00+00',
    generate_test_vector(0.02)
),
(
    '33333333-3333-3333-3333-333333333303'::uuid,
    'CASE-2024-0104',
    'FIR-2024-445',
    'Municipal Procurement Forgery Scam',
    'Forged municipal tender authorizations and fraudulent bank guarantee bonds submitted for public highway lighting contracts.',
    'Forgery',
    'Charge Sheet Filed',
    'Medium',
    2200000.00,
    '2024-06-10 11:00:00+00',
    '2024-07-01 15:45:00+00',
    generate_test_vector(0.03)
),
(
    '33333333-3333-3333-3333-333333333304'::uuid,
    'CASE-2024-0088',
    'FIR-2024-312',
    'Bribery & Kickback Scheme in Metro Infrastructure Project',
    'High-level corruption and quid pro quo facilitation in metro rail expansion contract allocation involving senior executives.',
    'Corruption',
    'Under Investigation',
    'Critical',
    35000000.00,
    '2024-03-05 16:20:00+00',
    '2024-04-12 09:15:00+00',
    generate_test_vector(0.04)
),
(
    '33333333-3333-3333-3333-333333333305'::uuid,
    'CASE-2023-0941',
    'FIR-2023-780',
    'Unauthorized Wire Transfer via Offshore Hawala Network',
    'Illegal cross-border capital flight through deceptive import-export trade mis-invoicing and unofficial remittance rings.',
    'Financial Fraud',
    'Closed',
    'High',
    12000000.00,
    '2023-08-14 10:00:00+00',
    '2023-09-02 14:00:00+00',
    generate_test_vector(0.05)
),
(
    '33333333-3333-3333-3333-333333333306'::uuid,
    'CASE-2023-0412',
    'FIR-2023-199',
    'Identity Theft & Credit Card Cloning Syndicate',
    'Point-of-sale skimming hardware deployed across regional gas stations, compromising over 1,200 payment cards.',
    'Cyber Crime',
    'Cold Case',
    'Low',
    850000.00,
    '2023-03-01 08:00:00+00',
    '2023-04-10 16:30:00+00',
    generate_test_vector(0.06)
)
ON CONFLICT (id) DO NOTHING;

-- ----------------------------------------------------------------------------
-- 4. CASE_ASSIGNMENTS
-- ----------------------------------------------------------------------------
INSERT INTO case_assignments (id, case_id, officer_id, role, assigned_date, relieved_date, is_current, assignment_notes) VALUES
(
    '44444444-4444-4444-4444-444444444401'::uuid,
    '33333333-3333-3333-3333-333333333301'::uuid, -- CASE-2025-0211
    '22222222-2222-2222-2222-222222222201'::uuid, -- Rajesh Sharma
    'Investigating Officer (IO)',
    '2025-02-10 10:00:00+00',
    NULL,
    true,
    'Designated primary lead investigator for corporate embezzlement probe.'
),
(
    '44444444-4444-4444-4444-444444444402'::uuid,
    '33333333-3333-3333-3333-333333333301'::uuid, -- CASE-2025-0211
    '22222222-2222-2222-2222-222222222206'::uuid, -- Kabir Khan
    'Forensic Specialist',
    '2025-02-12 11:30:00+00',
    NULL,
    true,
    'Assisting in digital analysis of financial accounting servers.'
),
(
    '44444444-4444-4444-4444-444444444403'::uuid,
    '33333333-3333-3333-3333-333333333302'::uuid, -- CASE-2025-0219
    '22222222-2222-2222-2222-222222222201'::uuid, -- Rajesh Sharma
    'Investigating Officer (IO)',
    '2025-02-18 11:30:00+00',
    NULL,
    true,
    'Assigned to direct crypto scam investigation.'
),
(
    '44444444-4444-4444-4444-444444444404'::uuid,
    '33333333-3333-3333-3333-333333333303'::uuid, -- CASE-2024-0104
    '22222222-2222-2222-2222-222222222202'::uuid, -- Amit Sharma
    'Investigating Officer (IO)',
    '2024-07-01 16:00:00+00',
    NULL,
    true,
    'Lead IO on municipal procurement forgery.'
),
(
    '44444444-4444-4444-4444-444444444405'::uuid,
    '33333333-3333-3333-3333-333333333304'::uuid, -- CASE-2024-0088
    '22222222-2222-2222-2222-222222222205'::uuid, -- Sunita Verma
    'Investigating Officer (IO)',
    '2024-04-12 10:00:00+00',
    NULL,
    true,
    'Lead IO on metro rail corruption investigation.'
),
(
    '44444444-4444-4444-4444-444444444406'::uuid,
    '33333333-3333-3333-3333-333333333304'::uuid, -- CASE-2024-0088
    '22222222-2222-2222-2222-222222222204'::uuid, -- Vikramaditya Rao
    'Supervisory Officer',
    '2024-04-12 10:00:00+00',
    NULL,
    true,
    'Directing supervisory review on high-value public contract investigation.'
),
(
    '44444444-4444-4444-4444-444444444407'::uuid,
    '33333333-3333-3333-3333-333333333305'::uuid, -- CASE-2023-0941
    '22222222-2222-2222-2222-222222222205'::uuid, -- Sunita Verma
    'Investigating Officer (IO)',
    '2023-09-02 14:00:00+00',
    '2024-02-15 17:00:00+00',
    false,
    'Case successfully prosecuted and closed.'
)
ON CONFLICT (id) DO NOTHING;

-- ----------------------------------------------------------------------------
-- 5. PARTIES
-- ----------------------------------------------------------------------------
INSERT INTO parties (id, party_type, full_name, alias_name, national_id_number, phone_number, email, residential_address, is_active) VALUES
(
    '55555555-5555-5555-5555-555555555501'::uuid,
    'Individual',
    'Ramesh Chander Singhania',
    'R.C. Singhania',
    'PAN-ABCPS8891K',
    '+91-98111-23456',
    'rc.singhania@apexcorp.com',
    'B-14/2, Sterling Towers, Malabar Hill, Metropolis',
    true
),
(
    '55555555-5555-5555-5555-555555555502'::uuid,
    'Corporate',
    'Apex Logistics Private Limited',
    'Apex Cargo Express',
    'CIN-U63090MH2012PTC231456',
    '+91-22-4099-8800',
    'legal@apexlogistics.in',
    'Plot 45, MIDC Industrial Area, Andheri East, Metropolis',
    true
),
(
    '55555555-5555-5555-5555-555555555503'::uuid,
    'Individual',
    'Devendra Saxena',
    'Dave Saxena',
    'PAN-BDGPS4412M',
    '+91-98222-34567',
    'd.saxena@whistleblower.org',
    'Flat 302, Green Meadows, Powai, Metropolis',
    true
),
(
    '55555555-5555-5555-5555-555555555504'::uuid,
    'Individual',
    'Arvind Patel',
    'Crypto Arvind',
    'PAN-CKHPP9931R',
    '+91-98333-45678',
    'arvind@bitvaultpool.io',
    'Villa 9, Palm Groves, Alibaug',
    true
),
(
    '55555555-5555-5555-5555-555555555505'::uuid,
    'Individual',
    'Meera Nair',
    NULL,
    'PAN-FKLPN1124L',
    '+91-98444-56789',
    'meera.nair@apexlogistics.in',
    'A-501, Orchid Heights, Goregaon West, Metropolis',
    true
),
(
    '55555555-5555-5555-5555-555555555506'::uuid,
    'Corporate',
    'BitVault Technologies LLP',
    'BitVault DeFi Protocol',
    'LLPIN-AAJ-9821',
    '+91-22-6712-4400',
    'support@bitvaultpool.io',
    'Level 8, Cyber City Tower B, Metropolis',
    true
)
ON CONFLICT (id) DO NOTHING;

-- ----------------------------------------------------------------------------
-- 6. CASE_PARTIES
-- ----------------------------------------------------------------------------
INSERT INTO case_parties (id, case_id, party_id, role_in_case, legal_sections, custody_status, tagged_date, notes) VALUES
(
    '66666666-6666-6666-6666-666666666601'::uuid,
    '33333333-3333-3333-3333-333333333301'::uuid, -- CASE-2025-0211
    '55555555-5555-5555-5555-555555555501'::uuid, -- Ramesh Singhania
    'Accused',
    'IPC 420 (Cheating), IPC 409 (Criminal Breach of Trust by Public Servant or Banker), IPC 120B (Criminal Conspiracy)',
    'Arrested',
    '2025-02-11 14:00:00+00',
    'Arrested following forensic audit finding 4.5M diverted.'
),
(
    '66666666-6666-6666-6666-666666666602'::uuid,
    '33333333-3333-3333-3333-333333333301'::uuid, -- CASE-2025-0211
    '55555555-5555-5555-5555-555555555502'::uuid, -- Apex Logistics
    'Victim',
    'N/A - Aggrieved Corporate Entity',
    'Not Applicable',
    '2025-02-10 10:00:00+00',
    'Victim entity from whose bank accounts funds were routed.'
),
(
    '66666666-6666-6666-6666-666666666603'::uuid,
    '33333333-3333-3333-3333-333333333301'::uuid, -- CASE-2025-0211
    '55555555-5555-5555-5555-555555555503'::uuid, -- Devendra Saxena
    'Complainant',
    'N/A - Whistleblower Informant',
    'Not Applicable',
    '2025-02-10 10:00:00+00',
    'Filed initial complaint and supplied internal memo records.'
),
(
    '66666666-6666-6666-6666-666666666604'::uuid,
    '33333333-3333-3333-3333-333333333301'::uuid, -- CASE-2025-0211
    '55555555-5555-5555-5555-555555555505'::uuid, -- Meera Nair
    'Key Witness',
    'N/A - Deposition under Section 164 CrPC',
    'Not Applicable',
    '2025-02-14 16:30:00+00',
    'Chief Financial Officer who confirmed unauthorized invoice signatures.'
),
(
    '66666666-6666-6666-6666-666666666605'::uuid,
    '33333333-3333-3333-3333-333333333302'::uuid, -- CASE-2025-0219
    '55555555-5555-5555-5555-555555555504'::uuid, -- Arvind Patel
    'Accused',
    'IPC 420 (Cheating), IT Act Section 66D (Cheating by Personation Using Computer Resource)',
    'At Large',
    '2025-02-18 11:30:00+00',
    'Founder of BitVault; currently absconding. Look Out Circular (LOC) issued.'
),
(
    '66666666-6666-6666-6666-666666666606'::uuid,
    '33333333-3333-3333-3333-333333333302'::uuid, -- CASE-2025-0219
    '55555555-5555-5555-5555-555555555506'::uuid, -- BitVault Technologies
    'Accused',
    'IPC 420 (Cheating), Banning of Unregulated Deposit Schemes Act 2019',
    'Not Applicable',
    '2025-02-18 11:30:00+00',
    'Entity accounts frozen across 4 commercial banks.'
)
ON CONFLICT (id) DO NOTHING;

-- ----------------------------------------------------------------------------
-- 7. EVIDENCE_RECORDS
-- ----------------------------------------------------------------------------
INSERT INTO evidence_records (id, case_id, evidence_number, evidence_type, title, description, chain_of_custody_officer_id, storage_location, collected_date, status, embedding) VALUES
(
    '77777777-7777-7777-7777-777777777701'::uuid,
    '33333333-3333-3333-3333-333333333301'::uuid, -- CASE-2025-0211
    'EVD-2025-0045',
    'Bank Statement',
    'Apex Logistics Chartered Account Ledger FY 2024-25',
    'Certified printouts and digital forensic export of SAP GL account 4100 showing unauthorized transfers to offshore entities.',
    '22222222-2222-2222-2222-222222222201'::uuid, -- Rajesh Sharma
    'Vault-A, Evidence Locker 14',
    '2025-02-12 14:00:00+00',
    'In Custody',
    generate_test_vector(0.11)
),
(
    '77777777-7777-7777-7777-777777777702'::uuid,
    '33333333-3333-3333-3333-333333333301'::uuid, -- CASE-2025-0211
    'EVD-2025-0046',
    'Hard Drive',
    'Seagate Barracuda 4TB HDD seized from Singhania Office',
    'Contains cloned email archives and private WhatsApp communications discussing fake vendor creation.',
    '22222222-2222-2222-2222-222222222206'::uuid, -- Kabir Khan
    'Cyber Lab Station 3',
    '2025-02-13 18:30:00+00',
    'Sent to Lab',
    generate_test_vector(0.12)
),
(
    '77777777-7777-7777-7777-777777777703'::uuid,
    '33333333-3333-3333-3333-333333333302'::uuid, -- CASE-2025-0219
    'EVD-2025-0078',
    'Digital Forensic',
    'Hardware Crypto Wallet (Ledger Nano X)',
    'Recovered from Arvind Patel premises containing public addresses holding 140 BTC in stolen deposits.',
    '22222222-2222-2222-2222-222222222206'::uuid, -- Kabir Khan
    'Vault-B, High-Security Locker 02',
    '2025-02-19 12:15:00+00',
    'In Custody',
    generate_test_vector(0.13)
),
(
    '77777777-7777-7777-7777-777777777704'::uuid,
    '33333333-3333-3333-3333-333333333303'::uuid, -- CASE-2024-0104
    'EVD-2024-0112',
    'Contract',
    'Forged Municipal Tender Certificate & Notarized Agreement',
    'Fraudulent stamp papers bearing duplicate government seal numbers and forged signatures of Chief Municipal Engineer.',
    '22222222-2222-2222-2222-222222222202'::uuid, -- Amit Sharma
    'Vault-A, Evidence Locker 08',
    '2024-07-05 10:30:00+00',
    'In Custody',
    generate_test_vector(0.14)
)
ON CONFLICT (id) DO NOTHING;

-- ----------------------------------------------------------------------------
-- 8. CASE_ACTIVITY_LOGS
-- ----------------------------------------------------------------------------
INSERT INTO case_activity_logs (id, case_id, logged_by_officer_id, activity_type, summary, activity_timestamp) VALUES
(
    '88888888-8888-8888-8888-888888888801'::uuid,
    '33333333-3333-3333-3333-333333333301'::uuid, -- CASE-2025-0211
    '22222222-2222-2222-2222-222222222201'::uuid, -- Rajesh Sharma
    'Search Warrant Executed',
    'Executed court-authorized search warrant at Apex Logistics HQ; seized 4 computers and 2 accounting folders.',
    '2025-02-11 08:30:00+00'
),
(
    '88888888-8888-8888-8888-888888888802'::uuid,
    '33333333-3333-3333-3333-333333333301'::uuid, -- CASE-2025-0211
    '22222222-2222-2222-2222-222222222201'::uuid, -- Rajesh Sharma
    'Arrest',
    'Arrested primary accused Ramesh Singhania from residence under Section 420/409 IPC; informed family as per guidelines.',
    '2025-02-11 14:15:00+00'
),
(
    '88888888-8888-8888-8888-888888888803'::uuid,
    '33333333-3333-3333-3333-333333333301'::uuid, -- CASE-2025-0211
    '22222222-2222-2222-2222-222222222206'::uuid, -- Kabir Khan
    'Lab Report Received',
    'Completed forensic imaging and MD5 checksum hash verification of Seagate 4TB HDD; identified 1,420 relevant emails.',
    '2025-02-15 17:00:00+00'
),
(
    '88888888-8888-8888-8888-888888888804'::uuid,
    '33333333-3333-3333-3333-333333333302'::uuid, -- CASE-2025-0219
    '22222222-2222-2222-2222-222222222201'::uuid, -- Rajesh Sharma
    'Raid',
    'Conducted coordinated raid on BitVault office at Level 8 Cyber City; seized hardware wallets, routers, and promotional banners.',
    '2025-02-19 10:00:00+00'
),
(
    '88888888-8888-8888-8888-888888888805'::uuid,
    '33333333-3333-3333-3333-333333333303'::uuid, -- CASE-2024-0104
    '22222222-2222-2222-2222-222222222202'::uuid, -- Amit Sharma
    'Charge Sheet Filed',
    'Final formal charge sheet filed before Special Judicial Magistrate Court with 35 forensic documentary exhibits.',
    '2024-11-15 11:30:00+00'
)
ON CONFLICT (id) DO NOTHING;

-- Cleanup temporary helper function
DROP FUNCTION IF EXISTS generate_test_vector(real);
