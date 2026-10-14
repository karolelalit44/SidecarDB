-- ============================================================================
-- Sandboxed Read-Only Role & Security Privileges
-- Layer 3 Database Sandboxing for NL-to-SQL Search
-- ============================================================================

DO $$
BEGIN
    IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'nl_search_user') THEN
        CREATE ROLE nl_search_user WITH LOGIN PASSWORD 'nl_search_pass_secret';
    END IF;
END
$$;

-- 1. Grant Connection and Schema Usage
GRANT CONNECT ON DATABASE investigation_db TO nl_search_user;
GRANT USAGE ON SCHEMA public TO nl_search_user;

-- 2. Grant Read-Only SELECT to all current and future tables
GRANT SELECT ON ALL TABLES IN SCHEMA public TO nl_search_user;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT SELECT ON TABLES TO nl_search_user;

-- 3. Explicitly Revoke all Mutation & Administration Privileges
REVOKE INSERT, UPDATE, DELETE, TRUNCATE, REFERENCES, TRIGGER ON ALL TABLES IN SCHEMA public FROM nl_search_user;
ALTER DEFAULT PRIVILEGES IN SCHEMA public REVOKE INSERT, UPDATE, DELETE, TRUNCATE, REFERENCES, TRIGGER ON TABLES FROM nl_search_user;

-- 4. Session & Safety Enforcement at Role Level
ALTER ROLE nl_search_user SET default_transaction_read_only = on;
ALTER ROLE nl_search_user SET statement_timeout = '3000ms';
ALTER ROLE nl_search_user SET idle_in_transaction_session_timeout = '5000ms';
ALTER ROLE nl_search_user SET search_path = public;
