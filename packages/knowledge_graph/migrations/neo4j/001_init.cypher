// 001_init.cypher — Base schema for multi-tenant Knowledge Graph (Neo4j 5+)
// Safe to run multiple times (IF NOT EXISTS).
// Usage (cypher-shell):
//   cypher-shell -a "$NEO4J_URI" -u "$NEO4J_USER" -p "$NEO4J_PASSWORD" -f migrations/neo4j/001_init.cypher

// ──────────────────────────────────────────────────────────────────────────────
// Node KEY constraints (primary keys per tenant)
// ──────────────────────────────────────────────────────────────────────────────
CREATE CONSTRAINT entity_nodekey IF NOT EXISTS
FOR (n:Entity) REQUIRE (n.tenant_id, n.id) IS NODE KEY;

CREATE CONSTRAINT metric_nodekey IF NOT EXISTS
FOR (n:Metric) REQUIRE (n.tenant_id, n.id) IS NODE KEY;

CREATE CONSTRAINT document_nodekey IF NOT EXISTS
FOR (n:Document) REQUIRE (n.tenant_id, n.id) IS NODE KEY;

CREATE CONSTRAINT evidence_nodekey IF NOT EXISTS
FOR (n:Evidence) REQUIRE (n.tenant_id, n.id) IS NODE KEY;

CREATE CONSTRAINT claim_nodekey IF NOT EXISTS
FOR (n:Claim) REQUIRE (n.tenant_id, n.id) IS NODE KEY;

CREATE CONSTRAINT observation_nodekey IF NOT EXISTS
FOR (n:Observation) REQUIRE (n.tenant_id, n.id) IS NODE KEY;

CREATE CONSTRAINT event_nodekey IF NOT EXISTS
FOR (n:Event) REQUIRE (n.tenant_id, n.id) IS NODE KEY;

CREATE CONSTRAINT location_nodekey IF NOT EXISTS
FOR (n:Location) REQUIRE (n.tenant_id, n.id) IS NODE KEY;

// ──────────────────────────────────────────────────────────────────────────────
// Property existence constraints (core props must exist)
// ──────────────────────────────────────────────────────────────────────────────
CREATE CONSTRAINT entity_id_exists IF NOT EXISTS
FOR (n:Entity) REQUIRE n.id IS NOT NULL;
CREATE CONSTRAINT entity_tenant_exists IF NOT EXISTS
FOR (n:Entity) REQUIRE n.tenant_id IS NOT NULL;

CREATE CONSTRAINT metric_id_exists IF NOT EXISTS
FOR (n:Metric) REQUIRE n.id IS NOT NULL;
CREATE CONSTRAINT metric_tenant_exists IF NOT EXISTS
FOR (n:Metric) REQUIRE n.tenant_id IS NOT NULL;

CREATE CONSTRAINT document_id_exists IF NOT EXISTS
FOR (n:Document) REQUIRE n.id IS NOT NULL;
CREATE CONSTRAINT document_tenant_exists IF NOT EXISTS
FOR (n:Document) REQUIRE n.tenant_id IS NOT NULL;

CREATE CONSTRAINT evidence_id_exists IF NOT EXISTS
FOR (n:Evidence) REQUIRE n.id IS NOT NULL;
CREATE CONSTRAINT evidence_tenant_exists IF NOT EXISTS
FOR (n:Evidence) REQUIRE n.tenant_id IS NOT NULL;

CREATE CONSTRAINT claim_id_exists IF NOT EXISTS
FOR (n:Claim) REQUIRE n.id IS NOT NULL;
CREATE CONSTRAINT claim_tenant_exists IF NOT EXISTS
FOR (n:Claim) REQUIRE n.tenant_id IS NOT NULL;

CREATE CONSTRAINT observation_id_exists IF NOT EXISTS
FOR (n:Observation) REQUIRE n.id IS NOT NULL;
CREATE CONSTRAINT observation_tenant_exists IF NOT EXISTS
FOR (n:Observation) REQUIRE n.tenant_id IS NOT NULL;

CREATE CONSTRAINT event_id_exists IF NOT EXISTS
FOR (n:Event) REQUIRE n.id IS NOT NULL;
CREATE CONSTRAINT event_tenant_exists IF NOT EXISTS
FOR (n:Event) REQUIRE n.tenant_id IS NOT NULL;

CREATE CONSTRAINT location_id_exists IF NOT EXISTS
FOR (n:Location) REQUIRE n.id IS NOT NULL;
CREATE CONSTRAINT location_tenant_exists IF NOT EXISTS
FOR (n:Location) REQUIRE n.tenant_id IS NOT NULL;

// ──────────────────────────────────────────────────────────────────────────────
// B-tree indexes for common filters & lookups
// ──────────────────────────────────────────────────────────────────────────────
CREATE INDEX entity_name_idx IF NOT EXISTS
FOR (n:Entity) ON (n.tenant_id, n.name);

CREATE INDEX metric_key_idx IF NOT EXISTS
FOR (n:Metric) ON (n.tenant_id, n.key);

CREATE INDEX claim_hash_idx IF NOT EXISTS
FOR (n:Claim) ON (n.tenant_id, n.hash);

CREATE INDEX observation_ts_idx IF NOT EXISTS
FOR (n:Observation) ON (n.tenant_id, n.ts);

CREATE INDEX document_published_idx IF NOT EXISTS
FOR (n:Document) ON (n.tenant_id, n.published_at);

CREATE INDEX location_codes_idx IF NOT EXISTS
FOR (n:Location) ON (n.tenant_id, n.iso2, n.iso3);

// ──────────────────────────────────────────────────────────────────────────────
// Full-text indexes for search (optional but useful)
// ──────────────────────────────────────────────────────────────────────────────
CREATE FULLTEXT INDEX fts_entity_name IF NOT EXISTS
FOR (n:Entity) ON EACH [n.name, n.alt_names];

CREATE FULLTEXT INDEX fts_claim_text IF NOT EXISTS
FOR (n:Claim) ON EACH [n.text, n.normalized_text];

// ──────────────────────────────────────────────────────────────────────────────
// Relationship property existence (ensures expected metadata is present)
// ──────────────────────────────────────────────────────────────────────────────
CREATE CONSTRAINT rel_valid_for_start_exists IF NOT EXISTS
FOR ()-[r:VALID_FOR]-() REQUIRE r.start IS NOT NULL;

CREATE CONSTRAINT rel_valid_for_end_exists IF NOT EXISTS
FOR ()-[r:VALID_FOR]-() REQUIRE r.end IS NOT NULL;

CREATE CONSTRAINT rel_scored_exists IF NOT EXISTS
FOR ()-[r:RELATED_TO]-() REQUIRE r.score IS NOT NULL;
