// 002_resolver_schema.cypher
// Neo4j 5.x — constraints + full-text indexes for the resolver

// ── Constraints ──────────────────────────────────────────────────────────────
CREATE CONSTRAINT entity_pk IF NOT EXISTS
FOR (e:Entity)
REQUIRE (e.tenant_id, e.id) IS UNIQUE;

CREATE CONSTRAINT metric_pk IF NOT EXISTS
FOR (m:Metric)
REQUIRE (m.tenant_id, m.id) IS UNIQUE;

CREATE CONSTRAINT entity_name_exists IF NOT EXISTS
FOR (e:Entity)
REQUIRE e.name IS NOT NULL;

CREATE CONSTRAINT metric_key_exists IF NOT EXISTS
FOR (m:Metric)
REQUIRE m.key IS NOT NULL;

CREATE CONSTRAINT metric_name_exists IF NOT EXISTS
FOR (m:Metric)
REQUIRE m.name IS NOT NULL;

// Optional but helpful: ensure metric key uniqueness per tenant
CREATE CONSTRAINT metric_key_unique IF NOT EXISTS
FOR (m:Metric)
REQUIRE (m.tenant_id, m.key) IS UNIQUE;

// ── Full-text indexes (used by Neo4jResolver FTS fallback) ───────────────────
// Index both `name` and `alt_names` (list) for entities.
CREATE FULLTEXT INDEX fts_entity_name IF NOT EXISTS
FOR (e:Entity)
ON EACH [e.name, e.alt_names];

// Index both `name` and `key` for metrics.
CREATE FULLTEXT INDEX fts_metric_name IF NOT EXISTS
FOR (m:Metric)
ON EACH [m.name, m.key];

// ── Smoke tests (safe to run; LIMIT keeps it light) ──────────────────────────
CALL db.indexes() YIELD name, type, entityType, state
RETURN name, type, entityType, state
ORDER BY name;

CALL db.index.fulltext.queryNodes('fts_entity_name', 'sainsbury') YIELD node, score
RETURN node.id AS id, node.name AS name, score
ORDER BY score DESC LIMIT 5;

CALL db.index.fulltext.queryNodes('fts_metric_name', 'scope 1') YIELD node, score
RETURN node.id AS id, node.name AS name, node.key AS key, score
ORDER BY score DESC LIMIT 5;
