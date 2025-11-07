// 003_resolver_backfill.cypher
// Purpose: clean up schema conflicts from 002 and backfill required properties
// Neo4j 5.x compatible (avoids deprecated procedures)

// ── 1) Ensure alt_names list exists on Entities ───────────────────────────────
MATCH (e:Entity)
WHERE e.alt_names IS NULL
SET e.alt_names = [];

// ── 2) Backfill missing Metric.name using key (some nodes were created without name) ──
MATCH (m:Metric)
WHERE m.name IS NULL
SET m.name = coalesce(m.key, 'UNKNOWN');

// ── 3) (Re)create existence constraint for Metric.name now that data is clean ───────
CREATE CONSTRAINT metric_name_exists IF NOT EXISTS
FOR (m:Metric)
REQUIRE m.name IS NOT NULL;

// NOTE on uniqueness for (tenant_id, key):
// You already have a RANGE index on (:Metric {tenant_id, key}). That's fine for lookups.
// If you later want a uniqueness constraint instead, first drop the index by *name*:
//   SHOW INDEXES YIELD name, labelsOrTypes, properties
//   WHERE 'Metric' IN labelsOrTypes AND properties = ['tenant_id','key']
//   RETURN name;
// Then run:  DROP INDEX <that_name> IF EXISTS;
// And then:  CREATE CONSTRAINT metric_key_unique IF NOT EXISTS
//            FOR (m:Metric) REQUIRE (m.tenant_id, m.key) IS UNIQUE;

// ── 4) Quick FTS sanity checks (safe, optional) ───────────────────────────────
// These queries return rows in Browser; if running via a script runner, they’ll just stream rows.
CALL db.index.fulltext.queryNodes('fts_entity_name', 'sainsbury') YIELD node, score
RETURN node.id AS id, node.name AS name, score
ORDER BY score DESC LIMIT 5;

CALL db.index.fulltext.queryNodes('fts_metric_name', 'scope 1') YIELD node, score
RETURN node.id AS id, node.name AS name, node.key AS key, score
ORDER BY score DESC LIMIT 5;
