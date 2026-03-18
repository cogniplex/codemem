-- Add temporal validity columns to graph_nodes, matching graph_edges.
-- NULL means "always valid" (backward compatible with existing nodes).
--
-- NOTE: ALTER TABLE ADD COLUMN has no IF NOT EXISTS in SQLite.
-- If this migration fails (e.g., columns already exist from partial upgrade),
-- the migration runner's transaction will roll back and retry on next startup.
ALTER TABLE graph_nodes ADD COLUMN valid_from INTEGER;
ALTER TABLE graph_nodes ADD COLUMN valid_to INTEGER;

CREATE INDEX IF NOT EXISTS idx_graph_nodes_temporal ON graph_nodes(valid_from, valid_to);
CREATE INDEX IF NOT EXISTS idx_graph_nodes_kind_temporal ON graph_nodes(kind, valid_from);
