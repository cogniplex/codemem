-- Migration 003: Add temporal dimension to graph edges
ALTER TABLE graph_edges ADD COLUMN valid_from INTEGER;
ALTER TABLE graph_edges ADD COLUMN valid_to INTEGER;
CREATE INDEX IF NOT EXISTS idx_graph_edges_temporal ON graph_edges(valid_from, valid_to);
