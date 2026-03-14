-- Add repo and git_ref columns for scope-aware storage.
-- Existing data defaults: repo = namespace (backward compatible), git_ref = 'main'.

ALTER TABLE memories ADD COLUMN repo TEXT;
ALTER TABLE memories ADD COLUMN git_ref TEXT DEFAULT 'main';

ALTER TABLE graph_nodes ADD COLUMN repo TEXT;
ALTER TABLE graph_nodes ADD COLUMN git_ref TEXT DEFAULT 'main';

ALTER TABLE graph_edges ADD COLUMN repo TEXT;
ALTER TABLE graph_edges ADD COLUMN git_ref TEXT DEFAULT 'main';

-- Backfill repo from namespace for existing data
UPDATE memories SET repo = namespace WHERE repo IS NULL AND namespace IS NOT NULL;
UPDATE graph_nodes SET repo = namespace WHERE repo IS NULL AND namespace IS NOT NULL;

-- Indexes for scope-filtered queries
CREATE INDEX IF NOT EXISTS idx_memories_repo_ref ON memories(repo, git_ref);
CREATE INDEX IF NOT EXISTS idx_graph_nodes_repo_ref ON graph_nodes(repo, git_ref);
CREATE INDEX IF NOT EXISTS idx_graph_edges_repo_ref ON graph_edges(repo, git_ref);
