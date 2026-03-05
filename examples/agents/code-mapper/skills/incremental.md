# Incremental Analysis (File Changes)

For re-analysis after file changes:

1. Ensure the codebase has been re-indexed from CLI first: `codemem index /path/to/project`
2. Use `get_node_memories` on each changed file's node to check for existing analysis:
   ```
   get_node_memories { "node_id": "file:<changed_file_path>" }
   ```
3. Compare stored `file_hash` in metadata with current hash
4. Files with stale hashes get re-analyzed with elevated priority
5. Files with matching hashes are skipped
6. Check for `pending-analysis` tagged memories from the Stop hook and prioritize those files:
   ```
   recall { "query": "pending analysis file changes", "k": 20 }
   ```
