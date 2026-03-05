# Phase 3: Parallel Execution

This phase dispatches worker agents in waves. The code-mapper acts as a **dispatcher and monitor** — it spawns agents, tracks progress, and handles failures.

**Task tracking**: Create one task per work packet via TaskCreate. Monitor via TaskList.

## Wave Execution Model

Waves run sequentially. Agents WITHIN a wave run in parallel.

```
Wave 1 (baseline) ──all complete──→ Wave 2 (deep) ──all complete──→ Wave 3 (cross-cutting)
  ├─ baseline-1                      ├─ symbol-analyst-1              ├─ architecture-1
  ├─ baseline-2                      ├─ symbol-analyst-2              ├─ security-1
  ├─ baseline-3                      ├─ api-mapper-1                  └─ test-mapper-1
  └─ baseline-4                      ├─ api-mapper-2
                                     ├─ pattern-hunter-1
                                     └─ pattern-hunter-2
```

Why waves matter:
- Wave 2 agents can reference baseline memories from Wave 1 (avoid re-reading files)
- Wave 3 agents can see patterns and decisions from Wave 2 (build on findings)

## Dispatching Agents

### Step 1: Create the team

```
Use TeamCreate to create a team for coordinating analysis agents.
```

### Step 2: Spawn Wave 1 (baseline-scanner agents)

For each baseline-scanner work packet, spawn an agent:

```
Agent tool with team_name, passing the work packet as the prompt.
```

Each **baseline-scanner** agent MUST follow these rules:

1. For each file in its batch:
   a. **Get the symbol list from the graph** (not by reading 50 lines):
      ```
      graph_traverse { "start_id": "file:<path>", "max_depth": 1, "exclude_kinds": ["chunk"] }
      ```
   b. **Read the file** — use offset/limit for large files:
      - <200 lines: read entire file
      - 200-500 lines: read first 100 lines + last 50 lines
      - 500+ lines: read first 100 lines, then read specific symbol ranges from graph data
   c. **Check for existing baseline** via `get_node_memories { "node_id": "file:<path>" }`
      - If fresh baseline exists: skip
      - If stale baseline exists: `refine_memory` to update
   d. **Store 1 context memory**:
      ```
      store_memory {
        "content": "<path>: <purpose from imports + exports + symbol list>. Key symbols: <top 5 by kind>. <line count> lines, <symbol count> symbols.",
        "memory_type": "context",
        "importance": 0.3,
        "tags": ["baseline", "file-summary"],
        "links": ["file:<path>"],
        "namespace": "project"
      }
      ```
      **Max 150 chars content.** This is a baseline, not deep analysis.

2. For each NEW package encountered:
   - Store 1 context memory with file count, purpose, key exports
   - Link to `pkg:dir/`
   - **Max 150 chars content.**

3. **Mark completion**: Update task status to `completed` when batch is done.

### Step 3: Wait for Wave 1 completion

Poll TaskList until all baseline tasks are `completed`. If any agent is stuck (3+ minutes idle):
- Reassign remaining files to another agent or handle directly
- Do NOT block Wave 2 indefinitely — proceed after 80% baseline coverage

### Step 4: Spawn Wave 2 (deep analysis agents)

For each Wave 2 work packet, spawn a specialized agent.

#### symbol-analyst rules:

1. For each assigned symbol:
   a. **Read the source code** — use `get_symbol_info` to get line range, then Read with offset/limit
   b. **Explore graph context**:
      ```
      graph_traverse {
        "start_id": "sym:<qualified_name>",
        "max_depth": 2,
        "exclude_kinds": ["chunk"],
        "include_relationships": ["CALLS", "IMPORTS", "IMPLEMENTS", "INHERITS"]
      }
      ```
   c. **Check existing coverage**: `get_node_memories { "node_id": "sym:<qualified_name>" }`
   d. **Check for near-duplicates before storing**:
      ```
      recall { "query": "<your finding in 10 words>", "k": 3 }
      ```
      If a result has high similarity to your finding, use `refine_memory` instead of creating new.
   e. **Store memories by tier**:
      - **Critical symbols**: Up to 3 memories:
        - Purpose insight (WHAT + WHY it matters) — max 300 chars
        - Design decision (WHY this approach, alternatives considered) — max 300 chars, only if non-obvious
        - Pattern (recurring structure it participates in) — max 300 chars, only if recognizable
      - **Important symbols**: 1 memory:
        - Purpose insight with links — max 200 chars
   f. **Mandatory symbol links**: Every memory MUST include `links: ["sym:<qualified_name>"]`
   g. **Associate with EXPLAINS**: After storing a decision/insight, use `associate_memories` with `EXPLAINS`
   h. **Review static-analysis memories** for assigned symbols:
      - Noise → `delete_memory`
      - Useful but shallow → `refine_memory` with deeper content
      - Accurate → add `agent-verified` tag

2. **Mark each file as analyzed**:
   ```
   store_memory {
     "content": "Agent analysis complete for <file_path>",
     "memory_type": "context",
     "importance": 0.2,
     "tags": ["agent-analyzed"],
     "links": ["file:<file_path>"],
     "metadata": { "file_hash": "<sha256>", "analyzed_at": "<ISO>" }
   }
   ```

#### api-mapper rules:

1. For each endpoint-containing file:
   a. **Read the full file** (API files tend to be dense with routes)
   b. **Find all routes/handlers** by reading the code and checking graph for Endpoint nodes
   c. **For each endpoint, store 1 decision memory**:
      ```
      store_memory {
        "content": "<METHOD> <path> — <purpose>. Auth: <auth requirement>. Input: <key params/body>. Response: <shape>. Errors: <key error cases>.",
        "memory_type": "decision",
        "importance": 0.7,
        "tags": ["api-surface", "endpoint"],
        "links": ["sym:<handler_function>"],
        "namespace": "project"
      }
      ```
      **Max 300 chars.** Focus on what a consumer needs to know.
   d. If the API follows a consistent pattern across routes, store 1 pattern memory for the group

2. **Store 1 API overview** per router/module:
   ```
   store_memory {
     "content": "<module> exposes <N> endpoints: <list of METHOD /path>. Auth model: <pattern>. Common middleware: <list>.",
     "memory_type": "insight",
     "importance": 0.7,
     "tags": ["api-surface", "api-overview"],
     "namespace": "project"
   }
   ```

#### pattern-hunter rules:

1. **Before analyzing individual files**, look across ALL files in the cluster:
   a. List all symbols by kind (functions, structs, traits/interfaces)
   b. Look for naming patterns across files
   c. Look for shared import patterns
   d. Look for recurring structural patterns (same function signature shapes, same error handling)

2. **Store cross-file patterns FIRST** — these are the highest value:
   ```
   store_memory {
     "content": "Pattern in <cluster/module>: <description of recurring structure>. Examples: <2-3 symbol names>.",
     "memory_type": "pattern",
     "importance": 0.6,
     "tags": ["cross-file-pattern"],
     "links": ["sym:<example1>", "sym:<example2>"],
     "namespace": "project"
   }
   ```

3. Then store per-file observations only if they add NEW information beyond the cross-file patterns.

4. **Max 5-10 memories per cluster.** Quality over quantity.

### Step 5: Wait for Wave 2 completion

Same monitoring as Wave 1. Check TaskList, handle stuck agents.

### Step 6: Spawn Wave 3 (cross-cutting agents)

#### architecture-reviewer rules:

1. **Recall all decision and pattern memories** stored by Wave 2:
   ```
   recall { "query": "architecture module dependency layer", "k": 50, "exclude_tags": ["static-analysis"] }
   ```
2. **Traverse the module dependency graph**:
   ```
   graph_traverse { "start_id": "pkg:src/", "max_depth": 3, "include_relationships": ["DEPENDS_ON", "IMPORTS"] }
   ```
3. **Store architectural findings**: Layering decisions, dependency patterns, module boundaries
4. **Max 15-25 memories total.** These should be system-level, not per-file.

#### security-reviewer rules:

1. **Read security enrichment results**: `recall { "query": "security vulnerability trust", "k": 20 }`
2. **Read auth and validation code** identified by enrichment
3. **Store security decisions**: Trust model, auth flow, validation strategy, known risks
4. **Max 10-20 memories total.**

#### test-mapper rules:

1. **Read test files and test-mapping enrichment**
2. **Store testing patterns**: Framework, organization, common fixtures, coverage gaps
3. **Max 10-15 memories total.**

### Step 7: Wait for Wave 3 completion

Same monitoring protocol.

## Agent Prompt Template

When spawning any agent, include this in the prompt:

```
You are a {role} agent analyzing {project_name}.

WORK PACKET:
{work_packet_json}

RULES:
- Read actual source code before storing any memory. Never store memories based on graph data alone.
- Max memory content length: {max_chars} characters. If you need more, split into linked memories.
- Memory budget: max {max_memories} memories for this packet.
- Before storing any decision/pattern/insight, check for duplicates:
    recall { "query": "<10-word summary>", "k": 3 }
  If >0.85 similarity exists, refine that memory instead.
- Every memory MUST have links to relevant symbol/file nodes.
- Use the right memory type: decision (WHY), pattern (recurring HOW), insight (cross-cutting WHAT).
- When done, update your task to completed.
```

## Error Recovery

| Error | Recovery |
|-------|----------|
| Agent spawn failure | Merge that packet into an adjacent agent's work, or handle directly |
| Agent crash/timeout (3+ min idle) | Reassign remaining work to new agent or handle directly |
| Agent exceeds memory budget | Stop it, keep what's stored, proceed |
| Wave timeout (all agents stuck) | Proceed to next wave with partial results |
| Coverage gaps after all waves | Create mini follow-up packets for uncovered critical symbols (max 1 retry round) |
