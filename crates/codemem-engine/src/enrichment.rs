//! Enrichment logic: store_insight, git history, security, performance,
//! complexity, architecture, test mapping, API surface, doc coverage,
//! change impact, code smells, hot+complex correlation, blame/ownership,
//! enhanced security scanning, and quality stratification.

use crate::scoring::truncate_content;
use crate::CodememEngine;
use codemem_core::{
    CodememError, Edge, GraphBackend, GraphNode, MemoryNode, MemoryType, NodeKind,
    RelationshipType, VectorBackend,
};
use serde_json::json;
use std::collections::{HashMap, HashSet};

/// Result from an enrichment operation.
pub struct EnrichResult {
    pub insights_stored: usize,
    pub details: serde_json::Value,
}

impl CodememEngine {
    /// Store an Insight memory through the full pipeline: storage, BM25, graph
    /// node, RELATES_TO edges to linked nodes, and vector embedding.
    /// Returns the memory ID if inserted, or None if it was a duplicate.
    /// Does NOT call `save_index()` -- callers should batch that at the end.
    ///
    // TODO: Steps 1/2/3/5 duplicate `persist_memory_inner`. Refactor to call
    // `persist_memory_no_save` for the core pipeline, then add the semantic dedup
    // pre-check (step 1b) and RELATES_TO edges (step 4) as pre/post steps.
    pub fn store_insight(
        &self,
        content: &str,
        track: &str,
        tags: &[&str],
        importance: f64,
        namespace: Option<&str>,
        links: &[String],
    ) -> Option<String> {
        let hash = codemem_storage::Storage::content_hash(content);
        let now = chrono::Utc::now();
        let id = uuid::Uuid::new_v4().to_string();
        let mut all_tags: Vec<String> =
            vec![format!("track:{track}"), "static-analysis".to_string()];
        all_tags.extend(tags.iter().map(|t| t.to_string()));

        let memory = MemoryNode {
            id: id.clone(),
            content: content.to_string(),
            memory_type: MemoryType::Insight,
            importance: importance.clamp(0.0, 1.0),
            confidence: self.config.enrichment.insight_confidence,
            access_count: 0,
            content_hash: hash,
            tags: all_tags.clone(),
            metadata: HashMap::from([
                ("track".into(), json!(track)),
                ("generated_by".into(), json!("enrichment_pipeline")),
            ]),
            namespace: namespace.map(String::from),
            created_at: now,
            updated_at: now,
            last_accessed_at: now,
        };

        // 1. Insert into storage (dedup by content hash)
        if self.storage.insert_memory(&memory).is_err() {
            return None; // duplicate or error -- skip silently
        }

        // 1b. Compute enriched embedding once (used for both dedup check and vector insert)
        let enriched = self.enrich_memory_text(
            content,
            MemoryType::Insight,
            &all_tags,
            namespace,
            Some(&id),
        );
        let stored_embedding = if let Ok(Some(emb_guard)) = self.lock_embeddings() {
            if let Ok(embedding) = emb_guard.embed(&enriched) {
                drop(emb_guard);
                // Semantic dedup: check top-3 nearest embeddings for near-duplicates
                if let Ok(vec) = self.lock_vector() {
                    let neighbors = vec.search(&embedding, 3).unwrap_or_default();
                    for (neighbor_id, similarity) in &neighbors {
                        if *neighbor_id == id {
                            continue;
                        }
                        if (*similarity as f64) > self.config.enrichment.dedup_similarity_threshold
                        {
                            // Too similar to an existing memory -- roll back
                            let _ = self.storage.delete_memory(&id);
                            return None;
                        }
                    }
                }
                Some(embedding)
            } else {
                None
            }
        } else {
            None
        };

        // 2. BM25 index
        if let Ok(mut bm25) = self.lock_bm25() {
            bm25.add_document(&id, content);
        }

        // 3. Graph node
        let graph_node = GraphNode {
            id: id.clone(),
            kind: NodeKind::Memory,
            label: truncate_content(content, 80),
            payload: HashMap::new(),
            centrality: 0.0,
            memory_id: Some(id.clone()),
            namespace: namespace.map(String::from),
        };
        let _ = self.storage.insert_graph_node(&graph_node);
        if let Ok(mut graph) = self.lock_graph() {
            let _ = graph.add_node(graph_node);
        }

        // 4. RELATES_TO edges to linked nodes
        if !links.is_empty() {
            if let Ok(mut graph) = self.lock_graph() {
                for link_id in links {
                    let edge = Edge {
                        id: format!("{id}-RELATES_TO-{link_id}"),
                        src: id.clone(),
                        dst: link_id.clone(),
                        relationship: RelationshipType::RelatesTo,
                        weight: 0.3,
                        properties: HashMap::new(),
                        created_at: now,
                        valid_from: None,
                        valid_to: None,
                    };
                    let _ = self.storage.insert_graph_edge(&edge);
                    let _ = graph.add_edge(edge);
                }
            }
        }

        // 4b. Auto-link to code nodes mentioned in content
        self.auto_link_to_code_nodes(&id, content, links);

        // 5. Vector embedding (reuse embedding from step 1b)
        if let Some(ref embedding) = stored_embedding {
            let _ = self.storage.store_embedding(&id, embedding);
            if let Ok(mut vec) = self.lock_vector() {
                let _ = vec.insert(&id, embedding);
            }
        }

        Some(id)
    }

    /// Enrich the graph with git history analysis: file activity, co-changes, contributors.
    pub fn enrich_git_history(
        &self,
        path: &str,
        days: u64,
        namespace: Option<&str>,
    ) -> Result<EnrichResult, CodememError> {
        // Run git log
        let output = std::process::Command::new("git")
            .args([
                "-C",
                path,
                "log",
                "--format=COMMIT:%H|%an|%aI",
                "--name-only",
                &format!("--since={days} days ago"),
            ])
            .output()
            .map_err(|e| CodememError::Internal(format!("Failed to run git: {e}")))?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(CodememError::Internal(format!("git log failed: {stderr}")));
        }

        let stdout = String::from_utf8_lossy(&output.stdout);

        // Parse commits
        struct Commit {
            author: String,
            date: chrono::DateTime<chrono::Utc>,
            files: Vec<String>,
        }

        let mut commits: Vec<Commit> = Vec::new();

        for block in stdout.split("COMMIT:").skip(1) {
            let mut lines = block.lines();
            if let Some(header) = lines.next() {
                let parts: Vec<&str> = header.splitn(3, '|').collect();
                if parts.len() >= 3 {
                    let author = parts[1].to_string();
                    let date = chrono::DateTime::parse_from_rfc3339(parts[2])
                        .map(|dt| dt.with_timezone(&chrono::Utc))
                        .unwrap_or_else(|_| chrono::Utc::now());
                    let files: Vec<String> = lines
                        .filter(|l| !l.trim().is_empty())
                        .map(|l| l.trim().to_string())
                        .collect();
                    if !files.is_empty() {
                        commits.push(Commit {
                            author,
                            date,
                            files,
                        });
                    }
                }
            }
        }

        let total_commits = commits.len();

        // Aggregate per-file stats
        struct FileStats {
            commit_count: usize,
            authors: HashSet<String>,
        }

        let mut file_stats: HashMap<String, FileStats> = HashMap::new();
        let mut author_file_count: HashMap<String, usize> = HashMap::new();
        let mut author_commit_count: HashMap<String, usize> = HashMap::new();

        // Co-change tracking with temporal info
        struct CoChangeInfo {
            count: usize,
            earliest: chrono::DateTime<chrono::Utc>,
            latest: chrono::DateTime<chrono::Utc>,
        }
        let mut co_change_info: HashMap<(String, String), CoChangeInfo> = HashMap::new();

        for commit in &commits {
            *author_commit_count
                .entry(commit.author.clone())
                .or_default() += 1;

            for file in &commit.files {
                let stats = file_stats.entry(file.clone()).or_insert(FileStats {
                    commit_count: 0,
                    authors: HashSet::new(),
                });
                stats.commit_count += 1;
                stats.authors.insert(commit.author.clone());
                *author_file_count.entry(commit.author.clone()).or_default() += 1;
            }

            // Track co-changes (pairs of files in same commit)
            // Skip bulk refactor commits (>50 files) to avoid O(N^2) explosion
            let mut sorted_files: Vec<&String> = commit.files.iter().collect();
            if sorted_files.len() > 50 {
                continue;
            }
            sorted_files.sort();
            for i in 0..sorted_files.len() {
                for j in (i + 1)..sorted_files.len() {
                    let key = (sorted_files[i].clone(), sorted_files[j].clone());
                    let entry = co_change_info.entry(key).or_insert(CoChangeInfo {
                        count: 0,
                        earliest: commit.date,
                        latest: commit.date,
                    });
                    entry.count += 1;
                    if commit.date < entry.earliest {
                        entry.earliest = commit.date;
                    }
                    if commit.date > entry.latest {
                        entry.latest = commit.date;
                    }
                }
            }
        }

        // Annotate graph nodes with git stats
        let mut files_annotated = 0;
        {
            let mut graph = self.lock_graph()?;

            for (file_path, stats) in &file_stats {
                let node_id = format!("file:{file_path}");
                if let Ok(Some(mut node)) = graph.get_node(&node_id) {
                    node.payload
                        .insert("git_commit_count".into(), json!(stats.commit_count));
                    node.payload
                        .insert("git_authors".into(), json!(stats.authors));
                    let churn_rate = if days > 0 {
                        stats.commit_count as f64 / (days as f64 / 30.0)
                    } else {
                        0.0
                    };
                    node.payload
                        .insert("git_churn_rate".into(), json!(churn_rate));
                    let _ = graph.add_node(node);
                    files_annotated += 1;
                }
            }
        }

        // Create co-change edges (threshold: 2+ co-occurrences) with temporal data
        let co_change_threshold = 2;
        let mut co_change_edges_created = 0;
        {
            let mut graph = self.lock_graph()?;

            for ((file_a, file_b), info) in &co_change_info {
                if info.count < co_change_threshold {
                    continue;
                }
                let src_id = format!("file:{file_a}");
                let dst_id = format!("file:{file_b}");

                if graph.get_node(&src_id).ok().flatten().is_none()
                    || graph.get_node(&dst_id).ok().flatten().is_none()
                {
                    continue;
                }

                let weight = if total_commits > 0 {
                    info.count as f64 / total_commits as f64
                } else {
                    0.0
                };

                let edge = Edge {
                    id: format!("cochange:{}:{}", file_a, file_b),
                    src: src_id,
                    dst: dst_id,
                    relationship: RelationshipType::CoChanged,
                    weight,
                    properties: HashMap::from([("commit_count".into(), json!(info.count))]),
                    created_at: chrono::Utc::now(),
                    valid_from: Some(info.earliest),
                    valid_to: Some(info.latest),
                };
                let _ = self.storage.insert_graph_edge(&edge);
                if graph.add_edge(edge).is_ok() {
                    co_change_edges_created += 1;
                }
            }
        }

        // Store insights
        let mut insights_stored = 0;

        // High-activity files
        for (file_path, stats) in &file_stats {
            if stats.commit_count > self.config.enrichment.git_min_commit_count {
                let mut sorted_authors: Vec<_> = stats.authors.iter().collect();
                sorted_authors.sort();
                let authors_str = sorted_authors
                    .iter()
                    .map(|s| s.as_str())
                    .collect::<Vec<_>>()
                    .join(", ");
                let content = format!(
                    "High activity: {} — {} commits in the last {} days by {}",
                    file_path, stats.commit_count, days, authors_str
                );
                let importance = (stats.commit_count as f64 / 100.0).clamp(0.2, 0.6);
                if self
                    .store_insight(
                        &content,
                        "activity",
                        &["git-history"],
                        importance,
                        namespace,
                        &[format!("file:{file_path}")],
                    )
                    .is_some()
                {
                    insights_stored += 1;
                }
            }
        }

        // Co-change patterns
        for ((file_a, file_b), info) in &co_change_info {
            if info.count >= self.config.enrichment.git_min_co_change_count {
                let content = format!(
                    "Co-change pattern: {} and {} change together in {} commits — likely coupled",
                    file_a, file_b, info.count
                );
                if self
                    .store_insight(
                        &content,
                        "activity",
                        &["git-history", "coupling"],
                        0.4,
                        namespace,
                        &[format!("file:{file_a}"), format!("file:{file_b}")],
                    )
                    .is_some()
                {
                    insights_stored += 1;
                }
            }
        }

        // Most active contributors
        let mut author_vec: Vec<_> = author_commit_count.iter().collect();
        author_vec.sort_by(|a, b| b.1.cmp(a.1));
        for (author, commit_count) in author_vec.iter().take(3) {
            let file_count = author_file_count.get(*author).unwrap_or(&0);
            let content = format!(
                "Most active contributor: {} with {} commits across {} files",
                author, commit_count, file_count
            );
            if self
                .store_insight(
                    &content,
                    "activity",
                    &["git-history", "contributor"],
                    0.5,
                    namespace,
                    &[],
                )
                .is_some()
            {
                insights_stored += 1;
            }
        }

        self.save_index();

        Ok(EnrichResult {
            insights_stored,
            details: json!({
                "total_commits": total_commits,
                "files_annotated": files_annotated,
                "co_change_edges_created": co_change_edges_created,
                "insights_stored": insights_stored,
                "unique_authors": author_commit_count.len(),
            }),
        })
    }

    /// Enrich the graph with security analysis: sensitive files, endpoints, security functions.
    pub fn enrich_security(&self, namespace: Option<&str>) -> Result<EnrichResult, CodememError> {
        use std::sync::LazyLock;

        static SECURITY_PATTERN: LazyLock<regex::Regex> = LazyLock::new(|| {
            regex::Regex::new(
                r"(?i)(auth|secret|key|password|token|credential|\.env|private|encrypt|decrypt|cert|permission|rbac|oauth|jwt|session|cookie|csrf|xss|injection|sanitize)"
            ).expect("valid regex")
        });

        static SECURITY_FN_PATTERN: LazyLock<regex::Regex> = LazyLock::new(|| {
            regex::Regex::new(
                r"(?i)(hash|verify|sign|encrypt|authenticate|authorize|validate_token|check_permission)"
            ).expect("valid regex")
        });

        let security_pattern = &*SECURITY_PATTERN;
        let security_fn_pattern = &*SECURITY_FN_PATTERN;

        let graph = self.lock_graph()?;
        let all_nodes = graph.get_all_nodes();
        drop(graph);

        let mut sensitive_files: Vec<String> = Vec::new();
        let mut endpoints: Vec<String> = Vec::new();
        let mut security_functions: Vec<(String, String, String)> = Vec::new();
        let mut nodes_to_annotate: Vec<(String, Vec<String>)> = Vec::new();

        for node in &all_nodes {
            match node.kind {
                NodeKind::File => {
                    if security_pattern.is_match(&node.label) {
                        sensitive_files.push(node.label.clone());
                        nodes_to_annotate.push((
                            node.id.clone(),
                            vec!["sensitive".into(), "auth_related".into()],
                        ));
                    }
                }
                NodeKind::Endpoint => {
                    endpoints.push(node.label.clone());
                    nodes_to_annotate.push((node.id.clone(), vec!["exposed_endpoint".into()]));
                }
                NodeKind::Function | NodeKind::Method => {
                    if security_fn_pattern.is_match(&node.label) {
                        let file = node
                            .payload
                            .get("file_path")
                            .and_then(|v| v.as_str())
                            .unwrap_or("unknown")
                            .to_string();
                        security_functions.push((node.label.clone(), file, node.id.clone()));
                        nodes_to_annotate.push((node.id.clone(), vec!["security_function".into()]));
                    }
                }
                _ => {}
            }
        }

        // Annotate nodes with security flags
        {
            let mut graph = self.lock_graph()?;
            for (node_id, flags) in &nodes_to_annotate {
                if let Ok(Some(mut node)) = graph.get_node(node_id) {
                    node.payload.insert("security_flags".into(), json!(flags));
                    let _ = graph.add_node(node);
                }
            }
        }

        // Store insights
        let mut insights_stored = 0;

        for file_path in &sensitive_files {
            let content = format!(
                "Sensitive file: {} — contains security-critical code (auth/credentials)",
                file_path
            );
            if self
                .store_insight(
                    &content,
                    "security",
                    &["severity:high"],
                    0.8,
                    namespace,
                    &[format!("file:{file_path}")],
                )
                .is_some()
            {
                insights_stored += 1;
            }
        }

        if !endpoints.is_empty() {
            let content = format!(
                "{} exposed API endpoints detected — review access controls",
                endpoints.len()
            );
            if self
                .store_insight(
                    &content,
                    "security",
                    &["severity:medium", "endpoints"],
                    0.7,
                    namespace,
                    &[],
                )
                .is_some()
            {
                insights_stored += 1;
            }
        }

        for (name, file, node_id) in &security_functions {
            let content = format!(
                "Security-critical function: {} in {} — ensure proper testing",
                name, file
            );
            if self
                .store_insight(
                    &content,
                    "security",
                    &["severity:medium"],
                    0.6,
                    namespace,
                    std::slice::from_ref(node_id),
                )
                .is_some()
            {
                insights_stored += 1;
            }
        }

        self.save_index();

        Ok(EnrichResult {
            insights_stored,
            details: json!({
                "sensitive_file_count": sensitive_files.len(),
                "endpoint_count": endpoints.len(),
                "security_function_count": security_functions.len(),
                "insights_stored": insights_stored,
            }),
        })
    }

    /// Enrich the graph with performance analysis: coupling, dependency depth, PageRank, complexity.
    pub fn enrich_performance(
        &self,
        top: usize,
        namespace: Option<&str>,
    ) -> Result<EnrichResult, CodememError> {
        // Collect data from graph into local variables, then drop lock
        let all_nodes;
        let mut coupling_data: Vec<(String, String, usize)> = Vec::new();
        let mut high_coupling_count = 0;
        let layers: Vec<Vec<String>>;
        let mut file_pagerank: Vec<(String, String, f64)> = Vec::new();
        {
            let graph = self.lock_graph()?;
            all_nodes = graph.get_all_nodes();

            // 1. Compute coupling (in-degree + out-degree) for each node
            for node in &all_nodes {
                let degree = graph.get_edges(&node.id).map(|e| e.len()).unwrap_or(0);
                coupling_data.push((node.id.clone(), node.label.clone(), degree));
                if degree > self.config.enrichment.perf_min_coupling_degree {
                    high_coupling_count += 1;
                }
            }

            // 2. Compute dependency depth via topological layers
            layers = graph.topological_layers();

            // 3. PageRank for critical path (File nodes only)
            for node in &all_nodes {
                if node.kind == NodeKind::File {
                    let pr = graph.get_pagerank(&node.id);
                    if pr > 0.0 {
                        file_pagerank.push((node.id.clone(), node.label.clone(), pr));
                    }
                }
            }
        }
        // Graph lock released here

        let max_depth = layers.len();
        file_pagerank.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));

        // 4. File complexity from symbol counts (computed from local all_nodes)
        let mut file_symbol_counts: HashMap<String, usize> = HashMap::new();
        for node in &all_nodes {
            match node.kind {
                NodeKind::Function
                | NodeKind::Method
                | NodeKind::Class
                | NodeKind::Interface
                | NodeKind::Type => {
                    if let Some(file_path) = node.payload.get("file_path").and_then(|v| v.as_str())
                    {
                        *file_symbol_counts.entry(file_path.to_string()).or_default() += 1;
                    }
                }
                _ => {}
            }
        }

        // Annotate graph nodes (short lock scope for writes only)
        {
            let mut graph = self.lock_graph()?;

            for (node_id, _label, degree) in &coupling_data {
                if let Ok(Some(mut node)) = graph.get_node(node_id) {
                    node.payload.insert("coupling_score".into(), json!(degree));
                    let _ = graph.add_node(node);
                }
            }

            for (layer_idx, layer) in layers.iter().enumerate() {
                for node_id in layer {
                    if let Ok(Some(mut node)) = graph.get_node(node_id) {
                        node.payload
                            .insert("dependency_layer".into(), json!(layer_idx));
                        let _ = graph.add_node(node);
                    }
                }
            }

            for (node_id, _label, rank) in file_pagerank.iter().take(top) {
                if let Ok(Some(mut node)) = graph.get_node(node_id) {
                    node.payload
                        .insert("critical_path_rank".into(), json!(rank));
                    let _ = graph.add_node(node);
                }
            }

            for (file_path, sym_count) in &file_symbol_counts {
                let node_id = format!("file:{file_path}");
                if let Ok(Some(mut node)) = graph.get_node(&node_id) {
                    node.payload.insert("symbol_count".into(), json!(sym_count));
                    let _ = graph.add_node(node);
                }
            }
        }

        // Store insights
        let mut insights_stored = 0;

        // High-coupling nodes
        coupling_data.sort_by(|a, b| b.2.cmp(&a.2));
        for (node_id, label, degree) in coupling_data.iter().take(top) {
            if *degree > self.config.enrichment.perf_min_coupling_degree {
                let content = format!(
                    "High coupling: {} has {} dependencies — refactoring risk",
                    label, degree
                );
                if self
                    .store_insight(
                        &content,
                        "performance",
                        &["coupling"],
                        0.7,
                        namespace,
                        std::slice::from_ref(node_id),
                    )
                    .is_some()
                {
                    insights_stored += 1;
                }
            }
        }

        // Deep dependency chain
        if max_depth > 5 {
            let content = format!(
                "Deep dependency chain: {} layers — impacts build and test times",
                max_depth
            );
            if self
                .store_insight(
                    &content,
                    "performance",
                    &["dependency-depth"],
                    0.6,
                    namespace,
                    &[],
                )
                .is_some()
            {
                insights_stored += 1;
            }
        }

        // Critical bottleneck (top PageRank file)
        if let Some((node_id, label, _)) = file_pagerank.first() {
            let content = format!(
                "Critical bottleneck: {} — highest centrality file, changes cascade widely",
                label
            );
            if self
                .store_insight(
                    &content,
                    "performance",
                    &["critical-path"],
                    0.8,
                    namespace,
                    std::slice::from_ref(node_id),
                )
                .is_some()
            {
                insights_stored += 1;
            }
        }

        // Complex files (high symbol count)
        let mut complex_files: Vec<_> = file_symbol_counts.iter().collect();
        complex_files.sort_by(|a, b| b.1.cmp(a.1));
        for (file_path, sym_count) in complex_files.iter().take(top) {
            if **sym_count > self.config.enrichment.perf_min_symbol_count {
                let content = format!("Complex file: {} — {} symbols", file_path, sym_count);
                if self
                    .store_insight(
                        &content,
                        "performance",
                        &["complexity"],
                        0.5,
                        namespace,
                        &[format!("file:{file_path}")],
                    )
                    .is_some()
                {
                    insights_stored += 1;
                }
            }
        }

        self.save_index();

        let critical_files: Vec<_> = file_pagerank
            .iter()
            .take(top)
            .map(|(_, label, score)| json!({"file": label, "pagerank": score}))
            .collect();

        Ok(EnrichResult {
            insights_stored,
            details: json!({
                "high_coupling_count": high_coupling_count,
                "max_depth": max_depth,
                "critical_files": critical_files,
                "insights_stored": insights_stored,
            }),
        })
    }

    // ── E1: Cyclomatic/Cognitive Complexity ─────────────────────────────

    /// Enrich the graph with cyclomatic and cognitive complexity metrics for functions/methods.
    ///
    /// For each Function/Method node, reads the source file, counts decision points
    /// (if/else/match/for/while/loop/&&/||) as cyclomatic complexity and measures
    /// max nesting depth as a cognitive complexity proxy. High-complexity functions
    /// (cyclomatic > 10) produce Insight memories.
    pub fn enrich_complexity(&self, namespace: Option<&str>) -> Result<EnrichResult, CodememError> {
        let all_nodes = {
            let graph = self.lock_graph()?;
            graph.get_all_nodes()
        };

        // Collect function/method nodes with file info
        struct SymbolInfo {
            node_id: String,
            label: String,
            file_path: String,
            line_start: usize,
            line_end: usize,
        }

        let mut symbols: Vec<SymbolInfo> = Vec::new();
        for node in &all_nodes {
            if !matches!(node.kind, NodeKind::Function | NodeKind::Method) {
                continue;
            }
            let file_path = match node.payload.get("file_path").and_then(|v| v.as_str()) {
                Some(fp) => fp.to_string(),
                None => continue,
            };
            let line_start = node
                .payload
                .get("line_start")
                .and_then(|v| v.as_u64())
                .unwrap_or(0) as usize;
            let line_end = node
                .payload
                .get("line_end")
                .and_then(|v| v.as_u64())
                .unwrap_or(0) as usize;
            if line_end <= line_start {
                continue;
            }
            symbols.push(SymbolInfo {
                node_id: node.id.clone(),
                label: node.label.clone(),
                file_path,
                line_start,
                line_end,
            });
        }

        // Cache file contents to avoid re-reading
        let mut file_cache: HashMap<String, Vec<String>> = HashMap::new();
        let mut annotated = 0usize;
        let mut insights_stored = 0usize;

        // Nodes to annotate (collected first, then applied in a single lock scope)
        struct ComplexityData {
            node_id: String,
            cyclomatic: usize,
            cognitive: usize,
        }
        let mut complexity_data: Vec<ComplexityData> = Vec::new();

        // Insights to store (collected first, then stored outside the lock)
        struct ComplexityInsight {
            content: String,
            importance: f64,
            node_id: String,
        }
        let mut pending_insights: Vec<ComplexityInsight> = Vec::new();

        for sym in &symbols {
            let lines = file_cache.entry(sym.file_path.clone()).or_insert_with(|| {
                std::fs::read_to_string(&sym.file_path)
                    .unwrap_or_default()
                    .lines()
                    .map(String::from)
                    .collect()
            });

            // Extract the function's lines (1-indexed to 0-indexed)
            let start = sym.line_start.saturating_sub(1);
            let end = sym.line_end.min(lines.len());
            if start >= end {
                continue;
            }
            let fn_lines = &lines[start..end];

            // Count cyclomatic complexity: decision points
            let mut cyclomatic: usize = 1; // base path
            let mut max_depth: usize = 0;
            let mut current_depth: usize = 0;

            for line in fn_lines {
                let trimmed = line.trim();

                // Count decision points
                for keyword in &[
                    "if ", "if(", "else if", "match ", "for ", "for(", "while ", "while(", "loop ",
                    "loop{",
                ] {
                    if trimmed.starts_with(keyword) || trimmed.contains(&format!(" {keyword}")) {
                        cyclomatic += 1;
                        break;
                    }
                }
                // Count logical operators as additional branches
                cyclomatic += trimmed.matches("&&").count();
                cyclomatic += trimmed.matches("||").count();

                // Track nesting depth via braces
                for ch in trimmed.chars() {
                    match ch {
                        '{' => {
                            current_depth += 1;
                            max_depth = max_depth.max(current_depth);
                        }
                        '}' => {
                            current_depth = current_depth.saturating_sub(1);
                        }
                        _ => {}
                    }
                }
            }

            complexity_data.push(ComplexityData {
                node_id: sym.node_id.clone(),
                cyclomatic,
                cognitive: max_depth,
            });
            annotated += 1;

            // High complexity threshold
            if cyclomatic > 10 {
                let importance = if cyclomatic > 20 { 0.9 } else { 0.7 };
                pending_insights.push(ComplexityInsight {
                    content: format!(
                        "High complexity: {} — cyclomatic={}, max_nesting={} in {}",
                        sym.label, cyclomatic, max_depth, sym.file_path
                    ),
                    importance,
                    node_id: sym.node_id.clone(),
                });
            }
        }

        // Annotate graph nodes
        {
            let mut graph = self.lock_graph()?;
            for data in &complexity_data {
                if let Ok(Some(mut node)) = graph.get_node(&data.node_id) {
                    node.payload
                        .insert("cyclomatic_complexity".into(), json!(data.cyclomatic));
                    node.payload
                        .insert("cognitive_complexity".into(), json!(data.cognitive));
                    let _ = graph.add_node(node);
                }
            }
        }

        // Store insights (outside graph lock)
        for insight in &pending_insights {
            if self
                .store_insight(
                    &insight.content,
                    "complexity",
                    &[],
                    insight.importance,
                    namespace,
                    std::slice::from_ref(&insight.node_id),
                )
                .is_some()
            {
                insights_stored += 1;
            }
        }

        self.save_index();

        Ok(EnrichResult {
            insights_stored,
            details: json!({
                "symbols_analyzed": annotated,
                "high_complexity_count": pending_insights.len(),
                "insights_stored": insights_stored,
            }),
        })
    }

    // ── E2: Architecture Inference ──────────────────────────────────────

    /// Infer architectural layers and patterns from the module dependency graph.
    ///
    /// Analyzes IMPORTS/CALLS/DEPENDS_ON edges between modules to detect layering
    /// (e.g., api -> service -> storage) and recognizes common directory patterns
    /// (controllers/, models/, views/, handlers/).
    pub fn enrich_architecture(
        &self,
        namespace: Option<&str>,
    ) -> Result<EnrichResult, CodememError> {
        let all_nodes;
        let mut module_deps: HashMap<String, HashSet<String>> = HashMap::new();

        {
            let graph = self.lock_graph()?;
            all_nodes = graph.get_all_nodes();

            // Build module dependency graph from IMPORTS/CALLS edges
            for node in &all_nodes {
                if !matches!(
                    node.kind,
                    NodeKind::File | NodeKind::Module | NodeKind::Package
                ) {
                    continue;
                }
                if let Ok(edges) = graph.get_edges(&node.id) {
                    for edge in &edges {
                        if !matches!(
                            edge.relationship,
                            RelationshipType::Imports
                                | RelationshipType::Calls
                                | RelationshipType::DependsOn
                        ) {
                            continue;
                        }
                        if edge.src == node.id {
                            module_deps
                                .entry(node.id.clone())
                                .or_default()
                                .insert(edge.dst.clone());
                        }
                    }
                }
            }
        }

        let mut insights_stored = 0;

        // Detect architectural layers by analyzing dependency direction
        // Extract top-level directory from node IDs
        fn top_dir(node_id: &str) -> Option<String> {
            let path = node_id
                .strip_prefix("file:")
                .or_else(|| node_id.strip_prefix("pkg:"))
                .unwrap_or(node_id);
            let parts: Vec<&str> = path.split('/').collect();
            if parts.len() >= 2 {
                Some(parts[0].to_string())
            } else {
                None
            }
        }

        // Build directory-level dependency counts
        let mut dir_deps: HashMap<String, HashSet<String>> = HashMap::new();
        for (src, dsts) in &module_deps {
            if let Some(src_dir) = top_dir(src) {
                for dst in dsts {
                    if let Some(dst_dir) = top_dir(dst) {
                        if src_dir != dst_dir {
                            dir_deps.entry(src_dir.clone()).or_default().insert(dst_dir);
                        }
                    }
                }
            }
        }

        // Detect layers: directories with no incoming deps are "top" layers
        let all_dirs: HashSet<String> = dir_deps
            .keys()
            .chain(dir_deps.values().flat_map(|v| v.iter()))
            .cloned()
            .collect();
        let dirs_with_incoming: HashSet<String> =
            dir_deps.values().flat_map(|v| v.iter()).cloned().collect();
        let top_layers: Vec<&String> = all_dirs
            .iter()
            .filter(|d| !dirs_with_incoming.contains(*d))
            .collect();
        let bottom_layers: Vec<&String> = all_dirs
            .iter()
            .filter(|d| !dir_deps.contains_key(*d))
            .collect();

        if !dir_deps.is_empty() {
            let mut layer_desc = String::new();
            if !top_layers.is_empty() {
                let mut sorted_top: Vec<&&String> = top_layers.iter().collect();
                sorted_top.sort();
                layer_desc.push_str(&format!(
                    "Top-level (entry points): {}",
                    sorted_top
                        .iter()
                        .map(|s| s.as_str())
                        .collect::<Vec<_>>()
                        .join(", ")
                ));
            }
            if !bottom_layers.is_empty() {
                if !layer_desc.is_empty() {
                    layer_desc.push_str("; ");
                }
                let mut sorted_bottom: Vec<&&String> = bottom_layers.iter().collect();
                sorted_bottom.sort();
                layer_desc.push_str(&format!(
                    "Foundation (no outbound deps): {}",
                    sorted_bottom
                        .iter()
                        .map(|s| s.as_str())
                        .collect::<Vec<_>>()
                        .join(", ")
                ));
            }
            let content = format!(
                "Architecture: {} module groups with layered dependencies. {}",
                all_dirs.len(),
                layer_desc
            );
            if self
                .store_insight(&content, "architecture", &[], 0.9, namespace, &[])
                .is_some()
            {
                insights_stored += 1;
            }
        }

        // Detect common architectural patterns from directory names
        let known_patterns = [
            ("controllers", "MVC Controller layer"),
            ("handlers", "Handler/Controller layer"),
            ("models", "Data model layer"),
            ("views", "View/Template layer"),
            ("services", "Service/Business logic layer"),
            ("api", "API layer"),
            ("routes", "Routing layer"),
            ("middleware", "Middleware layer"),
            ("utils", "Utility/Helper layer"),
            ("lib", "Library/Core layer"),
        ];

        let detected: Vec<&str> = known_patterns
            .iter()
            .filter(|(name, _)| {
                all_nodes
                    .iter()
                    .any(|n| n.kind == NodeKind::Package && n.label.contains(name))
            })
            .map(|(_, desc)| *desc)
            .collect();

        if !detected.is_empty() {
            let content = format!("Architecture patterns detected: {}", detected.join(", "));
            if self
                .store_insight(&content, "architecture", &[], 0.7, namespace, &[])
                .is_some()
            {
                insights_stored += 1;
            }
        }

        // Detect circular dependencies between directories
        for (dir, deps) in &dir_deps {
            for dep in deps {
                if let Some(back_deps) = dir_deps.get(dep) {
                    if back_deps.contains(dir) && dir < dep {
                        let content = format!(
                            "Circular dependency: {} and {} depend on each other — consider refactoring",
                            dir, dep
                        );
                        if self
                            .store_insight(
                                &content,
                                "architecture",
                                &["circular-dep"],
                                0.8,
                                namespace,
                                &[],
                            )
                            .is_some()
                        {
                            insights_stored += 1;
                        }
                    }
                }
            }
        }

        self.save_index();

        Ok(EnrichResult {
            insights_stored,
            details: json!({
                "module_count": all_dirs.len(),
                "dependency_edges": module_deps.values().map(|v| v.len()).sum::<usize>(),
                "top_layers": top_layers.len(),
                "bottom_layers": bottom_layers.len(),
                "patterns_detected": detected.len(),
                "insights_stored": insights_stored,
            }),
        })
    }

    // ── E3: Test-to-Code Mapping ────────────────────────────────────────

    /// Map test functions to the code they test and identify untested public functions.
    ///
    /// For Test-kind nodes, infers tested symbols by naming convention (`test_foo` -> `foo`)
    /// and by CALLS edges. Creates RELATES_TO edges between test and tested symbols.
    /// Produces Insight memories for files with untested public functions.
    pub fn enrich_test_mapping(
        &self,
        namespace: Option<&str>,
    ) -> Result<EnrichResult, CodememError> {
        let all_nodes;
        let mut test_edges_info: Vec<(String, String)> = Vec::new();

        {
            let graph = self.lock_graph()?;
            all_nodes = graph.get_all_nodes();

            // Collect test nodes and non-test function/method nodes
            let test_nodes: Vec<&GraphNode> = all_nodes
                .iter()
                .filter(|n| n.kind == NodeKind::Test)
                .collect();
            // Index by simple name (last segment of qualified name)
            let mut fn_by_simple_name: HashMap<String, Vec<&GraphNode>> = HashMap::new();
            for node in all_nodes
                .iter()
                .filter(|n| matches!(n.kind, NodeKind::Function | NodeKind::Method))
            {
                let simple = node
                    .label
                    .rsplit("::")
                    .next()
                    .unwrap_or(&node.label)
                    .to_string();
                fn_by_simple_name.entry(simple).or_default().push(node);
            }

            for test_node in &test_nodes {
                // Extract what this test might be testing from its name
                let test_name = test_node
                    .label
                    .rsplit("::")
                    .next()
                    .unwrap_or(&test_node.label);

                // Convention: test_foo tests foo, test_foo_bar tests foo_bar
                let tested_name = test_name
                    .strip_prefix("test_")
                    .or_else(|| test_name.strip_prefix("test"))
                    .unwrap_or("");

                if !tested_name.is_empty() {
                    // Check by simple name
                    if let Some(targets) = fn_by_simple_name.get(tested_name) {
                        for target in targets {
                            test_edges_info.push((test_node.id.clone(), target.id.clone()));
                        }
                    }
                }

                // Also check CALLS edges from the test to find tested symbols
                if let Ok(edges) = graph.get_edges(&test_node.id) {
                    for edge in &edges {
                        if edge.relationship == RelationshipType::Calls && edge.src == test_node.id
                        {
                            // Only link to function/method nodes
                            if let Ok(Some(dst_node)) = graph.get_node(&edge.dst) {
                                if matches!(dst_node.kind, NodeKind::Function | NodeKind::Method) {
                                    test_edges_info
                                        .push((test_node.id.clone(), dst_node.id.clone()));
                                }
                            }
                        }
                    }
                }
            }
        }

        // Dedup edges
        let unique_edges: HashSet<(String, String)> = test_edges_info.into_iter().collect();

        // Create RELATES_TO edges for test mappings
        let mut edges_created = 0;
        {
            let mut graph = self.lock_graph()?;
            let now = chrono::Utc::now();
            for (test_id, target_id) in &unique_edges {
                let edge_id = format!("test-map:{test_id}->{target_id}");
                // Skip if edge already exists
                if graph.get_node(test_id).ok().flatten().is_none()
                    || graph.get_node(target_id).ok().flatten().is_none()
                {
                    continue;
                }
                let edge = Edge {
                    id: edge_id,
                    src: test_id.clone(),
                    dst: target_id.clone(),
                    relationship: RelationshipType::RelatesTo,
                    weight: 0.8,
                    properties: HashMap::from([("test_mapping".into(), json!(true))]),
                    created_at: now,
                    valid_from: None,
                    valid_to: None,
                };
                let _ = self.storage.insert_graph_edge(&edge);
                if graph.add_edge(edge).is_ok() {
                    edges_created += 1;
                }
            }
        }

        // Identify untested public functions per file
        let tested_ids: HashSet<String> = unique_edges.iter().map(|(_, t)| t.clone()).collect();
        let mut untested_by_file: HashMap<String, Vec<String>> = HashMap::new();

        for node in &all_nodes {
            if !matches!(node.kind, NodeKind::Function | NodeKind::Method) {
                continue;
            }
            let visibility = node
                .payload
                .get("visibility")
                .and_then(|v| v.as_str())
                .unwrap_or("private");
            if visibility != "public" {
                continue;
            }
            if tested_ids.contains(&node.id) {
                continue;
            }
            let file_path = node
                .payload
                .get("file_path")
                .and_then(|v| v.as_str())
                .unwrap_or("unknown")
                .to_string();
            untested_by_file
                .entry(file_path)
                .or_default()
                .push(node.label.clone());
        }

        let mut insights_stored = 0;
        for (file_path, untested) in &untested_by_file {
            if untested.is_empty() {
                continue;
            }
            let names: Vec<&str> = untested.iter().take(10).map(|s| s.as_str()).collect();
            let suffix = if untested.len() > 10 {
                format!(" (and {} more)", untested.len() - 10)
            } else {
                String::new()
            };
            let content = format!(
                "Untested public functions in {}: {}{}",
                file_path,
                names.join(", "),
                suffix
            );
            if self
                .store_insight(
                    &content,
                    "testing",
                    &[],
                    0.6,
                    namespace,
                    &[format!("file:{file_path}")],
                )
                .is_some()
            {
                insights_stored += 1;
            }
        }

        self.save_index();

        Ok(EnrichResult {
            insights_stored,
            details: json!({
                "test_edges_created": edges_created,
                "files_with_untested": untested_by_file.len(),
                "insights_stored": insights_stored,
            }),
        })
    }

    // ── E4: API Surface Analysis ────────────────────────────────────────

    /// Analyze the public API surface of each module/package.
    ///
    /// Counts public vs private symbols per file and stores Insight memories
    /// summarizing the public API.
    pub fn enrich_api_surface(
        &self,
        namespace: Option<&str>,
    ) -> Result<EnrichResult, CodememError> {
        let all_nodes = {
            let graph = self.lock_graph()?;
            graph.get_all_nodes()
        };

        // Count public vs private symbols per file
        struct ApiStats {
            public: Vec<String>,
            private_count: usize,
        }
        let mut file_api: HashMap<String, ApiStats> = HashMap::new();

        for node in &all_nodes {
            if !matches!(
                node.kind,
                NodeKind::Function
                    | NodeKind::Method
                    | NodeKind::Class
                    | NodeKind::Interface
                    | NodeKind::Type
                    | NodeKind::Constant
            ) {
                continue;
            }
            let file_path = match node.payload.get("file_path").and_then(|v| v.as_str()) {
                Some(fp) => fp.to_string(),
                None => continue,
            };
            let visibility = node
                .payload
                .get("visibility")
                .and_then(|v| v.as_str())
                .unwrap_or("private");

            let stats = file_api.entry(file_path).or_insert(ApiStats {
                public: Vec::new(),
                private_count: 0,
            });
            if visibility == "public" {
                stats.public.push(node.label.clone());
            } else {
                stats.private_count += 1;
            }
        }

        let mut insights_stored = 0;
        let mut total_public = 0usize;
        let mut total_private = 0usize;

        for (file_path, stats) in &file_api {
            total_public += stats.public.len();
            total_private += stats.private_count;

            if stats.public.is_empty() {
                continue;
            }
            let names: Vec<&str> = stats.public.iter().take(15).map(|s| s.as_str()).collect();
            let suffix = if stats.public.len() > 15 {
                format!(" (and {} more)", stats.public.len() - 15)
            } else {
                String::new()
            };
            let ratio = stats.public.len() as f64
                / (stats.public.len() + stats.private_count).max(1) as f64;
            let content = format!(
                "API surface: {} — {} public, {} private (ratio {:.0}%). Exports: {}{}",
                file_path,
                stats.public.len(),
                stats.private_count,
                ratio * 100.0,
                names.join(", "),
                suffix
            );
            let importance = if ratio > 0.8 { 0.6 } else { 0.4 };
            if self
                .store_insight(
                    &content,
                    "api",
                    &[],
                    importance,
                    namespace,
                    &[format!("file:{file_path}")],
                )
                .is_some()
            {
                insights_stored += 1;
            }
        }

        self.save_index();

        Ok(EnrichResult {
            insights_stored,
            details: json!({
                "files_analyzed": file_api.len(),
                "total_public_symbols": total_public,
                "total_private_symbols": total_private,
                "insights_stored": insights_stored,
            }),
        })
    }

    // ── E5: Documentation Coverage ──────────────────────────────────────

    /// Analyze documentation coverage for public symbols.
    ///
    /// Checks if each public symbol has a non-empty `doc_comment` in its payload.
    /// Stores Insight memories for files with low documentation coverage.
    pub fn enrich_doc_coverage(
        &self,
        namespace: Option<&str>,
    ) -> Result<EnrichResult, CodememError> {
        let all_nodes = {
            let graph = self.lock_graph()?;
            graph.get_all_nodes()
        };

        struct DocStats {
            documented: usize,
            undocumented: Vec<String>,
        }
        let mut file_docs: HashMap<String, DocStats> = HashMap::new();

        for node in &all_nodes {
            if !matches!(
                node.kind,
                NodeKind::Function
                    | NodeKind::Method
                    | NodeKind::Class
                    | NodeKind::Interface
                    | NodeKind::Type
            ) {
                continue;
            }
            let visibility = node
                .payload
                .get("visibility")
                .and_then(|v| v.as_str())
                .unwrap_or("private");
            if visibility != "public" {
                continue;
            }
            let file_path = match node.payload.get("file_path").and_then(|v| v.as_str()) {
                Some(fp) => fp.to_string(),
                None => continue,
            };
            let has_doc = node
                .payload
                .get("doc_comment")
                .and_then(|v| v.as_str())
                .map(|s| !s.trim().is_empty())
                .unwrap_or(false);

            let stats = file_docs.entry(file_path).or_insert(DocStats {
                documented: 0,
                undocumented: Vec::new(),
            });
            if has_doc {
                stats.documented += 1;
            } else {
                stats.undocumented.push(node.label.clone());
            }
        }

        let mut insights_stored = 0;
        let mut total_documented = 0usize;
        let mut total_undocumented = 0usize;

        for (file_path, stats) in &file_docs {
            total_documented += stats.documented;
            total_undocumented += stats.undocumented.len();

            let total = stats.documented + stats.undocumented.len();
            if total == 0 {
                continue;
            }
            let coverage = stats.documented as f64 / total as f64;
            if coverage < 0.5 && !stats.undocumented.is_empty() {
                let names: Vec<&str> = stats
                    .undocumented
                    .iter()
                    .take(10)
                    .map(|s| s.as_str())
                    .collect();
                let suffix = if stats.undocumented.len() > 10 {
                    format!(" (and {} more)", stats.undocumented.len() - 10)
                } else {
                    String::new()
                };
                let content = format!(
                    "Undocumented public API: {} — {:.0}% coverage ({}/{} documented). Missing: {}{}",
                    file_path,
                    coverage * 100.0,
                    stats.documented,
                    total,
                    names.join(", "),
                    suffix
                );
                let importance = if coverage < 0.2 { 0.7 } else { 0.5 };
                if self
                    .store_insight(
                        &content,
                        "documentation",
                        &[],
                        importance,
                        namespace,
                        &[format!("file:{file_path}")],
                    )
                    .is_some()
                {
                    insights_stored += 1;
                }
            }
        }

        self.save_index();

        let total = total_documented + total_undocumented;
        let overall_coverage = if total > 0 {
            total_documented as f64 / total as f64
        } else {
            1.0
        };

        Ok(EnrichResult {
            insights_stored,
            details: json!({
                "files_analyzed": file_docs.len(),
                "total_public_documented": total_documented,
                "total_public_undocumented": total_undocumented,
                "overall_coverage": format!("{:.1}%", overall_coverage * 100.0),
                "insights_stored": insights_stored,
            }),
        })
    }

    // ── E6: Change Impact Prediction ────────────────────────────────────

    /// Predict the impact of changes to a given file by combining co-change edges,
    /// call graph edges, and test file associations.
    pub fn enrich_change_impact(
        &self,
        file_path: &str,
        namespace: Option<&str>,
    ) -> Result<EnrichResult, CodememError> {
        let graph = self.lock_graph()?;

        let file_id = format!("file:{file_path}");
        if graph.get_node(&file_id).ok().flatten().is_none() {
            return Err(CodememError::NotFound(format!(
                "File node not found: {file_path}"
            )));
        }

        let mut co_changed: Vec<String> = Vec::new();
        let mut callers: Vec<String> = Vec::new();
        let mut callees: Vec<String> = Vec::new();
        let mut test_files: Vec<String> = Vec::new();

        // Get edges for the file node
        if let Ok(edges) = graph.get_edges(&file_id) {
            for edge in &edges {
                match edge.relationship {
                    RelationshipType::CoChanged => {
                        let other = if edge.src == file_id {
                            &edge.dst
                        } else {
                            &edge.src
                        };
                        if let Some(path) = other.strip_prefix("file:") {
                            co_changed.push(path.to_string());
                        }
                    }
                    RelationshipType::Calls => {
                        let other = if edge.src == file_id {
                            callees.push(edge.dst.clone());
                            &edge.dst
                        } else {
                            callers.push(edge.src.clone());
                            &edge.src
                        };
                        let _ = other;
                    }
                    RelationshipType::RelatesTo => {
                        // Check if this is a test mapping edge
                        if edge.properties.contains_key("test_mapping") {
                            let other = if edge.src == file_id {
                                &edge.dst
                            } else {
                                &edge.src
                            };
                            if let Ok(Some(node)) = graph.get_node(other) {
                                if node.kind == NodeKind::Test {
                                    if let Some(fp) =
                                        node.payload.get("file_path").and_then(|v| v.as_str())
                                    {
                                        test_files.push(fp.to_string());
                                    }
                                }
                            }
                        }
                    }
                    _ => {}
                }
            }
        }

        // Also check symbols contained in this file for their callers
        let all_nodes = graph.get_all_nodes();
        for node in &all_nodes {
            if !matches!(node.kind, NodeKind::Function | NodeKind::Method) {
                continue;
            }
            let sym_file = node
                .payload
                .get("file_path")
                .and_then(|v| v.as_str())
                .unwrap_or("");
            if sym_file != file_path {
                continue;
            }
            if let Ok(edges) = graph.get_edges(&node.id) {
                for edge in &edges {
                    if edge.relationship == RelationshipType::Calls && edge.dst == node.id {
                        // Something calls this symbol
                        if let Ok(Some(caller_node)) = graph.get_node(&edge.src) {
                            if let Some(fp) = caller_node
                                .payload
                                .get("file_path")
                                .and_then(|v| v.as_str())
                            {
                                if fp != file_path {
                                    callers.push(fp.to_string());
                                }
                            }
                        }
                    }
                }
            }
        }

        drop(graph);

        // Dedup
        co_changed.sort();
        co_changed.dedup();
        callers.sort();
        callers.dedup();
        callees.sort();
        callees.dedup();
        test_files.sort();
        test_files.dedup();

        let impact_score = co_changed.len() + callers.len() + callees.len();

        let mut insights_stored = 0;

        if impact_score > 0 {
            let mut parts: Vec<String> = Vec::new();
            if !callers.is_empty() {
                parts.push(format!(
                    "{} caller files ({})",
                    callers.len(),
                    callers
                        .iter()
                        .take(5)
                        .cloned()
                        .collect::<Vec<_>>()
                        .join(", ")
                ));
            }
            if !co_changed.is_empty() {
                parts.push(format!(
                    "{} co-changed files ({})",
                    co_changed.len(),
                    co_changed
                        .iter()
                        .take(5)
                        .cloned()
                        .collect::<Vec<_>>()
                        .join(", ")
                ));
            }
            if !test_files.is_empty() {
                parts.push(format!(
                    "{} test files ({})",
                    test_files.len(),
                    test_files.join(", ")
                ));
            }
            let content = format!("Change impact for {}: {}", file_path, parts.join("; "));
            let importance = (impact_score as f64 / 20.0).clamp(0.4, 0.9);
            if self
                .store_insight(&content, "impact", &[], importance, namespace, &[file_id])
                .is_some()
            {
                insights_stored += 1;
            }
        }

        self.save_index();

        Ok(EnrichResult {
            insights_stored,
            details: json!({
                "file": file_path,
                "callers": callers.len(),
                "callees": callees.len(),
                "co_changed": co_changed.len(),
                "test_files": test_files.len(),
                "impact_score": impact_score,
                "insights_stored": insights_stored,
            }),
        })
    }

    // ── E7: Code Smell Detection ────────────────────────────────────────

    /// Detect common code smells: long functions (>50 lines), too many parameters (>5),
    /// deep nesting (>4 levels), and long files (>500 lines).
    ///
    /// Stores findings as Pattern memories with importance 0.5.
    pub fn enrich_code_smells(
        &self,
        namespace: Option<&str>,
    ) -> Result<EnrichResult, CodememError> {
        let all_nodes = {
            let graph = self.lock_graph()?;
            graph.get_all_nodes()
        };

        let mut smells_stored = 0;

        // Check functions/methods for long bodies and deep nesting
        let mut file_cache: HashMap<String, Vec<String>> = HashMap::new();

        for node in &all_nodes {
            if !matches!(node.kind, NodeKind::Function | NodeKind::Method) {
                continue;
            }
            let file_path = match node.payload.get("file_path").and_then(|v| v.as_str()) {
                Some(fp) => fp.to_string(),
                None => continue,
            };
            let line_start = node
                .payload
                .get("line_start")
                .and_then(|v| v.as_u64())
                .unwrap_or(0) as usize;
            let line_end = node
                .payload
                .get("line_end")
                .and_then(|v| v.as_u64())
                .unwrap_or(0) as usize;

            let fn_length = line_end.saturating_sub(line_start);

            // Long function (>50 lines)
            if fn_length > 50 {
                let content = format!(
                    "Code smell: Long function {} ({} lines) in {} — consider splitting",
                    node.label, fn_length, file_path
                );
                if self
                    .store_pattern_memory(&content, namespace, std::slice::from_ref(&node.id))
                    .is_some()
                {
                    smells_stored += 1;
                }
            }

            // Check parameter count from signature
            let signature = node
                .payload
                .get("signature")
                .and_then(|v| v.as_str())
                .unwrap_or("");
            if let Some(params_str) = signature
                .split('(')
                .nth(1)
                .and_then(|s| s.split(')').next())
            {
                let param_count = if params_str.trim().is_empty() {
                    0
                } else {
                    params_str.split(',').count()
                };
                if param_count > 5 {
                    let content = format!(
                        "Code smell: {} has {} parameters in {} — consider using a struct",
                        node.label, param_count, file_path
                    );
                    if self
                        .store_pattern_memory(&content, namespace, std::slice::from_ref(&node.id))
                        .is_some()
                    {
                        smells_stored += 1;
                    }
                }
            }

            // Check nesting depth
            if fn_length > 0 {
                let lines = file_cache.entry(file_path.clone()).or_insert_with(|| {
                    std::fs::read_to_string(&file_path)
                        .unwrap_or_default()
                        .lines()
                        .map(String::from)
                        .collect()
                });

                let start = line_start.saturating_sub(1);
                let end = line_end.min(lines.len());
                if start < end {
                    let mut max_depth = 0usize;
                    let mut depth = 0usize;
                    for line in &lines[start..end] {
                        for ch in line.chars() {
                            match ch {
                                '{' => {
                                    depth += 1;
                                    max_depth = max_depth.max(depth);
                                }
                                '}' => depth = depth.saturating_sub(1),
                                _ => {}
                            }
                        }
                    }
                    if max_depth > 4 {
                        let content = format!(
                            "Code smell: Deep nesting ({} levels) in {} in {} — consider extracting",
                            max_depth, node.label, file_path
                        );
                        if self
                            .store_pattern_memory(
                                &content,
                                namespace,
                                std::slice::from_ref(&node.id),
                            )
                            .is_some()
                        {
                            smells_stored += 1;
                        }
                    }
                }
            }
        }

        // Check for long files (>500 lines)
        for node in &all_nodes {
            if node.kind != NodeKind::File {
                continue;
            }
            let file_path = &node.label;
            let line_count = file_cache
                .get(file_path)
                .map(|lines| lines.len())
                .unwrap_or_else(|| {
                    std::fs::read_to_string(file_path)
                        .map(|s| s.lines().count())
                        .unwrap_or(0)
                });
            if line_count > 500 {
                let content = format!(
                    "Code smell: Long file {} ({} lines) — consider splitting into modules",
                    file_path, line_count
                );
                if self
                    .store_pattern_memory(&content, namespace, std::slice::from_ref(&node.id))
                    .is_some()
                {
                    smells_stored += 1;
                }
            }
        }

        self.save_index();

        Ok(EnrichResult {
            insights_stored: smells_stored,
            details: json!({
                "smells_detected": smells_stored,
            }),
        })
    }

    /// Store a Pattern memory for code smell detection (E7).
    /// Importance is fixed at 0.5 for code smells.
    fn store_pattern_memory(
        &self,
        content: &str,
        namespace: Option<&str>,
        links: &[String],
    ) -> Option<String> {
        let hash = codemem_storage::Storage::content_hash(content);
        let now = chrono::Utc::now();
        let id = uuid::Uuid::new_v4().to_string();
        let tags = vec![
            "static-analysis".to_string(),
            "track:code-smell".to_string(),
        ];

        let memory = MemoryNode {
            id: id.clone(),
            content: content.to_string(),
            memory_type: MemoryType::Pattern,
            importance: 0.5,
            confidence: self.config.enrichment.insight_confidence,
            access_count: 0,
            content_hash: hash,
            tags,
            metadata: HashMap::from([
                ("track".into(), json!("code-smell")),
                ("generated_by".into(), json!("enrichment_pipeline")),
            ]),
            namespace: namespace.map(String::from),
            created_at: now,
            updated_at: now,
            last_accessed_at: now,
        };

        if self.storage.insert_memory(&memory).is_err() {
            return None;
        }

        // Minimal pipeline: BM25 + graph node + links
        if let Ok(mut bm25) = self.lock_bm25() {
            bm25.add_document(&id, content);
        }

        let graph_node = GraphNode {
            id: id.clone(),
            kind: NodeKind::Memory,
            label: truncate_content(content, 80),
            payload: HashMap::new(),
            centrality: 0.0,
            memory_id: Some(id.clone()),
            namespace: namespace.map(String::from),
        };
        let _ = self.storage.insert_graph_node(&graph_node);
        if let Ok(mut graph) = self.lock_graph() {
            let _ = graph.add_node(graph_node);

            for link_id in links {
                let edge = Edge {
                    id: format!("{id}-RELATES_TO-{link_id}"),
                    src: id.clone(),
                    dst: link_id.clone(),
                    relationship: RelationshipType::RelatesTo,
                    weight: 0.3,
                    properties: HashMap::new(),
                    created_at: now,
                    valid_from: None,
                    valid_to: None,
                };
                let _ = self.storage.insert_graph_edge(&edge);
                let _ = graph.add_edge(edge);
            }
        }

        self.auto_link_to_code_nodes(&id, content, links);

        Some(id)
    }

    // ── E8: Hot+Complex Correlation ─────────────────────────────────────

    /// Cross-reference git churn with complexity to find high-risk files.
    ///
    /// Files that are BOTH high-churn AND high-complexity represent the highest
    /// maintenance risk. Requires E1 (complexity) and git enrichment to have run first.
    pub fn enrich_hot_complex(
        &self,
        namespace: Option<&str>,
    ) -> Result<EnrichResult, CodememError> {
        let all_nodes = {
            let graph = self.lock_graph()?;
            graph.get_all_nodes()
        };

        // Find files with git churn data
        let mut file_churn: HashMap<String, f64> = HashMap::new();
        for node in &all_nodes {
            if node.kind != NodeKind::File {
                continue;
            }
            if let Some(churn) = node.payload.get("git_churn_rate").and_then(|v| v.as_f64()) {
                if churn > 0.0 {
                    file_churn.insert(node.label.clone(), churn);
                }
            }
        }

        // Find functions with high complexity and aggregate per file
        let mut file_max_complexity: HashMap<String, (usize, String)> = HashMap::new();
        for node in &all_nodes {
            if !matches!(node.kind, NodeKind::Function | NodeKind::Method) {
                continue;
            }
            let cyclomatic = node
                .payload
                .get("cyclomatic_complexity")
                .and_then(|v| v.as_u64())
                .unwrap_or(0) as usize;
            if cyclomatic <= 5 {
                continue;
            }
            let file_path = match node.payload.get("file_path").and_then(|v| v.as_str()) {
                Some(fp) => fp.to_string(),
                None => continue,
            };
            let entry = file_max_complexity
                .entry(file_path)
                .or_insert((0, String::new()));
            if cyclomatic > entry.0 {
                *entry = (cyclomatic, node.label.clone());
            }
        }

        let mut insights_stored = 0;
        let mut hot_complex_files: Vec<serde_json::Value> = Vec::new();

        for (file_path, churn) in &file_churn {
            if let Some((complexity, fn_name)) = file_max_complexity.get(file_path) {
                // Both high churn and high complexity
                hot_complex_files.push(json!({
                    "file": file_path,
                    "churn_rate": churn,
                    "max_complexity": complexity,
                    "complex_function": fn_name,
                }));

                let content = format!(
                    "High-risk file: {} — churn rate {:.1} + max cyclomatic complexity {} (in {}). \
                     Prioritize refactoring",
                    file_path, churn, complexity, fn_name
                );
                if self
                    .store_insight(
                        &content,
                        "risk",
                        &["hot-complex"],
                        0.9,
                        namespace,
                        &[format!("file:{file_path}")],
                    )
                    .is_some()
                {
                    insights_stored += 1;
                }
            }
        }

        self.save_index();

        Ok(EnrichResult {
            insights_stored,
            details: json!({
                "hot_complex_files": hot_complex_files.len(),
                "files_with_churn": file_churn.len(),
                "files_with_complexity": file_max_complexity.len(),
                "insights_stored": insights_stored,
            }),
        })
    }

    // ── E9: Blame/Ownership Enrichment ──────────────────────────────────

    /// Enrich file nodes with primary owner and contributors from git blame.
    pub fn enrich_blame(
        &self,
        path: &str,
        namespace: Option<&str>,
    ) -> Result<EnrichResult, CodememError> {
        let file_nodes: Vec<String> = {
            let graph = self.lock_graph()?;
            graph
                .get_all_nodes()
                .into_iter()
                .filter(|n| n.kind == NodeKind::File)
                .map(|n| n.label.clone())
                .collect()
        };

        let mut files_annotated = 0;
        let mut insights_stored = 0;

        for file_path in &file_nodes {
            let output = std::process::Command::new("git")
                .args(["-C", path, "log", "--format=%an", "--", file_path])
                .output();

            let output = match output {
                Ok(o) if o.status.success() => o,
                _ => continue,
            };

            let stdout = String::from_utf8_lossy(&output.stdout);
            let mut author_counts: HashMap<String, usize> = HashMap::new();
            for line in stdout.lines() {
                let author = line.trim();
                if !author.is_empty() {
                    *author_counts.entry(author.to_string()).or_default() += 1;
                }
            }

            if author_counts.is_empty() {
                continue;
            }

            let mut sorted_authors: Vec<_> = author_counts.into_iter().collect();
            sorted_authors.sort_by(|a, b| b.1.cmp(&a.1));

            let primary_owner = sorted_authors[0].0.clone();
            let contributors: Vec<String> = sorted_authors.iter().map(|(a, _)| a.clone()).collect();

            // Annotate graph node
            let node_id = format!("file:{file_path}");
            {
                let mut graph = self.lock_graph()?;
                if let Ok(Some(mut node)) = graph.get_node(&node_id) {
                    node.payload
                        .insert("primary_owner".into(), json!(primary_owner));
                    node.payload
                        .insert("contributors".into(), json!(contributors));
                    let _ = graph.add_node(node);
                    files_annotated += 1;
                }
            }
        }

        // Store summary insight for files with many contributors (potential ownership ambiguity)
        let graph = self.lock_graph()?;
        for node in &graph.get_all_nodes() {
            if node.kind != NodeKind::File {
                continue;
            }
            if let Some(contribs) = node.payload.get("contributors").and_then(|v| v.as_array()) {
                if contribs.len() > 5 {
                    let primary = node
                        .payload
                        .get("primary_owner")
                        .and_then(|v| v.as_str())
                        .unwrap_or("unknown");
                    let content = format!(
                        "Shared ownership: {} has {} contributors (primary: {}) — may need clear ownership",
                        node.label, contribs.len(), primary
                    );
                    drop(graph);
                    if self
                        .store_insight(
                            &content,
                            "ownership",
                            &[],
                            0.5,
                            namespace,
                            std::slice::from_ref(&node.id),
                        )
                        .is_some()
                    {
                        insights_stored += 1;
                    }
                    // Re-acquire for next iteration — but we break here to avoid lock issues
                    break;
                }
            }
        }

        self.save_index();

        Ok(EnrichResult {
            insights_stored,
            details: json!({
                "files_annotated": files_annotated,
                "insights_stored": insights_stored,
            }),
        })
    }

    // ── E10: Enhanced Security Scanning ─────────────────────────────────

    /// Scan actual file contents for security issues: hardcoded credentials,
    /// SQL concatenation, unsafe blocks, etc.
    pub fn enrich_security_scan(
        &self,
        namespace: Option<&str>,
    ) -> Result<EnrichResult, CodememError> {
        use std::sync::LazyLock;

        static CREDENTIAL_PATTERN: LazyLock<regex::Regex> = LazyLock::new(|| {
            regex::Regex::new(
                r#"(?i)(password|secret|api_key|apikey|token|private_key)\s*[:=]\s*["'][^"']{8,}["']"#,
            )
            .expect("valid regex")
        });

        static SQL_CONCAT_PATTERN: LazyLock<regex::Regex> = LazyLock::new(|| {
            regex::Regex::new(
                r#"(?i)(SELECT|INSERT|UPDATE|DELETE|DROP)\s+.*\+\s*[a-zA-Z_]|format!\s*\(\s*"[^"]*(?:SELECT|INSERT|UPDATE|DELETE)"#,
            )
            .expect("valid regex")
        });

        static UNSAFE_PATTERN: LazyLock<regex::Regex> =
            LazyLock::new(|| regex::Regex::new(r"unsafe\s*\{").expect("valid regex"));

        let file_nodes: Vec<String> = {
            let graph = self.lock_graph()?;
            graph
                .get_all_nodes()
                .into_iter()
                .filter(|n| n.kind == NodeKind::File)
                .map(|n| n.label.clone())
                .collect()
        };

        let mut insights_stored = 0;
        let mut files_scanned = 0;

        for file_path in &file_nodes {
            let content = match std::fs::read_to_string(file_path) {
                Ok(c) => c,
                Err(_) => continue,
            };
            files_scanned += 1;

            // Check for hardcoded credentials
            if CREDENTIAL_PATTERN.is_match(&content) {
                let text = format!(
                    "Security: Potential hardcoded credential in {} — use environment variables or secrets manager",
                    file_path
                );
                if self
                    .store_insight(
                        &text,
                        "security",
                        &["severity:critical", "credentials"],
                        0.95,
                        namespace,
                        &[format!("file:{file_path}")],
                    )
                    .is_some()
                {
                    insights_stored += 1;
                }
            }

            // Check for SQL concatenation
            if SQL_CONCAT_PATTERN.is_match(&content) {
                let text = format!(
                    "Security: Potential SQL injection in {} — use parameterized queries",
                    file_path
                );
                if self
                    .store_insight(
                        &text,
                        "security",
                        &["severity:critical", "sql-injection"],
                        0.9,
                        namespace,
                        &[format!("file:{file_path}")],
                    )
                    .is_some()
                {
                    insights_stored += 1;
                }
            }

            // Check for unsafe blocks (Rust-specific)
            if file_path.ends_with(".rs") {
                let unsafe_count = UNSAFE_PATTERN.find_iter(&content).count();
                if unsafe_count > 0 {
                    let text = format!(
                        "Security: {} unsafe block(s) in {} — review for memory safety",
                        unsafe_count, file_path
                    );
                    let importance = if unsafe_count > 3 { 0.8 } else { 0.6 };
                    if self
                        .store_insight(
                            &text,
                            "security",
                            &["severity:medium", "unsafe"],
                            importance,
                            namespace,
                            &[format!("file:{file_path}")],
                        )
                        .is_some()
                    {
                        insights_stored += 1;
                    }
                }
            }
        }

        self.save_index();

        Ok(EnrichResult {
            insights_stored,
            details: json!({
                "files_scanned": files_scanned,
                "insights_stored": insights_stored,
            }),
        })
    }

    // ── E11: Quality Stratification ─────────────────────────────────────

    /// Categorize existing enrichment insights by signal strength and adjust importance.
    ///
    /// - Noise (< 0.3): basic counts, minor observations
    /// - Signal (0.5-0.7): moderate complexity, useful patterns
    /// - Critical (0.8-1.0): high-risk findings, security issues
    pub fn enrich_quality_stratification(
        &self,
        namespace: Option<&str>,
    ) -> Result<EnrichResult, CodememError> {
        // Query all static-analysis memories
        let all_ids = self.storage.list_memory_ids().unwrap_or_default();
        let id_refs: Vec<&str> = all_ids.iter().map(|s| s.as_str()).collect();
        let memories = self
            .storage
            .get_memories_batch(&id_refs)
            .unwrap_or_default();

        let mut noise_count = 0usize;
        let mut signal_count = 0usize;
        let mut critical_count = 0usize;
        let mut reclassified = 0usize;

        for memory in &memories {
            if !memory.tags.contains(&"static-analysis".to_string()) {
                continue;
            }
            // Apply namespace filter if specified
            if let Some(ns) = namespace {
                if memory.namespace.as_deref() != Some(ns) {
                    continue;
                }
            }

            let current_importance = memory.importance;
            let content_lower = memory.content.to_lowercase();

            // Determine signal strength based on content analysis
            let is_critical = content_lower.contains("security")
                || content_lower.contains("credential")
                || content_lower.contains("sql injection")
                || content_lower.contains("high-risk")
                || content_lower.contains("critical")
                || memory.tags.iter().any(|t| t.contains("severity:critical"));

            let is_signal = content_lower.contains("complexity")
                || content_lower.contains("untested")
                || content_lower.contains("coupling")
                || content_lower.contains("co-change")
                || content_lower.contains("architecture")
                || content_lower.contains("code smell");

            let target_importance = if is_critical {
                critical_count += 1;
                current_importance.max(0.8)
            } else if is_signal {
                signal_count += 1;
                current_importance.clamp(0.5, 0.7)
            } else {
                noise_count += 1;
                current_importance.min(0.3)
            };

            // Only update if importance actually changed
            if (target_importance - current_importance).abs() > 0.01 {
                let _ = self.storage.update_memory(
                    &memory.id,
                    &memory.content,
                    Some(target_importance),
                );
                reclassified += 1;
            }
        }

        let total = noise_count + signal_count + critical_count;

        Ok(EnrichResult {
            insights_stored: 0,
            details: json!({
                "total_analyzed": total,
                "noise": noise_count,
                "signal": signal_count,
                "critical": critical_count,
                "reclassified": reclassified,
            }),
        })
    }
}
