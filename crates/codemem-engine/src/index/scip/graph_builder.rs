//! SCIP graph builder: create nodes + edges from parsed SCIP data.
//!
//! Takes the intermediate structs from the reader and produces `GraphNode`s,
//! `Edge`s, and `MemoryNode`s (for hover documentation).

use std::collections::{HashMap, HashSet};

use chrono::Utc;
use codemem_core::{Edge, GraphNode, MemoryNode, MemoryType, NodeKind, RelationshipType};

use codemem_core::ScipConfig;

use super::{
    is_import_ref, is_read_ref, is_write_ref, ScipDefinition, ScipReadResult, ROLE_IMPORT,
    ROLE_READ_ACCESS, ROLE_WRITE_ACCESS,
};

/// Result of building graph structures from SCIP data.
#[derive(Debug, Clone, Default)]
pub struct ScipBuildResult {
    pub nodes: Vec<GraphNode>,
    pub edges: Vec<Edge>,
    pub memories: Vec<(MemoryNode, String)>, // (memory, related_node_id) for RELATES_TO edges
    pub ext_nodes_created: usize,
    pub files_covered: HashSet<String>,
    pub doc_memories_created: usize,
}

/// Build graph nodes, edges, and doc memories from a parsed SCIP result.
///
/// Respects `config.max_references_per_symbol`, `config.create_external_nodes`,
/// and `config.store_docs_as_memories` settings.
pub fn build_graph(
    scip: &ScipReadResult,
    namespace: Option<&str>,
    config: &ScipConfig,
) -> ScipBuildResult {
    let now = Utc::now();
    let ns = namespace.map(|s| s.to_string());

    let mut nodes = Vec::new();
    let mut edges = Vec::new();
    let mut memories: Vec<(MemoryNode, String)> = Vec::new();
    let mut ext_nodes_created = 0;
    let mut doc_memories_created = 0;

    // Filter out definitions from files outside the project root. SCIP indexers
    // may include build cache, vendored deps, or virtualenv paths that ast-grep
    // never walks. A source file path must be relative and stay within the project.
    // Also skip wildcard ambient module declarations (e.g., `declare module '*.css'`)
    // which act as catch-all type stubs — every matching import resolves to them,
    // creating thousands of useless edges with massive fan-in.
    // Stage 1: Filter by file path and wildcard module (cheap string checks).
    let path_filtered: Vec<&ScipDefinition> = scip
        .definitions
        .iter()
        .filter(|d| is_source_path(&d.file_path) && !is_wildcard_module(&d.qualified_name))
        .collect();

    // Stage 2: Parse SCIP symbols once, use for both noise filtering and containment chains.
    // This avoids double-parsing (is_noise_definition + extract_containment_chain both need it).
    let mut source_defs: Vec<&ScipDefinition> = Vec::with_capacity(path_filtered.len());
    let mut parsed_symbols: Vec<scip::types::Symbol> = Vec::with_capacity(path_filtered.len());
    for def in &path_filtered {
        let parsed = match scip::symbol::parse_symbol(&def.scip_symbol) {
            Ok(p) => p,
            Err(_) => {
                // Can't parse — keep it to be safe
                source_defs.push(def);
                parsed_symbols.push(scip::types::Symbol::default());
                continue;
            }
        };
        if is_noise_symbol(def, &parsed) {
            continue;
        }
        source_defs.push(def);
        parsed_symbols.push(parsed);
    }

    // Build a set of defined symbol strings -> qualified names for edge resolution.
    let mut symbol_to_qname: HashMap<&str, &str> = HashMap::new();
    for def in &source_defs {
        symbol_to_qname.insert(&def.scip_symbol, &def.qualified_name);
    }

    // Phase 1: Create sym: nodes from definitions.
    // Track created node IDs to avoid duplicates for synthetic parents.
    let mut created_node_ids: HashSet<String> = HashSet::new();
    let mut created_edge_ids: HashSet<String> = HashSet::new();
    // Tier 3 folding: map folded symbol qname → parent node ID for edge redirection.
    let mut folded_to_parent: HashMap<String, String> = HashMap::new();
    // Collect folded children to batch-add to parent payloads after all nodes are created.
    // Key: parent qname, Value: vec of (child label, tier3 category like "fields"/"type_params")
    let mut folded_children: HashMap<String, Vec<(String, &'static str)>> = HashMap::new();

    // Build containment chains from pre-parsed symbols (no re-parsing needed).
    let def_chains: Vec<Vec<(String, NodeKind)>> = parsed_symbols
        .iter()
        .map(extract_containment_chain_from_parsed)
        .collect();

    for (def_idx, def) in source_defs.iter().enumerate() {
        let kind = if def.is_test {
            NodeKind::Test
        } else {
            def.kind
        };

        // Node tiering: Tier 3 kinds get folded into parent metadata.
        let tier3_category = match kind {
            NodeKind::Field | NodeKind::Property => Some("fields"),
            NodeKind::TypeParameter => Some("type_params"),
            NodeKind::EnumVariant => Some("variants"),
            _ => None,
        };

        if let Some(category) = tier3_category {
            // Find parent from containment chain.
            let chain = &def_chains[def_idx];
            if chain.len() >= 2 {
                let parent_qname = &chain[chain.len() - 2].0;
                let leaf_name = def
                    .qualified_name
                    .rsplit([':', '.'])
                    .next()
                    .unwrap_or(&def.qualified_name);
                folded_children
                    .entry(parent_qname.clone())
                    .or_default()
                    .push((leaf_name.to_string(), category));
                folded_to_parent.insert(def.qualified_name.clone(), format!("sym:{parent_qname}"));
                // Also map the scip_symbol for reference resolution.
                symbol_to_qname.insert(&def.scip_symbol, &def.qualified_name);
                continue; // Don't create a node for this definition.
            }
        }

        let node_id = format!("sym:{}", def.qualified_name);

        let mut payload = HashMap::new();
        payload.insert(
            "scip_symbol".to_string(),
            serde_json::Value::String(def.scip_symbol.clone()),
        );
        payload.insert("line_start".to_string(), serde_json::json!(def.line_start));
        payload.insert("line_end".to_string(), serde_json::json!(def.line_end));
        payload.insert(
            "file_path".to_string(),
            serde_json::Value::String(def.file_path.clone()),
        );
        if def.is_test {
            payload.insert("is_test".to_string(), serde_json::json!(true));
        }
        if def.is_generated {
            payload.insert("is_generated".to_string(), serde_json::json!(true));
        }
        // Store type signature from first documentation line if available.
        if let Some(type_sig) = def.documentation.first() {
            payload.insert(
                "type_signature".to_string(),
                serde_json::Value::String(type_sig.clone()),
            );
        }
        payload.insert(
            "source".to_string(),
            serde_json::Value::String("scip".to_string()),
        );

        created_node_ids.insert(node_id.clone());
        nodes.push(GraphNode {
            id: node_id.clone(),
            kind,
            label: def.qualified_name.clone(),
            payload,
            centrality: 0.0,
            memory_id: None,
            namespace: ns.clone(),
            valid_from: None,
            valid_to: None,
        });

        // Create containment edges: either hierarchical (nested chain) or flat (file→sym).
        if config.hierarchical_containment {
            let chain = &def_chains[def_idx];
            let file_node_id = format!("file:{}", def.file_path);

            if chain.len() <= 1 {
                // No intermediate parents — just file→sym.
                let edge_id = format!("contains:{file_node_id}->{node_id}");
                if created_edge_ids.insert(edge_id.clone()) {
                    edges.push(Edge {
                        id: edge_id,
                        src: file_node_id,
                        dst: node_id.clone(),
                        relationship: RelationshipType::Contains,
                        weight: 0.1,
                        properties: scip_edge_properties(),
                        created_at: now,
                        valid_from: Some(now),
                        valid_to: None,
                    });
                }
            } else {
                // Build chain: file→top_parent→...→parent→leaf
                for (i, (seg_qname, seg_kind)) in chain.iter().enumerate() {
                    let seg_node_id = format!("sym:{seg_qname}");

                    // Create synthetic intermediate node if needed (not the leaf itself).
                    if seg_qname != &def.qualified_name
                        && created_node_ids.insert(seg_node_id.clone())
                    {
                        let mut syn_payload = HashMap::new();
                        syn_payload.insert(
                            "source".to_string(),
                            serde_json::Value::String("scip-synthetic".to_string()),
                        );
                        syn_payload.insert(
                            "file_path".to_string(),
                            serde_json::Value::String(def.file_path.clone()),
                        );
                        nodes.push(GraphNode {
                            id: seg_node_id.clone(),
                            kind: *seg_kind,
                            label: seg_qname.clone(),
                            payload: syn_payload,
                            centrality: 0.0,
                            memory_id: None,
                            namespace: ns.clone(),
                            valid_from: None,
                            valid_to: None,
                        });
                    }

                    // Create CONTAINS edge from parent to this segment.
                    let parent_id = if i == 0 {
                        file_node_id.clone()
                    } else {
                        format!("sym:{}", chain[i - 1].0)
                    };

                    let edge_id = format!("contains:{parent_id}->{seg_node_id}");
                    if created_edge_ids.insert(edge_id.clone()) {
                        edges.push(Edge {
                            id: edge_id,
                            src: parent_id,
                            dst: seg_node_id,
                            relationship: RelationshipType::Contains,
                            weight: 0.1,
                            properties: scip_edge_properties(),
                            created_at: now,
                            valid_from: Some(now),
                            valid_to: None,
                        });
                    }
                }
            }
        } else {
            // Flat containment: file → symbol (original behavior).
            let file_node_id = format!("file:{}", def.file_path);
            edges.push(Edge {
                id: format!("contains:{file_node_id}->{node_id}"),
                src: file_node_id,
                dst: node_id.clone(),
                relationship: RelationshipType::Contains,
                weight: 0.1,
                properties: scip_edge_properties(),
                created_at: now,
                valid_from: Some(now),
                valid_to: None,
            });
        }

        // Create hover doc memories (if enabled in config).
        if config.store_docs_as_memories && !def.documentation.is_empty() {
            let doc_text = def.documentation.join("\n");
            let mem_id = format!("scip-doc:{}", def.qualified_name);
            let memory = MemoryNode {
                id: mem_id,
                content: doc_text,
                memory_type: MemoryType::Context,
                importance: 0.4,
                confidence: 1.0,
                access_count: 0,
                content_hash: String::new(), // Will be computed by engine on persist.
                tags: vec!["scip-doc".to_string(), "auto-generated".to_string()],
                metadata: HashMap::new(),
                namespace: ns.clone(),
                session_id: None,
                repo: None,
                git_ref: None,
                expires_at: None,
                created_at: now,
                updated_at: now,
                last_accessed_at: now,
            };
            memories.push((memory, node_id.clone()));
            doc_memories_created += 1;
        }

        // Create edges from SCIP relationships.
        for rel in &def.relationships {
            if rel.target_symbol.is_empty() {
                continue;
            }
            // Resolve target to qualified name if it's a known symbol.
            let target_node_id =
                if let Some(qname) = symbol_to_qname.get(rel.target_symbol.as_str()) {
                    format!("sym:{qname}")
                } else {
                    // Target might be external — try to parse as external node ID.
                    match parse_external_node_id(&rel.target_symbol) {
                        Some(ext_id) => ext_id,
                        None => continue,
                    }
                };

            if rel.is_implementation {
                edges.push(Edge {
                    id: format!("implements:{node_id}->{target_node_id}"),
                    src: node_id.clone(),
                    dst: target_node_id.clone(),
                    relationship: RelationshipType::Implements,
                    weight: 0.8,
                    properties: scip_edge_properties(),
                    created_at: now,
                    valid_from: Some(now),
                    valid_to: None,
                });
                // If the source is a method, also create OVERRIDES edge.
                if def.kind == NodeKind::Method {
                    edges.push(Edge {
                        id: format!("overrides:{node_id}->{target_node_id}"),
                        src: node_id.clone(),
                        dst: target_node_id.clone(),
                        relationship: RelationshipType::Overrides,
                        weight: 0.8,
                        properties: scip_edge_properties(),
                        created_at: now,
                        valid_from: Some(now),
                        valid_to: None,
                    });
                }
            }
            if rel.is_type_definition {
                edges.push(Edge {
                    id: format!("typedef:{node_id}->{target_node_id}"),
                    src: node_id.clone(),
                    dst: target_node_id.clone(),
                    relationship: RelationshipType::TypeDefinition,
                    weight: 0.6,
                    properties: scip_edge_properties(),
                    created_at: now,
                    valid_from: Some(now),
                    valid_to: None,
                });
            }
            // `is_reference` on a relationship indicates a superclass/supertype
            // reference (e.g., class Dog extends Animal — Dog's SymbolInformation
            // has a relationship to Animal with is_reference=true). Map to Inherits.
            if rel.is_reference && !rel.is_implementation {
                edges.push(Edge {
                    id: format!("inherits:{node_id}->{target_node_id}"),
                    src: node_id.clone(),
                    dst: target_node_id,
                    relationship: RelationshipType::Inherits,
                    weight: 0.8,
                    properties: scip_edge_properties(),
                    created_at: now,
                    valid_from: Some(now),
                    valid_to: None,
                });
            }
        }
    }

    // Apply folded Tier 3 children to parent node payloads.
    for node in &mut nodes {
        let qname = node.label.as_str();
        if let Some(children) = folded_children.get(qname) {
            let mut fields = Vec::new();
            let mut type_params = Vec::new();
            let mut variants = Vec::new();
            for (name, category) in children {
                match *category {
                    "fields" => fields.push(serde_json::Value::String(name.clone())),
                    "type_params" => type_params.push(serde_json::Value::String(name.clone())),
                    "variants" => variants.push(serde_json::Value::String(name.clone())),
                    _ => {}
                }
            }
            if !fields.is_empty() {
                node.payload
                    .insert("fields".to_string(), serde_json::Value::Array(fields));
            }
            if !type_params.is_empty() {
                node.payload.insert(
                    "type_params".to_string(),
                    serde_json::Value::Array(type_params),
                );
            }
            if !variants.is_empty() {
                node.payload
                    .insert("variants".to_string(), serde_json::Value::Array(variants));
            }
        }
    }

    // Phase 2: Create pkg: nodes from external symbols (if enabled in config).
    // Instead of one node per external symbol (thousands), we aggregate to one node
    // per external *package* — this gives the API surface graph ("which modules depend
    // on which packages") without polluting the graph with individual library symbols.
    if config.create_external_nodes {
        let mut pkg_nodes_created: HashSet<String> = HashSet::new();
        for ext in &scip.externals {
            if ext.package_manager.is_empty() || ext.package_name.is_empty() {
                continue;
            }
            let node_id = format!("pkg:{}:{}", ext.package_manager, ext.package_name);
            if !pkg_nodes_created.insert(node_id.clone()) {
                continue; // Already created this package node
            }

            let mut payload = HashMap::new();
            payload.insert(
                "package_manager".to_string(),
                serde_json::Value::String(ext.package_manager.clone()),
            );
            payload.insert(
                "package_name".to_string(),
                serde_json::Value::String(ext.package_name.clone()),
            );
            payload.insert(
                "package_version".to_string(),
                serde_json::Value::String(ext.package_version.clone()),
            );
            payload.insert(
                "source".to_string(),
                serde_json::Value::String("scip".to_string()),
            );

            nodes.push(GraphNode {
                id: node_id,
                kind: NodeKind::External,
                label: ext.package_name.clone(),
                payload,
                centrality: 0.0,
                memory_id: None,
                namespace: ns.clone(),
                valid_from: None,
                valid_to: None,
            });
            ext_nodes_created += 1;
        }
    } // end if create_external_nodes

    // Phase 3: Create edges from references.
    // Pre-index source definitions by file path for O(1) lookup in find_enclosing_def.
    // Exclude Tier 3 folded definitions — they have no graph nodes and would only
    // inflate the linear scan in find_enclosing_def_indexed.
    let mut defs_by_file: HashMap<&str, Vec<&ScipDefinition>> = HashMap::new();
    for def in &source_defs {
        if folded_to_parent.contains_key(&def.qualified_name) {
            continue;
        }
        defs_by_file
            .entry(def.file_path.as_str())
            .or_default()
            .push(def);
    }

    // Filter references to source files only.
    let source_refs: Vec<&super::ScipReference> = scip
        .references
        .iter()
        .filter(|r| is_source_path(&r.file_path))
        .collect();

    // Count references per (symbol, file) to enforce per-kind fan-out limits.
    // Intentionally per-file, not global: a utility function referenced 30 times in
    // file A and 30 times in file B stays under the limit in each file independently.
    // Global counting would require a second pass; per-file is cheaper and still
    // prevents the worst offenders (e.g., `log()` called 200 times in one file).
    let mut ref_counts: HashMap<(&str, &str), usize> = HashMap::new();
    for r in &source_refs {
        *ref_counts
            .entry((&r.scip_symbol, &r.file_path))
            .or_insert(0) += 1;
    }

    // Build scip_symbol → NodeKind map for per-kind fan-out limits.
    let symbol_to_kind: HashMap<&str, NodeKind> = source_defs
        .iter()
        .map(|d| (d.scip_symbol.as_str(), d.kind))
        .collect();

    for r in &source_refs {
        // Skip high fan-out symbols using per-kind limits.
        let count = ref_counts
            .get(&(r.scip_symbol.as_str(), r.file_path.as_str()))
            .copied()
            .unwrap_or(0);
        let target_kind = symbol_to_kind.get(r.scip_symbol.as_str()).copied();
        let limit = match target_kind {
            Some(NodeKind::Module) => config.fan_out_limits.module,
            Some(NodeKind::Function) => config.fan_out_limits.function,
            Some(NodeKind::Method) => config.fan_out_limits.method,
            Some(NodeKind::Class | NodeKind::Trait | NodeKind::Interface) => {
                config.fan_out_limits.class
            }
            _ => config.max_references_per_symbol,
        };
        if count > limit {
            continue;
        }

        // R5: Filter noise calls using the blocklist.
        if crate::index::blocklist::is_blocked_call_scip(&r.scip_symbol) {
            continue;
        }

        // Resolve the referenced symbol to a node ID.
        let mut target_node_id = if let Some(qname) = symbol_to_qname.get(r.scip_symbol.as_str()) {
            format!("sym:{qname}")
        } else {
            // Might reference an external symbol.
            match parse_external_node_id(&r.scip_symbol) {
                Some(ext_id) => ext_id,
                None => continue,
            }
        };

        // Redirect folded Tier 3 symbols to their parent node.
        if let Some(qname) = symbol_to_qname.get(r.scip_symbol.as_str()) {
            if let Some(parent_id) = folded_to_parent.get(*qname) {
                target_node_id = parent_id.clone();
            }
        }

        // Find the enclosing definition in the same file to use as source.
        // If we can't find one, use the file node.
        let mut source_node_id = find_enclosing_def_indexed(&defs_by_file, &r.file_path, r.line)
            .map(|def| format!("sym:{}", def.qualified_name))
            .unwrap_or_else(|| format!("file:{}", r.file_path));

        // Redirect if the enclosing def was itself folded.
        if let Some(parent_id) = source_node_id
            .strip_prefix("sym:")
            .and_then(|qn| folded_to_parent.get(qn))
        {
            source_node_id = parent_id.clone();
        }

        // Don't create self-edges.
        if source_node_id == target_node_id {
            continue;
        }

        // Pick the most specific role for each reference. Priority:
        //   IMPORT > WRITE > READ > generic CALLS
        // A reference can have multiple role flags (e.g., IMPORT + READ_ACCESS),
        // but we emit one edge per reference to avoid double-counting in
        // PageRank — the more specific role subsumes the less specific one.
        //
        // scip-go workaround: scip-go sets READ_ACCESS on ALL references
        // without semantic differentiation. When ONLY READ_ACCESS is set
        // (no IMPORT, WRITE), fall through to CALLS.
        let semantic_mask = ROLE_IMPORT | ROLE_WRITE_ACCESS | ROLE_READ_ACCESS;
        let is_scip_go_generic = r.role_bitmask & semantic_mask == ROLE_READ_ACCESS;

        let (rel, weight) = if is_import_ref(r.role_bitmask) {
            (RelationshipType::Imports, 0.5)
        } else if is_write_ref(r.role_bitmask) {
            (RelationshipType::Writes, 0.4)
        } else if is_read_ref(r.role_bitmask) && !is_scip_go_generic {
            (RelationshipType::Reads, 0.3)
        } else {
            (RelationshipType::Calls, 1.0)
        };

        let edge_prefix = rel.to_string().to_lowercase();
        edges.push(Edge {
            id: format!(
                "{edge_prefix}:{source_node_id}->{target_node_id}:{}:{}",
                r.file_path, r.line
            ),
            src: source_node_id.clone(),
            dst: target_node_id.clone(),
            relationship: rel,
            weight,
            properties: scip_edge_properties(),
            created_at: now,
            valid_from: Some(now),
            valid_to: None,
        });

        // For non-import references to type-like symbols, also create a
        // DependsOn edge. This captures "function X uses type Y" which is
        // critical for blast-radius analysis. This is the ONE case where
        // we emit two edges per reference — the structural relationship
        // (Calls/Reads/Writes) AND the type dependency are distinct signals.
        if !is_import_ref(r.role_bitmask) {
            let is_type_target = matches!(
                target_kind,
                Some(
                    NodeKind::Class
                        | NodeKind::Trait
                        | NodeKind::Interface
                        | NodeKind::Type
                        | NodeKind::Enum
                )
            );
            if is_type_target {
                edges.push(Edge {
                    id: format!(
                        "depends:{source_node_id}->{target_node_id}:{}:{}",
                        r.file_path, r.line
                    ),
                    src: source_node_id,
                    dst: target_node_id,
                    relationship: RelationshipType::DependsOn,
                    weight: 0.7,
                    properties: scip_edge_properties(),
                    created_at: now,
                    valid_from: Some(now),
                    valid_to: None,
                });
            }
        }
    }

    // Deduplicate edges by ID (keep first occurrence).
    let mut seen_edge_ids = HashSet::new();
    edges.retain(|e| seen_edge_ids.insert(e.id.clone()));

    // Collapse intra-class edges: methods of the same class calling each other
    // are replaced by metadata on the parent class node.
    if config.collapse_intra_class_edges && config.hierarchical_containment {
        // Build sym:child → sym:parent map from containment edges.
        let mut child_to_parent: HashMap<&str, &str> = HashMap::new();
        for edge in &edges {
            if edge.relationship == RelationshipType::Contains
                && edge.src.starts_with("sym:")
                && edge.dst.starts_with("sym:")
            {
                child_to_parent.insert(&edge.dst, &edge.src);
            }
        }

        // Build node ID → kind map so we only collapse within class-like parents.
        let node_kind_map: HashMap<&str, NodeKind> =
            nodes.iter().map(|n| (n.id.as_str(), n.kind)).collect();

        // Find edges where src and dst share the same class/struct/trait parent.
        let mut intra_class_counts: HashMap<String, Vec<(String, String)>> = HashMap::new();
        let mut intra_edge_ids: HashSet<String> = HashSet::new();
        for edge in &edges {
            if !matches!(
                edge.relationship,
                RelationshipType::Calls | RelationshipType::Reads | RelationshipType::Writes
            ) {
                continue;
            }
            let src_parent = child_to_parent.get(edge.src.as_str());
            let dst_parent = child_to_parent.get(edge.dst.as_str());
            if let (Some(sp), Some(dp)) = (src_parent, dst_parent) {
                // Only collapse within class-like parents, not modules.
                let parent_kind = node_kind_map.get(sp).copied();
                let is_class_like = matches!(
                    parent_kind,
                    Some(NodeKind::Class | NodeKind::Trait | NodeKind::Interface | NodeKind::Enum)
                );
                if sp == dp && is_class_like {
                    // Same parent — mark for collapsing.
                    let src_leaf = edge.src.rsplit([':', '.']).next().unwrap_or(&edge.src);
                    let dst_leaf = edge.dst.rsplit([':', '.']).next().unwrap_or(&edge.dst);
                    intra_class_counts
                        .entry(sp.to_string())
                        .or_default()
                        .push((src_leaf.to_string(), dst_leaf.to_string()));
                    intra_edge_ids.insert(edge.id.clone());
                }
            }
        }

        // Remove intra-class edges and add metadata to parent nodes.
        if !intra_edge_ids.is_empty() {
            edges.retain(|e| !intra_edge_ids.contains(&e.id));
            for node in &mut nodes {
                if let Some(calls) = intra_class_counts.get(&node.id) {
                    let call_entries: Vec<serde_json::Value> = calls
                        .iter()
                        .map(|(from, to)| serde_json::json!({"from": from, "to": to}))
                        .collect();
                    node.payload.insert(
                        "intra_class_calls".to_string(),
                        serde_json::Value::Array(call_entries),
                    );
                }
            }
        }
    }

    let files_covered: HashSet<String> = scip.covered_files.iter().cloned().collect();

    // Ensure every node referenced by an edge exists. SCIP edges reference:
    // - file: nodes (from CONTAINS edges) — may not exist if ast-grep didn't walk the file
    // - ext: nodes (from reference/relationship edges) — may not be in scip.externals
    // Without these, FK constraints cause entire edge batches to fail silently.
    let existing_node_ids: HashSet<&str> = nodes.iter().map(|n| n.id.as_str()).collect();
    let mut missing_ids: HashSet<String> = HashSet::new();
    for edge in &edges {
        if !existing_node_ids.contains(edge.src.as_str()) {
            missing_ids.insert(edge.src.clone());
        }
        if !existing_node_ids.contains(edge.dst.as_str()) {
            missing_ids.insert(edge.dst.clone());
        }
    }
    for missing_id in &missing_ids {
        let (kind, label) = if let Some(file_path) = missing_id.strip_prefix("file:") {
            (NodeKind::File, file_path.to_string())
        } else if let Some(pkg_rest) = missing_id.strip_prefix("pkg:") {
            // pkg:{manager}:{name} — use package name as label
            let label = pkg_rest.rsplit(':').next().unwrap_or(pkg_rest).to_string();
            ext_nodes_created += 1;
            (NodeKind::External, label)
        } else if missing_id.starts_with("ext:") {
            // Legacy ext: IDs from relationships — still create stub if needed
            let label = missing_id
                .rsplit(':')
                .next()
                .unwrap_or(missing_id)
                .to_string();
            ext_nodes_created += 1;
            (NodeKind::External, label)
        } else if let Some(qname) = missing_id.strip_prefix("sym:") {
            // Interface methods, abstract methods, or symbols defined in external
            // code that SCIP references but didn't emit a definition for.
            let label = qname.rsplit([':', '.']).next().unwrap_or(qname).to_string();
            (NodeKind::Method, label)
        } else {
            continue; // Don't create stubs for unknown prefixes
        };
        let mut payload = HashMap::new();
        payload.insert(
            "source".to_string(),
            serde_json::Value::String("scip".to_string()),
        );
        nodes.push(GraphNode {
            id: missing_id.clone(),
            kind,
            label,
            payload,
            centrality: 0.0,
            memory_id: None,
            namespace: ns.clone(),
            valid_from: None,
            valid_to: None,
        });
    }

    // Filter edges whose endpoints were removed by noise filtering.
    // Noise definitions (parameters, locals, typeLiterals) are filtered from nodes
    // but SCIP references may still point to them, causing FK constraint failures.
    // We also allow edges to file: and pkg: nodes which are created by the persistence
    // layer (not in this build result's nodes vec).
    let valid_node_ids: HashSet<&str> = nodes.iter().map(|n| n.id.as_str()).collect();
    let edge_count_before = edges.len();
    edges.retain(|e| {
        let src_ok = valid_node_ids.contains(e.src.as_str())
            || e.src.starts_with("file:")
            || e.src.starts_with("pkg:");
        let dst_ok = valid_node_ids.contains(e.dst.as_str())
            || e.dst.starts_with("file:")
            || e.dst.starts_with("pkg:");
        src_ok && dst_ok
    });
    let edges_dropped = edge_count_before - edges.len();
    if edges_dropped > 0 {
        tracing::debug!("Dropped {edges_dropped} SCIP edges referencing filtered noise nodes");
    }

    ScipBuildResult {
        nodes,
        edges,
        memories,
        ext_nodes_created,
        files_covered,
        doc_memories_created,
    }
}

/// Find the innermost definition that encloses a given line in a file,
/// using a pre-indexed HashMap for O(defs_in_file) instead of O(all_defs).
fn find_enclosing_def_indexed<'a>(
    defs_by_file: &HashMap<&str, Vec<&'a ScipDefinition>>,
    file_path: &str,
    line: u32,
) -> Option<&'a ScipDefinition> {
    defs_by_file
        .get(file_path)?
        .iter()
        .filter(|d| d.line_start <= line && d.line_end >= line)
        .min_by_key(|d| d.line_end - d.line_start)
        .copied()
}

/// Check if a file path looks like a project source file (relative, no escape).
///
/// Rejects paths that escape the project root (`..`), absolute paths, and common
/// non-source directories (build caches, vendor dirs, virtualenvs, node_modules).
fn is_source_path(path: &str) -> bool {
    // Must be relative and not escape project root
    if path.starts_with('/') || path.starts_with("..") {
        return false;
    }
    // Reject common non-source paths across languages
    let reject_dirs = [
        "node_modules/",
        ".venv/",
        "site-packages/",
        "__pycache__/",
        ".gradle/",
        ".m2/",
        "/go-build/",
        "vendor/", // Go vendored deps
        "dist/",
        "build/",
    ];
    if reject_dirs.iter().any(|r| path.contains(r)) {
        return false;
    }
    // Reject generated code directories and output files
    if path.contains("__generated__") || path.contains(".generated.") {
        return false;
    }
    // Reject bundled/minified JS output
    if path.ends_with(".bundle.js")
        || path.ends_with(".min.js")
        || path.ends_with(".min.css")
        || path.contains("/webpack_bundles/")
    {
        return false;
    }
    true
}

/// Check if a SCIP definition is a noise symbol using structural descriptor analysis.
///
/// Uses SCIP's descriptor suffix metadata to identify symbols that are structurally
/// noise for a knowledge graph: parameters, type parameters, local variables inside
/// functions, positional disambiguators, and anonymous type members.
///
/// This is more reliable than name-based heuristics because it classifies based on
/// what the symbol *is* (its role in the code structure) rather than what it's *named*.
fn is_noise_symbol(def: &ScipDefinition, parsed: &scip::types::Symbol) -> bool {
    // Skip generated code (SCIP provides this flag)
    if def.is_generated {
        return true;
    }

    // typeLiteral in any descriptor name (TS anonymous inline type members)
    if parsed
        .descriptors
        .iter()
        .any(|d| d.name.contains("typeLiteral"))
    {
        return true;
    }

    let leaf = match parsed.descriptors.last() {
        Some(d) => d,
        None => return false,
    };

    use scip::types::descriptor::Suffix;
    match leaf.suffix.enum_value() {
        // Parameters and type parameters are never graph-worthy
        Ok(Suffix::Parameter | Suffix::TypeParameter) => return true,
        // Term descriptor: check context to distinguish fields from locals
        Ok(Suffix::Term) => {
            // Term inside a Method = local variable / destructured param.
            // A function's internal bindings (let x = ...) are almost never
            // useful in a structural knowledge graph.
            let parent_suffix = parsed
                .descriptors
                .iter()
                .rev()
                .nth(1)
                .and_then(|d| d.suffix.enum_value().ok());
            if matches!(parent_suffix, Some(Suffix::Method)) {
                return true;
            }
            // Positional disambiguator: any Term whose name ends in digits.
            // SCIP appends digits to disambiguate anonymous/positional symbols
            // (e.g., `name0`, `key21`). No name-list needed — the trailing
            // digit + Term suffix is sufficient.
            if has_trailing_digits(&leaf.name) {
                return true;
            }
        }
        _ => {}
    }

    false
}

/// Check if a name ends with ASCII digits (SCIP positional disambiguator).
fn has_trailing_digits(name: &str) -> bool {
    name.len() > 1 && name.ends_with(|c: char| c.is_ascii_digit())
}

/// Try to parse a SCIP symbol string into a package-level external node ID.
///
/// Returns `pkg:{manager}:{name}` — collapsing all symbols from the same package
/// into a single node. This gives the API surface graph without per-symbol noise.
fn parse_external_node_id(scip_symbol: &str) -> Option<String> {
    let parsed = scip::symbol::parse_symbol(scip_symbol).ok()?;
    let package = parsed.package.as_ref()?;
    if package.manager.is_empty() || package.name.is_empty() {
        return None;
    }
    Some(format!("pkg:{}:{}", package.manager, package.name))
}

/// Extract the containment chain from a pre-parsed SCIP symbol.
///
/// For a symbol like `rust-analyzer cargo foo 1.0 auth/middleware/validate_token().`,
/// returns: `[("auth", Module), ("auth::middleware", Module), ("auth::middleware::validate_token", Function)]`.
///
/// The chain represents the hierarchical nesting: file→auth→auth::middleware→validate_token.
fn extract_containment_chain_from_parsed(parsed: &scip::types::Symbol) -> Vec<(String, NodeKind)> {
    // Detect separator from scheme
    let scheme = &parsed.scheme;
    let sep = if scheme == "rust-analyzer" || scheme == "lsif-clang" {
        "::"
    } else {
        "."
    };

    let mut chain = Vec::new();
    let mut cumulative_parts: Vec<&str> = Vec::new();
    let leaf_kind = super::infer_kind_from_parsed(parsed);

    for desc in &parsed.descriptors {
        if desc.name.is_empty() {
            continue;
        }
        cumulative_parts.push(&desc.name);
        let qname = cumulative_parts.join(sep);
        // For intermediate segments, use descriptor suffix to determine kind.
        let seg_kind = if cumulative_parts.len() < parsed.descriptors.len() {
            use scip::types::descriptor::Suffix;
            match desc.suffix.enum_value() {
                Ok(Suffix::Package | Suffix::Namespace) => NodeKind::Module,
                Ok(Suffix::Type) => NodeKind::Class,
                Ok(Suffix::Method) => NodeKind::Method,
                Ok(Suffix::Macro) => NodeKind::Macro,
                _ => NodeKind::Module,
            }
        } else {
            leaf_kind
        };
        chain.push((qname, seg_kind));
    }

    chain
}

/// Check if a qualified name represents a wildcard ambient module declaration
/// (e.g., TypeScript `declare module '*.css'`).
///
/// These are type-system catch-alls — every matching import resolves to them,
/// creating thousands of fan-in edges with no semantic value.
fn is_wildcard_module(qualified_name: &str) -> bool {
    // SCIP represents these as qualified names containing `'*` (the glob pattern
    // is part of the module name in the TS declaration).
    qualified_name.contains("'*")
}

/// Standard edge properties for SCIP-derived edges (allocated once, cloned per edge).
/// SCIP base confidence for multi-layer fusion.
/// ast-grep = 0.10, SCIP = 0.15, LSP = 0.20 (per north-star).
const SCIP_BASE_CONFIDENCE: f64 = 0.15;

fn scip_edge_properties() -> HashMap<String, serde_json::Value> {
    use std::sync::LazyLock;
    static PROPS: LazyLock<HashMap<String, serde_json::Value>> = LazyLock::new(|| {
        let mut props = HashMap::new();
        props.insert(
            "source".to_string(),
            serde_json::Value::String("scip".to_string()),
        );
        props.insert(
            "confidence".to_string(),
            serde_json::json!(SCIP_BASE_CONFIDENCE),
        );
        props.insert("source_layers".to_string(), serde_json::json!(["scip"]));
        props
    });
    PROPS.clone()
}

#[cfg(test)]
#[path = "../tests/scip_graph_builder_tests.rs"]
mod tests;
