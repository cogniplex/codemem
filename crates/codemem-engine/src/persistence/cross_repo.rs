//! Cross-repo persistence: register packages, store unresolved refs,
//! run forward/backward linking, persist cross-namespace edges, and
//! detect API endpoints.

use super::CrossRepoPersistResult;
use crate::index::api_surface;
use crate::index::linker::{self, CrossRepoEdge, PendingRef, RegisteredPackage};
use crate::index::manifest::ManifestResult;
use crate::index::resolver::UnresolvedRef;
use crate::index::symbol::{Reference, Symbol};
use codemem_core::{CodememError, Edge, GraphBackend, RelationshipType};
use std::collections::HashMap;

impl super::super::CodememEngine {
    /// Persist cross-repo linking data after `persist_index_results`.
    ///
    /// This method runs 3 phases:
    /// 1. Register packages + store unresolved refs
    /// 2. Forward/backward cross-repo linking
    /// 3. API endpoint + client call detection
    pub fn persist_cross_repo_data(
        &self,
        manifests: &ManifestResult,
        unresolved: &[UnresolvedRef],
        symbols: &[Symbol],
        references: &[Reference],
        namespace: &str,
    ) -> Result<CrossRepoPersistResult, CodememError> {
        let mut result = CrossRepoPersistResult::default();

        // 1. Register packages from manifests into the package registry.
        let packages = linker::extract_packages(manifests, namespace);
        for pkg in &packages {
            if let Err(e) = self.storage.upsert_package_registry(
                &pkg.package_name,
                &pkg.namespace,
                &pkg.version,
                &pkg.manifest,
            ) {
                tracing::warn!("Failed to register package {}: {e}", pkg.package_name);
            } else {
                result.packages_registered += 1;
            }
        }

        // 2. Store unresolved refs for future backward linking by other namespaces.
        {
            let batch: Vec<codemem_core::UnresolvedRefData> = unresolved
                .iter()
                .map(|uref| codemem_core::UnresolvedRefData {
                    source_qualified_name: uref.source_node.clone(),
                    target_name: uref.target_name.clone(),
                    namespace: namespace.to_string(),
                    file_path: uref.file_path.clone(),
                    line: uref.line,
                    ref_kind: uref.ref_kind.clone(),
                    package_hint: uref.package_hint.clone(),
                })
                .collect();
            match self.storage.store_unresolved_refs_batch(&batch) {
                Ok(count) => result.unresolved_refs_stored = count,
                Err(e) => tracing::warn!("Failed to store unresolved refs batch: {e}"),
            }
        }

        // 3. Load existing registered packages and pending refs from storage.
        let all_registry: Vec<RegisteredPackage> = self
            .storage
            .list_registered_packages()
            .unwrap_or_default()
            .into_iter()
            .map(|(name, ns, manifest)| RegisteredPackage {
                package_name: name,
                namespace: ns,
                version: String::new(),
                manifest,
            })
            .collect();

        let package_names: Vec<String> = packages.iter().map(|p| p.package_name.clone()).collect();

        // Convert resolver UnresolvedRef -> linker PendingRef for this namespace.
        let this_ns_pending: Vec<PendingRef> = unresolved
            .iter()
            .map(|uref| PendingRef {
                id: format!("uref:{namespace}:{}:{}", uref.source_node, uref.target_name),
                namespace: namespace.to_string(),
                source_node: uref.source_node.clone(),
                target_name: uref.target_name.clone(),
                package_hint: uref.package_hint.clone(),
                ref_kind: uref.ref_kind.clone(),
                file_path: Some(uref.file_path.clone()),
                line: Some(uref.line),
            })
            .collect();

        // Load ALL pending refs from storage (for backward linking).
        let all_pending: Vec<PendingRef> = self
            .storage
            .list_pending_unresolved_refs()
            .unwrap_or_default()
            .into_iter()
            .map(|r| PendingRef {
                id: r.id,
                namespace: r.namespace,
                source_node: r.source_node,
                target_name: r.target_name,
                package_hint: r.package_hint,
                ref_kind: r.ref_kind,
                file_path: Some(r.file_path),
                line: Some(r.line),
            })
            .collect();

        // 4. Forward link: resolve our unresolved refs against other namespaces.
        //    Pre-build namespace→SymbolMatch index to avoid O(N*M) scans.
        let ns_symbol_index: HashMap<String, Vec<linker::SymbolMatch>> = {
            let graph = self.lock_graph()?;
            let mut index: HashMap<String, Vec<linker::SymbolMatch>> = HashMap::new();
            for n in graph.get_all_nodes() {
                if !n.id.starts_with("sym:") {
                    continue;
                }
                let Some(ref ns) = n.namespace else {
                    continue;
                };
                let vis_str = n
                    .payload
                    .get("visibility")
                    .and_then(|v| v.as_str())
                    .unwrap_or("private");
                let visibility = match vis_str {
                    "public" | "Public" => crate::index::symbol::Visibility::Public,
                    "crate" | "Crate" => crate::index::symbol::Visibility::Crate,
                    "protected" | "Protected" => crate::index::symbol::Visibility::Protected,
                    _ => crate::index::symbol::Visibility::Private,
                };
                let kind = n
                    .payload
                    .get("symbol_kind")
                    .and_then(|v| v.as_str())
                    .unwrap_or("unknown")
                    .to_string();
                index
                    .entry(ns.clone())
                    .or_default()
                    .push(linker::SymbolMatch {
                        qualified_name: n.label.clone(),
                        visibility,
                        kind,
                    });
            }
            index
        };

        let resolve_fn = |target_ns: &str, target_name: &str| -> Vec<linker::SymbolMatch> {
            let Some(symbols) = ns_symbol_index.get(target_ns) else {
                return Vec::new();
            };
            symbols
                .iter()
                .filter(|s| {
                    let label = &s.qualified_name;
                    // Exact match
                    if label == target_name {
                        return true;
                    }
                    // Suffix match with separator check (. or ::)
                    if label.ends_with(target_name) {
                        let prefix = &label[..label.len() - target_name.len()];
                        return prefix.ends_with('.') || prefix.ends_with("::");
                    }
                    false
                })
                .cloned()
                .collect()
        };

        let forward_result =
            linker::forward_link(namespace, &this_ns_pending, &all_registry, &resolve_fn);
        for edge in &forward_result.forward_edges {
            if let Err(e) = self.persist_cross_repo_edge(edge) {
                tracing::warn!("Failed to persist forward edge: {e}");
            } else {
                result.forward_edges_created += 1;
            }
        }

        // 5. Backward link: resolve other namespaces' pending refs against our symbols.
        let backward_result =
            linker::backward_link(namespace, &package_names, &all_pending, symbols);
        for edge in &backward_result.backward_edges {
            if let Err(e) = self.persist_cross_repo_edge(edge) {
                tracing::warn!("Failed to persist backward edge: {e}");
            } else {
                result.backward_edges_created += 1;
            }
        }

        // 5b. Clean up resolved refs so they don't accumulate.
        let all_resolved: Vec<&str> = forward_result
            .resolved_ref_ids
            .iter()
            .chain(backward_result.resolved_ref_ids.iter())
            .map(|s| s.as_str())
            .collect();
        for ref_id in &all_resolved {
            if let Err(e) = self.storage.delete_unresolved_ref(ref_id) {
                tracing::warn!("Failed to delete resolved ref {ref_id}: {e}");
            }
        }

        // ── Phase 3: API Surface ────────────────────────────────────────────
        // 6. Detect API endpoints and client calls.
        let endpoints = api_surface::detect_endpoints(symbols, namespace);
        result.endpoints_detected = endpoints.len();
        for ep in &endpoints {
            if let Err(e) = self.storage.store_api_endpoint(
                ep.method.as_deref().unwrap_or("ANY"),
                &ep.path,
                &ep.handler,
                namespace,
            ) {
                tracing::warn!(
                    "Failed to store endpoint {} {}: {e}",
                    ep.method.as_deref().unwrap_or("ANY"),
                    ep.path
                );
            }
        }

        let client_calls = api_surface::detect_client_calls(references);
        result.client_calls_detected = client_calls.len();
        for call in &client_calls {
            if let Err(e) = self.storage.store_api_client_call(
                &call.client_library,
                call.method.as_deref(),
                &call.caller,
                namespace,
            ) {
                tracing::warn!(
                    "Failed to store client call to {}: {e}",
                    call.client_library
                );
            }
        }

        Ok(result)
    }

    /// Persist a cross-repo edge into the graph_edges table and in-memory graph.
    fn persist_cross_repo_edge(&self, edge: &CrossRepoEdge) -> Result<(), CodememError> {
        let now = chrono::Utc::now();
        let relationship = match edge.relationship.as_str() {
            "Calls" => RelationshipType::Calls,
            "Imports" => RelationshipType::Imports,
            "Inherits" => RelationshipType::Inherits,
            "Implements" => RelationshipType::Implements,
            "DependsOn" => RelationshipType::DependsOn,
            _ => RelationshipType::RelatesTo,
        };

        let graph_edge = Edge {
            id: edge.id.clone(),
            src: edge.source.clone(),
            dst: edge.target.clone(),
            relationship,
            weight: edge.confidence.min(1.0) * 0.7,
            valid_from: Some(now),
            valid_to: None,
            properties: {
                let mut props = HashMap::new();
                props.insert(
                    "src_namespace".to_string(),
                    serde_json::Value::String(edge.source_namespace.clone()),
                );
                props.insert(
                    "dst_namespace".to_string(),
                    serde_json::Value::String(edge.target_namespace.clone()),
                );
                props.insert("cross_namespace".to_string(), serde_json::Value::Bool(true));
                props.insert("confidence".to_string(), serde_json::json!(edge.confidence));
                props
            },
            created_at: now,
        };

        self.storage.insert_graph_edge(&graph_edge)?;
        let mut graph = self.lock_graph()?;
        let _ = graph.add_edge(graph_edge);
        Ok(())
    }

    // ── Query helpers for tool_get_cross_repo ────────────────────────────

    /// Get all cross-namespace edges touching a given namespace.
    pub fn get_cross_namespace_edges(&self, namespace: &str) -> Result<Vec<Edge>, CodememError> {
        self.storage
            .graph_edges_for_namespace_with_cross(namespace, true)
    }

    /// Count unresolved refs for a namespace.
    pub fn count_unresolved_refs(&self, namespace: &str) -> Result<usize, CodememError> {
        self.storage.count_unresolved_refs(namespace)
    }

    /// List registered packages for a namespace.
    pub fn get_registered_packages(
        &self,
        namespace: &str,
    ) -> Result<Vec<RegisteredPackage>, CodememError> {
        let tuples = self
            .storage
            .list_registered_packages_for_namespace(namespace)?;
        Ok(tuples
            .into_iter()
            .map(|(name, ns, manifest)| RegisteredPackage {
                package_name: name,
                namespace: ns,
                version: String::new(),
                manifest,
            })
            .collect())
    }

    /// List detected API endpoints for a namespace.
    pub fn get_detected_endpoints(
        &self,
        namespace: &str,
    ) -> Result<Vec<api_surface::DetectedEndpoint>, CodememError> {
        let tuples = self.storage.list_api_endpoints(namespace)?;
        Ok(tuples
            .into_iter()
            .map(
                |(method, path, handler, _ns)| api_surface::DetectedEndpoint {
                    id: format!("ep:{namespace}:{method}:{path}"),
                    method: if method == "ANY" { None } else { Some(method) },
                    path,
                    handler,
                    file_path: String::new(),
                    line: 0,
                },
            )
            .collect())
    }
}
