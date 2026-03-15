//! Cross-repo linker: connects symbols across namespaces via
//! package registry matching and symbol resolution.

use crate::index::manifest::ManifestResult;
use crate::index::symbol::{Symbol, Visibility};

/// A package registered in the cross-repo registry.
#[derive(Debug, Clone)]
pub struct RegisteredPackage {
    pub package_name: String,
    pub namespace: String,
    pub version: String,
    pub manifest: String,
}

/// An unresolved reference awaiting cross-repo resolution.
#[derive(Debug, Clone)]
pub struct PendingRef {
    pub id: String,
    pub namespace: String,
    pub source_node: String,
    pub target_name: String,
    pub package_hint: Option<String>,
    pub ref_kind: String,
    pub file_path: Option<String>,
    pub line: Option<usize>,
}

/// A cross-namespace edge produced by the linker.
#[derive(Debug, Clone)]
pub struct CrossRepoEdge {
    /// Edge ID: "xref:{src_ns}/{src_sym}->{dst_ns}/{dst_sym}"
    pub id: String,
    /// Source node ID (e.g., "sym:handler.process")
    pub source: String,
    /// Target node ID (e.g., "sym:validate")
    pub target: String,
    /// Relationship type string (e.g., "Calls", "Imports")
    pub relationship: String,
    /// Confidence of the cross-repo resolution
    pub confidence: f64,
    /// Source namespace
    pub source_namespace: String,
    /// Target namespace
    pub target_namespace: String,
}

/// Result of a cross-repo linking pass.
#[derive(Debug, Default)]
pub struct LinkResult {
    /// Packages registered in this pass.
    pub packages_registered: usize,
    /// Cross-namespace edges created via forward linking.
    pub forward_edges: Vec<CrossRepoEdge>,
    /// Cross-namespace edges created via backward linking.
    pub backward_edges: Vec<CrossRepoEdge>,
    /// IDs of unresolved refs that were resolved (to be deleted).
    pub resolved_ref_ids: Vec<String>,
}

/// A matched symbol for cross-repo resolution.
#[derive(Debug, Clone)]
pub struct SymbolMatch {
    pub qualified_name: String,
    pub visibility: Visibility,
    pub kind: String,
}

/// Extract packages from manifests for registry insertion.
pub fn extract_packages(manifests: &ManifestResult, namespace: &str) -> Vec<RegisteredPackage> {
    manifests
        .packages
        .iter()
        .map(|(name, manifest_path)| {
            // Find version from dependencies (self-reference)
            let version = manifests
                .dependencies
                .iter()
                .find(|d| d.name == *name)
                .map(|d| d.version.clone())
                .unwrap_or_default();
            RegisteredPackage {
                package_name: name.clone(),
                namespace: namespace.to_string(),
                version,
                manifest: manifest_path.clone(),
            }
        })
        .collect()
}

/// Forward link: resolve this namespace's pending refs against known packages.
///
/// For each pending ref from `namespace` that has a `package_hint` matching
/// a registered package in another namespace, attempt to match the target_name
/// against the provided symbols from that namespace.
pub fn forward_link(
    namespace: &str,
    pending_refs: &[PendingRef],
    registry: &[RegisteredPackage],
    resolve_fn: &dyn Fn(&str, &str) -> Vec<SymbolMatch>,
) -> LinkResult {
    let mut result = LinkResult::default();

    for pending_ref in pending_refs {
        // Only process refs from this namespace
        if pending_ref.namespace != namespace {
            continue;
        }

        // Skip refs without a package hint
        let package_hint = match &pending_ref.package_hint {
            Some(hint) => hint,
            None => continue,
        };

        // Find matching registry entries in OTHER namespaces
        let matching_entries: Vec<&RegisteredPackage> = registry
            .iter()
            .filter(|entry| entry.package_name == *package_hint && entry.namespace != namespace)
            .collect();

        // Try all matching namespaces and pick the best overall match,
        // rather than stopping at the first namespace that resolves.
        let mut best_edge: Option<(CrossRepoEdge, f64)> = None;
        for entry in matching_entries {
            let matches = resolve_fn(&entry.namespace, &pending_ref.target_name);
            if let Some(best) = pick_best_match(&matches) {
                let confidence = match_confidence_for_symbol(best);
                if best_edge.as_ref().is_none_or(|(_, c)| confidence > *c) {
                    best_edge = Some((
                        CrossRepoEdge {
                            id: make_edge_id(
                                namespace,
                                &pending_ref.source_node,
                                &entry.namespace,
                                &best.qualified_name,
                            ),
                            source: pending_ref.source_node.clone(),
                            target: format!("sym:{}", best.qualified_name),
                            relationship: ref_kind_to_relationship(&pending_ref.ref_kind)
                                .to_string(),
                            confidence,
                            source_namespace: namespace.to_string(),
                            target_namespace: entry.namespace.clone(),
                        },
                        confidence,
                    ));
                }
            }
        }
        if let Some((edge, _)) = best_edge {
            result.forward_edges.push(edge);
            result.resolved_ref_ids.push(pending_ref.id.clone());
        }
    }

    result
}

/// Backward link: resolve OTHER namespaces' pending refs against THIS namespace's symbols.
///
/// Only considers refs whose `package_hint` matches one of our `package_names`,
/// or refs with no package hint (best-effort matching).
pub fn backward_link(
    namespace: &str,
    package_names: &[String],
    pending_refs_for_packages: &[PendingRef],
    symbols: &[Symbol],
) -> LinkResult {
    let mut result = LinkResult::default();

    for pending_ref in pending_refs_for_packages {
        // Don't self-link
        if pending_ref.namespace == namespace {
            continue;
        }

        // Only consider refs that target one of our packages.
        // Skip refs with no package_hint — they would match any symbol and produce
        // false-positive cross-repo edges for common names.
        let Some(ref hint) = pending_ref.package_hint else {
            continue;
        };
        if !package_names.iter().any(|p| p == hint) {
            continue;
        }

        if let Some((qualified_name, confidence)) = match_symbol(&pending_ref.target_name, symbols)
        {
            let edge = CrossRepoEdge {
                id: make_edge_id(
                    &pending_ref.namespace,
                    &pending_ref.source_node,
                    namespace,
                    &qualified_name,
                ),
                source: pending_ref.source_node.clone(),
                target: format!("sym:{qualified_name}"),
                relationship: ref_kind_to_relationship(&pending_ref.ref_kind).to_string(),
                confidence,
                source_namespace: pending_ref.namespace.clone(),
                target_namespace: namespace.to_string(),
            };
            result.backward_edges.push(edge);
            result.resolved_ref_ids.push(pending_ref.id.clone());
        }
    }

    result
}

/// Match a target_name against a set of symbols with confidence scoring.
///
/// Strategy:
/// 1. Exact qualified name match -> confidence 1.0
/// 2. Suffix match (e.g., "validate" matches "utils.validate") -> confidence 0.85
/// 3. Simple name match -> confidence 0.7, prefers public symbols and shortest qualified name
/// 4. None
pub fn match_symbol(target_name: &str, symbols: &[Symbol]) -> Option<(String, f64)> {
    // 1. Exact qualified name match
    if let Some(sym) = symbols.iter().find(|s| s.qualified_name == target_name) {
        let boost = visibility_boost(sym.visibility);
        return Some((sym.qualified_name.clone(), (1.0 + boost).min(1.0)));
    }

    // 2. Suffix match: target matches the last segment(s) of a qualified name
    let suffix_matches: Vec<&Symbol> = symbols
        .iter()
        .filter(|s| {
            // Check if qualified_name ends with the target after a separator
            let qn = &s.qualified_name;
            qn.ends_with(target_name)
                && (qn.len() == target_name.len()
                    || qn[..qn.len() - target_name.len()].ends_with('.')
                    || qn[..qn.len() - target_name.len()].ends_with("::"))
        })
        .collect();

    if !suffix_matches.is_empty() {
        // Prefer public symbols
        let public_matches: Vec<&&Symbol> = suffix_matches
            .iter()
            .filter(|s| s.visibility == Visibility::Public)
            .collect();

        let best = if !public_matches.is_empty() {
            public_matches
                .iter()
                .min_by_key(|s| s.qualified_name.len())
                .unwrap()
        } else {
            suffix_matches
                .iter()
                .min_by_key(|s| s.qualified_name.len())
                .unwrap()
        };

        let boost = visibility_boost(best.visibility);
        return Some((best.qualified_name.clone(), (0.85 + boost).min(1.0)));
    }

    // 3. Simple name match
    let simple_name = simple_name_of(target_name);
    let name_matches: Vec<&Symbol> = symbols.iter().filter(|s| s.name == simple_name).collect();

    if !name_matches.is_empty() {
        let best = pick_best_by_visibility(&name_matches);
        let boost = visibility_boost(best.visibility);
        return Some((best.qualified_name.clone(), (0.7 + boost).min(1.0)));
    }

    None
}

/// Build a cross-repo edge ID.
fn make_edge_id(src_ns: &str, src_sym: &str, dst_ns: &str, dst_sym: &str) -> String {
    format!("xref:{src_ns}/{src_sym}->{dst_ns}/{dst_sym}")
}

/// Map ref_kind string to relationship type string.
fn ref_kind_to_relationship(ref_kind: &str) -> &str {
    match ref_kind {
        "call" => "Calls",
        "import" => "Imports",
        "inherits" => "Inherits",
        "implements" => "Implements",
        "type_usage" => "DependsOn",
        _ => "RelatesTo",
    }
}

// ── Internal helpers ─────────────────────────────────────────────────────

/// Extract the simple (unqualified) name from a potentially qualified name.
fn simple_name_of(name: &str) -> &str {
    // Try :: separator first (Rust-style), then . (most other languages)
    name.rsplit("::")
        .next()
        .unwrap_or(name)
        .rsplit('.')
        .next()
        .unwrap_or(name)
}

/// Visibility boost for scoring: public symbols get a small confidence bump.
fn visibility_boost(vis: Visibility) -> f64 {
    match vis {
        Visibility::Public => 0.05,
        Visibility::Crate => 0.02,
        Visibility::Protected => 0.01,
        Visibility::Private => 0.0,
    }
}

/// Pick the best symbol from a set of name matches by visibility, then shortest qualified name.
fn pick_best_by_visibility<'a>(candidates: &[&'a Symbol]) -> &'a Symbol {
    candidates
        .iter()
        .max_by(|a, b| {
            let vis_ord = visibility_rank(a.visibility).cmp(&visibility_rank(b.visibility));
            // If same visibility, prefer shortest qualified name
            vis_ord.then_with(|| b.qualified_name.len().cmp(&a.qualified_name.len()))
        })
        .unwrap()
}

/// Rank visibility for sorting (higher = better).
fn visibility_rank(vis: Visibility) -> u8 {
    match vis {
        Visibility::Public => 4,
        Visibility::Crate => 3,
        Visibility::Protected => 2,
        Visibility::Private => 1,
    }
}

/// Pick the best match from resolved symbols (highest confidence via visibility).
fn pick_best_match(matches: &[SymbolMatch]) -> Option<&SymbolMatch> {
    matches.iter().max_by(|a, b| {
        let va = visibility_rank(a.visibility);
        let vb = visibility_rank(b.visibility);
        va.cmp(&vb)
            .then_with(|| b.qualified_name.len().cmp(&a.qualified_name.len()))
    })
}

/// Compute confidence for a `SymbolMatch` from the resolve callback.
fn match_confidence_for_symbol(m: &SymbolMatch) -> f64 {
    0.85 + visibility_boost(m.visibility)
}

#[cfg(test)]
#[path = "tests/linker_tests.rs"]
mod tests;
