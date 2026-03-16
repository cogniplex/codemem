//! Cross-repo linking: package registry, unresolved references, and API endpoints.

use crate::{MapStorageErr, Storage};
use codemem_core::{CodememError, Edge};
use rusqlite::params;

/// A row in the `package_registry` table.
#[derive(Debug, Clone)]
pub struct PackageRegistryEntry {
    pub package_name: String,
    pub namespace: String,
    pub version: String,
    pub manifest: String,
}

/// A row in the `unresolved_refs` table.
#[derive(Debug, Clone)]
pub struct UnresolvedRefEntry {
    pub id: String,
    pub namespace: String,
    pub source_node: String,
    pub target_name: String,
    pub package_hint: Option<String>,
    pub ref_kind: String,
    pub file_path: Option<String>,
    pub line: Option<i64>,
    pub created_at: i64,
}

/// A row in the `api_endpoints` table.
#[derive(Debug, Clone)]
pub struct ApiEndpointEntry {
    pub id: String,
    pub namespace: String,
    pub method: Option<String>,
    pub path: String,
    pub handler: Option<String>,
    pub schema: String,
}

/// A row in the `api_client_calls` table.
#[derive(Debug, Clone)]
pub struct ApiClientCallEntry {
    pub id: String,
    pub namespace: String,
    pub method: Option<String>,
    pub target: String,
    pub caller: String,
    pub library: String,
}

/// A row in the `event_channels` table.
#[derive(Debug, Clone)]
pub struct EventChannelEntry {
    pub id: String,
    pub namespace: String,
    pub channel: String,
    pub direction: String,
    pub protocol: String,
    pub message_schema: String,
    pub description: String,
    pub handler: String,
    pub spec_file: String,
}

impl Storage {
    // ── Package Registry ─────────────────────────────────────────────────

    /// Insert or update a package registry entry.
    pub fn upsert_package_registry(
        &self,
        package_name: &str,
        namespace: &str,
        version: &str,
        manifest: &str,
    ) -> Result<(), CodememError> {
        let conn = self.conn()?;
        conn.execute(
            "INSERT OR REPLACE INTO package_registry (package_name, namespace, version, manifest)
             VALUES (?1, ?2, ?3, ?4)",
            params![package_name, namespace, version, manifest],
        )
        .storage_err()?;
        Ok(())
    }

    /// Get all packages registered in a namespace.
    pub fn get_packages_for_namespace(
        &self,
        namespace: &str,
    ) -> Result<Vec<PackageRegistryEntry>, CodememError> {
        let conn = self.conn()?;
        let mut stmt = conn
            .prepare(
                "SELECT package_name, namespace, version, manifest
                 FROM package_registry WHERE namespace = ?1",
            )
            .storage_err()?;
        let rows = stmt
            .query_map(params![namespace], |row| {
                Ok(PackageRegistryEntry {
                    package_name: row.get(0)?,
                    namespace: row.get(1)?,
                    version: row.get(2)?,
                    manifest: row.get(3)?,
                })
            })
            .storage_err()?;
        let mut entries = Vec::new();
        for row in rows {
            entries.push(row.storage_err()?);
        }
        Ok(entries)
    }

    /// Find all namespaces that provide a given package.
    pub fn find_namespace_for_package(
        &self,
        package_name: &str,
    ) -> Result<Vec<PackageRegistryEntry>, CodememError> {
        let conn = self.conn()?;
        let mut stmt = conn
            .prepare(
                "SELECT package_name, namespace, version, manifest
                 FROM package_registry WHERE package_name = ?1",
            )
            .storage_err()?;
        let rows = stmt
            .query_map(params![package_name], |row| {
                Ok(PackageRegistryEntry {
                    package_name: row.get(0)?,
                    namespace: row.get(1)?,
                    version: row.get(2)?,
                    manifest: row.get(3)?,
                })
            })
            .storage_err()?;
        let mut entries = Vec::new();
        for row in rows {
            entries.push(row.storage_err()?);
        }
        Ok(entries)
    }

    /// Delete all package registry entries for a namespace. Returns count deleted.
    pub fn delete_package_registry_for_namespace(
        &self,
        namespace: &str,
    ) -> Result<usize, CodememError> {
        let conn = self.conn()?;
        let deleted = conn
            .execute(
                "DELETE FROM package_registry WHERE namespace = ?1",
                params![namespace],
            )
            .storage_err()?;
        Ok(deleted)
    }

    // ── Unresolved Refs ──────────────────────────────────────────────────

    /// Insert a single unresolved reference.
    pub fn insert_unresolved_ref(&self, entry: &UnresolvedRefEntry) -> Result<(), CodememError> {
        let conn = self.conn()?;
        conn.execute(
            "INSERT OR REPLACE INTO unresolved_refs
             (id, namespace, source_node, target_name, package_hint, ref_kind, file_path, line, created_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)",
            params![
                entry.id,
                entry.namespace,
                entry.source_node,
                entry.target_name,
                entry.package_hint,
                entry.ref_kind,
                entry.file_path,
                entry.line,
                entry.created_at,
            ],
        )
        .storage_err()?;
        Ok(())
    }

    /// Batch insert unresolved references, respecting SQLite 999-param limit.
    pub fn insert_unresolved_refs_batch(
        &self,
        refs: &[UnresolvedRefEntry],
    ) -> Result<(), CodememError> {
        if refs.is_empty() {
            return Ok(());
        }
        let conn = self.conn()?;
        let tx = conn.unchecked_transaction().storage_err()?;

        const COLS: usize = 9;
        const BATCH: usize = 999 / COLS; // 111

        for chunk in refs.chunks(BATCH) {
            let mut placeholders = String::new();
            for (r, _) in chunk.iter().enumerate() {
                if r > 0 {
                    placeholders.push(',');
                }
                placeholders.push('(');
                for c in 0..COLS {
                    if c > 0 {
                        placeholders.push(',');
                    }
                    placeholders.push('?');
                    placeholders.push_str(&(r * COLS + c + 1).to_string());
                }
                placeholders.push(')');
            }

            let sql = format!(
                "INSERT OR REPLACE INTO unresolved_refs
                 (id, namespace, source_node, target_name, package_hint, ref_kind, file_path, line, created_at)
                 VALUES {placeholders}"
            );

            let mut param_values: Vec<Box<dyn rusqlite::types::ToSql>> = Vec::new();
            for entry in chunk {
                param_values.push(Box::new(entry.id.clone()));
                param_values.push(Box::new(entry.namespace.clone()));
                param_values.push(Box::new(entry.source_node.clone()));
                param_values.push(Box::new(entry.target_name.clone()));
                param_values.push(Box::new(entry.package_hint.clone()));
                param_values.push(Box::new(entry.ref_kind.clone()));
                param_values.push(Box::new(entry.file_path.clone()));
                param_values.push(Box::new(entry.line));
                param_values.push(Box::new(entry.created_at));
            }
            let param_refs: Vec<&dyn rusqlite::types::ToSql> =
                param_values.iter().map(|p| p.as_ref()).collect();

            tx.execute(&sql, param_refs.as_slice()).storage_err()?;
        }

        tx.commit().storage_err()?;
        Ok(())
    }

    /// Get all unresolved references for a namespace.
    pub fn get_unresolved_refs_for_namespace(
        &self,
        namespace: &str,
    ) -> Result<Vec<UnresolvedRefEntry>, CodememError> {
        let conn = self.conn()?;
        let mut stmt = conn
            .prepare(
                "SELECT id, namespace, source_node, target_name, package_hint, ref_kind, file_path, line, created_at
                 FROM unresolved_refs WHERE namespace = ?1",
            )
            .storage_err()?;
        let rows = stmt
            .query_map(params![namespace], |row| {
                Ok(UnresolvedRefEntry {
                    id: row.get(0)?,
                    namespace: row.get(1)?,
                    source_node: row.get(2)?,
                    target_name: row.get(3)?,
                    package_hint: row.get(4)?,
                    ref_kind: row.get(5)?,
                    file_path: row.get(6)?,
                    line: row.get(7)?,
                    created_at: row.get(8)?,
                })
            })
            .storage_err()?;
        let mut entries = Vec::new();
        for row in rows {
            entries.push(row.storage_err()?);
        }
        Ok(entries)
    }

    /// Get all unresolved references with a given package hint.
    pub fn get_unresolved_refs_for_package_hint(
        &self,
        package_hint: &str,
    ) -> Result<Vec<UnresolvedRefEntry>, CodememError> {
        let conn = self.conn()?;
        let mut stmt = conn
            .prepare(
                "SELECT id, namespace, source_node, target_name, package_hint, ref_kind, file_path, line, created_at
                 FROM unresolved_refs WHERE package_hint = ?1",
            )
            .storage_err()?;
        let rows = stmt
            .query_map(params![package_hint], |row| {
                Ok(UnresolvedRefEntry {
                    id: row.get(0)?,
                    namespace: row.get(1)?,
                    source_node: row.get(2)?,
                    target_name: row.get(3)?,
                    package_hint: row.get(4)?,
                    ref_kind: row.get(5)?,
                    file_path: row.get(6)?,
                    line: row.get(7)?,
                    created_at: row.get(8)?,
                })
            })
            .storage_err()?;
        let mut entries = Vec::new();
        for row in rows {
            entries.push(row.storage_err()?);
        }
        Ok(entries)
    }

    /// Delete a single unresolved reference by ID.
    pub fn delete_unresolved_ref(&self, id: &str) -> Result<(), CodememError> {
        let conn = self.conn()?;
        conn.execute("DELETE FROM unresolved_refs WHERE id = ?1", params![id])
            .storage_err()?;
        Ok(())
    }

    /// Batch delete unresolved references by IDs.
    pub fn delete_unresolved_refs_batch(&self, ids: &[String]) -> Result<(), CodememError> {
        if ids.is_empty() {
            return Ok(());
        }
        let conn = self.conn()?;
        let tx = conn.unchecked_transaction().storage_err()?;

        // 1 param per id, batch by 999
        for chunk in ids.chunks(999) {
            let placeholders: Vec<String> = (1..=chunk.len()).map(|i| format!("?{i}")).collect();
            let sql = format!(
                "DELETE FROM unresolved_refs WHERE id IN ({})",
                placeholders.join(",")
            );
            let param_refs: Vec<&dyn rusqlite::types::ToSql> = chunk
                .iter()
                .map(|s| s as &dyn rusqlite::types::ToSql)
                .collect();
            tx.execute(&sql, param_refs.as_slice()).storage_err()?;
        }

        tx.commit().storage_err()?;
        Ok(())
    }

    /// Delete all unresolved references for a namespace. Returns count deleted.
    pub fn delete_unresolved_refs_for_namespace(
        &self,
        namespace: &str,
    ) -> Result<usize, CodememError> {
        let conn = self.conn()?;
        let deleted = conn
            .execute(
                "DELETE FROM unresolved_refs WHERE namespace = ?1",
                params![namespace],
            )
            .storage_err()?;
        Ok(deleted)
    }

    // ── API Endpoints ────────────────────────────────────────────────────

    /// Insert or update an API endpoint.
    pub fn upsert_api_endpoint(&self, endpoint: &ApiEndpointEntry) -> Result<(), CodememError> {
        let conn = self.conn()?;
        conn.execute(
            "INSERT OR REPLACE INTO api_endpoints (id, namespace, method, path, handler, schema)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
            params![
                endpoint.id,
                endpoint.namespace,
                endpoint.method,
                endpoint.path,
                endpoint.handler,
                endpoint.schema,
            ],
        )
        .storage_err()?;
        Ok(())
    }

    /// Get all API endpoints for a namespace.
    pub fn get_api_endpoints_for_namespace(
        &self,
        namespace: &str,
    ) -> Result<Vec<ApiEndpointEntry>, CodememError> {
        let conn = self.conn()?;
        let mut stmt = conn
            .prepare(
                "SELECT id, namespace, method, path, handler, schema
                 FROM api_endpoints WHERE namespace = ?1",
            )
            .storage_err()?;
        let rows = stmt
            .query_map(params![namespace], |row| {
                Ok(ApiEndpointEntry {
                    id: row.get(0)?,
                    namespace: row.get(1)?,
                    method: row.get(2)?,
                    path: row.get(3)?,
                    handler: row.get(4)?,
                    schema: row.get(5)?,
                })
            })
            .storage_err()?;
        let mut entries = Vec::new();
        for row in rows {
            entries.push(row.storage_err()?);
        }
        Ok(entries)
    }

    /// Get all API endpoints with an exact path match.
    pub fn get_api_endpoints_for_path(
        &self,
        path: &str,
    ) -> Result<Vec<ApiEndpointEntry>, CodememError> {
        let conn = self.conn()?;
        let mut stmt = conn
            .prepare(
                "SELECT id, namespace, method, path, handler, schema
                 FROM api_endpoints WHERE path = ?1",
            )
            .storage_err()?;
        let rows = stmt
            .query_map(params![path], |row| {
                Ok(ApiEndpointEntry {
                    id: row.get(0)?,
                    namespace: row.get(1)?,
                    method: row.get(2)?,
                    path: row.get(3)?,
                    handler: row.get(4)?,
                    schema: row.get(5)?,
                })
            })
            .storage_err()?;
        let mut entries = Vec::new();
        for row in rows {
            entries.push(row.storage_err()?);
        }
        Ok(entries)
    }

    /// Find API endpoints whose path matches a LIKE pattern.
    pub fn find_api_endpoints_by_path_pattern(
        &self,
        path_pattern: &str,
    ) -> Result<Vec<ApiEndpointEntry>, CodememError> {
        let conn = self.conn()?;
        let mut stmt = conn
            .prepare(
                "SELECT id, namespace, method, path, handler, schema
                 FROM api_endpoints WHERE path LIKE ?1",
            )
            .storage_err()?;
        let rows = stmt
            .query_map(params![path_pattern], |row| {
                Ok(ApiEndpointEntry {
                    id: row.get(0)?,
                    namespace: row.get(1)?,
                    method: row.get(2)?,
                    path: row.get(3)?,
                    handler: row.get(4)?,
                    schema: row.get(5)?,
                })
            })
            .storage_err()?;
        let mut entries = Vec::new();
        for row in rows {
            entries.push(row.storage_err()?);
        }
        Ok(entries)
    }

    /// Delete all API endpoints for a namespace. Returns count deleted.
    pub fn delete_api_endpoints_for_namespace(
        &self,
        namespace: &str,
    ) -> Result<usize, CodememError> {
        let conn = self.conn()?;
        let deleted = conn
            .execute(
                "DELETE FROM api_endpoints WHERE namespace = ?1",
                params![namespace],
            )
            .storage_err()?;
        Ok(deleted)
    }

    // ── API Client Calls ─────────────────────────────────────────────────

    /// Insert or update an API client call.
    pub fn upsert_api_client_call(
        &self,
        id: &str,
        namespace: &str,
        method: Option<&str>,
        target: &str,
        caller: &str,
        library: &str,
    ) -> Result<(), CodememError> {
        let conn = self.conn()?;
        conn.execute(
            "INSERT OR REPLACE INTO api_client_calls (id, namespace, method, target, caller, library)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
            params![id, namespace, method, target, caller, library],
        )
        .storage_err()?;
        Ok(())
    }

    /// Get all API client calls for a namespace.
    pub fn get_api_client_calls_for_namespace(
        &self,
        namespace: &str,
    ) -> Result<Vec<ApiClientCallEntry>, CodememError> {
        let conn = self.conn()?;
        let mut stmt = conn
            .prepare(
                "SELECT id, namespace, method, target, caller, library
                 FROM api_client_calls WHERE namespace = ?1",
            )
            .storage_err()?;
        let rows = stmt
            .query_map(params![namespace], |row| {
                Ok(ApiClientCallEntry {
                    id: row.get(0)?,
                    namespace: row.get(1)?,
                    method: row.get(2)?,
                    target: row.get(3)?,
                    caller: row.get(4)?,
                    library: row.get(5)?,
                })
            })
            .storage_err()?;
        let mut entries = Vec::new();
        for row in rows {
            entries.push(row.storage_err()?);
        }
        Ok(entries)
    }

    // ── Cross-namespace Edge Queries ─────────────────────────────────────

    // ── Event Channels ───────────────────────────────────────────────

    /// Insert or update an event channel entry.
    pub fn upsert_event_channel(&self, entry: &EventChannelEntry) -> Result<(), CodememError> {
        let conn = self.conn()?;
        conn.execute(
            "INSERT OR REPLACE INTO event_channels (id, namespace, channel, direction, protocol, message_schema, description, handler, spec_file)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)",
            params![
                entry.id,
                entry.namespace,
                entry.channel,
                entry.direction,
                entry.protocol,
                entry.message_schema,
                entry.description,
                entry.handler,
                entry.spec_file,
            ],
        )
        .storage_err()?;
        Ok(())
    }

    /// Get all event channels for a namespace.
    pub fn get_event_channels_for_namespace(
        &self,
        namespace: &str,
    ) -> Result<Vec<EventChannelEntry>, CodememError> {
        let conn = self.conn()?;
        let mut stmt = conn
            .prepare(
                "SELECT id, namespace, channel, direction, protocol, message_schema, description, handler, spec_file
                 FROM event_channels WHERE namespace = ?1",
            )
            .storage_err()?;
        let rows = stmt
            .query_map(params![namespace], |row| {
                Ok(EventChannelEntry {
                    id: row.get(0)?,
                    namespace: row.get(1)?,
                    channel: row.get(2)?,
                    direction: row.get(3)?,
                    protocol: row.get(4)?,
                    message_schema: row.get(5)?,
                    description: row.get(6)?,
                    handler: row.get(7)?,
                    spec_file: row.get(8)?,
                })
            })
            .storage_err()?;
        let mut entries = Vec::new();
        for row in rows {
            entries.push(row.storage_err()?);
        }
        Ok(entries)
    }

    /// Get all event channels across all namespaces.
    pub fn get_all_event_channels(&self) -> Result<Vec<EventChannelEntry>, CodememError> {
        let conn = self.conn()?;
        let mut stmt = conn
            .prepare(
                "SELECT id, namespace, channel, direction, protocol, message_schema, description, handler, spec_file
                 FROM event_channels",
            )
            .storage_err()?;
        let rows = stmt
            .query_map([], |row| {
                Ok(EventChannelEntry {
                    id: row.get(0)?,
                    namespace: row.get(1)?,
                    channel: row.get(2)?,
                    direction: row.get(3)?,
                    protocol: row.get(4)?,
                    message_schema: row.get(5)?,
                    description: row.get(6)?,
                    handler: row.get(7)?,
                    spec_file: row.get(8)?,
                })
            })
            .storage_err()?;
        let mut entries = Vec::new();
        for row in rows {
            entries.push(row.storage_err()?);
        }
        Ok(entries)
    }

    // ── Cross-namespace Edge Queries ─────────────────────────────────────

    /// Get edges where at least one endpoint (src or dst) belongs to the given
    /// namespace and the edge has `cross_namespace = true` in its properties.
    /// This is semantically equivalent to `graph_edges_for_namespace_with_cross(ns, true)`
    /// but additionally filters for edges explicitly marked as cross-namespace.
    pub fn get_cross_namespace_edges(&self, namespace: &str) -> Result<Vec<Edge>, CodememError> {
        // Delegate to the unified method and filter for cross_namespace property.
        let all_edges = self.graph_edges_for_namespace_with_cross(namespace, true)?;
        Ok(all_edges
            .into_iter()
            .filter(|e| {
                e.properties
                    .get("cross_namespace")
                    .and_then(|v| v.as_bool())
                    .unwrap_or(false)
            })
            .collect())
    }
}

#[cfg(test)]
#[path = "tests/cross_repo_tests.rs"]
mod tests;
