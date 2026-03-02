//! Agent recipe runner — predefined MCP tool sequences streamed via SSE.

use axum::{
    extract::{Json, State},
    response::{
        sse::{Event, KeepAlive, Sse},
        IntoResponse,
    },
};
use serde_json::json;
use std::convert::Infallible;
use std::sync::Arc;

use crate::types::{RecipeListResponse, RecipeStep, RunRecipeRequest};
use crate::AppState;

/// Predefined recipes: each is a sequence of MCP tool calls.
fn get_recipes() -> Vec<RecipeListResponse> {
    vec![
        RecipeListResponse {
            id: "full-analysis".into(),
            name: "Full Analysis".into(),
            description: "Index a repo, detect patterns, generate insights, and find clusters"
                .into(),
            steps: vec![
                RecipeStep {
                    tool: "index_codebase".into(),
                    description: "Scan and index all source files".into(),
                },
                RecipeStep {
                    tool: "enrich_git_history".into(),
                    description: "Enrich graph with git history and co-change edges".into(),
                },
                RecipeStep {
                    tool: "enrich_security".into(),
                    description: "Scan for security-sensitive files and endpoints".into(),
                },
                RecipeStep {
                    tool: "enrich_performance".into(),
                    description: "Analyze coupling, dependency depth, and critical path".into(),
                },
                RecipeStep {
                    tool: "detect_patterns".into(),
                    description: "Detect cross-session patterns".into(),
                },
                RecipeStep {
                    tool: "pattern_insights".into(),
                    description: "Generate pattern insights summary".into(),
                },
                RecipeStep {
                    tool: "get_clusters".into(),
                    description: "Find semantic clusters in the graph".into(),
                },
            ],
        },
        RecipeListResponse {
            id: "quick-index".into(),
            name: "Quick Index".into(),
            description: "Index a repo and show symbol stats".into(),
            steps: vec![
                RecipeStep {
                    tool: "index_codebase".into(),
                    description: "Scan and index all source files".into(),
                },
                RecipeStep {
                    tool: "codemem_stats".into(),
                    description: "Show memory and graph statistics".into(),
                },
            ],
        },
        RecipeListResponse {
            id: "graph-analysis".into(),
            name: "Graph Analysis".into(),
            description: "Analyze graph structure: PageRank, clusters, and impact".into(),
            steps: vec![
                RecipeStep {
                    tool: "get_pagerank".into(),
                    description: "Compute PageRank scores for all nodes".into(),
                },
                RecipeStep {
                    tool: "get_clusters".into(),
                    description: "Find semantic clusters".into(),
                },
                RecipeStep {
                    tool: "detect_patterns".into(),
                    description: "Detect recurring patterns".into(),
                },
            ],
        },
        RecipeListResponse {
            id: "consolidate-all".into(),
            name: "Full Consolidation".into(),
            description: "Run all consolidation cycles: decay, creative, cluster, summarize"
                .into(),
            steps: vec![
                RecipeStep {
                    tool: "consolidate_decay".into(),
                    description: "Apply importance decay to old memories".into(),
                },
                RecipeStep {
                    tool: "consolidate_creative".into(),
                    description: "Find creative connections between memories".into(),
                },
                RecipeStep {
                    tool: "consolidate_cluster".into(),
                    description: "Cluster similar memories".into(),
                },
                RecipeStep {
                    tool: "consolidate_summarize".into(),
                    description: "Summarize memory clusters".into(),
                },
            ],
        },
    ]
}

/// GET /api/agents/recipes — list available recipes.
pub async fn list_recipes() -> impl IntoResponse {
    Json(get_recipes())
}

/// POST /api/agents/run — execute a recipe, streaming results via SSE.
pub async fn run_recipe(
    State(state): State<Arc<AppState>>,
    Json(req): Json<RunRecipeRequest>,
) -> Sse<impl tokio_stream::Stream<Item = Result<Event, Infallible>>> {
    let recipes = get_recipes();
    let recipe = recipes.into_iter().find(|r| r.id == req.recipe);

    let stream = async_stream::stream! {
        let recipe = match recipe {
            Some(r) => r,
            None => {
                yield Ok::<_, Infallible>(
                    Event::default()
                        .event("error")
                        .json_data(json!({ "error": format!("Unknown recipe: {}", req.recipe) }))
                        .unwrap_or_else(|_| Event::default().data("error"))
                );
                return;
            }
        };

        yield Ok::<_, Infallible>(
            Event::default()
                .event("recipe_start")
                .json_data(json!({
                    "recipe": recipe.id,
                    "name": recipe.name,
                    "total_steps": recipe.steps.len(),
                }))
                .unwrap_or_else(|_| Event::default().data("error"))
        );

        for (i, step) in recipe.steps.iter().enumerate() {
            yield Ok::<_, Infallible>(
                Event::default()
                    .event("step_start")
                    .json_data(json!({
                        "step": i,
                        "tool": step.tool,
                        "description": step.description,
                    }))
                    .unwrap_or_else(|_| Event::default().data("error"))
            );

            // Build tool params based on the tool name
            let params = build_tool_params(&step.tool, req.repo_id.as_deref(), req.namespace.as_deref(), &*state);

            // Execute via MCP server handle_request
            let request_params = json!({
                "name": step.tool,
                "arguments": params,
            });
            let id = serde_json::Value::Number(serde_json::Number::from(i as u64 + 1));
            let response = state.server.handle_request(
                "tools/call",
                Some(&request_params),
                id,
            );

            // Extract result from JSON-RPC response
            let (success, result_text) = extract_result(&response);

            yield Ok::<_, Infallible>(
                Event::default()
                    .event("step_result")
                    .json_data(json!({
                        "step": i,
                        "tool": step.tool,
                        "success": success,
                        "result": result_text,
                    }))
                    .unwrap_or_else(|_| Event::default().data("error"))
            );

            // Small pause between steps so the client can render
            tokio::time::sleep(std::time::Duration::from_millis(100)).await;
        }

        yield Ok::<_, Infallible>(
            Event::default()
                .event("recipe_complete")
                .json_data(json!({ "recipe": req.recipe }))
                .unwrap_or_else(|_| Event::default().data("error"))
        );
    };

    Sse::new(stream).keep_alive(KeepAlive::default())
}

/// Build tool arguments based on the tool name and optional repo/namespace context.
fn build_tool_params(
    tool: &str,
    repo_id: Option<&str>,
    namespace: Option<&str>,
    state: &AppState,
) -> serde_json::Value {
    match tool {
        "index_codebase" => {
            let mut params = json!({});
            // index_codebase requires `path`, not namespace/repo_id
            let raw_path = if let Some(rid) = repo_id {
                state.storage_direct().get_repo(rid).ok().flatten().map(|r| r.path)
            } else {
                namespace.map(|ns| ns.to_string())
            };
            if let Some(p) = raw_path {
                // Expand ~ to actual home directory
                let expanded = if p.starts_with("~/") {
                    if let Some(home) = std::env::var("HOME").ok() {
                        format!("{}{}", home, &p[1..])
                    } else { p }
                } else { p };
                params["path"] = json!(expanded);
            }
            params
        }
        "enrich_git_history" => {
            let mut params = json!({});
            let raw_path = if let Some(rid) = repo_id {
                state.storage_direct().get_repo(rid).ok().flatten().map(|r| r.path)
            } else {
                namespace.map(|ns| ns.to_string())
            };
            if let Some(p) = raw_path {
                let expanded = if p.starts_with("~/") {
                    if let Some(home) = std::env::var("HOME").ok() {
                        format!("{}{}", home, &p[1..])
                    } else { p }
                } else { p };
                params["path"] = json!(expanded);
            }
            if let Some(ns) = namespace {
                params["namespace"] = json!(ns);
            }
            params
        }
        "enrich_security" | "enrich_performance" => {
            let mut params = json!({});
            if let Some(ns) = namespace {
                params["namespace"] = json!(ns);
            }
            params
        }
        "detect_patterns" | "pattern_insights" => {
            let mut params = json!({});
            if let Some(ns) = namespace {
                params["namespace"] = json!(ns);
            }
            params
        }
        "get_clusters" => {
            // Louvain doesn't support namespace filtering — only pass resolution
            json!({ "resolution": 1.0 })
        }
        "get_pagerank" => {
            // Correct param name is `top_k`, not `top`
            json!({ "top_k": 20 })
        }
        "codemem_stats" => json!({}),
        "consolidate_decay" | "consolidate_creative" | "consolidate_cluster"
        | "consolidate_summarize" => json!({}),
        _ => json!({}),
    }
}

/// Extract success/text from a JSON-RPC response.
fn extract_result(response: &codemem_mcp::types::JsonRpcResponse) -> (bool, String) {
    let resp_json = serde_json::to_value(response).unwrap_or_default();

    if let Some(error) = resp_json.get("error") {
        let msg = error
            .get("message")
            .and_then(|m| m.as_str())
            .unwrap_or("Unknown error");
        return (false, msg.to_string());
    }

    if let Some(result) = resp_json.get("result") {
        // MCP tool results have a "content" array with text items
        if let Some(content) = result.get("content") {
            if let Some(arr) = content.as_array() {
                let texts: Vec<&str> = arr
                    .iter()
                    .filter_map(|item| item.get("text").and_then(|t| t.as_str()))
                    .collect();
                if !texts.is_empty() {
                    return (true, texts.join("\n"));
                }
            }
        }
        return (true, serde_json::to_string_pretty(result).unwrap_or_default());
    }

    (false, "No result".to_string())
}
