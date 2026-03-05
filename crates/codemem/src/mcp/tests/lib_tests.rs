use super::*;
use test_helpers::*;

#[test]
fn handle_initialize() {
    let server = test_server();
    let resp = server.handle_request("initialize", None, json!(1));
    assert!(resp.result.is_some());
    assert!(resp.error.is_none());

    let result = resp.result.unwrap();
    assert_eq!(result["protocolVersion"], "2024-11-05");
    assert_eq!(result["serverInfo"]["name"], "codemem");
}

#[test]
fn handle_tools_list_returns_32_tools() {
    let server = test_server();
    let resp = server.handle_request("tools/list", None, json!(2));
    let result = resp.result.unwrap();
    let tools = result["tools"].as_array().unwrap();
    assert_eq!(tools.len(), 32);

    let names: Vec<&str> = tools.iter().filter_map(|t| t["name"].as_str()).collect();
    // Memory CRUD (7)
    assert!(names.contains(&"store_memory"));
    assert!(names.contains(&"recall"));
    assert!(names.contains(&"delete_memory"));
    assert!(names.contains(&"associate_memories"));
    assert!(names.contains(&"refine_memory"));
    assert!(names.contains(&"split_memory"));
    assert!(names.contains(&"merge_memories"));
    // Graph & Structure (9)
    assert!(names.contains(&"graph_traverse"));
    assert!(names.contains(&"summary_tree"));
    assert!(names.contains(&"codemem_status"));
    assert!(names.contains(&"index_codebase"));
    assert!(names.contains(&"search_code"));
    assert!(names.contains(&"get_symbol_info"));
    assert!(names.contains(&"get_symbol_graph"));
    assert!(names.contains(&"find_important_nodes"));
    assert!(names.contains(&"find_related_groups"));
    assert!(names.contains(&"get_node_memories"));
    assert!(names.contains(&"node_coverage"));
    assert!(names.contains(&"get_cross_repo"));
    // Consolidation & Patterns (3)
    assert!(names.contains(&"consolidate"));
    assert!(names.contains(&"detect_patterns"));
    assert!(names.contains(&"get_decision_chain"));
    // Namespace Management (3)
    assert!(names.contains(&"list_namespaces"));
    assert!(names.contains(&"namespace_stats"));
    assert!(names.contains(&"delete_namespace"));
    // Session & Context (2)
    assert!(names.contains(&"session_checkpoint"));
    assert!(names.contains(&"session_context"));
    // Enrichment (5)
    assert!(names.contains(&"enrich_codebase"));
    assert!(names.contains(&"analyze_codebase"));
    assert!(names.contains(&"enrich_git_history"));
    assert!(names.contains(&"enrich_security"));
    assert!(names.contains(&"enrich_performance"));
}

#[test]
fn handle_unknown_method() {
    let server = test_server();
    let resp = server.handle_request("some/unknown", None, json!(5));
    assert!(resp.error.is_some());
    assert_eq!(resp.error.unwrap().code, -32601);
}

#[test]
fn handle_ping() {
    let server = test_server();
    let resp = server.handle_request("ping", None, json!(6));
    assert!(resp.result.is_some());
}

// ── Legacy Alias Tests ──────────────────────────────────────────────

#[test]
fn legacy_recall_memory_alias_works() {
    let server = test_server();
    // The old "recall_memory" name should dispatch to "recall"
    let params = json!({"name": "recall_memory", "arguments": {"query": "test"}});
    let resp = server.handle_request("tools/call", Some(&params), json!(10));
    let result = resp.result.unwrap();
    // Should not be an "unknown tool" error
    assert_ne!(result["isError"], true);
}

#[test]
fn removed_tools_return_error() {
    let server = test_server();
    for tool_name in &["set_scoring_weights", "export_memories", "import_memories"] {
        let params = json!({"name": tool_name, "arguments": {}});
        let resp = server.handle_request("tools/call", Some(&params), json!(20));
        let result = resp.result.unwrap();
        assert_eq!(result["isError"], true, "{tool_name} should return error");
        let text = result["content"][0]["text"].as_str().unwrap();
        assert!(
            text.contains("removed"),
            "{tool_name} error should mention 'removed'"
        );
    }
}
