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
fn handle_tools_list_returns_43_tools() {
    let server = test_server();
    let resp = server.handle_request("tools/list", None, json!(2));
    let result = resp.result.unwrap();
    let tools = result["tools"].as_array().unwrap();
    assert_eq!(tools.len(), 43);

    let names: Vec<&str> = tools.iter().filter_map(|t| t["name"].as_str()).collect();
    assert!(names.contains(&"store_memory"));
    assert!(names.contains(&"recall_memory"));
    assert!(names.contains(&"graph_traverse"));
    assert!(names.contains(&"codemem_health"));
    assert!(names.contains(&"index_codebase"));
    assert!(names.contains(&"search_symbols"));
    assert!(names.contains(&"get_symbol_info"));
    assert!(names.contains(&"get_dependencies"));
    assert!(names.contains(&"get_impact"));
    assert!(names.contains(&"get_clusters"));
    assert!(names.contains(&"get_cross_repo"));
    assert!(names.contains(&"get_pagerank"));
    assert!(names.contains(&"search_code"));
    assert!(names.contains(&"set_scoring_weights"));
    assert!(names.contains(&"export_memories"));
    assert!(names.contains(&"import_memories"));
    assert!(names.contains(&"recall_with_expansion"));
    assert!(names.contains(&"list_namespaces"));
    assert!(names.contains(&"namespace_stats"));
    assert!(names.contains(&"delete_namespace"));
    assert!(names.contains(&"consolidate_decay"));
    assert!(names.contains(&"consolidate_creative"));
    assert!(names.contains(&"consolidate_cluster"));
    assert!(names.contains(&"consolidate_forget"));
    assert!(names.contains(&"consolidation_status"));
    assert!(names.contains(&"recall_with_impact"));
    assert!(names.contains(&"get_decision_chain"));
    assert!(names.contains(&"detect_patterns"));
    assert!(names.contains(&"pattern_insights"));
    assert!(names.contains(&"refine_memory"));
    assert!(names.contains(&"split_memory"));
    assert!(names.contains(&"merge_memories"));
    assert!(names.contains(&"consolidate_summarize"));
    assert!(names.contains(&"codemem_metrics"));
    assert!(names.contains(&"session_checkpoint"));
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
