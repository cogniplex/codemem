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
fn handle_tools_list_returns_33_tools() {
    let server = test_server();
    let resp = server.handle_request("tools/list", None, json!(2));
    let result = resp.result.unwrap();
    let tools = result["tools"].as_array().unwrap();
    assert_eq!(tools.len(), 33);

    let names: Vec<&str> = tools.iter().filter_map(|t| t["name"].as_str()).collect();
    // Memory CRUD (7)
    assert!(names.contains(&"store_memory"));
    assert!(names.contains(&"recall"));
    assert!(names.contains(&"delete_memory"));
    assert!(names.contains(&"associate_memories"));
    assert!(names.contains(&"refine_memory"));
    assert!(names.contains(&"split_memory"));
    assert!(names.contains(&"merge_memories"));
    // Graph & Structure (8)
    assert!(names.contains(&"graph_traverse"));
    assert!(names.contains(&"summary_tree"));
    assert!(names.contains(&"codemem_status"));
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
    // Test Impact (1)
    assert!(names.contains(&"test_impact"));
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
