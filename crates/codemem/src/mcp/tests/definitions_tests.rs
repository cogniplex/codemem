use crate::mcp::definitions::tool_definitions;
use std::collections::HashSet;

#[test]
fn tool_definitions_returns_expected_count() {
    let defs = tool_definitions();
    // The dispatch table in mod.rs has 32 tool entries (including the unknown fallback).
    // definitions.rs should define all of them.
    assert!(
        defs.len() >= 32,
        "Expected at least 32 tool definitions, got {}",
        defs.len()
    );
}

#[test]
fn every_definition_has_required_fields() {
    let defs = tool_definitions();
    for (i, def) in defs.iter().enumerate() {
        assert!(
            def.get("name").is_some(),
            "Tool definition at index {i} is missing 'name'"
        );
        assert!(
            def["name"].as_str().is_some(),
            "Tool definition at index {i} has non-string 'name'"
        );
        assert!(
            def.get("description").is_some(),
            "Tool '{}' is missing 'description'",
            def["name"]
        );
        assert!(
            def["description"].as_str().is_some(),
            "Tool '{}' has non-string 'description'",
            def["name"]
        );
        assert!(
            def.get("inputSchema").is_some(),
            "Tool '{}' is missing 'inputSchema'",
            def["name"]
        );
        assert!(
            def["inputSchema"].is_object(),
            "Tool '{}' has non-object 'inputSchema'",
            def["name"]
        );
    }
}

#[test]
fn no_duplicate_tool_names() {
    let defs = tool_definitions();
    let mut seen = HashSet::new();
    for def in &defs {
        let name = def["name"].as_str().unwrap();
        assert!(
            seen.insert(name.to_string()),
            "Duplicate tool name: {name}"
        );
    }
}

#[test]
fn input_schema_has_type_object() {
    let defs = tool_definitions();
    for def in &defs {
        let name = def["name"].as_str().unwrap();
        let schema = &def["inputSchema"];
        assert_eq!(
            schema["type"].as_str(),
            Some("object"),
            "Tool '{name}' inputSchema type should be 'object'"
        );
    }
}

#[test]
fn store_memory_has_required_content() {
    let defs = tool_definitions();
    let store = defs
        .iter()
        .find(|d| d["name"] == "store_memory")
        .expect("store_memory tool should exist");

    let schema = &store["inputSchema"];
    let required = schema["required"]
        .as_array()
        .expect("store_memory should have required array");
    let required_strs: Vec<&str> = required.iter().filter_map(|v| v.as_str()).collect();
    assert!(
        required_strs.contains(&"content"),
        "store_memory should require 'content'"
    );

    let props = schema["properties"].as_object().unwrap();
    assert!(props.contains_key("content"));
    assert!(props.contains_key("memory_type"));
    assert!(props.contains_key("tags"));
    assert!(props.contains_key("namespace"));
    assert!(props.contains_key("importance"));
}

#[test]
fn recall_has_required_query() {
    let defs = tool_definitions();
    let recall = defs
        .iter()
        .find(|d| d["name"] == "recall")
        .expect("recall tool should exist");

    let required = recall["inputSchema"]["required"]
        .as_array()
        .expect("recall should have required array");
    let required_strs: Vec<&str> = required.iter().filter_map(|v| v.as_str()).collect();
    assert!(
        required_strs.contains(&"query"),
        "recall should require 'query'"
    );
}

#[test]
fn enrichment_tools_are_defined() {
    let defs = tool_definitions();
    let names: HashSet<String> = defs
        .iter()
        .map(|d| d["name"].as_str().unwrap().to_string())
        .collect();

    for expected in [
        "enrich_codebase",
        "analyze_codebase",
        "enrich_git_history",
        "enrich_security",
        "enrich_performance",
    ] {
        assert!(
            names.contains(expected),
            "Missing enrichment tool: {expected}"
        );
    }
}

#[test]
fn enrich_codebase_requires_path() {
    let defs = tool_definitions();
    let tool = defs
        .iter()
        .find(|d| d["name"] == "enrich_codebase")
        .expect("enrich_codebase tool should exist");

    let required = tool["inputSchema"]["required"]
        .as_array()
        .expect("enrich_codebase should have required array");
    let required_strs: Vec<&str> = required.iter().filter_map(|v| v.as_str()).collect();
    assert!(
        required_strs.contains(&"path"),
        "enrich_codebase should require 'path'"
    );
}

#[test]
fn all_dispatch_tools_have_definitions() {
    // These are the tool names from dispatch_tool_inner in mod.rs
    let dispatched = vec![
        "store_memory",
        "recall",
        "delete_memory",
        "associate_memories",
        "refine_memory",
        "split_memory",
        "merge_memories",
        "graph_traverse",
        "summary_tree",
        "codemem_status",
        "index_codebase",
        "search_code",
        "get_symbol_info",
        "get_symbol_graph",
        "find_important_nodes",
        "find_related_groups",
        "get_cross_repo",
        "get_node_memories",
        "node_coverage",
        "consolidate",
        "detect_patterns",
        "get_decision_chain",
        "list_namespaces",
        "namespace_stats",
        "delete_namespace",
        "session_checkpoint",
        "session_context",
        "enrich_codebase",
        "analyze_codebase",
        "enrich_git_history",
        "enrich_security",
        "enrich_performance",
    ];

    let defs = tool_definitions();
    let defined_names: HashSet<String> = defs
        .iter()
        .map(|d| d["name"].as_str().unwrap().to_string())
        .collect();

    for tool_name in &dispatched {
        assert!(
            defined_names.contains(*tool_name),
            "Dispatched tool '{tool_name}' has no definition in tool_definitions()"
        );
    }
}

#[test]
fn description_is_nonempty() {
    let defs = tool_definitions();
    for def in &defs {
        let name = def["name"].as_str().unwrap();
        let desc = def["description"].as_str().unwrap();
        assert!(
            !desc.is_empty(),
            "Tool '{name}' has an empty description"
        );
    }
}

#[test]
fn tools_list_rpc_returns_definitions() {
    let server = crate::mcp::McpServer::for_testing();
    let resp = server.handle_request("tools/list", None, serde_json::json!(1));
    assert!(resp.error.is_none());
    let result = resp.result.unwrap();
    let tools = result["tools"].as_array().unwrap();
    assert!(
        tools.len() >= 32,
        "tools/list should return at least 32 tools, got {}",
        tools.len()
    );
}
