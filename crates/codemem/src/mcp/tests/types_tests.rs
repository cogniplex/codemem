use super::*;

#[test]
fn parse_json_rpc_request() {
    let json = r#"{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}"#;
    let req: JsonRpcRequest = serde_json::from_str(json).unwrap();
    assert_eq!(req.method, "initialize");
    assert!(req.id.is_some());
}

#[test]
fn parse_notification_no_id() {
    let json = r#"{"jsonrpc":"2.0","method":"notifications/initialized"}"#;
    let req: JsonRpcRequest = serde_json::from_str(json).unwrap();
    assert!(req.id.is_none());
}

#[test]
fn tool_result_serialization() {
    let result = ToolResult::text("hello");
    let json = serde_json::to_value(&result).unwrap();
    assert_eq!(json["content"][0]["type"], "text");
    assert_eq!(json["content"][0]["text"], "hello");
    assert_eq!(json["isError"], false);
}

#[test]
fn tool_error_serialization() {
    let result = ToolResult::tool_error("something went wrong");
    let json = serde_json::to_value(&result).unwrap();
    assert_eq!(json["isError"], true);
}
