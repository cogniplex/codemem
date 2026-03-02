use super::*;

#[test]
fn write_response_newline_delimited() {
    let resp = JsonRpcResponse::success(json!(1), json!({"ok": true}));
    let mut buf = Vec::new();
    write_response(&mut buf, &resp).unwrap();
    let output = String::from_utf8(buf).unwrap();
    assert!(output.ends_with('\n'));
    assert!(!output.contains("Content-Length"));
}
