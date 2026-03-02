use super::*;

#[test]
fn navigate_top_level() {
    let json = serde_json::json!({"scoring": {"vector_similarity": 0.25}});
    let v = navigate_json(&json, "scoring").unwrap();
    assert!(v.is_object());
}

#[test]
fn navigate_nested() {
    let json = serde_json::json!({"scoring": {"vector_similarity": 0.25}});
    let v = navigate_json(&json, "scoring.vector_similarity").unwrap();
    assert_eq!(v.as_f64().unwrap(), 0.25);
}

#[test]
fn navigate_missing() {
    let json = serde_json::json!({"scoring": {}});
    assert!(navigate_json(&json, "scoring.nonexistent").is_none());
}

#[test]
fn set_json_path_works() {
    let mut json = serde_json::json!({"scoring": {"vector_similarity": 0.25}});
    set_json_path(
        &mut json,
        "scoring.vector_similarity",
        serde_json::json!(0.5),
    )
    .unwrap();
    assert_eq!(json["scoring"]["vector_similarity"], 0.5);
}

#[test]
fn navigate_empty_path_returns_none() {
    let json = serde_json::json!({"scoring": {"vector_similarity": 0.25}});
    assert!(navigate_json(&json, "").is_none());
}

#[test]
fn navigate_three_levels_deep() {
    let json = serde_json::json!({"a": {"b": {"c": 42}}});
    let v = navigate_json(&json, "a.b.c").unwrap();
    assert_eq!(v.as_i64().unwrap(), 42);
}

#[test]
fn set_json_path_top_level_key() {
    let mut json = serde_json::json!({"debug": false});
    set_json_path(&mut json, "debug", serde_json::json!(true)).unwrap();
    assert_eq!(json["debug"], true);
}

#[test]
fn set_json_path_unknown_key_errors() {
    let mut json = serde_json::json!({"scoring": {}});
    let err = set_json_path(&mut json, "scoring.nonexistent", serde_json::json!(1.0)).unwrap_err();
    assert!(err.to_string().contains("Unknown config key"));
}
