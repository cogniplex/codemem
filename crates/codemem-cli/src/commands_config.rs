//! `codemem config get/set` — read and modify configuration.

use codemem_core::CodememConfig;

pub(crate) fn cmd_config_get(key: &str) -> anyhow::Result<()> {
    let config = CodememConfig::load_or_default();
    let json = serde_json::to_value(&config)?;

    let value = navigate_json(&json, key);
    match value {
        Some(v) => {
            let pretty = serde_json::to_string_pretty(v)?;
            println!("{pretty}");
        }
        None => {
            anyhow::bail!("Unknown config key: {key}");
        }
    }
    Ok(())
}

pub(crate) fn cmd_config_set(key: &str, value: &str) -> anyhow::Result<()> {
    let mut config = CodememConfig::load_or_default();
    let mut json = serde_json::to_value(&config)?;

    // Parse the value as JSON first, fall back to string
    let new_value: serde_json::Value = serde_json::from_str(value)
        .unwrap_or_else(|_| serde_json::Value::String(value.to_string()));

    set_json_path(&mut json, key, new_value)?;

    // Deserialize back into config to validate
    config = serde_json::from_value(json)?;

    let config_path = CodememConfig::default_path();
    config.save(&config_path)?;
    eprintln!("Updated {key} and saved to {}", config_path.display());
    Ok(())
}

/// Navigate a JSON value by a dot-separated path.
fn navigate_json<'a>(value: &'a serde_json::Value, path: &str) -> Option<&'a serde_json::Value> {
    let parts: Vec<&str> = path.split('.').collect();
    let mut current = value;
    for part in &parts {
        current = current.get(part)?;
    }
    Some(current)
}

/// Set a value at a dot-separated JSON path.
fn set_json_path(
    root: &mut serde_json::Value,
    path: &str,
    value: serde_json::Value,
) -> anyhow::Result<()> {
    let parts: Vec<&str> = path.split('.').collect();
    if parts.is_empty() {
        anyhow::bail!("Empty key path");
    }

    let mut current = root;
    for part in &parts[..parts.len() - 1] {
        current = current
            .get_mut(*part)
            .ok_or_else(|| anyhow::anyhow!("Unknown config section: {part}"))?;
    }

    let last = parts.last().unwrap();
    if let Some(obj) = current.as_object_mut() {
        if obj.contains_key(*last) {
            obj.insert((*last).to_string(), value);
        } else {
            anyhow::bail!("Unknown config key: {last}");
        }
    } else {
        anyhow::bail!("Config path does not lead to an object");
    }

    Ok(())
}

#[cfg(test)]
mod tests {
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
        let err = set_json_path(
            &mut json,
            "scoring.nonexistent",
            serde_json::json!(1.0),
        )
        .unwrap_err();
        assert!(err.to_string().contains("Unknown config key"));
    }
}
