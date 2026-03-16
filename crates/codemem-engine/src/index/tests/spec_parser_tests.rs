use super::*;
use std::io::Write;
use tempfile::TempDir;

// ── Helpers ──────────────────────────────────────────────────────────────

fn write_temp_file(dir: &TempDir, name: &str, content: &str) -> std::path::PathBuf {
    let path = dir.path().join(name);
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).unwrap();
    }
    let mut f = std::fs::File::create(&path).unwrap();
    f.write_all(content.as_bytes()).unwrap();
    path
}

fn write_json_file(dir: &TempDir, name: &str, value: &serde_json::Value) -> std::path::PathBuf {
    let content = serde_json::to_string_pretty(value).unwrap();
    write_temp_file(dir, name, &content)
}

// ── OpenAPI 3.x Tests ───────────────────────────────────────────────────

#[test]
fn test_parse_openapi3_json() {
    let dir = TempDir::new().unwrap();
    let spec = serde_json::json!({
        "openapi": "3.0.3",
        "info": { "title": "Pet Store", "version": "1.0.0" },
        "paths": {
            "/pets": {
                "get": {
                    "operationId": "listPets",
                    "summary": "List all pets",
                    "responses": {
                        "200": {
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "array",
                                        "items": { "$ref": "#/components/schemas/Pet" }
                                    }
                                }
                            }
                        }
                    }
                },
                "post": {
                    "operationId": "createPet",
                    "description": "Create a pet",
                    "requestBody": {
                        "content": {
                            "application/json": {
                                "schema": { "$ref": "#/components/schemas/Pet" }
                            }
                        }
                    },
                    "responses": {
                        "201": {
                            "content": {
                                "application/json": {
                                    "schema": { "$ref": "#/components/schemas/Pet" }
                                }
                            }
                        }
                    }
                }
            },
            "/pets/{petId}": {
                "get": {
                    "operationId": "showPetById",
                    "summary": "Info for a specific pet",
                    "responses": {
                        "200": {
                            "content": {
                                "application/json": {
                                    "schema": { "$ref": "#/components/schemas/Pet" }
                                }
                            }
                        }
                    }
                }
            }
        }
    });

    let path = write_json_file(&dir, "openapi.json", &spec);
    let result = parse_openapi(&path).unwrap();

    assert_eq!(result.title.as_deref(), Some("Pet Store"));
    assert_eq!(result.version.as_deref(), Some("1.0.0"));
    assert_eq!(result.endpoints.len(), 3);

    let get_pets = result
        .endpoints
        .iter()
        .find(|e| e.method == "GET" && e.path == "/pets")
        .unwrap();
    assert_eq!(get_pets.operation_id.as_deref(), Some("listPets"));
    assert_eq!(get_pets.description.as_deref(), Some("List all pets"));
    assert!(get_pets.response_schema.is_some());
    assert!(get_pets.request_schema.is_none());

    let create_pet = result
        .endpoints
        .iter()
        .find(|e| e.method == "POST")
        .unwrap();
    assert_eq!(create_pet.operation_id.as_deref(), Some("createPet"));
    assert!(create_pet.request_schema.is_some());
    assert!(create_pet.response_schema.is_some());

    let get_by_id = result
        .endpoints
        .iter()
        .find(|e| e.path == "/pets/{petId}")
        .unwrap();
    assert_eq!(get_by_id.operation_id.as_deref(), Some("showPetById"));
}

#[test]
fn test_parse_openapi3_yaml() {
    let dir = TempDir::new().unwrap();
    let content = "\
openapi: \"3.0.0\"
info:
  title: Users API
  version: \"2.0\"
paths:
  /users:
    get:
      operationId: getUsers
      summary: Get all users
      responses:
        \"200\":
          content:
            application/json:
              schema:
                type: array
";

    let path = write_temp_file(&dir, "openapi.yaml", content);
    let result = parse_openapi(&path).unwrap();

    assert_eq!(result.title.as_deref(), Some("Users API"));
    assert_eq!(result.version.as_deref(), Some("2.0"));
    assert_eq!(result.endpoints.len(), 1);
    assert_eq!(result.endpoints[0].method, "GET");
    assert_eq!(result.endpoints[0].path, "/users");
}

// ── Swagger 2.0 Tests ───────────────────────────────────────────────────

#[test]
fn test_parse_swagger2_json() {
    let dir = TempDir::new().unwrap();
    let spec = serde_json::json!({
        "swagger": "2.0",
        "info": { "title": "Legacy API", "version": "1.0" },
        "paths": {
            "/items": {
                "post": {
                    "operationId": "createItem",
                    "summary": "Create item",
                    "parameters": [
                        {
                            "in": "body",
                            "name": "body",
                            "schema": { "$ref": "#/definitions/Item" }
                        }
                    ],
                    "responses": {
                        "200": {
                            "schema": { "$ref": "#/definitions/Item" }
                        }
                    }
                }
            }
        }
    });

    let path = write_json_file(&dir, "swagger.json", &spec);
    let result = parse_openapi(&path).unwrap();

    assert_eq!(result.title.as_deref(), Some("Legacy API"));
    assert_eq!(result.endpoints.len(), 1);

    let ep = &result.endpoints[0];
    assert_eq!(ep.method, "POST");
    assert_eq!(ep.path, "/items");
    assert!(ep.request_schema.is_some());
    assert!(ep.response_schema.is_some());
}

// ── AsyncAPI 2.x Tests ──────────────────────────────────────────────────

#[test]
fn test_parse_asyncapi2_yaml() {
    let dir = TempDir::new().unwrap();
    let content = "\
asyncapi: \"2.6.0\"
info:
  title: Events Service
  version: \"1.0.0\"
servers:
  production:
    url: broker.example.com
    protocol: kafka
channels:
  user.created:
    publish:
      operationId: publishUserCreated
      description: User was created
      message:
        payload:
          type: object
          properties:
            userId:
              type: string
    subscribe:
      operationId: onUserCreated
      description: Listen for user creation
      message:
        payload:
          type: object
";

    let path = write_temp_file(&dir, "asyncapi.yaml", content);
    let result = parse_asyncapi(&path).unwrap();

    assert_eq!(result.title.as_deref(), Some("Events Service"));
    assert_eq!(result.version.as_deref(), Some("1.0.0"));
    assert_eq!(result.channels.len(), 2);

    let pub_ch = result
        .channels
        .iter()
        .find(|c| c.direction == "publish")
        .unwrap();
    assert_eq!(pub_ch.channel, "user.created");
    assert_eq!(pub_ch.operation_id.as_deref(), Some("publishUserCreated"));
    assert_eq!(pub_ch.protocol.as_deref(), Some("kafka"));
    assert!(pub_ch.message_schema.is_some());

    let sub_ch = result
        .channels
        .iter()
        .find(|c| c.direction == "subscribe")
        .unwrap();
    assert_eq!(sub_ch.operation_id.as_deref(), Some("onUserCreated"));
}

// ── AsyncAPI 3.0 Tests ──────────────────────────────────────────────────

#[test]
fn test_parse_asyncapi3_json() {
    let dir = TempDir::new().unwrap();
    let spec = serde_json::json!({
        "asyncapi": "3.0.0",
        "info": { "title": "Order Events", "version": "2.0.0" },
        "servers": {
            "main": { "host": "rabbitmq.example.com", "protocol": "amqp" }
        },
        "channels": {
            "orderChannel": {
                "messages": {
                    "OrderCreated": {
                        "payload": {
                            "type": "object",
                            "properties": { "orderId": { "type": "string" } }
                        }
                    }
                }
            }
        },
        "operations": {
            "sendOrder": {
                "action": "send",
                "channel": { "$ref": "#/channels/orderChannel" },
                "operationId": "sendOrder",
                "description": "Send order event",
                "messages": {
                    "OrderCreated": {
                        "payload": { "type": "object" }
                    }
                }
            },
            "receiveOrder": {
                "action": "receive",
                "channel": { "$ref": "#/channels/orderChannel" },
                "operationId": "receiveOrder"
            }
        }
    });

    let path = write_json_file(&dir, "asyncapi.json", &spec);
    let result = parse_asyncapi(&path).unwrap();

    assert_eq!(result.title.as_deref(), Some("Order Events"));
    assert_eq!(result.version.as_deref(), Some("2.0.0"));
    assert_eq!(result.channels.len(), 2);

    let send = result
        .channels
        .iter()
        .find(|c| c.direction == "publish")
        .unwrap();
    assert_eq!(send.channel, "orderChannel");
    assert_eq!(send.operation_id.as_deref(), Some("sendOrder"));
    assert_eq!(send.protocol.as_deref(), Some("amqp"));
    assert!(send.message_schema.is_some());

    let recv = result
        .channels
        .iter()
        .find(|c| c.direction == "subscribe")
        .unwrap();
    assert_eq!(recv.operation_id.as_deref(), Some("receiveOrder"));
    // Falls back to channel-level message since operation has no messages
    assert!(recv.message_schema.is_some());
}

// ── Edge Cases ───────────────────────────────────────────────────────────

#[test]
fn test_nonexistent_file_returns_none() {
    let path = std::path::Path::new("/tmp/does-not-exist-codemem-test.json");
    assert!(parse_openapi(path).is_none());
    assert!(parse_asyncapi(path).is_none());
}

#[test]
fn test_invalid_json_returns_none() {
    let dir = TempDir::new().unwrap();
    let path = write_temp_file(&dir, "broken.json", "{ not valid json }}}");
    assert!(parse_openapi(&path).is_none());
}

#[test]
fn test_valid_json_but_not_spec_returns_none() {
    let dir = TempDir::new().unwrap();
    let spec = serde_json::json!({ "name": "hello", "version": "1.0" });
    let path = write_json_file(&dir, "config.json", &spec);
    assert!(parse_openapi(&path).is_none());
    assert!(parse_asyncapi(&path).is_none());
}

#[test]
fn test_openapi_missing_paths_returns_empty_endpoints() {
    let dir = TempDir::new().unwrap();
    let spec = serde_json::json!({
        "openapi": "3.0.0",
        "info": { "title": "Empty", "version": "0.1" }
    });
    let path = write_json_file(&dir, "openapi.json", &spec);
    let result = parse_openapi(&path).unwrap();
    assert!(result.endpoints.is_empty());
    assert_eq!(result.title.as_deref(), Some("Empty"));
}

#[test]
fn test_asyncapi_missing_channels_returns_empty() {
    let dir = TempDir::new().unwrap();
    let spec = serde_json::json!({
        "asyncapi": "2.0.0",
        "info": { "title": "No channels", "version": "0.1" }
    });
    let path = write_json_file(&dir, "asyncapi.json", &spec);
    let result = parse_asyncapi(&path).unwrap();
    assert!(result.channels.is_empty());
}

#[test]
fn test_openapi_path_normalization() {
    let dir = TempDir::new().unwrap();
    let spec = serde_json::json!({
        "openapi": "3.0.0",
        "info": { "title": "Test", "version": "1" },
        "paths": {
            "users/{userId}/orders": {
                "get": { "operationId": "getUserOrders" }
            }
        }
    });
    let path = write_json_file(&dir, "api.json", &spec);
    let result = parse_openapi(&path).unwrap();
    assert_eq!(result.endpoints.len(), 1);
    // normalize_path_pattern adds leading slash if missing
    assert!(result.endpoints[0].path.starts_with('/'));
}

// ── Directory Scanning ───────────────────────────────────────────────────

#[test]
fn test_scan_api_specs_finds_well_known_files() {
    let dir = TempDir::new().unwrap();

    let openapi_spec = serde_json::json!({
        "openapi": "3.0.0",
        "info": { "title": "Scan Test", "version": "1" },
        "paths": {
            "/health": { "get": { "operationId": "healthCheck" } }
        }
    });

    let asyncapi_spec = serde_json::json!({
        "asyncapi": "2.0.0",
        "info": { "title": "Events", "version": "1" },
        "channels": {
            "events": {
                "publish": { "operationId": "pubEvent" }
            }
        }
    });

    write_json_file(&dir, "openapi.json", &openapi_spec);
    write_json_file(&dir, "asyncapi.json", &asyncapi_spec);
    // Non-spec file should be ignored
    let non_spec = serde_json::json!({ "key": "value" });
    write_json_file(&dir, "config.json", &non_spec);

    let results = scan_api_specs(dir.path());
    assert_eq!(results.len(), 2);

    let has_openapi = results
        .iter()
        .any(|r| matches!(r, SpecFileResult::OpenApi(_)));
    let has_asyncapi = results
        .iter()
        .any(|r| matches!(r, SpecFileResult::AsyncApi(_)));
    assert!(has_openapi);
    assert!(has_asyncapi);
}

#[test]
fn test_scan_detects_non_well_known_spec_by_peeking() {
    let dir = TempDir::new().unwrap();

    let spec = serde_json::json!({
        "openapi": "3.0.0",
        "info": { "title": "Custom Named", "version": "1" },
        "paths": {}
    });

    write_json_file(&dir, "my-api-spec.json", &spec);

    let results = scan_api_specs(dir.path());
    assert_eq!(results.len(), 1);

    if let SpecFileResult::OpenApi(result) = &results[0] {
        assert_eq!(result.title.as_deref(), Some("Custom Named"));
    } else {
        panic!("Expected OpenApi result");
    }
}

#[test]
fn test_scan_ignores_non_spec_yaml() {
    let dir = TempDir::new().unwrap();
    let content = "name: my-config\nversion: 1\nkey: value\n";
    write_temp_file(&dir, "config.yaml", content);

    let results = scan_api_specs(dir.path());
    assert!(results.is_empty());
}
