use super::*;
use crate::index::symbol::{Reference, ReferenceKind, Symbol, SymbolKind, Visibility};

fn make_symbol(
    name: &str,
    qn: &str,
    attrs: Vec<&str>,
    kind: SymbolKind,
    parent: Option<&str>,
) -> Symbol {
    Symbol {
        name: name.to_string(),
        qualified_name: qn.to_string(),
        kind,
        signature: String::new(),
        visibility: Visibility::Public,
        file_path: "test.py".to_string(),
        line_start: 1,
        line_end: 10,
        doc_comment: None,
        parent: parent.map(|s| s.to_string()),
        parameters: Vec::new(),
        return_type: None,
        is_async: false,
        attributes: attrs.into_iter().map(|s| s.to_string()).collect(),
        throws: Vec::new(),
        generic_params: None,
        is_abstract: false,
    }
}

fn make_ref(source: &str, target: &str, kind: ReferenceKind) -> Reference {
    Reference {
        source_qualified_name: source.to_string(),
        target_name: target.to_string(),
        kind,
        file_path: "test.py".to_string(),
        line: 1,
    }
}

// ── Endpoint detection tests ──

#[test]
fn detect_flask_route() {
    let sym = make_symbol(
        "get_users",
        "app.get_users",
        vec![r#"@app.route("/users", methods=["GET"])"#],
        SymbolKind::Function,
        None,
    );
    let eps = detect_endpoints(&[sym], "myapp");
    assert_eq!(eps.len(), 1);
    assert_eq!(eps[0].method, Some("GET".to_string()));
    assert_eq!(eps[0].path, "/users");
    assert_eq!(eps[0].handler, "app.get_users");
    assert!(eps[0].id.starts_with("ep:myapp:GET:"));
}

#[test]
fn detect_fastapi_route() {
    let sym = make_symbol(
        "get_user",
        "api.get_user",
        vec![r#"@router.get("/users/{id}")"#],
        SymbolKind::Function,
        None,
    );
    let eps = detect_endpoints(&[sym], "myapi");
    assert_eq!(eps.len(), 1);
    assert_eq!(eps[0].method, Some("GET".to_string()));
    assert_eq!(eps[0].path, "/users/{id}");
}

#[test]
fn detect_spring_get_mapping() {
    let sym = make_symbol(
        "getUsers",
        "UserController.getUsers",
        vec![r#"@GetMapping("/api/users")"#],
        SymbolKind::Method,
        Some("UserController"),
    );
    let eps = detect_endpoints(&[sym], "svc");
    assert_eq!(eps.len(), 1);
    assert_eq!(eps[0].method, Some("GET".to_string()));
    assert_eq!(eps[0].path, "/api/users");
    assert_eq!(eps[0].handler, "UserController.getUsers");
}

#[test]
fn detect_spring_request_mapping() {
    let sym = make_symbol(
        "createUser",
        "UserController.createUser",
        vec![r#"@RequestMapping(value="/users", method=RequestMethod.POST)"#],
        SymbolKind::Method,
        Some("UserController"),
    );
    let eps = detect_endpoints(&[sym], "svc");
    assert_eq!(eps.len(), 1);
    assert_eq!(eps[0].method, Some("POST".to_string()));
    assert_eq!(eps[0].path, "/users");
}

#[test]
fn detect_nestjs_decorator() {
    let sym = make_symbol(
        "findAll",
        "UsersController.findAll",
        vec![r#"@Get("/users")"#],
        SymbolKind::Method,
        Some("UsersController"),
    );
    let eps = detect_endpoints(&[sym], "nest");
    assert_eq!(eps.len(), 1);
    assert_eq!(eps[0].method, Some("GET".to_string()));
    assert_eq!(eps[0].path, "/users");
}

#[test]
fn detect_nestjs_no_path() {
    // NestJS @Post() with no path defaults to "/"
    let sym = make_symbol(
        "create",
        "UsersController.create",
        vec!["@Post()"],
        SymbolKind::Method,
        Some("UsersController"),
    );
    let eps = detect_endpoints(&[sym], "nest");
    assert_eq!(eps.len(), 1);
    assert_eq!(eps[0].method, Some("POST".to_string()));
    assert_eq!(eps[0].path, "/");
}

#[test]
fn detect_django_cbv_methods() {
    let sym = make_symbol(
        "get",
        "UserView.get",
        vec![],
        SymbolKind::Method,
        Some("UserView"),
    );
    let eps = detect_endpoints(&[sym], "django");
    assert_eq!(eps.len(), 1);
    assert_eq!(eps[0].method, Some("GET".to_string()));
    assert_eq!(eps[0].path, "view:UserView");
    assert_eq!(eps[0].handler, "UserView.get");
}

#[test]
fn no_endpoint_for_plain_function() {
    let sym = make_symbol("helper", "utils.helper", vec![], SymbolKind::Function, None);
    let eps = detect_endpoints(&[sym], "app");
    assert!(eps.is_empty());
}

// ── Client call detection tests ──

#[test]
fn detect_python_requests_get() {
    let refs = vec![make_ref("my_func", "requests.get", ReferenceKind::Call)];
    let calls = detect_client_calls(&refs);
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].client_library, "requests");
    assert_eq!(calls[0].method, Some("GET".to_string()));
    assert_eq!(calls[0].caller, "my_func");
}

#[test]
fn detect_python_httpx_post() {
    let refs = vec![make_ref("sender", "httpx.post", ReferenceKind::Call)];
    let calls = detect_client_calls(&refs);
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].client_library, "httpx");
    assert_eq!(calls[0].method, Some("POST".to_string()));
}

#[test]
fn detect_fetch_call() {
    let refs = vec![make_ref("loadData", "fetch", ReferenceKind::Call)];
    let calls = detect_client_calls(&refs);
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].client_library, "fetch");
    assert_eq!(calls[0].method, None); // method is in options arg
}

#[test]
fn detect_axios_post() {
    let refs = vec![make_ref("submitForm", "axios.post", ReferenceKind::Call)];
    let calls = detect_client_calls(&refs);
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].client_library, "axios");
    assert_eq!(calls[0].method, Some("POST".to_string()));
}

#[test]
fn detect_go_http_get() {
    let refs = vec![make_ref("fetchData", "http.Get", ReferenceKind::Call)];
    let calls = detect_client_calls(&refs);
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].client_library, "net/http");
    assert_eq!(calls[0].method, Some("GET".to_string()));
}

#[test]
fn detect_go_http_new_request() {
    let refs = vec![make_ref("doReq", "http.NewRequest", ReferenceKind::Call)];
    let calls = detect_client_calls(&refs);
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].client_library, "net/http");
    assert_eq!(calls[0].method, None); // method passed as argument
}

#[test]
fn detect_java_rest_template() {
    let refs = vec![make_ref(
        "callService",
        "restTemplate.getForObject",
        ReferenceKind::Call,
    )];
    let calls = detect_client_calls(&refs);
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].client_library, "restTemplate");
}

#[test]
fn ignore_non_call_references() {
    let refs = vec![make_ref("mod", "requests.get", ReferenceKind::Import)];
    let calls = detect_client_calls(&refs);
    assert!(calls.is_empty());
}

#[test]
fn ignore_unrelated_call() {
    let refs = vec![make_ref("foo", "bar.baz", ReferenceKind::Call)];
    let calls = detect_client_calls(&refs);
    assert!(calls.is_empty());
}

// ── Path normalization tests ──

#[test]
fn normalize_flask_params() {
    assert_eq!(normalize_path_pattern("/users/<int:id>"), "/users/{id}");
}

#[test]
fn normalize_flask_simple_param() {
    assert_eq!(normalize_path_pattern("/users/<name>"), "/users/{name}");
}

#[test]
fn normalize_express_params() {
    assert_eq!(normalize_path_pattern("/users/:id"), "/users/{id}");
}

#[test]
fn normalize_already_normalized() {
    assert_eq!(normalize_path_pattern("/users/{id}"), "/users/{id}");
}

#[test]
fn normalize_adds_leading_slash() {
    assert_eq!(normalize_path_pattern("users"), "/users");
}

#[test]
fn normalize_removes_trailing_slash() {
    assert_eq!(normalize_path_pattern("/users/"), "/users");
}

#[test]
fn normalize_root_path_preserved() {
    assert_eq!(normalize_path_pattern("/"), "/");
}

#[test]
fn normalize_multiple_params() {
    assert_eq!(
        normalize_path_pattern("/orgs/:orgId/users/:userId"),
        "/orgs/{orgId}/users/{userId}"
    );
}

// ── Endpoint matching tests ──

#[test]
fn match_exact_path() {
    let eps = vec![DetectedEndpoint {
        id: "ep:ns:GET:/users".to_string(),
        method: Some("GET".to_string()),
        path: "/users".to_string(),
        handler: "getUsers".to_string(),
        file_path: "app.py".to_string(),
        line: 1,
    }];
    let result = match_endpoint("/users", Some("GET"), &eps);
    assert!(result.is_some());
    let (ep, confidence) = result.unwrap();
    assert_eq!(ep.path, "/users");
    assert!(confidence > 0.9);
}

#[test]
fn match_with_param_substitution() {
    let eps = vec![DetectedEndpoint {
        id: "ep:ns:GET:/users/{id}".to_string(),
        method: Some("GET".to_string()),
        path: "/users/{id}".to_string(),
        handler: "getUser".to_string(),
        file_path: "app.py".to_string(),
        line: 1,
    }];
    let result = match_endpoint("/users/123", None, &eps);
    assert!(result.is_some());
    let (ep, confidence) = result.unwrap();
    assert_eq!(ep.path, "/users/{id}");
    assert!(confidence >= 0.9);
}

#[test]
fn match_method_boost() {
    let eps = vec![
        DetectedEndpoint {
            id: "ep:ns:GET:/users".to_string(),
            method: Some("GET".to_string()),
            path: "/users".to_string(),
            handler: "listUsers".to_string(),
            file_path: "app.py".to_string(),
            line: 1,
        },
        DetectedEndpoint {
            id: "ep:ns:POST:/users".to_string(),
            method: Some("POST".to_string()),
            path: "/users".to_string(),
            handler: "createUser".to_string(),
            file_path: "app.py".to_string(),
            line: 10,
        },
    ];
    // With POST method, should prefer the POST endpoint
    let result = match_endpoint("/users", Some("POST"), &eps);
    assert!(result.is_some());
    let (ep, _) = result.unwrap();
    assert_eq!(ep.handler, "createUser");
}

#[test]
fn no_match_below_threshold() {
    let eps = vec![DetectedEndpoint {
        id: "ep:ns:GET:/completely/different".to_string(),
        method: Some("GET".to_string()),
        path: "/completely/different".to_string(),
        handler: "other".to_string(),
        file_path: "app.py".to_string(),
        line: 1,
    }];
    let result = match_endpoint("/users", None, &eps);
    assert!(result.is_none());
}

#[test]
fn paths_match_with_params_same_length() {
    assert!(paths_match_with_params("/users/42", "/users/{id}"));
    assert!(paths_match_with_params(
        "/orgs/acme/users/42",
        "/orgs/{org}/users/{id}"
    ));
}

#[test]
fn paths_match_with_params_different_length() {
    assert!(!paths_match_with_params("/users", "/users/{id}"));
    assert!(!paths_match_with_params("/users/42/posts", "/users/{id}"));
}

#[test]
fn match_empty_endpoints_returns_none() {
    let result = match_endpoint("/users", Some("GET"), &[]);
    assert!(result.is_none());
}
