use super::*;
use clap::Parser;

#[test]
fn truncate_str_short() {
    assert_eq!(truncate_str("hi", 10), "hi");
}

#[test]
fn truncate_str_exact() {
    assert_eq!(truncate_str("hello", 5), "hello");
}

#[test]
fn truncate_str_long() {
    assert_eq!(truncate_str("hello world", 5), "hello...");
}

#[test]
fn truncate_str_empty() {
    assert_eq!(truncate_str("", 5), "");
}

#[test]
fn parse_search_command() {
    let cli = Cli::try_parse_from(["codemem", "search", "query text"]).unwrap();
    match cli.command {
        Commands::Search {
            query,
            k,
            namespace,
        } => {
            assert_eq!(query, "query text");
            assert_eq!(k, 10); // default
            assert!(namespace.is_none());
        }
        _ => panic!("Expected Search command"),
    }
}

#[test]
fn parse_search_with_options() {
    let cli = Cli::try_parse_from([
        "codemem",
        "search",
        "my query",
        "-k",
        "5",
        "--namespace",
        "my-project",
    ])
    .unwrap();
    match cli.command {
        Commands::Search {
            query,
            k,
            namespace,
        } => {
            assert_eq!(query, "my query");
            assert_eq!(k, 5);
            assert_eq!(namespace, Some("my-project".to_string()));
        }
        _ => panic!("Expected Search command"),
    }
}

#[test]
fn parse_stats_command() {
    let cli = Cli::try_parse_from(["codemem", "stats"]).unwrap();
    assert!(matches!(cli.command, Commands::Stats));
}

#[test]
fn parse_init_command() {
    let cli = Cli::try_parse_from(["codemem", "init"]).unwrap();
    match cli.command {
        Commands::Init { path, skip_model } => {
            assert!(path.is_none());
            assert!(!skip_model);
        }
        _ => panic!("Expected Init command"),
    }
}

#[test]
fn parse_init_with_skip_model() {
    let cli = Cli::try_parse_from(["codemem", "init", "--skip-model"]).unwrap();
    match cli.command {
        Commands::Init { skip_model, .. } => {
            assert!(skip_model);
        }
        _ => panic!("Expected Init command"),
    }
}

#[test]
fn parse_export_command() {
    let cli = Cli::try_parse_from([
        "codemem",
        "export",
        "--namespace",
        "test-ns",
        "--memory-type",
        "decision",
    ])
    .unwrap();
    match cli.command {
        Commands::Export {
            namespace,
            memory_type,
            output,
            format,
        } => {
            assert_eq!(namespace, Some("test-ns".to_string()));
            assert_eq!(memory_type, Some("decision".to_string()));
            assert!(output.is_none());
            assert_eq!(format, "jsonl"); // default
        }
        _ => panic!("Expected Export command"),
    }
}

#[test]
fn parse_export_with_format() {
    let cli = Cli::try_parse_from(["codemem", "export", "--format", "csv"]).unwrap();
    match cli.command {
        Commands::Export { format, .. } => {
            assert_eq!(format, "csv");
        }
        _ => panic!("Expected Export command"),
    }
}

#[test]
fn parse_import_command() {
    let cli = Cli::try_parse_from(["codemem", "import", "--skip-duplicates"]).unwrap();
    match cli.command {
        Commands::Import {
            input,
            skip_duplicates,
        } => {
            assert!(input.is_none());
            assert!(skip_duplicates);
        }
        _ => panic!("Expected Import command"),
    }
}

#[test]
fn parse_consolidate_command() {
    let cli = Cli::try_parse_from(["codemem", "consolidate", "-c", "creative"]).unwrap();
    match cli.command {
        Commands::Consolidate { cycle, status } => {
            assert_eq!(cycle, "creative");
            assert!(!status);
        }
        _ => panic!("Expected Consolidate command"),
    }
}

#[test]
fn parse_sessions_list() {
    let cli = Cli::try_parse_from(["codemem", "sessions", "list"]).unwrap();
    match cli.command {
        Commands::Sessions { action } => match action {
            SessionAction::List { namespace } => {
                assert!(namespace.is_none());
            }
            _ => panic!("Expected List action"),
        },
        _ => panic!("Expected Sessions command"),
    }
}

#[test]
fn parse_sessions_end() {
    let cli =
        Cli::try_parse_from(["codemem", "sessions", "end", "sess-123", "-s", "done"]).unwrap();
    match cli.command {
        Commands::Sessions { action } => match action {
            SessionAction::End { id, summary } => {
                assert_eq!(id, "sess-123");
                assert_eq!(summary, Some("done".to_string()));
            }
            _ => panic!("Expected End action"),
        },
        _ => panic!("Expected Sessions command"),
    }
}

#[test]
fn parse_unknown_command_fails() {
    assert!(Cli::try_parse_from(["codemem", "unknown"]).is_err());
}

#[test]
fn codemem_db_path_returns_valid_path() {
    let path = codemem_db_path();
    assert!(path.ends_with("codemem.db"));
    assert!(path.to_string_lossy().contains(".codemem"));
}
