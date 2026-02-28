//! HCL (HashiCorp Configuration Language) extractor using tree-sitter-hcl.

use crate::extractor::LanguageExtractor;
use crate::symbol::{Reference, ReferenceKind, Symbol, SymbolKind, Visibility};
use tree_sitter::{Node, Tree};

/// HCL language extractor for tree-sitter-based code indexing.
pub struct HclExtractor;

impl HclExtractor {
    pub fn new() -> Self {
        Self
    }
}

impl Default for HclExtractor {
    fn default() -> Self {
        Self::new()
    }
}

impl LanguageExtractor for HclExtractor {
    fn language_name(&self) -> &str {
        "hcl"
    }

    fn file_extensions(&self) -> &[&str] {
        &["tf", "hcl", "tfvars"]
    }

    fn tree_sitter_language(&self) -> tree_sitter::Language {
        tree_sitter_hcl::LANGUAGE.into()
    }

    fn extract_symbols(&self, tree: &Tree, source: &[u8], file_path: &str) -> Vec<Symbol> {
        let mut symbols = Vec::new();
        let root = tree.root_node();
        extract_symbols_recursive(root, source, file_path, &[], &mut symbols);
        symbols
    }

    fn extract_references(&self, tree: &Tree, source: &[u8], file_path: &str) -> Vec<Reference> {
        let mut references = Vec::new();
        let root = tree.root_node();
        extract_references_recursive(root, source, file_path, &[], &mut references);
        references
    }
}

// ── Symbol Extraction ─────────────────────────────────────────────────────

fn extract_symbols_recursive(
    node: Node,
    source: &[u8],
    file_path: &str,
    scope: &[String],
    symbols: &mut Vec<Symbol>,
) {
    match node.kind() {
        "block" => {
            extract_block_symbol(node, source, file_path, scope, symbols);
            return; // Don't default-recurse; extract_block_symbol handles children
        }
        "attribute" => {
            // Only extract top-level attributes (scope is empty)
            if scope.is_empty() {
                if let Some(sym) = extract_attribute_symbol(node, source, file_path, scope) {
                    symbols.push(sym);
                }
            }
            return;
        }
        _ => {}
    }

    // Default recursion for other node types
    for i in 0..node.child_count() {
        if let Some(child) = node.child(i as u32) {
            extract_symbols_recursive(child, source, file_path, scope, symbols);
        }
    }
}

/// Extract a symbol from an HCL block node.
///
/// HCL blocks have this structure: `identifier (string_lit | identifier)* block_start body? block_end`
/// For example: `resource "aws_s3_bucket" "my_bucket" { ... }`
///   - child 0: identifier "resource"
///   - child 1: string_lit "aws_s3_bucket"
///   - child 2: string_lit "my_bucket"
///   - child 3: block_start "{"
///   - child 4: body (optional)
///   - child 5: block_end "}"
fn extract_block_symbol(
    node: Node,
    source: &[u8],
    file_path: &str,
    scope: &[String],
    symbols: &mut Vec<Symbol>,
) {
    let (block_type, labels) = extract_block_type_and_labels(node, source);

    match block_type.as_str() {
        "resource" | "data" => {
            // resource "type" "name" or data "type" "name"
            // name = last label, qualified_name = block_type.label0.label1...
            let name = labels.last().cloned().unwrap_or_else(|| block_type.clone());
            let qualified_name = build_qualified_name(&block_type, &labels, scope);
            let signature = extract_block_signature(node, source);
            let doc_comment = extract_hcl_comment(node, source);

            symbols.push(Symbol {
                name,
                qualified_name: qualified_name.clone(),
                kind: SymbolKind::Class,
                signature,
                visibility: Visibility::Public,
                file_path: file_path.to_string(),
                line_start: node.start_position().row,
                line_end: node.end_position().row,
                doc_comment,
                parent: scope.last().cloned(),
            });

            // Recurse into body with updated scope for nested blocks
            let mut new_scope = scope.to_vec();
            new_scope.push(qualified_name);
            if let Some(body) = find_body(node) {
                for i in 0..body.child_count() {
                    if let Some(child) = body.child(i as u32) {
                        extract_symbols_recursive(child, source, file_path, &new_scope, symbols);
                    }
                }
            }
        }
        "module" | "provider" => {
            let name = labels.first().cloned().unwrap_or_else(|| block_type.clone());
            let qualified_name = build_qualified_name(&block_type, &labels, scope);
            let signature = extract_block_signature(node, source);
            let doc_comment = extract_hcl_comment(node, source);

            symbols.push(Symbol {
                name,
                qualified_name: qualified_name.clone(),
                kind: SymbolKind::Module,
                signature,
                visibility: Visibility::Public,
                file_path: file_path.to_string(),
                line_start: node.start_position().row,
                line_end: node.end_position().row,
                doc_comment,
                parent: scope.last().cloned(),
            });

            let mut new_scope = scope.to_vec();
            new_scope.push(qualified_name);
            if let Some(body) = find_body(node) {
                for i in 0..body.child_count() {
                    if let Some(child) = body.child(i as u32) {
                        extract_symbols_recursive(child, source, file_path, &new_scope, symbols);
                    }
                }
            }
        }
        "variable" | "output" => {
            let name = labels.first().cloned().unwrap_or_else(|| block_type.clone());
            let qualified_name = build_qualified_name(&block_type, &labels, scope);
            let signature = extract_block_signature(node, source);
            let doc_comment = extract_hcl_comment(node, source);

            symbols.push(Symbol {
                name,
                qualified_name,
                kind: SymbolKind::Constant,
                signature,
                visibility: Visibility::Public,
                file_path: file_path.to_string(),
                line_start: node.start_position().row,
                line_end: node.end_position().row,
                doc_comment,
                parent: scope.last().cloned(),
            });
            // No recursion needed for variable/output bodies
        }
        "locals" => {
            // locals blocks contain attribute definitions that are constants
            let doc_comment = extract_hcl_comment(node, source);
            if let Some(body) = find_body(node) {
                for i in 0..body.child_count() {
                    if let Some(child) = body.child(i as u32) {
                        if child.kind() == "attribute" {
                            if let Some(attr_name) = extract_attribute_name(child, source) {
                                let qualified_name = if scope.is_empty() {
                                    format!("locals.{}", attr_name)
                                } else {
                                    format!("{}.locals.{}", scope.join("."), attr_name)
                                };
                                let sig = node_text(child, source)
                                    .lines()
                                    .next()
                                    .unwrap_or("")
                                    .trim()
                                    .to_string();

                                symbols.push(Symbol {
                                    name: attr_name,
                                    qualified_name,
                                    kind: SymbolKind::Constant,
                                    signature: sig,
                                    visibility: Visibility::Public,
                                    file_path: file_path.to_string(),
                                    line_start: child.start_position().row,
                                    line_end: child.end_position().row,
                                    doc_comment: doc_comment.clone(),
                                    parent: scope.last().cloned(),
                                });
                            }
                        }
                    }
                }
            }
        }
        _ => {
            // Other block types (terraform, lifecycle, provisioner, etc.)
            // Extract as a generic block if it has labels
            if !labels.is_empty() {
                let name = labels.last().cloned().unwrap_or_else(|| block_type.clone());
                let qualified_name = build_qualified_name(&block_type, &labels, scope);
                let signature = extract_block_signature(node, source);
                let doc_comment = extract_hcl_comment(node, source);

                symbols.push(Symbol {
                    name,
                    qualified_name: qualified_name.clone(),
                    kind: SymbolKind::Module,
                    signature,
                    visibility: Visibility::Public,
                    file_path: file_path.to_string(),
                    line_start: node.start_position().row,
                    line_end: node.end_position().row,
                    doc_comment,
                    parent: scope.last().cloned(),
                });

                let mut new_scope = scope.to_vec();
                new_scope.push(qualified_name);
                if let Some(body) = find_body(node) {
                    for i in 0..body.child_count() {
                        if let Some(child) = body.child(i as u32) {
                            extract_symbols_recursive(
                                child, source, file_path, &new_scope, symbols,
                            );
                        }
                    }
                }
            } else {
                // Unlabeled blocks (like terraform {}, lifecycle {}) — recurse into body
                if let Some(body) = find_body(node) {
                    for i in 0..body.child_count() {
                        if let Some(child) = body.child(i as u32) {
                            extract_symbols_recursive(child, source, file_path, scope, symbols);
                        }
                    }
                }
            }
        }
    }
}

/// Extract a symbol from a top-level attribute (e.g., `terraform_version = "1.0"`).
fn extract_attribute_symbol(
    node: Node,
    source: &[u8],
    file_path: &str,
    scope: &[String],
) -> Option<Symbol> {
    let name = extract_attribute_name(node, source)?;
    let qualified_name = if scope.is_empty() {
        name.clone()
    } else {
        format!("{}.{}", scope.join("."), name)
    };
    let sig = node_text(node, source)
        .lines()
        .next()
        .unwrap_or("")
        .trim()
        .to_string();
    let doc_comment = extract_hcl_comment(node, source);

    Some(Symbol {
        name,
        qualified_name,
        kind: SymbolKind::Constant,
        signature: sig,
        visibility: Visibility::Public,
        file_path: file_path.to_string(),
        line_start: node.start_position().row,
        line_end: node.end_position().row,
        doc_comment,
        parent: scope.last().cloned(),
    })
}

// ── Reference Extraction ──────────────────────────────────────────────────

fn extract_references_recursive(
    node: Node,
    source: &[u8],
    file_path: &str,
    scope: &[String],
    references: &mut Vec<Reference>,
) {
    match node.kind() {
        "variable_expr" => {
            // variable_expr contains a single identifier child.
            // Build the full dotted reference by consuming subsequent get_attr siblings.
            let full_ref = build_dotted_reference(node, source);
            if !full_ref.is_empty() {
                let source_qn = if scope.is_empty() {
                    file_path.to_string()
                } else {
                    scope.join(".")
                };

                references.push(Reference {
                    source_qualified_name: source_qn,
                    target_name: full_ref,
                    kind: ReferenceKind::Call,
                    file_path: file_path.to_string(),
                    line: node.start_position().row,
                });
            }
            return; // No children to recurse into meaningfully
        }
        "block" => {
            // Update scope when entering a block
            let (block_type, labels) = extract_block_type_and_labels(node, source);
            let qn = build_qualified_name(&block_type, &labels, scope);
            let mut new_scope = scope.to_vec();
            new_scope.push(qn);
            for i in 0..node.child_count() {
                if let Some(child) = node.child(i as u32) {
                    extract_references_recursive(child, source, file_path, &new_scope, references);
                }
            }
            return;
        }
        _ => {}
    }

    // Default recursion
    for i in 0..node.child_count() {
        if let Some(child) = node.child(i as u32) {
            extract_references_recursive(child, source, file_path, scope, references);
        }
    }
}

/// Build a dotted reference from a `variable_expr` node and its subsequent `get_attr` siblings
/// within the same parent `expression` node.
///
/// For `var.name`, the AST looks like:
///   expression
///     variable_expr
///       identifier "var"
///     get_attr
///       identifier "name"
///
/// We start from the variable_expr, read its identifier, then look at subsequent siblings
/// that are `get_attr` nodes.
fn build_dotted_reference(variable_expr_node: Node, source: &[u8]) -> String {
    let base = node_text(variable_expr_node, source);
    if base.is_empty() {
        return String::new();
    }

    let mut parts = vec![base];

    // Walk subsequent siblings in the parent to collect get_attr chains
    let mut sibling = variable_expr_node.next_sibling();
    while let Some(sib) = sibling {
        if sib.kind() == "get_attr" {
            // get_attr contains "." and an identifier
            for i in 0..sib.child_count() {
                if let Some(child) = sib.child(i as u32) {
                    if child.kind() == "identifier" {
                        parts.push(node_text(child, source));
                    }
                }
            }
            sibling = sib.next_sibling();
        } else {
            break;
        }
    }

    parts.join(".")
}

// ── Helper Functions ──────────────────────────────────────────────────────

fn node_text(node: Node, source: &[u8]) -> String {
    node.utf8_text(source).unwrap_or("").to_string()
}

/// Extract the block type (first identifier) and labels (subsequent string_lit/identifier
/// children before the block_start) from an HCL block node.
fn extract_block_type_and_labels(node: Node, source: &[u8]) -> (String, Vec<String>) {
    let mut block_type = String::new();
    let mut labels = Vec::new();
    let mut found_type = false;

    for i in 0..node.child_count() {
        if let Some(child) = node.child(i as u32) {
            match child.kind() {
                "identifier" if !found_type => {
                    block_type = node_text(child, source);
                    found_type = true;
                }
                "identifier" => {
                    labels.push(node_text(child, source));
                }
                "string_lit" => {
                    // String labels are quoted; strip the quotes
                    let raw = node_text(child, source);
                    let unquoted = raw
                        .trim_start_matches('"')
                        .trim_end_matches('"')
                        .to_string();
                    labels.push(unquoted);
                }
                "block_start" => break, // Stop at opening brace
                _ => {}
            }
        }
    }

    (block_type, labels)
}

/// Build a qualified name from block type, labels, and parent scope.
/// Example: scope=[], block_type="resource", labels=["aws_s3_bucket", "my_bucket"]
///   -> "resource.aws_s3_bucket.my_bucket"
fn build_qualified_name(block_type: &str, labels: &[String], scope: &[String]) -> String {
    let mut parts: Vec<&str> = scope.iter().map(|s| s.as_str()).collect();
    parts.push(block_type);
    for label in labels {
        parts.push(label.as_str());
    }
    parts.join(".")
}

/// Find the `body` child node inside a block.
fn find_body(node: Node) -> Option<Node> {
    for i in 0..node.child_count() {
        if let Some(child) = node.child(i as u32) {
            if child.kind() == "body" {
                return Some(child);
            }
        }
    }
    None
}

/// Extract the name (first identifier) from an attribute node.
fn extract_attribute_name(node: Node, source: &[u8]) -> Option<String> {
    for i in 0..node.child_count() {
        if let Some(child) = node.child(i as u32) {
            if child.kind() == "identifier" {
                return Some(node_text(child, source));
            }
        }
    }
    None
}

/// Extract the block header signature (everything up to the opening brace).
fn extract_block_signature(node: Node, source: &[u8]) -> String {
    let text = node_text(node, source);
    if let Some(pos) = text.find('{') {
        text[..pos].trim().to_string()
    } else {
        let first_line = text.lines().next().unwrap_or(&text);
        first_line.trim().to_string()
    }
}

/// Extract a doc comment from comment nodes preceding this node.
///
/// In HCL's tree-sitter grammar, comments are "extra" nodes that can appear at
/// any level. For a top-level block, the comment is a sibling of the `body` node
/// (not the block itself, since the block is a child of `body`). We check the
/// node's own previous sibling first, then fall back to checking the parent
/// (`body`) node's previous sibling if the block is the first child of `body`.
fn extract_hcl_comment(node: Node, source: &[u8]) -> Option<String> {
    // Try direct previous sibling first
    let result = collect_preceding_comments(node, source);
    if result.is_some() {
        return result;
    }

    // If no comment found as direct sibling, check the parent node's siblings.
    // This handles the case where:
    //   config_file -> comment, body -> block
    // The comment is a sibling of body, not of block.
    if let Some(parent) = node.parent() {
        if parent.kind() == "body" {
            return collect_preceding_comments(parent, source);
        }
    }

    None
}

/// Collect consecutive comment nodes preceding the given node.
fn collect_preceding_comments(node: Node, source: &[u8]) -> Option<String> {
    let mut comments = Vec::new();
    let mut prev = node.prev_sibling();

    while let Some(sibling) = prev {
        if sibling.kind() == "comment" {
            let text = node_text(sibling, source);
            let cleaned = clean_hcl_comment(&text);
            comments.push(cleaned);
            prev = sibling.prev_sibling();
        } else {
            break;
        }
    }

    if comments.is_empty() {
        return None;
    }

    // Comments were collected in reverse order (bottom-up), so reverse them
    comments.reverse();
    Some(comments.join("\n"))
}

/// Clean a single HCL comment line, stripping the comment prefix.
fn clean_hcl_comment(raw: &str) -> String {
    let trimmed = raw.trim();
    if let Some(rest) = trimmed.strip_prefix("//") {
        rest.trim_start().to_string()
    } else if let Some(rest) = trimmed.strip_prefix('#') {
        rest.trim_start().to_string()
    } else if trimmed.starts_with("/*") && trimmed.ends_with("*/") {
        // Block comment: strip delimiters
        let inner = &trimmed[2..trimmed.len() - 2];
        inner.trim().to_string()
    } else {
        trimmed.to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::extractor::LanguageExtractor;
    use tree_sitter::Parser;

    fn parse_hcl(source: &str) -> Tree {
        let mut parser = Parser::new();
        let lang = tree_sitter_hcl::LANGUAGE;
        parser
            .set_language(&lang.into())
            .expect("failed to set HCL language");
        parser
            .parse(source.as_bytes(), None)
            .expect("failed to parse")
    }

    #[test]
    fn extract_resource_block() {
        let source = r#"
resource "aws_s3_bucket" "my_bucket" {
  bucket = "my-unique-bucket"
  acl    = "private"
}
"#;
        let tree = parse_hcl(source);
        let extractor = HclExtractor::new();
        let symbols = extractor.extract_symbols(&tree, source.as_bytes(), "main.tf");

        let resource = symbols.iter().find(|s| s.name == "my_bucket").unwrap();
        assert_eq!(resource.kind, SymbolKind::Class);
        assert_eq!(
            resource.qualified_name,
            "resource.aws_s3_bucket.my_bucket"
        );
        assert_eq!(resource.visibility, Visibility::Public);
        assert!(
            resource.signature.contains("resource"),
            "signature: {}",
            resource.signature
        );
        assert!(
            resource.signature.contains("aws_s3_bucket"),
            "signature: {}",
            resource.signature
        );
    }

    #[test]
    fn extract_variable_and_output() {
        let source = r#"
variable "region" {
  description = "AWS region"
  type        = string
  default     = "us-east-1"
}

output "bucket_arn" {
  value = aws_s3_bucket.my_bucket.arn
}
"#;
        let tree = parse_hcl(source);
        let extractor = HclExtractor::new();
        let symbols = extractor.extract_symbols(&tree, source.as_bytes(), "variables.tf");

        let variable = symbols.iter().find(|s| s.name == "region").unwrap();
        assert_eq!(variable.kind, SymbolKind::Constant);
        assert_eq!(variable.qualified_name, "variable.region");
        assert_eq!(variable.visibility, Visibility::Public);

        let output = symbols.iter().find(|s| s.name == "bucket_arn").unwrap();
        assert_eq!(output.kind, SymbolKind::Constant);
        assert_eq!(output.qualified_name, "output.bucket_arn");
    }

    #[test]
    fn extract_module_block() {
        let source = r#"
module "vpc" {
  source  = "terraform-aws-modules/vpc/aws"
  version = "3.0.0"

  name = "my-vpc"
  cidr = "10.0.0.0/16"
}
"#;
        let tree = parse_hcl(source);
        let extractor = HclExtractor::new();
        let symbols = extractor.extract_symbols(&tree, source.as_bytes(), "main.tf");

        let module = symbols.iter().find(|s| s.name == "vpc").unwrap();
        assert_eq!(module.kind, SymbolKind::Module);
        assert_eq!(module.qualified_name, "module.vpc");
        assert_eq!(module.visibility, Visibility::Public);
        assert!(
            module.signature.contains("module"),
            "signature: {}",
            module.signature
        );
    }

    #[test]
    fn extract_data_source() {
        let source = r#"
data "aws_ami" "latest" {
  most_recent = true
  owners      = ["amazon"]
}
"#;
        let tree = parse_hcl(source);
        let extractor = HclExtractor::new();
        let symbols = extractor.extract_symbols(&tree, source.as_bytes(), "data.tf");

        let data = symbols.iter().find(|s| s.name == "latest").unwrap();
        assert_eq!(data.kind, SymbolKind::Class);
        assert_eq!(data.qualified_name, "data.aws_ami.latest");
    }

    #[test]
    fn extract_provider_block() {
        let source = r#"
provider "aws" {
  region = "us-east-1"
}
"#;
        let tree = parse_hcl(source);
        let extractor = HclExtractor::new();
        let symbols = extractor.extract_symbols(&tree, source.as_bytes(), "providers.tf");

        let provider = symbols.iter().find(|s| s.name == "aws").unwrap();
        assert_eq!(provider.kind, SymbolKind::Module);
        assert_eq!(provider.qualified_name, "provider.aws");
    }

    #[test]
    fn extract_locals_block() {
        let source = r#"
locals {
  environment = "production"
  project     = "codemem"
}
"#;
        let tree = parse_hcl(source);
        let extractor = HclExtractor::new();
        let symbols = extractor.extract_symbols(&tree, source.as_bytes(), "locals.tf");

        let env = symbols.iter().find(|s| s.name == "environment").unwrap();
        assert_eq!(env.kind, SymbolKind::Constant);
        assert_eq!(env.qualified_name, "locals.environment");

        let proj = symbols.iter().find(|s| s.name == "project").unwrap();
        assert_eq!(proj.kind, SymbolKind::Constant);
        assert_eq!(proj.qualified_name, "locals.project");
    }

    #[test]
    fn extract_references_from_expressions() {
        let source = r#"
resource "aws_instance" "web" {
  ami           = var.ami_id
  instance_type = var.instance_type
  subnet_id     = module.vpc.public_subnets[0]
}
"#;
        let tree = parse_hcl(source);
        let extractor = HclExtractor::new();
        let references = extractor.extract_references(&tree, source.as_bytes(), "main.tf");

        let calls: Vec<_> = references
            .iter()
            .filter(|r| r.kind == ReferenceKind::Call)
            .collect();

        assert!(
            calls.iter().any(|r| r.target_name.starts_with("var.ami_id")),
            "expected var.ami_id reference, got: {:#?}",
            calls
        );
        assert!(
            calls
                .iter()
                .any(|r| r.target_name.starts_with("var.instance_type")),
            "expected var.instance_type reference, got: {:#?}",
            calls
        );
        assert!(
            calls
                .iter()
                .any(|r| r.target_name.starts_with("module.vpc")),
            "expected module.vpc reference, got: {:#?}",
            calls
        );
    }

    #[test]
    fn extract_hcl_doc_comments() {
        let source = r#"
# The main S3 bucket for storing artifacts
resource "aws_s3_bucket" "artifacts" {
  bucket = "my-artifacts"
}
"#;
        let tree = parse_hcl(source);
        let extractor = HclExtractor::new();
        let symbols = extractor.extract_symbols(&tree, source.as_bytes(), "main.tf");

        let bucket = symbols.iter().find(|s| s.name == "artifacts").unwrap();
        let doc = bucket
            .doc_comment
            .as_ref()
            .expect("expected doc comment on resource");
        assert!(
            doc.contains("main S3 bucket"),
            "doc: {}",
            doc
        );
    }

    #[test]
    fn file_extensions_include_tf_hcl_tfvars() {
        let extractor = HclExtractor::new();
        let exts = extractor.file_extensions();
        assert!(exts.contains(&"tf"));
        assert!(exts.contains(&"hcl"));
        assert!(exts.contains(&"tfvars"));
    }

    #[test]
    fn language_name_is_hcl() {
        let extractor = HclExtractor::new();
        assert_eq!(extractor.language_name(), "hcl");
    }
}
