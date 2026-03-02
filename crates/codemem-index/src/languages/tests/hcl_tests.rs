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
    assert_eq!(resource.qualified_name, "resource.aws_s3_bucket.my_bucket");
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
        calls
            .iter()
            .any(|r| r.target_name.starts_with("var.ami_id")),
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
    assert!(doc.contains("main S3 bucket"), "doc: {}", doc);
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
