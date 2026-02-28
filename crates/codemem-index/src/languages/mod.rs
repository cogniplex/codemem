//! Language registry for code indexing.
//!
//! Each language implements the `LanguageExtractor` trait and is registered here.

pub mod cpp;
pub mod csharp;
pub mod go;
pub mod hcl;
pub mod java;
pub mod kotlin;
pub mod php;
pub mod python;
pub mod ruby;
pub mod rust;
pub mod scala;
pub mod swift;
pub mod typescript;

use crate::extractor::LanguageExtractor;

/// Returns all available language extractors.
pub fn all_extractors() -> Vec<Box<dyn LanguageExtractor>> {
    vec![
        Box::new(rust::RustExtractor::new()),
        Box::new(typescript::TypeScriptExtractor::new()),
        Box::new(python::PythonExtractor::new()),
        Box::new(go::GoExtractor::new()),
        Box::new(cpp::CppExtractor::new()),
        Box::new(java::JavaExtractor::new()),
        Box::new(csharp::CSharpExtractor::new()),
        Box::new(ruby::RubyExtractor::new()),
        Box::new(kotlin::KotlinExtractor::new()),
        Box::new(swift::SwiftExtractor::new()),
        Box::new(php::PhpExtractor::new()),
        Box::new(scala::ScalaExtractor::new()),
        Box::new(hcl::HclExtractor::new()),
    ]
}

/// Find an extractor for a given file extension.
pub fn extractor_for_extension(ext: &str) -> Option<Box<dyn LanguageExtractor>> {
    all_extractors()
        .into_iter()
        .find(|extractor| extractor.file_extensions().contains(&ext))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn finds_rust_extractor() {
        let ext = extractor_for_extension("rs");
        assert!(ext.is_some());
        assert_eq!(ext.unwrap().language_name(), "rust");
    }

    #[test]
    fn finds_typescript_extractor() {
        let ext = extractor_for_extension("ts");
        assert!(ext.is_some());
        assert_eq!(ext.unwrap().language_name(), "typescript");
    }

    #[test]
    fn finds_tsx_extractor() {
        let ext = extractor_for_extension("tsx");
        assert!(ext.is_some());
        assert_eq!(ext.unwrap().language_name(), "typescript");
    }

    #[test]
    fn finds_python_extractor() {
        let ext = extractor_for_extension("py");
        assert!(ext.is_some());
        assert_eq!(ext.unwrap().language_name(), "python");
    }

    #[test]
    fn finds_go_extractor() {
        let ext = extractor_for_extension("go");
        assert!(ext.is_some());
        assert_eq!(ext.unwrap().language_name(), "go");
    }

    #[test]
    fn finds_cpp_extractor() {
        let ext = extractor_for_extension("cpp");
        assert!(ext.is_some());
        assert_eq!(ext.unwrap().language_name(), "cpp");
    }

    #[test]
    fn finds_c_extractor() {
        let ext = extractor_for_extension("c");
        assert!(ext.is_some());
        assert_eq!(ext.unwrap().language_name(), "cpp");
    }

    #[test]
    fn finds_java_extractor() {
        let ext = extractor_for_extension("java");
        assert!(ext.is_some());
        assert_eq!(ext.unwrap().language_name(), "java");
    }

    #[test]
    fn finds_csharp_extractor() {
        let ext = extractor_for_extension("cs");
        assert!(ext.is_some());
        assert_eq!(ext.unwrap().language_name(), "csharp");
    }

    #[test]
    fn finds_ruby_extractor() {
        let ext = extractor_for_extension("rb");
        assert!(ext.is_some());
        assert_eq!(ext.unwrap().language_name(), "ruby");
    }

    #[test]
    fn finds_kotlin_extractor() {
        let ext = extractor_for_extension("kt");
        assert!(ext.is_some());
        assert_eq!(ext.unwrap().language_name(), "kotlin");
    }

    #[test]
    fn finds_swift_extractor() {
        let ext = extractor_for_extension("swift");
        assert!(ext.is_some());
        assert_eq!(ext.unwrap().language_name(), "swift");
    }

    #[test]
    fn finds_php_extractor() {
        let ext = extractor_for_extension("php");
        assert!(ext.is_some());
        assert_eq!(ext.unwrap().language_name(), "php");
    }

    #[test]
    fn finds_scala_extractor() {
        let ext = extractor_for_extension("scala");
        assert!(ext.is_some());
        assert_eq!(ext.unwrap().language_name(), "scala");
    }

    #[test]
    fn finds_hcl_extractor() {
        let ext = extractor_for_extension("tf");
        assert!(ext.is_some());
        assert_eq!(ext.unwrap().language_name(), "hcl");
    }

    #[test]
    fn finds_js_extractor() {
        let ext = extractor_for_extension("js");
        assert!(ext.is_some());
        assert_eq!(ext.unwrap().language_name(), "typescript");
    }

    #[test]
    fn finds_jsx_extractor() {
        let ext = extractor_for_extension("jsx");
        assert!(ext.is_some());
        assert_eq!(ext.unwrap().language_name(), "typescript");
    }

    #[test]
    fn returns_none_for_unknown() {
        let ext = extractor_for_extension("xyz");
        assert!(ext.is_none());
    }
}
