//! Language registry for code indexing.
//!
//! Each language implements the `LanguageExtractor` trait and is registered here.

pub mod cpp;
pub mod go;
pub mod java;
pub mod python;
pub mod rust;
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
    fn returns_none_for_unknown() {
        let ext = extractor_for_extension("xyz");
        assert!(ext.is_none());
    }
}
