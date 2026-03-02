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
#[path = "tests/mod_tests.rs"]
mod tests;
