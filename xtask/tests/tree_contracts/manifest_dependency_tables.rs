//! `manifest_walk` reads every dependency table cargo reads, wherever it nests.
//!
//! WHY: two gates derived a dependency roster from a manifest and each read a
//! different set of tables. The example gate scanned lines and saw only the
//! three root tables, so it left `[target.'cfg(unix)'.dependencies]` out of the
//! `[patch.crates-io]` section it writes and the example built that crate from
//! the registry. The feature-matrix gate walked the document recursively and
//! saw all of them. One reader now answers both, and this pins the tables it
//! reaches against a document that puts a dependency in each of them.

use xtask::manifest_walk::{dependency_names, optional_dependency_names};

/// A manifest with a dependency in every table shape cargo accepts.
const NESTED: &str = "\
[package]
name = \"probe\"

[dependencies]
root-normal = \"1\"
root-optional = { version = \"1\", optional = true }

[dev-dependencies]
root-dev = \"1\"

[build-dependencies]
root-build = \"1\"

[target.'cfg(unix)'.dependencies]
unix-normal = \"1\"
unix-optional = { version = \"1\", optional = true }

[target.'cfg(windows)'.dev-dependencies]
windows-dev = \"1\"

[lints.rust]
unsafe_code = \"forbid\"
";

#[test]
fn every_nested_dependency_table_is_read_once_and_sorted() {
    assert_eq!(
        dependency_names(&toml::from_str(NESTED).expect("Fix: the fixture must be valid TOML.")),
        vec![
            "root-build".to_string(),
            "root-dev".to_string(),
            "root-normal".to_string(),
            "root-optional".to_string(),
            "unix-normal".to_string(),
            "unix-optional".to_string(),
            "windows-dev".to_string(),
        ]
    );
}

#[test]
fn an_optional_dependency_is_optional_under_a_target_selector_too() {
    assert_eq!(
        optional_dependency_names(
            &toml::from_str(NESTED).expect("Fix: the fixture must be valid TOML.")
        ),
        vec!["root-optional".to_string(), "unix-optional".to_string()]
    );
}

/// A key that is not a dependency table contributes nothing, however deeply it
/// nests: `[lints.rust]` holds names, and reading it would patch a lint.
#[test]
fn a_table_that_is_not_a_dependency_table_contributes_nothing() {
    let only_lints: toml::Table = toml::from_str("[lints.rust]\nunsafe_code = \"forbid\"\n")
        .expect("Fix: the fixture must be valid TOML.");

    assert!(dependency_names(&only_lints).is_empty());
    assert!(optional_dependency_names(&only_lints).is_empty());
}
