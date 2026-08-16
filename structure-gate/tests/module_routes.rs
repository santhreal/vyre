//! Which files a crate root compiles, and what enables each one.
//!
//! WHY: two gates answered this question separately and disagreed. One read a
//! dialect's features from the crate root alone, so a file that `encoding/mod.rs`
//! declares behind the neural-network gates looked reachable without them and 13
//! legitimate imports were reported as undeclared coupling. The same reader
//! judged a `#[cfg(test)]` module file as production source for 6 more. The other
//! walked the `mod` declarations and was right, so the walk is shared and pinned
//! here.
//!
//! What these do not catch: a module declared inside an inline `mod x { .. }`
//! block. The walk reads out-of-line declarations, which is every module that
//! has a file of its own.

use std::fs;
use std::path::Path;

use structure_gate::source_scan::{ModuleRoute, gating_features, module_routes};

fn write(path: &Path, text: &str) {
    fs::create_dir_all(path.parent().expect("Fix: fixture path must have a parent"))
        .expect("Fix: fixture directory must be creatable");
    fs::write(path, text).expect("Fix: fixture must be writable");
}

fn route<'a>(routes: &'a [ModuleRoute], suffix: &str) -> Option<&'a ModuleRoute> {
    routes.iter().find(|route| {
        route
            .path
            .to_string_lossy()
            .replace('\\', "/")
            .ends_with(suffix)
    })
}

/// A file's route carries every gate on the way down to it.
#[test]
fn a_nested_declaration_joins_the_gates_above_it() {
    let dir = tempfile::tempdir().expect("Fix: fixture directory must exist");
    let src = dir.path().join("src");
    write(&src.join("lib.rs"), "#[cfg(feature = \"encoding\")]\npub mod encoding;\n");
    write(
        &src.join("encoding/mod.rs"),
        "#[cfg(any(\n    feature = \"nn-linear\",\n    feature = \"nn-attention\"\n))]\npub mod paging;\n",
    );
    write(&src.join("encoding/paging.rs"), "pub fn page() {}\n");

    let routes = module_routes(&src);

    assert_eq!(
        route(&routes, "encoding/paging.rs").map(|route| route.features.clone()),
        Some(vec![
            "encoding".to_string(),
            "nn-linear".to_string(),
            "nn-attention".to_string(),
        ])
    );
}

/// A module only a test build reaches is not a compiled module.
#[test]
fn a_test_gated_module_file_is_not_on_any_route() {
    let dir = tempfile::tempdir().expect("Fix: fixture directory must exist");
    let src = dir.path().join("src");
    write(&src.join("lib.rs"), "pub mod matching;\n");
    write(
        &src.join("matching/mod.rs"),
        "pub mod dfa;\n#[cfg(test)]\nmod region_tests;\n",
    );
    write(&src.join("matching/dfa.rs"), "pub fn run() {}\n");
    write(&src.join("matching/region_tests.rs"), "fn helper() {}\n");

    let routes = module_routes(&src);

    assert!(route(&routes, "matching/dfa.rs").is_some());
    assert!(
        route(&routes, "region_tests.rs").is_none(),
        "a cfg(test) module is test source: {:?}",
        routes.iter().map(|route| route.path.clone()).collect::<Vec<_>>()
    );
}

/// A directory no declaration names carries no module.
#[test]
fn a_directory_no_declaration_names_is_not_a_module() {
    let dir = tempfile::tempdir().expect("Fix: fixture directory must exist");
    let src = dir.path().join("src");
    write(&src.join("lib.rs"), "pub mod kept;\n");
    write(&src.join("kept.rs"), "pub fn kept() {}\n");
    write(&src.join("departed/mod.rs"), "pub fn gone() {}\n");

    let routes = module_routes(&src);

    assert!(route(&routes, "kept.rs").is_some());
    assert!(route(&routes, "departed/mod.rs").is_none());
}

/// A `mod x;` is read through whatever visibility it carries.
#[test]
fn a_module_declaration_is_read_through_its_visibility() {
    let dir = tempfile::tempdir().expect("Fix: fixture directory must exist");
    let src = dir.path().join("src");
    write(
        &src.join("lib.rs"),
        "pub(crate) mod builder;\n    mod inner;\npub mod math;\n// mod commented;\nmod tests {\n}\n",
    );
    for name in ["builder", "inner", "math", "commented"] {
        write(&src.join(format!("{name}.rs")), "pub fn f() {}\n");
    }

    let routes = module_routes(&src);
    let named: Vec<String> = routes
        .iter()
        .filter_map(|route| {
            route
                .path
                .file_name()
                .map(|name| name.to_string_lossy().to_string())
        })
        .collect();

    assert!(named.contains(&"builder.rs".to_string()), "{named:?}");
    assert!(named.contains(&"inner.rs".to_string()), "{named:?}");
    assert!(named.contains(&"math.rs".to_string()), "{named:?}");
    assert!(!named.contains(&"commented.rs".to_string()), "{named:?}");
}

/// A `cfg` naming `test` beside a feature still compiles in a feature build.
#[test]
fn a_test_predicate_beside_a_feature_still_compiles() {
    let source = "#[cfg(any(test, feature = \"cpu-parity\"))]\npub fn kept() {}\n";
    let test_only = "#[cfg(test)]\nfn helper() {}\n";
    let named_latest = "#[cfg(feature = \"latest\")]\npub fn kept() {}\n";

    assert_eq!(
        gating_features(source, 1),
        Some(vec!["cpu-parity".to_string()])
    );
    assert_eq!(gating_features(test_only, 1), None);
    assert_eq!(
        gating_features(named_latest, 1),
        Some(vec!["latest".to_string()]),
        "a feature whose name contains `test` is not the test predicate"
    );
}

/// `not(test)` is production source, not test source.
#[test]
fn a_module_excluded_from_a_test_build_is_production_source() {
    let dir = tempfile::tempdir().expect("Fix: fixture directory must exist");
    let src = dir.path().join("src");
    write(&src.join("lib.rs"), "#[cfg(not(test))]\npub mod platform;\n");
    write(&src.join("platform.rs"), "pub fn run() {}\n");

    let routes = module_routes(&src);

    let platform = route(&routes, "platform.rs").expect("a not(test) module compiles");
    assert!(platform.features.is_empty(), "{:?}", platform.features);
    assert_eq!(
        gating_features("#[cfg(not(test))]\npub fn kept() {}\n", 1),
        Some(Vec::new())
    );
    assert_eq!(
        gating_features("#[cfg(not(not(test)))]\nfn helper() {}\n", 1),
        None,
        "a doubly negated test predicate is the test predicate again"
    );
}

/// A cycle in the module tree ends the walk instead of running forever.
#[cfg(unix)]
#[test]
fn a_symlinked_cycle_reads_each_file_once() {
    let dir = tempfile::tempdir().expect("Fix: fixture directory must exist");
    let src = dir.path().join("src");
    write(&src.join("lib.rs"), "pub mod region;\n");
    write(&src.join("region/mod.rs"), "pub mod region;\n");
    std::os::unix::fs::symlink(src.join("region"), src.join("region/region"))
        .expect("Fix: fixture symlink must be creatable");

    let routes = module_routes(&src);

    assert_eq!(
        routes.len(),
        2,
        "the cycle must be read once: {:?}",
        routes.iter().map(|route| route.path.clone()).collect::<Vec<_>>()
    );
}

/// A module the walk cannot read is still a module on that route.
///
/// WHY: a source over the read cap aborted the whole walk, so one generated
/// file dropped into a crate took down every rule that reads the module tree.
/// Dropping the route instead is the other wrong answer: the reader that has to
/// read the file then sees no module at all and reports nothing.
#[test]
fn a_module_over_the_read_cap_stays_on_its_route() {
    let dir = tempfile::tempdir().expect("Fix: fixture directory must exist");
    let src = dir.path().join("src");
    write(&src.join("lib.rs"), "pub mod generated;\npub mod kept;\n");
    write(&src.join("kept.rs"), "pub fn kept() {}\n");
    let mut oversized = String::with_capacity(4 * 1024 * 1024 + 64);
    while oversized.len() <= 4 * 1024 * 1024 {
        oversized.push_str("// padding past the read cap\n");
    }
    write(&src.join("generated.rs"), &oversized);

    let routes = module_routes(&src);

    assert!(route(&routes, "generated.rs").is_some());
    assert!(route(&routes, "kept.rs").is_some());
}
