//! What the placement reader answers over a fixture checkout.
//!
//! WHY: placement decides which crate each registered operation is documented
//! against, and every wrong answer this suite pins was shipped once. A prefix
//! read as a placement fact attributed 130 operations to the crate they left. A
//! bare mention of an id read as a definition reported 132 operations as defined
//! in two crates at once. A directory read as a module kept a moved domain
//! alive, because git tracks files and leaves the directory behind.
//!
//! What it does not catch: the reader's own helpers. Attribute parsing, module
//! declaration reading and literal scanning are crate-private and pinned beside
//! them.

use std::collections::BTreeSet;
use std::fs;
use std::path::Path;

use xtask_registry::docs::operation_schema::placement::{read, Placement};

fn write(path: &Path, text: &str) {
    fs::create_dir_all(path.parent().expect("Fix: fixture path must have a parent"))
        .expect("Fix: fixture directory must be creatable");
    fs::write(path, text).expect("Fix: fixture must be writable");
}

fn workspace(root: &Path, members: &[&str]) {
    let named: Vec<(&str, &str)> = members
        .iter()
        .map(|member| {
            let leaf = member.rsplit('/').next().unwrap_or(member);
            (*member, leaf)
        })
        .collect();
    workspace_named(root, &named);
}

/// A workspace whose members declare a package name of their own.
fn workspace_named(root: &Path, members: &[(&str, &str)]) {
    let listed = members
        .iter()
        .map(|(path, _)| format!("\"{path}\""))
        .collect::<Vec<_>>()
        .join(", ");
    write(
        &root.join("Cargo.toml"),
        &format!("[workspace]\nmembers = [{listed}]\n"),
    );
    for (path, name) in members {
        write(
            &root.join(path).join("Cargo.toml"),
            &format!("[package]\nname = \"{name}\"\nversion = \"0.1.0\"\n"),
        );
    }
}

#[test]
fn a_registration_places_the_operation_in_its_crate_behind_its_module_features() {
    let dir = tempfile::tempdir().expect("Fix: fixture directory must exist");
    let root = dir.path();
    workspace(root, &["libs"]);
    write(
        &root.join("libs/src/lib.rs"),
        "#[cfg(any(\n    feature = \"math-dialect\",\n    feature = \"math-kernels\"\n))]\npub mod math;\n",
    );
    write(&root.join("libs/src/math/mod.rs"), "pub mod square;\n");
    write(
        &root.join("libs/src/math/square.rs"),
        "const OP_ID: &str = \"libs::math::square\";\ninventory::submit! {\n    OperationRegistration::library_unconstrained(OP_ID, builder)\n}\n",
    );

    let ids = BTreeSet::from(["libs::math::square"]);
    let mut errors = Vec::new();
    let placements = read(root, &ids, &mut errors);

    assert_eq!(errors, Vec::<String>::new());
    assert_eq!(
        placements.get("libs::math::square"),
        Some(&Placement {
            crate_name: "libs".to_string(),
            features: vec!["math-dialect".to_string(), "math-kernels".to_string()],
        })
    );
}

/// A gate that lists ids is not a definition site.
#[test]
fn a_crate_that_only_names_the_id_is_not_the_defining_crate() {
    let dir = tempfile::tempdir().expect("Fix: fixture directory must exist");
    let root = dir.path();
    workspace(root, &["libs", "tooling"]);
    write(&root.join("libs/src/lib.rs"), "pub mod square;\n");
    write(
        &root.join("libs/src/square.rs"),
        "inventory::submit! {\n    OperationRegistration::library_unconstrained(\"libs::square\", builder)\n}\n",
    );
    write(&root.join("tooling/src/lib.rs"), "pub mod audit;\n");
    write(
        &root.join("tooling/src/audit.rs"),
        "const WAIVED: [&str; 1] = [\"libs::square\"];\n",
    );

    let ids = BTreeSet::from(["libs::square"]);
    let mut errors = Vec::new();
    let placements = read(root, &ids, &mut errors);

    assert_eq!(errors, Vec::<String>::new());
    assert_eq!(
        placements
            .get("libs::square")
            .map(|found| &found.crate_name),
        Some(&"libs".to_string())
    );
}

/// A macro that registers takes the id at the invocation, so the file that
/// spells it is the definition even though it names no registration.
#[test]
fn an_id_passed_to_a_registering_macro_places_in_the_invoking_module() {
    let dir = tempfile::tempdir().expect("Fix: fixture directory must exist");
    let root = dir.path();
    workspace(root, &["libs"]);
    write(
        &root.join("libs/src/lib.rs"),
        "#[cfg(feature = \"bitset\")]\npub mod bitset;\n",
    );
    write(
        &root.join("libs/src/bitset/mod.rs"),
        "pub mod word;\n\ndefine_op! {\n    op_id: \"libs::bitset::or_into\",\n}\n",
    );
    write(
        &root.join("libs/src/bitset/word.rs"),
        "inventory::submit! {\n    OperationRegistration::library_unconstrained(\"libs::bitset::and\", builder)\n}\n",
    );

    let ids = BTreeSet::from(["libs::bitset::or_into"]);
    let mut errors = Vec::new();
    let placements = read(root, &ids, &mut errors);

    assert_eq!(errors, Vec::<String>::new());
    assert_eq!(
        placements.get("libs::bitset::or_into"),
        Some(&Placement {
            crate_name: "libs".to_string(),
            features: vec!["bitset".to_string()],
        })
    );
}

/// A registration reachable only under `cfg(test)` is a fixture.
#[test]
fn a_test_only_module_is_not_a_definition_site() {
    let dir = tempfile::tempdir().expect("Fix: fixture directory must exist");
    let root = dir.path();
    workspace(root, &["libs"]);
    write(
        &root.join("libs/src/lib.rs"),
        "pub mod real;\n#[cfg(test)]\nmod fixtures;\n",
    );
    write(
        &root.join("libs/src/real.rs"),
        "inventory::submit! {\n    OperationRegistration::library_unconstrained(\"libs::real\", builder)\n}\n",
    );
    write(
        &root.join("libs/src/fixtures.rs"),
        "inventory::submit! {\n    OperationRegistration::library_unconstrained(\"libs::ghost\", builder)\n}\n",
    );

    let ids = BTreeSet::from(["libs::real", "libs::ghost"]);
    let mut errors = Vec::new();
    let placements = read(root, &ids, &mut errors);

    assert!(placements.get("libs::real").is_some());
    assert_eq!(placements.get("libs::ghost"), None);
    assert_eq!(errors.len(), 1, "{errors:?}");
    assert!(
        errors[0].contains("operation `libs::ghost` has no definition site"),
        "{errors:?}"
    );
}

/// A domain that moved leaves its directory behind in every checkout that
/// pulled the deletion, so only the declaration answers.
#[test]
fn a_directory_no_declaration_names_carries_no_module() {
    let dir = tempfile::tempdir().expect("Fix: fixture directory must exist");
    let root = dir.path();
    workspace(root, &["libs"]);
    write(
        &root.join("libs/src/lib.rs"),
        "#[cfg(feature = \"reduce\")]\npub mod reduce;\n",
    );
    write(&root.join("libs/src/reduce.rs"), "pub fn sum() {}\n");
    write(
        &root.join("libs/src/matching/mod.rs"),
        "inventory::submit! {\n    OperationRegistration::library_unconstrained(\"libs::matching::dfa\", builder)\n}\n",
    );

    let ids = BTreeSet::from(["libs::matching::dfa"]);
    let mut errors = Vec::new();
    let placements = read(root, &ids, &mut errors);

    assert_eq!(placements.get("libs::matching::dfa"), None);
    assert_eq!(errors.len(), 1, "{errors:?}");
}

/// A manifest the reader cannot parse names itself.
///
/// WHY: the reader answered an unparseable manifest with an empty member roster,
/// so every registered operation came back with no definition site and the whole
/// schema read as a broken registry. The document parse itself was the defect:
/// one TOML value was parsed where a document was meant, which cannot succeed on
/// any manifest at all.
#[test]
fn an_unparseable_workspace_manifest_is_an_error_naming_the_file() {
    let dir = tempfile::tempdir().expect("Fix: fixture directory must exist");
    let root = dir.path();
    write(&root.join("Cargo.toml"), "[workspace\nmembers = [\n");
    write(
        &root.join("libs/src/lib.rs"),
        "inventory::submit! {\n    OperationRegistration::library_unconstrained(\"libs::real\", builder)\n}\n",
    );

    let ids = BTreeSet::from(["libs::real"]);
    let mut errors = Vec::new();
    let placements = read(root, &ids, &mut errors);

    assert_eq!(placements.get("libs::real"), None);
    assert!(
        errors.iter().any(|error| error.contains("cannot parse")
            && error.contains("Cargo.toml")
            && error.contains("Fix: repair the workspace manifest")),
        "{errors:?}"
    );
}

/// A workspace whose members carry no crate root names that, once.
#[test]
fn a_member_roster_that_reaches_no_crate_root_is_an_error() {
    let dir = tempfile::tempdir().expect("Fix: fixture directory must exist");
    let root = dir.path();
    workspace(root, &["libs", "tooling"]);
    write(&root.join("libs/src/main.rs"), "fn main() {}\n");

    let ids = BTreeSet::from(["libs::real"]);
    let mut errors = Vec::new();
    read(root, &ids, &mut errors);

    assert!(
        errors
            .iter()
            .any(|error| error.contains("names 2 member(s) and none carries a readable")),
        "{errors:?}"
    );
}

/// A placement names the package, not the directory it is built from.
///
/// WHY: the reader carried the workspace member path, so a crate under a
/// grouping directory was reported as `conform/vyre-conform`, which names no
/// package a reader can pass to cargo and no crate the registry knows.
#[test]
fn a_placement_names_the_package_not_the_member_directory() {
    let dir = tempfile::tempdir().expect("Fix: fixture directory must exist");
    let root = dir.path();
    workspace_named(root, &[("conform/vyre-conform", "vyre-conform")]);
    write(
        &root.join("conform/vyre-conform/src/lib.rs"),
        "inventory::submit! {\n    OperationRegistration::library_unconstrained(\"vyre-conform::probe\", builder)\n}\n",
    );

    let ids = BTreeSet::from(["vyre-conform::probe"]);
    let mut errors = Vec::new();
    let placements = read(root, &ids, &mut errors);

    assert_eq!(errors, Vec::<String>::new());
    assert_eq!(
        placements.get("vyre-conform::probe"),
        Some(&Placement {
            crate_name: "vyre-conform".to_string(),
            features: Vec::new(),
        })
    );
    assert_eq!(
        placements.directory("vyre-conform"),
        Some("conform/vyre-conform")
    );
}

/// A member manifest without a package name is an error naming the file.
#[test]
fn a_member_without_a_package_name_is_an_error_naming_the_manifest() {
    let dir = tempfile::tempdir().expect("Fix: fixture directory must exist");
    let root = dir.path();
    write(
        &root.join("Cargo.toml"),
        "[workspace]\nmembers = [\"libs\"]\n",
    );
    write(
        &root.join("libs/Cargo.toml"),
        "[package]\nversion = \"0\"\n",
    );
    write(
        &root.join("libs/src/lib.rs"),
        "inventory::submit! {\n    OperationRegistration::library_unconstrained(\"libs::real\", builder)\n}\n",
    );

    let ids = BTreeSet::from(["libs::real"]);
    let mut errors = Vec::new();
    read(root, &ids, &mut errors);

    let named = root.join("libs").join("Cargo.toml");
    assert!(
        errors
            .iter()
            .any(|error| error.contains(&named.display().to_string())
                && error.contains("declares no `package.name`")),
        "{errors:?}"
    );
}

/// A source file over the read cap is named, not skipped in silence.
///
/// WHY: a skipped file registers nothing, so the operations it defines came
/// back with no definition site and the message named the operation rather than
/// the file nobody read.
#[test]
fn a_source_file_over_the_read_cap_is_an_error_naming_the_file() {
    let dir = tempfile::tempdir().expect("Fix: fixture directory must exist");
    let root = dir.path();
    workspace(root, &["libs"]);
    let mut oversized = String::with_capacity(4 * 1024 * 1024 + 128);
    oversized.push_str(
        "inventory::submit! {\n    OperationRegistration::library_unconstrained(\"libs::real\", builder)\n}\n",
    );
    while oversized.len() <= 4 * 1024 * 1024 {
        oversized.push_str("// padding to cross the read cap\n");
    }
    write(&root.join("libs/src/lib.rs"), &oversized);

    let ids = BTreeSet::from(["libs::real"]);
    let mut errors = Vec::new();
    read(root, &ids, &mut errors);

    let named = root.join("libs").join("src").join("lib.rs");
    assert!(
        errors
            .iter()
            .any(|error| error.contains(&named.display().to_string())
                && error.contains("over the 4194304 byte read cap")),
        "{errors:?}"
    );
}

#[test]
fn an_intrinsic_registration_places_the_operation_in_its_defining_crate() {
    let dir = tempfile::tempdir().expect("Fix: fixture directory must exist");
    let root = dir.path();
    workspace(root, &["primitives"]);
    write(&root.join("primitives/src/lib.rs"), "pub mod hardware;\n");
    write(
        &root.join("primitives/src/hardware.rs"),
        "const OP_ID: &str = \"primitives::hardware::bit_reverse_u32\";\n\
         inventory::submit! {\n\
             OperationRegistration::intrinsic_unconstrained(OP_ID, SIG, None, None, None)\n\
         }\n",
    );

    let ids = BTreeSet::from(["primitives::hardware::bit_reverse_u32"]);
    let mut errors = Vec::new();
    let placements = read(root, &ids, &mut errors);

    assert_eq!(errors, Vec::<String>::new());
    assert_eq!(
        placements.get("primitives::hardware::bit_reverse_u32"),
        Some(&Placement {
            crate_name: "primitives".to_string(),
            features: Vec::new(),
        })
    );
}

#[test]
fn a_macro_generated_intrinsic_places_the_operation_in_its_defining_crate() {
    let dir = tempfile::tempdir().expect("Fix: fixture directory must exist");
    let root = dir.path();
    workspace(root, &["primitives"]);
    write(&root.join("primitives/src/lib.rs"), "pub mod hardware;\n");
    write(
        &root.join("primitives/src/hardware.rs"),
        "define_unary_u32_hardware_intrinsic!(\n\
             bit_reverse_u32,\n\
             \"primitives::hardware::bit_reverse_u32\",\n\
             Expr::reverse_bits,\n\
         );\n",
    );

    let ids = BTreeSet::from(["primitives::hardware::bit_reverse_u32"]);
    let mut errors = Vec::new();
    let placements = read(root, &ids, &mut errors);

    assert_eq!(errors, Vec::<String>::new());
    assert_eq!(
        placements.get("primitives::hardware::bit_reverse_u32"),
        Some(&Placement {
            crate_name: "primitives".to_string(),
            features: Vec::new(),
        })
    );
}
