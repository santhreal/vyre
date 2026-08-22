//! The registry reader every rule takes its roster from.
//!
//! A rule that resolves a file to the wrong crate holds it to the wrong owner,
//! and a rule whose reader accepts a row missing `owner` covers that crate with
//! an empty string nobody notices. Both fail here rather than in the rule.
//!
//! The checkout cases are held against `docs/CRATE_OWNERSHIP.toml` as it stands,
//! so a member that stops declaring a field, or a directory that stops matching
//! the tree, is reported by this file and not by whichever rule read it next.

use std::path::Path;

use structure_gate::crate_ownership::{Registry, REGISTRY};
use structure_gate::workspace_root;

/// The declared checkout resolves every member it declares.
#[test]
fn every_declared_member_resolves_from_its_own_directory() {
    let registry = Registry::read(&workspace_root()).expect("Fix: the registry must be readable");
    assert!(
        !registry.rows().is_empty(),
        "Fix: {REGISTRY} declares no member, so every rule reading it covers nothing"
    );
    for row in registry.rows() {
        let owning = registry.owning_crate(&row.path).unwrap_or_else(|| {
            panic!(
                "Fix: {REGISTRY} declares `{}` at `{}` and that path resolves to no member",
                row.package, row.path
            )
        });
        assert_eq!(
            owning.package, row.package,
            "Fix: `{}` resolves to `{}` instead of itself, so files under it are held to the \
             wrong owner",
            row.path, owning.package
        );
        assert!(
            structure_gate::source_scan::carries_rust_source(&workspace_root().join(&row.path)),
            "Fix: {REGISTRY} declares `{}` at `{}` and no such directory exists",
            row.package,
            row.path
        );
    }
}

/// A directory prefix that is not a path component belongs to neither member.
///
/// `conform/vyre-conform` is a prefix of the `conform/vyre-conform-spec` string,
/// so a reader comparing raw prefixes hands every spec file to the harness crate
/// and its owner never sees them.
#[test]
fn a_sibling_sharing_a_name_prefix_keeps_its_own_files() {
    let registry = Registry::read(&workspace_root()).expect("Fix: the registry must be readable");
    let spec = registry
        .owning_crate("conform/vyre-conform-spec/src/lib.rs")
        .expect("Fix: the spec crate must own files under its own directory");
    assert_eq!(spec.package, "vyre-conform-spec");
    let harness = registry
        .owning_crate("conform/vyre-conform/src/lib.rs")
        .expect("Fix: the harness crate must own files under its own directory");
    assert_eq!(harness.package, "vyre-conform");
}

/// A member nested inside another member's directory keeps the files under it.
#[test]
fn the_longest_declared_directory_owns_the_file() {
    let registry = Registry::parse(
        "schema_version = 2\n\n\
         [[crate]]\npackage = \"outer\"\npath = \"tools\"\nowner = \"outer-owner\"\nlayer = \"tooling\"\n\n\
         [[crate]]\npackage = \"inner\"\npath = \"tools/inner\"\nowner = \"inner-owner\"\nlayer = \"tooling\"\n",
    )
    .expect("Fix: the fixture registry must be readable");
    assert_eq!(
        registry
            .owning_crate("tools/inner/src/lib.rs")
            .map(|row| row.owner.as_str()),
        Some("inner-owner")
    );
    assert_eq!(
        registry
            .owning_crate("tools/src/lib.rs")
            .map(|row| row.owner.as_str()),
        Some("outer-owner")
    );
}

/// A row and a query written with different separators resolve to one member.
///
/// The reader normalised the query and, in the exact-match arm, not the row, so
/// a row declared with backslashes owned every file under its directory and not
/// the directory itself. Both spellings of both sides answer the same row here.
#[test]
fn a_declared_directory_owns_its_files_whichever_separator_wrote_it() {
    let registry = Registry::parse(
        "schema_version = 2\n\n\
         [[crate]]\npackage = \"win\"\npath = \"tools\\\\win\"\nowner = \"win-owner\"\nlayer = \"tooling\"\n",
    )
    .expect("Fix: the fixture registry must be readable");
    for query in [
        "tools/win",
        "tools\\win",
        "tools/win/src/lib.rs",
        "tools\\win\\src\\lib.rs",
    ] {
        assert_eq!(
            registry.owning_crate(query).map(|row| row.owner.as_str()),
            Some("win-owner"),
            "Fix: `{query}` must resolve to the member that declares its directory"
        );
    }
}

/// A file under no declared member resolves to no owner.
#[test]
fn a_file_outside_every_declared_directory_has_no_owner() {
    let registry = Registry::read(&workspace_root()).expect("Fix: the registry must be readable");
    assert_eq!(registry.owning_crate("docs/ARCHITECTURE.md"), None);
}

/// A row missing a field is reported, not read as an empty owner.
#[test]
fn a_row_missing_a_required_field_fails_closed() {
    for (field, row) in [
        (
            "path",
            "[[crate]]\npackage = \"a\"\nowner = \"a-owner\"\nlayer = \"a-layer\"\n",
        ),
        (
            "owner",
            "[[crate]]\npackage = \"a\"\npath = \"a\"\nlayer = \"a-layer\"\n",
        ),
        (
            "layer",
            "[[crate]]\npackage = \"a\"\npath = \"a\"\nowner = \"a-owner\"\n",
        ),
    ] {
        let error = Registry::parse(row).expect_err("Fix: an incomplete row must be reported");
        assert!(
            error.contains(&format!("declares no `{field}`")),
            "Fix: a row with no `{field}` reported {error:?}"
        );
    }
    let error =
        Registry::parse("[[crate]]\npath = \"a\"\nowner = \"a-owner\"\nlayer = \"a-layer\"\n")
            .expect_err("Fix: a row with no package must be reported");
    assert!(
        error.contains("entry with no `package`"),
        "Fix: a row with no package reported {error:?}"
    );
}

/// A registry that declares nothing is reported rather than read as an empty
/// roster every rule then passes against.
#[test]
fn an_empty_registry_fails_closed() {
    let error =
        Registry::parse("schema_version = 2\n").expect_err("Fix: an empty registry must fail");
    assert!(
        error.contains("declares no [[crate]] entries"),
        "Fix: an empty registry reported {error:?}"
    );
    let error = Registry::read(Path::new("/nonexistent-checkout"))
        .expect_err("Fix: a missing registry must fail");
    assert!(
        error.contains("cannot read"),
        "Fix: a missing registry reported {error:?}"
    );
}
