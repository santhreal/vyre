//! The source-tree digest identifies a package's bytes, not its directory order.
//!
//! Artifact caches mix the digest into their keys, so two trees that differ in
//! any byte, path, or manifest must not share a value, and one tree must produce
//! the same value twice. A digest that ignored a nested file, or that framed
//! contents without their path, would let a cache answer a compiler change with
//! the artifact the previous compiler wrote.
//!
//! What this does not catch: a cache that never mixes the digest in. The driver
//! crates own that side.

use std::path::Path;

use tempfile::TempDir;
use vyre_foundation::source_digest::{
    cargo_directives, source_tree_digest, SourceDigestError, MAX_SOURCE_DIGEST_BYTES,
};

const DOMAIN: &[u8] = b"vyre-source-digest-test-v1\0";

fn package(files: &[(&str, &[u8])]) -> TempDir {
    let root = TempDir::new().expect("Fix: the test needs a temporary directory.");
    write_file(
        &root.path().join("Cargo.toml"),
        b"[package]\nname = \"fixture\"\n",
    );
    for (relative, contents) in files {
        write_file(&root.path().join("src").join(relative), contents);
    }
    root
}

fn write_file(path: &Path, contents: &[u8]) {
    let parent = path.parent().expect("Fix: a fixture path needs a parent.");
    std::fs::create_dir_all(parent).expect("Fix: the fixture directory must be creatable.");
    std::fs::write(path, contents).expect("Fix: the fixture file must be writable.");
}

fn digest(root: &Path) -> String {
    source_tree_digest(root, DOMAIN).expect("Fix: the fixture package must digest.")
}

#[test]
fn one_tree_digests_to_one_value() {
    let tree = package(&[
        ("lib.rs", b"fn main() {}"),
        ("nested/deep.rs", b"const A: u8 = 1;"),
    ]);
    assert_eq!(
        digest(tree.path()),
        digest(tree.path()),
        "Fix: the digest of an unchanged tree changed between calls, so a cache key built from it \
         invalidates on every build and no cache ever hits."
    );
}

#[test]
fn a_changed_byte_in_a_nested_file_changes_the_digest() {
    let before = package(&[
        ("lib.rs", b"fn main() {}"),
        ("nested/deep.rs", b"const A: u8 = 1;"),
    ]);
    let after = package(&[
        ("lib.rs", b"fn main() {}"),
        ("nested/deep.rs", b"const A: u8 = 2;"),
    ]);
    assert_ne!(
        digest(before.path()),
        digest(after.path()),
        "Fix: a nested source edit left the digest unchanged, so an artifact cache answers a \
         compiler change with the artifact the previous compiler wrote."
    );
}

#[test]
fn an_added_file_changes_the_digest() {
    let before = package(&[("lib.rs", b"fn main() {}")]);
    let after = package(&[("lib.rs", b"fn main() {}"), ("added.rs", b"")]);
    assert_ne!(
        digest(before.path()),
        digest(after.path()),
        "Fix: adding a source file left the digest unchanged, so the walk is not covering the tree."
    );
}

#[test]
fn renaming_a_file_changes_the_digest() {
    // The contents and their order are identical in both trees, so only the
    // framed path can tell them apart. A digest that hashed contents alone would
    // let a module move keep the cache key it had before the move, while the
    // emitted artifact names the new path.
    let before = package(&[("a.rs", b"const A: u8 = 1;"), ("b.rs", b"const B: u8 = 2;")]);
    let after = package(&[("a.rs", b"const A: u8 = 1;"), ("c.rs", b"const B: u8 = 2;")]);
    assert_ne!(
        digest(before.path()),
        digest(after.path()),
        "Fix: two trees whose contents match in order but whose paths differ digested alike, so \
         the digest frames contents without their path and a rename is invisible to it."
    );
}

#[test]
fn a_changed_manifest_changes_the_digest() {
    let before = package(&[("lib.rs", b"fn main() {}")]);
    let after = package(&[("lib.rs", b"fn main() {}")]);
    write_file(
        &after.path().join("Cargo.toml"),
        b"[package]\nname = \"fixture\"\nversion = \"2\"\n",
    );
    assert_ne!(
        digest(before.path()),
        digest(after.path()),
        "Fix: a manifest edit left the digest unchanged, so a dependency or feature change that \
         alters the emitted artifact keeps the old cache key."
    );
}

#[test]
fn a_different_domain_changes_the_digest() {
    let tree = package(&[("lib.rs", b"fn main() {}")]);
    let other = source_tree_digest(tree.path(), b"vyre-source-digest-other-v1\0")
        .expect("Fix: the fixture package must digest.");
    assert_ne!(
        digest(tree.path()),
        other,
        "Fix: two domains produced one value, so two caches that hash different subjects collide."
    );
}

#[test]
fn an_empty_source_directory_is_refused() {
    let tree = package(&[]);
    std::fs::create_dir_all(tree.path().join("src")).expect("Fix: the fixture needs src.");
    let error = source_tree_digest(tree.path(), DOMAIN)
        .expect_err("Fix: an empty source tree identifies nothing and must be refused.");
    assert!(
        matches!(error, SourceDigestError::Empty { .. }),
        "Fix: an empty source directory reported {error:?} instead of refusing; a digest over zero \
         files is one constant shared by every compiler version."
    );
}

#[test]
fn a_missing_source_directory_is_refused() {
    let tree = package(&[]);
    let error = source_tree_digest(tree.path(), DOMAIN)
        .expect_err("Fix: a package without sources must be refused, not digested.");
    assert!(
        matches!(error, SourceDigestError::Directory { .. }),
        "Fix: a missing source directory reported {error:?} instead of naming the directory it \
         could not list."
    );
}

#[test]
fn the_cap_bounds_the_whole_tree_not_one_file() {
    let half = MAX_SOURCE_DIGEST_BYTES / 2 + 1;
    let tree = package(&[]);
    let filler = vec![b'x'; usize::try_from(half).expect("Fix: the cap must fit in usize.")];
    write_file(&tree.path().join("src").join("first.rs"), &filler);
    write_file(&tree.path().join("src").join("second.rs"), &filler);
    let error = source_tree_digest(tree.path(), DOMAIN)
        .expect_err("Fix: a tree over the cap must be refused, not read.");
    let SourceDigestError::TooLarge { path, limit } = error else {
        panic!(
            "Fix: two files that together cross the cap reported {error:?}, so the cap is applied \
             per file and a build script can still read an unbounded tree."
        );
    };
    assert_eq!(
        limit, MAX_SOURCE_DIGEST_BYTES,
        "Fix: the refusal must state the cap it applied."
    );
    assert_eq!(
        path,
        tree.path().join("src").join("second.rs"),
        "Fix: the refusal must name the file that crossed the cap."
    );
}

/// WHY: the cap used to be checked after the file was already in memory, so the
/// read itself was unbounded. A source path that names a stream with no end
/// answers `read` forever: the walk never returns and the build hangs instead of
/// refusing. The bound now belongs to the read, so an endless input terminates
/// with the cap it crossed.
///
/// What this does not catch: a sparse regular file, which reports its length and
/// ends on its own.
#[cfg(unix)]
#[test]
fn an_endless_source_file_terminates_with_the_cap() {
    let tree = package(&[("lib.rs", b"fn main() {}")]);
    let endless = tree.path().join("src").join("endless.rs");
    std::os::unix::fs::symlink("/dev/zero", &endless)
        .expect("Fix: the test needs a symlink to an endless stream.");
    let error = source_tree_digest(tree.path(), DOMAIN)
        .expect_err("Fix: an endless source file must be refused, not read to its end.");
    let SourceDigestError::TooLarge { path, limit } = error else {
        panic!(
            "Fix: an endless source file reported {error:?} instead of crossing the cap, so the \
             read is not bounded by the budget."
        );
    };
    assert_eq!(
        limit, MAX_SOURCE_DIGEST_BYTES,
        "Fix: the refusal must state the cap it applied."
    );
    assert_eq!(
        path, endless,
        "Fix: the refusal must name the file that crossed the cap."
    );
}

#[test]
fn the_directives_ask_cargo_to_rerun_for_the_tree_it_digested() {
    let tree = package(&[("lib.rs", b"fn main() {}")]);
    let directives = cargo_directives(tree.path(), "VYRE_FIXTURE_DIGEST", DOMAIN)
        .expect("Fix: the fixture package must digest.");
    let expected = format!(
        "cargo:rerun-if-changed={}\ncargo:rerun-if-changed={}\ncargo:rustc-env=VYRE_FIXTURE_DIGEST={}\n",
        tree.path().join("src").display(),
        tree.path().join("Cargo.toml").display(),
        digest(tree.path())
    );
    assert_eq!(
        directives, expected,
        "Fix: the stamped directives must name both rerun triggers and the digest. A missing \
         trigger pins the previous digest through an edit, which is the stale-cache defect the \
         digest exists to close."
    );
}

#[test]
fn a_variable_name_cargo_cannot_pass_through_is_refused() {
    let tree = package(&[("lib.rs", b"fn main() {}")]);
    for name in ["", "lowercase", "HAS SPACE", "HAS=EQUALS"] {
        let error = cargo_directives(tree.path(), name, DOMAIN).expect_err(
            "Fix: a variable name cargo cannot pass through must be refused, not stamped.",
        );
        assert!(
            matches!(error, SourceDigestError::EnvVarName { .. }),
            "Fix: {name:?} reported {error:?} instead of naming the variable; a directive cargo \
             cannot parse leaves the crate reading an unset variable."
        );
    }
}
