//! Manifest parsing, crate identification, and workspace path resolution.
//!
//! Every gate that resolves a name against the tree or judges source layout
//! relies on root-relative directory resolution and member manifests. This
//! module reads workspace manifests, locates member and crate source roots,
//! and scans discarding imports and inventory registration submitters.

use std::fs;
use std::path::{Path, PathBuf};

use toml::Value;
use walkdir::WalkDir;

use crate::module_layout::CrateRoot;
use crate::source_scan::code_offsets;

/// Largest source or manifest file this gate will read.
///
/// The gate walks whatever tree it is pointed at, so an unbounded
/// `read_to_string` lets one pathological file decide the process's memory.
/// Every read in this crate goes through here.
pub const MAX_SOURCE_BYTES: u64 = 16_777_216;

/// Read a source or manifest file, refusing anything over [`MAX_SOURCE_BYTES`].
pub(crate) fn read_source_bounded(path: &Path) -> std::io::Result<String> {
    use std::io::Read as _;

    let file = fs::File::open(path)?;
    let length = file.metadata()?.len();
    if length > MAX_SOURCE_BYTES {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!(
                "`{}` is {length} bytes; refusing to read more than {MAX_SOURCE_BYTES}",
                path.display()
            ),
        ));
    }
    let mut text = String::with_capacity(length as usize);
    file.take(MAX_SOURCE_BYTES + 1).read_to_string(&mut text)?;
    Ok(text)
}

/// The checkout-relative directory of a workspace member, by package name.
///
/// A member's directory is not always its package name: `vyre-conform` lives at
/// `conform/vyre-conform`. The roster is read from the root manifest at run
/// time, so a gate that needs its own crate directory gets it without a
/// compiled-in manifest path, which would name whichever checkout built the
/// binary.
///
/// # Panics
/// Panics when no member directory's manifest declares `package`.
#[must_use]
pub fn member_directory(root: &Path, package: &str) -> PathBuf {
    for member in workspace_members(root) {
        let member_dir = root.join(&member);
        let manifest_path = member_dir.join("Cargo.toml");
        let Ok(text) = read_source_bounded(&manifest_path) else {
            continue;
        };
        let declared = toml::from_str::<toml::Table>(&text).ok().and_then(|table| {
            Value::Table(table)
                .get("package")
                .and_then(|pkg| pkg.get("name"))
                .and_then(Value::as_str)
                .map(str::to_string)
        });
        if declared.as_deref() == Some(package) {
            return member_dir;
        }
    }
    panic!(
        "Fix: no workspace member under {} declares package `{package}`; the roster in the root \
         Cargo.toml is what this resolves against.",
        root.display()
    );
}

/// The workspace member roster, as the root manifest declares it.
///
/// Every gate that resolves a name against the tree needs this list, so it has
/// one owner: a second copy drifts the moment a member is added under a path
/// one copy filters and the other does not.
///
/// # Panics
///
/// Panics when the root manifest cannot be read or parsed.
#[must_use]
pub fn workspace_members(root: &Path) -> Vec<String> {
    workspace_paths(root, "members")
}

/// The paths the root manifest excludes from the workspace.
///
/// `exclude` is the other half of the roster: a directory that is neither a
/// member nor excluded is a directory cargo will pull in the day it grows a
/// manifest. Reading it beside [`workspace_members`] keeps both answers coming
/// from one parse of one file.
///
/// # Panics
///
/// Panics when the root manifest cannot be read or parsed.
#[must_use]
pub fn workspace_excludes(root: &Path) -> Vec<String> {
    workspace_paths(root, "exclude")
}

/// One `[workspace]` array of paths, empty when the key is absent.
///
/// # Panics
///
/// Panics when the root manifest cannot be read or parsed. Every gate in this
/// crate answers for the roster that manifest declares, so a gate that carried
/// on with an empty roster would report a clean tree it never read.
pub(crate) fn workspace_paths(root: &Path, key: &str) -> Vec<String> {
    let manifest_path = root.join("Cargo.toml");
    let text = read_source_bounded(&manifest_path)
        .unwrap_or_else(|error| panic!("Fix: cannot read {}: {error}", manifest_path.display()));
    let table: toml::Table = toml::from_str(&text)
        .unwrap_or_else(|error| panic!("Fix: parse {}: {error}", manifest_path.display()));
    Value::Table(table)
        .get("workspace")
        .and_then(|workspace| workspace.get(key))
        .and_then(Value::as_array)
        .map(|entries| {
            entries
                .iter()
                .filter_map(Value::as_str)
                .map(str::to_string)
                .collect()
        })
        .unwrap_or_default()
}

/// Crate identifier for a crate name, e.g. `vyre_libs` for `vyre-libs`.
#[must_use]
pub fn crate_ident(crate_name: &str) -> String {
    crate_name.replace('-', "_")
}

/// This crate's own sources. Its tests carry example registrations that name
/// other crates on purpose, so scanning itself would report its own fixtures.
pub(crate) const SELF_CRATE: &str = "structure-gate";

pub(crate) fn source_files(root: &Path, member: &str) -> Vec<PathBuf> {
    if member == SELF_CRATE {
        return Vec::new();
    }
    source_tree_files(&root.join(member))
}

/// Every `.rs` file under one source tree.
pub(crate) fn source_tree_files(directory: &Path) -> Vec<PathBuf> {
    let mut files: Vec<PathBuf> = WalkDir::new(directory)
        .into_iter()
        .filter_map(Result::ok)
        .filter(|e| e.file_type().is_file() && e.path().extension().is_some_and(|ext| ext == "rs"))
        .map(walkdir::DirEntry::into_path)
        .collect();
    files.sort();
    files
}

/// Every crate in the checkout that keeps its sources in a `src/` directory.
///
/// Read from the tree rather than the workspace roster, because the layout
/// rules judge tree shape and a crate outside the workspace grows the same
/// pairs and the same nameless modules: the external extension examples are
/// separate packages on purpose. A directory earns a place here by declaring
/// `[package]` and holding Rust source under `src/`, so a crate added anywhere
/// in the checkout is judged without an edit here. A `src/` emptied by a
/// deletion is not a crate root: the directory survives the pull that removed
/// every file in it. This crate is included; [`source_files`]
/// exempts it only because its registration fixtures name other crates, and a
/// rule over file names has no such fixtures.
pub(crate) fn crate_source_roots(root: &Path) -> Vec<CrateRoot> {
    let mut roots: Vec<CrateRoot> = WalkDir::new(root)
        .into_iter()
        .filter_map(Result::ok)
        .filter(|e| e.file_type().is_file() && e.file_name() == "Cargo.toml")
        .filter_map(|entry| {
            let directory = entry.path().parent()?;
            let ident = manifest_crate_ident(entry.path())?;
            crate::source_scan::carries_rust_source(&directory.join("src")).then(|| CrateRoot {
                directory: relative(root, directory),
                ident,
            })
        })
        .collect();
    roots.sort_by(|left, right| left.directory.cmp(&right.directory));
    roots
}

/// The identifier a manifest's library carries, or `None` for no package.
///
/// `[lib] name` wins where it is written, because that is the name a consumer
/// and `cargo public-api` both use; the package name is the default Cargo
/// applies when it is not.
pub(crate) fn manifest_crate_ident(manifest: &Path) -> Option<String> {
    let text = read_source_bounded(manifest).ok()?;
    let table: toml::Table = toml::from_str(&text).ok()?;
    let value = Value::Table(table);
    let package = value.get("package")?.get("name")?.as_str()?;
    let name = value
        .get("lib")
        .and_then(|lib| lib.get("name"))
        .and_then(Value::as_str)
        .unwrap_or(package);
    Some(crate_ident(name))
}

pub(crate) fn relative(root: &Path, path: &Path) -> String {
    path.strip_prefix(root)
        .unwrap_or(path)
        .to_string_lossy()
        .replace('\\', "/")
}

/// Workspace root, resolved from the directory the gate was invoked in.
///
/// Never compiled in. A target directory shared by several checkouts computes the
/// same unit hash for all of them, so cargo hands one checkout a binary another
/// one built; a path baked into that binary then names the wrong tree, and
/// `VYRE_CHECKOUT_ROOT` did not prevent it, because cargo does not export a
/// `relative = true` config variable to the process it runs. The tree the
/// operator invoked cargo in is the tree the gate must answer for.
///
/// # Panics
///
/// Panics when no ancestor of the working directory declares a `[workspace]`.
#[must_use]
pub fn workspace_root() -> PathBuf {
    let start = std::env::current_dir()
        .expect("Fix: the working directory must be readable to locate the vyre checkout");
    workspace_root_from(&start).unwrap_or_else(|| {
        panic!(
            "Fix: run this from inside the vyre checkout; no ancestor of `{}` has a Cargo.toml \
             declaring [workspace].",
            start.display()
        )
    })
}

/// The nearest ancestor of `start`, inclusive, whose manifest declares a workspace.
#[must_use]
pub fn workspace_root_from(start: &Path) -> Option<PathBuf> {
    start
        .ancestors()
        .find(|directory| {
            read_source_bounded(&directory.join("Cargo.toml")).is_ok_and(|text| {
                text.lines()
                    .any(|line| line.trim_start().starts_with("[workspace]"))
            })
        })
        .map(Path::to_path_buf)
}

/// True when this source text submits an inventory registration.
///
/// Comments are skipped, so a doc comment explaining the linkage rule is not
/// mistaken for a registration.
#[must_use]
pub fn submits_registrations(text: &str) -> bool {
    for offset in code_offsets(text) {
        let rest = &text[offset..];
        if rest.starts_with("inventory::submit!") {
            return true;
        }
    }
    false
}

/// Crate identifiers named by a discarding import, as written.
///
/// Only a bare crate identifier counts. `use std::io::Read as _;` imports a
/// trait into scope, which is the legitimate use of the form and references a
/// symbol at every call site.
#[must_use]
pub fn discarding_imports(text: &str) -> Vec<String> {
    let mut imports = Vec::new();
    for offset in code_offsets(text) {
        let rest = &text[offset..];
        if rest.starts_with("use ") {
            if let Some(crate_name) = discarded_crate(&rest["use ".len()..]) {
                imports.push(crate_name);
            }
        }
    }
    imports
}

/// Crate identifier of a `use <crate> as _;` statement starting at `rest`.
fn discarded_crate(rest: &str) -> Option<String> {
    let mut tokens = rest.split_whitespace();
    let imported = tokens.next()?;
    let as_token = tokens.next()?;
    let underscore = tokens.next()?;
    if as_token != "as" {
        return None;
    }
    if !underscore.starts_with('_') {
        return None;
    }
    let without_semi = imported.strip_suffix(';').unwrap_or(imported);
    if without_semi.contains("::") {
        return None;
    }
    Some(without_semi.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_trait_import_is_not_a_discarding_crate_import() {
        let named = discarding_imports("fn read() {\n    use std::io::Read as _;\n}\n");

        assert!(named.is_empty(), "{named:?}");
    }

    #[test]
    fn a_crate_named_only_inside_a_comment_is_not_an_import() {
        let named = discarding_imports(
            "/// Naming the crate with `use vyre_libs as _;` references nothing.\npub fn anchor() {}\n",
        );

        assert!(named.is_empty(), "{named:?}");
    }

    #[test]
    fn a_discarding_crate_import_is_read_from_source() {
        let named = discarding_imports(
            "#[cfg(feature = \"gpu\")]\nuse vyre_driver_metal as _;\nuse vyre_libs as _;\n",
        );

        assert_eq!(named, vec!["vyre_driver_metal", "vyre_libs"]);
    }

    #[test]
    fn a_submission_inside_a_comment_does_not_make_a_crate_a_submitter() {
        assert!(!submits_registrations(
            "// This crate reads the registry; inventory::submit! lives in the driver.\n"
        ));
        assert!(submits_registrations(
            "inventory::submit! {\n    ExampleRegistration { id: \"example\" }\n}\n"
        ));
        assert!(submits_registrations(
            "submit_hardware_intrinsic! {\n    id: \"example\",\n}\n"
        ));
        assert!(submits_registrations(
            "define_unary_u32_hardware_intrinsic!(foo, \"example\", expr);\n"
        ));
    }

    #[test]
    fn manifest_with_leading_comments_parses_workspace_members() {
        let temp_dir = tempfile::tempdir().expect("tempdir");
        let manifest_path = temp_dir.path().join("Cargo.toml");
        let manifest_content = r#"# Workspace members are listed explicitly.
# Comments at top of file.
[workspace]
resolver = "2"
members = [
    # Core compiler
    "vyre",
    "vyre-libs",
]
"#;
        std::fs::write(&manifest_path, manifest_content).expect("write manifest");
        let members = workspace_members(temp_dir.path());
        assert_eq!(members, vec!["vyre", "vyre-libs"]);
    }

    #[test]
    fn member_directory_resolves_package_from_manifest_with_comments() {
        let temp_dir = tempfile::tempdir().expect("tempdir");
        let root = temp_dir.path();
        std::fs::write(
            root.join("Cargo.toml"),
            "# Workspace root\n[workspace]\nmembers = [\"nested/pkg-a\"]\n",
        )
        .expect("write root Cargo.toml");
        let pkg_dir = root.join("nested").join("pkg-a");
        std::fs::create_dir_all(&pkg_dir).expect("create pkg dir");
        std::fs::write(
            pkg_dir.join("Cargo.toml"),
            "# Package manifest\n[package]\nname = \"pkg-a\"\nversion = \"0.1.0\"\n",
        )
        .expect("write member Cargo.toml");

        let found = member_directory(root, "pkg-a");
        assert_eq!(found, pkg_dir);
    }
}
