//! Every `#[path]` module declaration in the workspace resolves inside the
//! Cargo package that declares it.
//!
//! # The class this closes
//!
//! A `#[path]` attribute whose value climbs out of its package makes one source
//! file the textual body of several test targets at once. Ten support sources
//! under a workspace-root `tests/support/` were included 46 times by 41 test
//! files that way. Each copy compiled under a different feature resolution,
//! needed a crate-wide dead-code allowance because a fixture one target skipped
//! was unused there, and was absent from the consuming package archive, so a
//! published-source test lost the contract body while the monorepo build kept
//! passing. A shared contract belongs to one library target, which is what
//! `vyre-test-support` is.
//!
//! The finding is structural, so it is checked structurally. The package set is
//! discovered by reading manifests on each run, not listed here, so a crate
//! added tomorrow is judged tomorrow, and the root workspace manifest's
//! `members` list is held to that discovery: a member whose manifest stopped
//! declaring a package would otherwise silently drop out of the owner set and
//! turn every escape from it into a pass. The declaration set is enumerated
//! from source the same way, and both enumerations verify a known anchor,
//! because the failure mode of a walk is finding nothing and an empty set
//! satisfies every per-item assertion.
//!
//! # What it does not catch
//!
//! Resolution tracks inline `mod name { … }` blocks using Rust 2021 submodule
//! directory rules (crate roots and `mod.rs` resolve inline `mod name` to
//! `<dir>/name`, while non-root `<dir>/foo.rs` resolves inline `mod name` to
//! `<dir>/foo/name`).
//!
//! It reads attributes as text with comments masked out, so it judges what the
//! source says rather than what a `cfg` selects: a declaration behind a
//! `#[cfg]` that never activates is still held to the rule. A `#[path]` value
//! assembled by a macro, or one appearing inside a string literal directly
//! ahead of the word `mod`, is outside what a text scan can decide.
//!
//! It says nothing about whether the module a package owns is the right owner
//! for the contract inside it. Duplication across packages that is copied
//! rather than included escapes a `#[path]` audit entirely.

use std::collections::{BTreeMap, BTreeSet};
use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};

use xtask::workspace_root as vyre_workspace_root;

/// Directory names the walk never enters.
///
/// Build output and vendored trees hold generated Rust that no package owns,
/// and `.internal` is not a published tree.
const PRUNED: &[&str] = &["target", ".git", "node_modules", ".internal"];

/// Known workspace member that manifest discovery must always resolve.
const ANCHOR_MEMBER: &str = "crates/vyre-runtime";

/// File inside `xtask` that declares this module via `#[path]`, proving the
/// attribute scanner is non-vacuous regardless of how many crates exist.
const ANCHOR_DECLARATION_FILE: &str = "tools/xtask/tests/all_tests.rs";
/// One `#[path]` module declaration, resolved.
struct Declaration {
    /// File that declares the module, relative to the workspace root.
    file: PathBuf,
    /// Line of the `#[path]` attribute in that file, 1-based.
    line: usize,
    /// Attribute value, verbatim.
    value: String,
    /// Absolute path the attribute names, relative to the declaring file's
    /// directory.
    target: PathBuf,
}

#[test]
fn every_path_module_resolves_inside_its_own_package() {
    let root = vyre_workspace_root();
    let members = declared_members(&root);
    let (packages, rs_files) = discover_workspace(&root, true);
    let declarations = path_declarations(&root, &packages, &rs_files);
    assert!(
        !members.is_empty()
            && members.iter().any(|m| m == ANCHOR_MEMBER)
            && packages.contains_key(&root.join(ANCHOR_MEMBER)),
        "Fix: manifest discovery under {} did not resolve `{ANCHOR_MEMBER}` (members: {}, packages: {}).",
        root.display(),
        members.len(),
        packages.len()
    );

    assert!(
        declarations
            .iter()
            .any(|d| d.file == Path::new(ANCHOR_DECLARATION_FILE)),
        "Fix: `#[path]` scan under {} did not find `{ANCHOR_DECLARATION_FILE}`, so the source scan is vacuous.",
        root.display()
    );
    let mut findings = Vec::new();
    for declaration in &declarations {
        let declaring = root.join(&declaration.file);
        let Some(home) = owning_package(&packages, &declaring) else {
            findings.push(format!(
                "{}:{} declares `#[path = \"{}\"]` from a file no Cargo package owns",
                declaration.file.display(),
                declaration.line,
                declaration.value
            ));
            continue;
        };
        match owning_package(&packages, &declaration.target) {
            Some(reached) if reached == home => {}
            Some(reached) => findings.push(format!(
                "{}:{} declares `#[path = \"{}\"]`, which resolves into package {} while the file belongs to {}. Move the shared body into a library target both packages depend on, such as vyre-test-support, and reach it with `use`.",
                declaration.file.display(),
                declaration.line,
                declaration.value,
                relative(&root, reached).display(),
                relative(&root, home).display()
            )),
            None => findings.push(format!(
                "{}:{} declares `#[path = \"{}\"]`, which resolves to {} outside every Cargo package. A module body outside a package compiles into each including target separately and is absent from the package archive.",
                declaration.file.display(),
                declaration.line,
                declaration.value,
                relative(&root, &declaration.target).display()
            )),
        }
        if !declaration.target.exists() {
            findings.push(format!(
                "{}:{} declares `#[path = \"{}\"]`, which names no file at {}.",
                declaration.file.display(),
                declaration.line,
                declaration.value,
                relative(&root, &declaration.target).display()
            ));
        }
    }

    assert!(
        findings.is_empty(),
        "Fix: {} `#[path]` declaration finding(s):\n{}",
        findings.len(),
        findings.join("\n")
    );
}

#[test]
fn every_declared_workspace_member_is_a_discovered_package() {
    let root = vyre_workspace_root();
    let packages: BTreeSet<PathBuf> = package_roots(&root).into_keys().collect();

    let members = declared_members(&root);
    assert!(
        !members.is_empty() && members.iter().any(|m| m == ANCHOR_MEMBER),
        "Fix: root manifest at {} did not yield `{ANCHOR_MEMBER}` in workspace members ({members:?}).",
        root.join("Cargo.toml").display()
    );

    let undiscovered: Vec<&String> = members
        .iter()
        .filter(|member| !packages.contains(&root.join(member)))
        .collect();
    assert!(
        undiscovered.is_empty(),
        "Fix: workspace member(s) {undiscovered:?} were not discovered as Cargo packages. A member whose manifest no longer declares `[package]` drops out of the owner set, and every `#[path]` escaping it would then read as in-package."
    );
}

/// Masking preserves byte offsets and never splits a character.
///
/// The scan reports a declaration by the byte offset the mask hands back, so a
/// mask that changed a length would misreport every later line. Non-ASCII text
/// is the boundary: a source carrying `⚠` inside a string literal used to abort
/// the whole scan on a byte index that was not a character boundary, which the
/// corpus tests read as an escape-free tree only because they never finished.
#[test]
fn masking_keeps_byte_offsets_across_non_ascii_sources() {
    let cases = [
        "// ⚠ warning ─────\nmod a;\n",
        "let s = \"⚠ warning\";\nmod a;\n",
        "let s = \"\\⚠\";\nmod a;\n",
        "let s = r#\"⚠ raw \"# ;\nmod a;\n",
        "/* ⚠ block ─── */\nmod a;\n",
        "let s = \"unterminated ⚠",
    ];
    for source in cases {
        let masked = mask_comments(source);
        assert_eq!(
            masked.len(),
            source.len(),
            "Fix: masking {source:?} changed the byte length, so every offset the scan reports after it names the wrong column."
        );
        assert_eq!(
            masked.lines().count(),
            source.lines().count(),
            "Fix: masking {source:?} changed the line count, so the scan reports declarations on the wrong line."
        );
    }

    let literal = "let s = \"⚠\";";
    let comment = "// ⚠ tail";
    assert_eq!(
        mask_comments(&format!("{literal} {comment}\n")),
        format!("{literal} {}\n", " ".repeat(comment.len())),
        "Fix: masking must keep a literal's bytes verbatim and replace each comment byte with one space."
    );
}

#[test]
fn inline_mod_resolves_path_against_enclosing_module_directory() {
    let pkg = Path::new("/ws/crates/demo");
    let root_test = Path::new("/ws/crates/demo/tests/all_tests.rs");
    let root_src = r#"
        #[path = "top.rs"]
        pub mod top;
        pub mod lower {
            #[path = "nested.rs"]
            pub mod nested;
            #[path = "../../../escaped.rs"]
            pub mod escaped;
        }
    "#;
    let found_root = scan_file_declarations(Some(pkg), root_test, &mask_comments(root_src));
    assert_eq!(found_root.len(), 3);
    assert_eq!(
        found_root[0].2,
        PathBuf::from("/ws/crates/demo/tests/top.rs")
    );
    assert_eq!(
        found_root[1].2,
        PathBuf::from("/ws/crates/demo/tests/lower/nested.rs")
    );
    assert_eq!(found_root[2].2, PathBuf::from("/ws/crates/escaped.rs"));
    assert!(!found_root[2].2.starts_with(pkg));

    let non_root = Path::new("/ws/crates/demo/src/helper.rs");
    let non_root_src = r#"
        mod inner {
            #[path = "leaf.rs"]
            mod leaf;
        }
    "#;
    let found_non_root = scan_file_declarations(Some(pkg), non_root, &mask_comments(non_root_src));
    assert_eq!(found_non_root.len(), 1);
    assert_eq!(
        found_non_root[0].2,
        PathBuf::from("/ws/crates/demo/src/helper/inner/leaf.rs")
    );
}

/// Every directory under `root` whose `Cargo.toml` declares a package, mapped to
/// that manifest's path.
///
/// Discovery is per-manifest rather than per-member so that a package excluded
/// from the workspace, such as an example crate, still owns its own sources.
fn package_roots(root: &Path) -> BTreeMap<PathBuf, PathBuf> {
    discover_workspace(root, false).0
}

/// Single-walk discovery of package manifests and (when `collect_rs` is set)
/// `.rs` source files under `root`, using `DirEntry::file_type()` to avoid
/// per-entry `stat` round-trips.
fn discover_workspace(root: &Path, collect_rs: bool) -> (BTreeMap<PathBuf, PathBuf>, Vec<PathBuf>) {
    let mut roots = BTreeMap::new();
    let mut rs_files = Vec::new();
    let mut pending = vec![root.to_path_buf()];
    while let Some(dir) = pending.pop() {
        let Ok(entries) = std::fs::read_dir(&dir) else {
            continue;
        };
        for entry in entries.flatten() {
            let file_name = entry.file_name();
            let Some(name) = file_name.to_str() else {
                continue;
            };
            if PRUNED.contains(&name) {
                continue;
            }
            let Ok(ft) = entry.file_type() else {
                continue;
            };
            let path = entry.path();
            let is_dir = ft.is_dir() || (ft.is_symlink() && path.is_dir());
            if is_dir {
                pending.push(path);
            } else if name == "Cargo.toml" {
                let text = std::fs::read_to_string(&path)
                    .unwrap_or_else(|error| panic!("Fix: read {}: {error}", path.display()));
                if declares_a_package(&text) {
                    roots.insert(dir.clone(), path);
                }
            } else if collect_rs && name.ends_with(".rs") {
                rs_files.push(path);
            }
        }
    }
    rs_files.sort();
    (roots, rs_files)
}

/// True when `manifest` has a `[package]` table.
///
/// Matched on a line of its own so that `[package.metadata.cargo-machete]` and
/// the `[workspace.package]` inheritance table in the root manifest do not read
/// as a package declaration.
fn declares_a_package(manifest: &str) -> bool {
    manifest.lines().any(|line| line.trim() == "[package]")
}

/// The `members` entries of the root workspace manifest.
fn declared_members(root: &Path) -> BTreeSet<String> {
    let manifest = root.join("Cargo.toml");
    let text = std::fs::read_to_string(&manifest)
        .unwrap_or_else(|error| panic!("Fix: read {}: {error}", manifest.display()));
    // A document, not a value expression: `toml::Value`'s `FromStr` parses one
    // expression, so a manifest opening with a comment reads as trailing content.
    let parsed = toml::from_str::<toml::Table>(&text)
        .unwrap_or_else(|error| panic!("Fix: parse {}: {error}", manifest.display()));
    let members = parsed
        .get("workspace")
        .and_then(|workspace| workspace.get("members"))
        .and_then(toml::Value::as_array)
        .unwrap_or_else(|| {
            panic!(
                "Fix: {} declares no `[workspace] members` array; the owner set cannot be pinned to it.",
                manifest.display()
            )
        });
    members
        .iter()
        .map(|member| {
            member
                .as_str()
                .unwrap_or_else(|| {
                    panic!(
                        "Fix: {} lists a non-string workspace member {member}.",
                        manifest.display()
                    )
                })
                .to_string()
        })
        .collect()
}

/// The package whose directory is the longest prefix of `path`.
fn owning_package<'a>(packages: &'a BTreeMap<PathBuf, PathBuf>, path: &Path) -> Option<&'a Path> {
    packages
        .keys()
        .filter(|root| path.starts_with(root))
        .max_by_key(|root| root.as_os_str().len())
        .map(PathBuf::as_path)
}

/// True when `entry` is a crate root or `mod.rs` inside `pkg_root`, so an
/// inline `mod child { … }` resolves submodules under `entry.parent()/child`
/// rather than `entry.parent()/<stem>/child`.
fn is_root_or_mod_rs(pkg_root: Option<&Path>, entry: &Path) -> bool {
    if entry.file_name().and_then(|n| n.to_str()) == Some("mod.rs") {
        return true;
    }
    let Some(pkg) = pkg_root else {
        return false;
    };
    let Ok(rel) = entry.strip_prefix(pkg) else {
        return false;
    };
    let parts: Vec<&str> = rel.iter().filter_map(|c| c.to_str()).collect();
    matches!(
        parts.as_slice(),
        ["build.rs"]
            | ["src", "lib.rs"]
            | ["src", "main.rs"]
            | ["src", "bin", _]
            | ["tests", _]
            | ["benches", _]
            | ["examples", _]
    )
}

/// Every `#[path]` module declaration under `root`, resolved against its
/// enclosing module directory.
fn path_declarations(
    root: &Path,
    packages: &BTreeMap<PathBuf, PathBuf>,
    rs_files: &[PathBuf],
) -> Vec<Declaration> {
    let workers = rs_files.len().clamp(1, 8);
    let cursor = AtomicUsize::new(0);
    let mut chunks: Vec<Vec<Declaration>> = (0..workers).map(|_| Vec::new()).collect();
    std::thread::scope(|scope| {
        for out in &mut chunks {
            let cursor = &cursor;
            scope.spawn(move || {
                let mut buf = Vec::with_capacity(64 * 1024);
                loop {
                    let idx = cursor.fetch_add(1, Ordering::Relaxed);
                    let Some(entry) = rs_files.get(idx) else {
                        break;
                    };
                    buf.clear();
                    std::fs::File::open(entry)
                        .and_then(|mut f| f.read_to_end(&mut buf))
                        .unwrap_or_else(|error| panic!("Fix: read {}: {error}", entry.display()));
                    if !buf.windows(2).any(|w| w == b"#[") || !buf.windows(4).any(|w| w == b"path")
                    {
                        continue;
                    }
                    let text = std::str::from_utf8(&buf).unwrap_or_else(|error| {
                        panic!("Fix: invalid UTF-8 in {}: {error}", entry.display())
                    });
                    let masked = mask_comments(text);
                    let pkg_root = owning_package(packages, entry);
                    for (offset, value, target) in scan_file_declarations(pkg_root, entry, &masked)
                    {
                        out.push(Declaration {
                            file: relative(root, entry).to_path_buf(),
                            line: masked[..offset].matches('\n').count() + 1,
                            target,
                            value,
                        });
                    }
                }
            });
        }
    });
    let mut declarations: Vec<Declaration> = chunks.into_iter().flatten().collect();
    declarations.sort_by(|a, b| (&a.file, a.line).cmp(&(&b.file, b.line)));
    declarations
}

/// Byte offset, verbatim `#[path]` string, and lexical target path of every
/// `#[path = "…"]` module declaration in `masked`.
fn scan_file_declarations(
    pkg_root: Option<&Path>,
    entry: &Path,
    masked: &str,
) -> Vec<(usize, String, PathBuf)> {
    let file_dir = entry.parent().unwrap_or_else(|| Path::new("."));
    let file_submod_root = if is_root_or_mod_rs(pkg_root, entry) {
        file_dir.to_path_buf()
    } else {
        match entry.file_stem() {
            Some(stem) => file_dir.join(stem),
            None => file_dir.to_path_buf(),
        }
    };

    let mut found = Vec::new();
    let bytes = masked.as_bytes();
    let mut brace_depth = 0usize;
    let mut mod_scopes: Vec<(usize, PathBuf)> = Vec::new();
    let mut pending_path: Option<(usize, String)> = None;
    let mut at = 0usize;

    while at < bytes.len() {
        if bytes[at].is_ascii_whitespace() {
            at += 1;
            continue;
        }
        if let Some(end) = raw_string_end(bytes, at) {
            at = end;
            continue;
        }
        if bytes[at] == b'"' {
            at = skip_quoted_string(masked, at);
            continue;
        }
        if bytes[at] == b'\'' {
            at = skip_char_or_lifetime(bytes, at);
            continue;
        }
        if bytes[at..].starts_with(b"#[") {
            let start = at;
            if let Some((end, path_val)) = parse_attribute(masked, start) {
                if let Some(val) = path_val {
                    pending_path = Some((start, val));
                }
                at = end;
                continue;
            }
        }
        match bytes[at] {
            b'{' => {
                brace_depth += 1;
                pending_path = None;
                at += 1;
                continue;
            }
            b'}' => {
                brace_depth = brace_depth.saturating_sub(1);
                while mod_scopes.last().is_some_and(|(d, _)| *d > brace_depth) {
                    mod_scopes.pop();
                }
                pending_path = None;
                at += 1;
                continue;
            }
            b';' => {
                pending_path = None;
                at += 1;
                continue;
            }
            _ => {}
        }
        if let Some((ident, next_at)) = scan_ident(masked, at) {
            at = next_at;
            if ident == "pub" {
                while at < bytes.len() && bytes[at].is_ascii_whitespace() {
                    at += 1;
                }
                if bytes.get(at) == Some(&b'(') {
                    if let Some(close) = bytes[at..].iter().position(|&b| b == b')') {
                        at += close + 1;
                    }
                }
                continue;
            }
            if ident == "mod" {
                while at < bytes.len() && bytes[at].is_ascii_whitespace() {
                    at += 1;
                }
                let mod_name = if let Some((name, after_name)) = scan_ident(masked, at) {
                    at = after_name;
                    name.strip_prefix("r#").unwrap_or(name)
                } else {
                    ""
                };
                while at < bytes.len() && bytes[at].is_ascii_whitespace() {
                    at += 1;
                }
                let base_dir = mod_scopes.last().map_or(file_dir, |(_, dir)| dir.as_path());
                let is_inline = bytes.get(at) == Some(&b'{');
                if let Some((attr_start, val)) = pending_path.take() {
                    let resolved = normalize(&base_dir.join(&val));
                    found.push((attr_start, val, resolved.clone()));
                    if is_inline {
                        brace_depth += 1;
                        at += 1;
                        let inline_dir = if resolved.extension().is_some() {
                            resolved.parent().unwrap_or(&resolved).to_path_buf()
                        } else {
                            resolved
                        };
                        mod_scopes.push((brace_depth, inline_dir));
                    }
                } else if is_inline && !mod_name.is_empty() {
                    brace_depth += 1;
                    at += 1;
                    let sub_dir = match mod_scopes.last() {
                        Some((_, dir)) => dir.join(mod_name),
                        None => file_submod_root.join(mod_name),
                    };
                    mod_scopes.push((brace_depth, normalize(&sub_dir)));
                }
                continue;
            }
            pending_path = None;
            continue;
        }
        at += char_width(bytes[at]);
    }
    found
}

fn skip_quoted_string(masked: &str, start: usize) -> usize {
    let bytes = masked.as_bytes();
    let mut at = start + 1;
    while at < bytes.len() {
        if bytes[at] == b'\\' && at + 1 < bytes.len() {
            at = (at + 1 + char_width(bytes[at + 1])).min(bytes.len());
            continue;
        }
        let b = bytes[at];
        at += char_width(b);
        if b == b'"' {
            break;
        }
    }
    at
}

fn skip_char_or_lifetime(bytes: &[u8], start: usize) -> usize {
    if bytes.get(start + 1) == Some(&b'\\') {
        let mut at = start + 2;
        while at < bytes.len() && at <= start + 12 {
            let b = bytes[at];
            at += char_width(b);
            if b == b'\'' {
                return at;
            }
        }
        return start + 1;
    }
    if start + 1 < bytes.len() {
        let after_ch = start + 1 + char_width(bytes[start + 1]);
        if bytes.get(after_ch) == Some(&b'\'') {
            return after_ch + 1;
        }
    }
    start + 1
}

fn parse_attribute(masked: &str, start: usize) -> Option<(usize, Option<String>)> {
    let bytes = masked.as_bytes();
    let mut at = start + 2;
    let mut bracket_depth = 1usize;
    while at < bytes.len() && bracket_depth > 0 {
        if let Some(end) = raw_string_end(bytes, at) {
            at = end;
            continue;
        }
        if bytes[at] == b'"' {
            at = skip_quoted_string(masked, at);
            continue;
        }
        match bytes[at] {
            b'[' => {
                bracket_depth += 1;
                at += 1;
            }
            b']' => {
                bracket_depth -= 1;
                at += 1;
            }
            b => at += char_width(b),
        }
    }
    if bracket_depth != 0 {
        return None;
    }
    let inner = masked[start + 2..at - 1].trim();
    let path_val = inner
        .strip_prefix("path")
        .map(str::trim_start)
        .and_then(|s| s.strip_prefix('='))
        .map(str::trim)
        .and_then(|s| s.strip_prefix('"')?.strip_suffix('"'))
        .filter(|v| !v.is_empty() && !v.contains('\\'))
        .map(str::to_owned);
    Some((at, path_val))
}

fn scan_ident(masked: &str, start: usize) -> Option<(&str, usize)> {
    let bytes = masked.as_bytes();
    let mut at = start;
    if bytes[at..].starts_with(b"r#")
        && bytes
            .get(at + 2)
            .is_some_and(|b| b.is_ascii_alphabetic() || *b == b'_')
    {
        at += 2;
    } else if !(bytes[at].is_ascii_alphabetic() || bytes[at] == b'_') {
        return None;
    }
    while at < bytes.len() && (bytes[at].is_ascii_alphanumeric() || bytes[at] == b'_') {
        at += 1;
    }
    Some((&masked[start..at], at))
}

/// `source` with every comment replaced by spaces, keeping byte offsets and line
/// breaks.
///
/// String and raw-string bodies are preserved so that a `//` inside a literal
/// does not swallow the rest of the line.
fn mask_comments(source: &str) -> String {
    let bytes = source.as_bytes();
    let mut out = String::with_capacity(source.len());
    let mut at = 0;
    while at < bytes.len() {
        if bytes[at..].starts_with(b"//") {
            while at < bytes.len() && bytes[at] != b'\n' {
                out.push(' ');
                at += 1;
            }
            continue;
        }
        if bytes[at..].starts_with(b"/*") {
            let mut depth = 1_usize;
            out.push_str("  ");
            at += 2;
            while at < bytes.len() && depth > 0 {
                if bytes[at..].starts_with(b"/*") {
                    depth += 1;
                    out.push_str("  ");
                    at += 2;
                } else if bytes[at..].starts_with(b"*/") {
                    depth -= 1;
                    out.push_str("  ");
                    at += 2;
                } else {
                    out.push(if bytes[at] == b'\n' { '\n' } else { ' ' });
                    at += 1;
                }
            }
            continue;
        }
        if let Some(end) = raw_string_end(bytes, at) {
            out.push_str(&source[at..end]);
            at = end;
            continue;
        }
        if bytes[at] == b'"' {
            out.push('"');
            at += 1;
            // Whole characters, not bytes. A literal holding a non-ASCII
            // character, `"⚠ ..."` among them, sliced mid-sequence and panicked
            // on a byte index that is not a character boundary.
            while at < bytes.len() {
                if bytes[at] == b'\\' && at + 1 < bytes.len() {
                    let width = 1 + char_width(bytes[at + 1]);
                    let end = (at + width).min(bytes.len());
                    out.push_str(&source[at..end]);
                    at = end;
                    continue;
                }
                let byte = bytes[at];
                let end = at + char_width(byte);
                out.push_str(&source[at..end]);
                at = end;
                if byte == b'"' {
                    break;
                }
            }
            continue;
        }
        let width = char_width(bytes[at]);
        out.push_str(&source[at..at + width]);
        at += width;
    }
    out
}

/// End offset of the raw string literal starting at `at`, if one starts there.
fn raw_string_end(bytes: &[u8], at: usize) -> Option<usize> {
    if bytes[at] != b'r' {
        return None;
    }
    let mut hashes = 0;
    let mut cursor = at + 1;
    while bytes.get(cursor) == Some(&b'#') {
        hashes += 1;
        cursor += 1;
    }
    if bytes.get(cursor) != Some(&b'"') {
        return None;
    }
    cursor += 1;
    let mut closing = Vec::with_capacity(hashes + 1);
    closing.push(b'"');
    closing.extend(std::iter::repeat_n(b'#', hashes));
    while cursor < bytes.len() {
        if bytes[cursor..].starts_with(&closing) {
            return Some(cursor + closing.len());
        }
        cursor += 1;
    }
    Some(bytes.len())
}

/// Byte length of the UTF-8 sequence beginning with `lead`.
fn char_width(lead: u8) -> usize {
    match lead {
        0x00..=0x7F => 1,
        0xC0..=0xDF => 2,
        0xE0..=0xEF => 3,
        _ => 4,
    }
}

/// `path` with `.` and `..` components folded away, without touching the disk.
///
/// A `#[path]` value is resolved lexically because the rule is about which
/// package directory contains the target, and a symlink-following canonicalize
/// would answer for a different tree.
fn normalize(path: &Path) -> PathBuf {
    let mut folded = PathBuf::new();
    for component in path.components() {
        match component {
            std::path::Component::CurDir => {}
            std::path::Component::ParentDir => {
                folded.pop();
            }
            other => folded.push(other),
        }
    }
    folded
}

/// `path` with the workspace root prefix dropped, for a readable message.
fn relative<'a>(root: &Path, path: &'a Path) -> &'a Path {
    path.strip_prefix(root).unwrap_or(path)
}
