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
//! from source the same way, and both enumerations carry a floor, because the
//! failure mode of a walk is finding nothing and an empty set satisfies every
//! per-item assertion.
//!
//! # What it does not catch
//!
//! Resolution assumes the declaration sits at the top level of its file, which
//! is the only form this workspace uses. Rust resolves a `#[path]` inside an
//! inline `mod outer { … }` block against `outer/` instead, so a nested
//! declaration is reported against the wrong base directory; a target that then
//! names no file is reported as a finding rather than passing quietly.
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
use std::path::{Path, PathBuf};

use vyre_test_support::monorepo::vyre_workspace_root;

/// Directory names the walk never enters.
///
/// Build output and vendored trees hold generated Rust that no package owns,
/// and `.internal` is not a published tree.
const PRUNED: &[&str] = &["target", ".git", "node_modules", ".internal"];

/// Minimum packages the manifest discovery must find.
///
/// The workspace declared 34 members when this floor was set. A discovery that
/// returns a handful would make almost every file unowned and every escape
/// unclassifiable, so it fails here instead.
const PACKAGE_FLOOR: usize = 30;

/// Minimum `#[path]` module declarations the source scan must find.
///
/// The workspace carried 199 when this floor was set, every one of them
/// in-package. A scan that returns nothing proves nothing.
const DECLARATION_FLOOR: usize = 100;

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
    let packages = package_roots(&root);
    assert!(
        packages.len() >= PACKAGE_FLOOR,
        "Fix: only {} Cargo package(s) were discovered under {}; at least {PACKAGE_FLOOR} are expected, so the manifest walk is broken and no escape could be classified.",
        packages.len(),
        root.display()
    );

    let declarations = path_declarations(&root);
    assert!(
        declarations.len() >= DECLARATION_FLOOR,
        "Fix: only {} `#[path]` module declaration(s) were found under {}; at least {DECLARATION_FLOOR} are expected, so the source scan is broken and every escape would pass.",
        declarations.len(),
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
        members.len() >= PACKAGE_FLOOR,
        "Fix: the root manifest at {} lists only {} workspace member(s); at least {PACKAGE_FLOOR} are expected, so the members list was misparsed and the owner set it pins is meaningless.",
        root.join("Cargo.toml").display(),
        members.len()
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

/// Every directory under `root` whose `Cargo.toml` declares a package, mapped to
/// that manifest's path.
///
/// Discovery is per-manifest rather than per-member so that a package excluded
/// from the workspace, such as an example crate, still owns its own sources.
fn package_roots(root: &Path) -> BTreeMap<PathBuf, PathBuf> {
    let mut roots = BTreeMap::new();
    let mut pending = vec![root.to_path_buf()];
    while let Some(dir) = pending.pop() {
        let manifest = dir.join("Cargo.toml");
        if manifest.is_file() {
            let text = std::fs::read_to_string(&manifest)
                .unwrap_or_else(|error| panic!("Fix: read {}: {error}", manifest.display()));
            if declares_a_package(&text) {
                roots.insert(dir.clone(), manifest);
            }
        }
        for entry in read_dir(&dir) {
            if entry.is_dir() {
                pending.push(entry);
            }
        }
    }
    roots
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

/// Every `#[path]` module declaration under `root`, resolved against the
/// declaring file's directory.
fn path_declarations(root: &Path) -> Vec<Declaration> {
    let mut declarations = Vec::new();
    let mut pending = vec![root.to_path_buf()];
    while let Some(dir) = pending.pop() {
        for entry in read_dir(&dir) {
            if entry.is_dir() {
                pending.push(entry);
                continue;
            }
            if entry.extension().and_then(|ext| ext.to_str()) != Some("rs") {
                continue;
            }
            let text = std::fs::read_to_string(&entry)
                .unwrap_or_else(|error| panic!("Fix: read {}: {error}", entry.display()));
            let masked = mask_comments(&text);
            for (offset, value) in path_attributes(&masked) {
                declarations.push(Declaration {
                    file: relative(root, &entry).to_path_buf(),
                    line: masked[..offset].matches('\n').count() + 1,
                    target: normalize(&dir.join(&value)),
                    value,
                });
            }
        }
    }
    declarations
}

/// Byte offset and value of every `#[path = "…"]` attribute in `masked` that a
/// `mod` declaration follows.
///
/// An attribute on anything else changes no module resolution, and the doc
/// comments of the xtask gates that inspect this attribute quote it verbatim, so
/// the following item is what separates a declaration from a mention.
fn path_attributes(masked: &str) -> Vec<(usize, String)> {
    let mut found = Vec::new();
    let bytes = masked.as_bytes();
    let mut at = 0;
    while let Some(hit) = masked[at..].find("#[") {
        let start = at + hit;
        at = start + 2;
        let Some(close) = masked[at..].find(']') else {
            break;
        };
        let inner = masked[at..at + close].trim();
        at += close + 1;
        let Some(literal) = inner.strip_prefix("path") else {
            continue;
        };
        let literal = literal.trim_start();
        let Some(literal) = literal.strip_prefix('=') else {
            continue;
        };
        let literal = literal.trim();
        let Some(value) = literal
            .strip_prefix('"')
            .and_then(|rest| rest.strip_suffix('"'))
        else {
            continue;
        };
        if value.contains('\\') || value.is_empty() {
            continue;
        }
        if declares_a_module(bytes, at) {
            found.push((start, value.to_string()));
        }
    }
    found
}

/// True when the item beginning at `at` is a module declaration, skipping any
/// further attributes and a visibility qualifier.
fn declares_a_module(bytes: &[u8], mut at: usize) -> bool {
    loop {
        while at < bytes.len() && bytes[at].is_ascii_whitespace() {
            at += 1;
        }
        if bytes[at..].starts_with(b"#[") {
            let Some(close) = bytes[at..].iter().position(|byte| *byte == b']') else {
                return false;
            };
            at += close + 1;
            continue;
        }
        if bytes[at..].starts_with(b"pub") {
            at += 3;
            while at < bytes.len() && bytes[at].is_ascii_whitespace() {
                at += 1;
            }
            if bytes[at..].starts_with(b"(") {
                let Some(close) = bytes[at..].iter().position(|byte| *byte == b')') else {
                    return false;
                };
                at += close + 1;
            }
            continue;
        }
        return bytes[at..].starts_with(b"mod")
            && bytes
                .get(at + 3)
                .is_some_and(|byte| byte.is_ascii_whitespace());
    }
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

/// Entries of `dir`, or nothing when the directory is pruned or unreadable.
fn read_dir(dir: &Path) -> Vec<PathBuf> {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return Vec::new();
    };
    entries
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .filter(|path| {
            path.file_name()
                .and_then(|name| name.to_str())
                .is_some_and(|name| !PRUNED.contains(&name))
        })
        .collect()
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
