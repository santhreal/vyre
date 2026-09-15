//! Contract tests for physical kernel IR and legalization isolation.
//!
//! Verifies:
//! 1. No physical kernel IR type in `vyre-lower::descriptor` names a type the schedule owners
//!    declare. Lowering realizes a selected schedule, so a descriptor that names one of the
//!    selector's types has taken a second copy of the decision.
//! 2. No backend name appears as a string literal anywhere in `vyre-lower/src`. Hardware reaches
//!    lowering as a fact vector; a name in a branch is a per-target path nothing authenticated.
//! 3. Lowering from selected schedules is deterministic and byte-identical on repeat runs.
//!
//! Both sets are derived at run time. The semantic types are the `pub struct`, `pub enum`, and
//! `pub trait` declarations under `vyre-foundation/src/execution_plan/` and `vyre-megakernel/src/`,
//! and the backend names are the suffixes of the workspace's `vyre-driver-*` and `vyre-emit-*`
//! members. A type or a backend added to either is judged without a roster in this file going
//! stale, which is the failure a pinned list of sixteen names cannot report.
//!
//! Comments and string literals are removed before rule 1 matches, so prose naming a selector
//! type is not a finding while a code reference is. Rule 2 judges the literals the same scan
//! separates out. The scanner reads Rust's own lexical forms - line and block comments, strings,
//! raw strings, char literals, and lifetimes - and a construct it cannot classify is kept as code,
//! so an unknown form is reported rather than skipped.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

use vyre_lower::{verify_descriptor, KernelOpKind};

/// Workspace root, whether the test runs from the workspace or from the crate directory.
fn workspace_root() -> PathBuf {
    let here = Path::new(".");
    if here.join("docs/CRATE_OWNERSHIP.toml").exists() {
        return here.to_path_buf();
    }
    let parent = Path::new("..");
    assert!(
        parent.join("docs/CRATE_OWNERSHIP.toml").exists(),
        "cannot locate the workspace root from the test working directory"
    );
    parent.to_path_buf()
}

/// Every `.rs` file under `dir`, recursively.
fn rust_sources(dir: &Path) -> Vec<PathBuf> {
    let mut found = Vec::new();
    let mut pending = vec![dir.to_path_buf()];
    while let Some(next) = pending.pop() {
        let entries = std::fs::read_dir(&next)
            .unwrap_or_else(|error| panic!("cannot read {}: {error}", next.display()));
        for entry in entries {
            let path = entry.expect("valid dir entry").path();
            if path.is_dir() {
                pending.push(path);
            } else if path.extension().and_then(|ext| ext.to_str()) == Some("rs") {
                found.push(path);
            }
        }
    }
    found.sort();
    found
}

/// Split Rust source into code with comments and literals blanked, and the string literals.
fn split_code_and_literals(source: &str) -> (String, Vec<String>) {
    let bytes: Vec<char> = source.chars().collect();
    let mut code = String::with_capacity(source.len());
    let mut literals = Vec::new();
    let mut index = 0usize;

    while index < bytes.len() {
        let current = bytes[index];
        let next = bytes.get(index + 1).copied();

        if current == '/' && next == Some('/') {
            while index < bytes.len() && bytes[index] != '\n' {
                index += 1;
            }
            continue;
        }
        if current == '/' && next == Some('*') {
            index += 2;
            let mut depth = 1usize;
            while index < bytes.len() && depth > 0 {
                if bytes[index] == '/' && bytes.get(index + 1) == Some(&'*') {
                    depth += 1;
                    index += 2;
                } else if bytes[index] == '*' && bytes.get(index + 1) == Some(&'/') {
                    depth -= 1;
                    index += 2;
                } else {
                    index += 1;
                }
            }
            code.push(' ');
            continue;
        }
        if current == 'r' && (next == Some('"') || next == Some('#')) {
            let mut hashes = 0usize;
            let mut probe = index + 1;
            while bytes.get(probe) == Some(&'#') {
                hashes += 1;
                probe += 1;
            }
            if bytes.get(probe) == Some(&'"') {
                probe += 1;
                let mut literal = String::new();
                let mut closing = vec!['"'];
                closing.resize(hashes + 1, '#');
                while probe < bytes.len() && !bytes[probe..].starts_with(closing.as_slice()) {
                    literal.push(bytes[probe]);
                    probe += 1;
                }
                literals.push(literal);
                index = (probe + closing.len()).min(bytes.len());
                code.push(' ');
                continue;
            }
        }
        if current == '"' {
            index += 1;
            let mut literal = String::new();
            while index < bytes.len() && bytes[index] != '"' {
                if bytes[index] == '\\' {
                    index += 1;
                }
                if index < bytes.len() {
                    literal.push(bytes[index]);
                    index += 1;
                }
            }
            index = (index + 1).min(bytes.len());
            literals.push(literal);
            code.push(' ');
            continue;
        }
        if current == '\'' {
            // A char literal closes within three characters; anything else is a lifetime.
            let escaped = next == Some('\\');
            let closes_at = if escaped {
                bytes[index + 1..]
                    .iter()
                    .position(|&character| character == '\'')
                    .map(|offset| index + 1 + offset)
            } else if bytes.get(index + 2) == Some(&'\'') {
                Some(index + 2)
            } else {
                None
            };
            if let Some(end) = closes_at {
                index = end + 1;
                code.push(' ');
                continue;
            }
        }
        code.push(current);
        index += 1;
    }
    (code, literals)
}

/// The public nominal types declared under `dir`, mapped to the file declaring each.
fn declared_public_types(dir: &Path) -> BTreeMap<String, PathBuf> {
    let mut declared = BTreeMap::new();
    for path in rust_sources(dir) {
        let source = std::fs::read_to_string(&path)
            .unwrap_or_else(|error| panic!("cannot read {}: {error}", path.display()));
        let (code, _) = split_code_and_literals(&source);
        for line in code.lines() {
            let trimmed = line.trim();
            let Some(rest) = ["pub struct ", "pub enum ", "pub trait "]
                .into_iter()
                .find_map(|keyword| trimmed.strip_prefix(keyword))
            else {
                continue;
            };
            let name: String = rest
                .chars()
                .take_while(|character| character.is_alphanumeric() || *character == '_')
                .collect();
            if name.starts_with(char::is_uppercase) {
                declared.entry(name).or_insert_with(|| path.clone());
            }
        }
    }
    declared
}

/// Identifiers appearing in code, with comments and literals already removed.
fn code_identifiers(code: &str) -> BTreeSet<String> {
    code.split(|character: char| !character.is_alphanumeric() && character != '_')
        .filter(|word| !word.is_empty())
        .map(str::to_string)
        .collect()
}

/// The backend names the workspace publishes, derived from its driver and emitter members.
fn backend_names(root: &Path) -> BTreeSet<String> {
    let mut names = BTreeSet::new();
    let entries = std::fs::read_dir(root).expect("workspace root is readable");
    for entry in entries {
        let path = entry.expect("valid dir entry").path();
        if !path.is_dir() {
            continue;
        }
        let Some(member) = path.file_name().and_then(|name| name.to_str()) else {
            continue;
        };
        for prefix in ["vyre-driver-", "vyre-emit-"] {
            if let Some(suffix) = member.strip_prefix(prefix) {
                names.insert(suffix.to_string());
            }
        }
    }
    assert!(
        names.len() >= 5,
        "derived {} backend name(s) from {}, which cannot be the whole backend set",
        names.len(),
        root.display()
    );
    names
}

#[test]
fn no_physical_kernel_ir_type_names_a_schedule_owner_type() {
    let root = workspace_root();
    let mut semantic = declared_public_types(&root.join("vyre-foundation/src/execution_plan"));
    semantic.extend(declared_public_types(&root.join("vyre-megakernel/src")));
    assert!(
        semantic.len() >= 50,
        "derived only {} schedule-owner type(s), which cannot be the whole set",
        semantic.len()
    );

    let descriptor_dir = root.join("vyre-lower/src/descriptor");
    let sources = rust_sources(&descriptor_dir);
    assert!(
        sources.len() >= 5,
        "expected at least 5 descriptor source files, found {}",
        sources.len()
    );

    let mut findings = Vec::new();
    for path in sources {
        let source = std::fs::read_to_string(&path)
            .unwrap_or_else(|error| panic!("cannot read {}: {error}", path.display()));
        let (code, _) = split_code_and_literals(&source);
        for identifier in code_identifiers(&code) {
            if let Some(declared_in) = semantic.get(&identifier) {
                findings.push(format!(
                    "{}: names `{identifier}`, declared by the schedule owner at {}",
                    path.display(),
                    declared_in.display()
                ));
            }
        }
    }

    assert!(
        findings.is_empty(),
        "physical kernel IR realizes a selected schedule and states no selector type:\n{}",
        findings.join("\n")
    );
}

#[test]
fn no_backend_name_appears_as_a_literal_in_lowering() {
    let root = workspace_root();
    let names = backend_names(&root);
    let mut findings = Vec::new();

    for path in rust_sources(&root.join("vyre-lower/src")) {
        let source = std::fs::read_to_string(&path)
            .unwrap_or_else(|error| panic!("cannot read {}: {error}", path.display()));
        let (_, literals) = split_code_and_literals(&source);
        for literal in literals {
            if names.contains(&literal.to_ascii_lowercase()) {
                findings.push(format!("{}: names backend `{literal}`", path.display()));
            }
        }
    }

    assert!(
        findings.is_empty(),
        "hardware reaches lowering as a fact vector, never as a backend name:\n{}",
        findings.join("\n")
    );
}

#[test]
fn repeat_lowering_and_verification_produces_byte_identical_descriptors() {
    use vyre_lower::descriptor_builder::{body, descriptor, effect, global_rw, lit, op};
    use vyre_lower::LiteralValue;

    let desc = descriptor("multi_entry_phase_0")
        .slot(global_rw(
            0,
            vyre_foundation::ir::DataType::U32,
            "input_matrix_a",
        ))
        .slot(global_rw(
            1,
            vyre_foundation::ir::DataType::U32,
            "output_matrix_b",
        ))
        .dispatch(128, 1, 1)
        .body(
            body()
                .literals([LiteralValue::U32(0), LiteralValue::U32(42)])
                .op(lit(0, 0))
                .op(lit(1, 1))
                .op(op(
                    KernelOpKind::BinOpKind(vyre_foundation::ir::BinOp::Add),
                    [0, 1],
                    2,
                ))
                .op(effect(KernelOpKind::StoreGlobal, [1, 0, 2])),
        )
        .build();

    // Verify descriptor twice
    let verified_1 = verify_descriptor(&desc).expect("first verification succeeds");
    let verified_2 = verify_descriptor(&desc).expect("second verification succeeds");

    // Serialize both verified descriptors to JSON
    let json_1 = serde_json::to_vec(&verified_1).expect("serialization 1");
    let json_2 = serde_json::to_vec(&verified_2).expect("serialization 2");

    // Exact byte equality check
    assert_eq!(
        json_1, json_2,
        "repeat descriptor lowering and canonicalization must produce byte-identical results with no rediscovery"
    );
    assert_eq!(verified_1, verified_2);
}
