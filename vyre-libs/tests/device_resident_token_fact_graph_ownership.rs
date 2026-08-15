//! One owner, and one vocabulary, for the device-resident token/fact graph.
//!
//! WHY: `vyre-driver-cuda` carried a second copy of this graph's device layout.
//! It restated the resident byte envelope and the CSR out-degree profile the
//! composition crate already computes, and in exchange `vyre-libs` had started
//! naming that one backend in its own error text. Both halves are the same
//! defect: a layout with two owners drifts, and the drift is invisible because
//! each half reads correct on its own.
//!
//! Two contracts are pinned here.
//!
//! 1. **Neutrality.** No file under `vyre-libs/src/device/` may name a concrete
//!    backend. The banned vocabulary is derived from the workspace roster at run
//!    time, so adding `vyre-driver-<target>` or `vyre-emit-<dialect>` extends it
//!    without anyone remembering to. A new backend whose name leaks into the
//!    device module turns this red on the commit that adds it.
//!
//! 2. **Ownership.** The resident layout, the degree-profile ranks and the
//!    profile bucket count are declared exactly once in the workspace, in
//!    `vyre-libs`. Every other crate may read them; none may declare them. The
//!    crate list is the workspace roster, so a new driver crate that reintroduces
//!    the restatement is caught without being named here.
//!
//! What this does NOT catch: a backend that copies the arithmetic under
//! different identifiers. Nothing short of reading the diff catches that, and
//! `xtask dup-scan` is the measure that makes it expensive.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

use vyre_test_support::monorepo::{vyre_crate_directory, vyre_workspace_root};

/// Directory that owns every device-residency contract in the composition crate.
const DEVICE_MODULE: &str = "src/device";

/// Where the token/fact graph layout is declared, relative to `vyre-libs`.
const OWNER_FILE: &str = "src/device/device_resident_token_fact_graph.rs";

/// The crate that owns the layout. Everything else may only read it.
const OWNER_CRATE: &str = "vyre-libs";

/// Declarations that constitute owning the resident layout. A second crate that
/// declares any of these has restated the layout rather than consumed it.
const OWNED_DECLARATIONS: [&str; 4] = [
    "struct DeviceResidentTokenFactGraphLayout",
    "const TOKEN_FACT_DEGREE_PROFILE_RANKS",
    "const TOKEN_FACT_DEGREE_PROFILE_BUCKETS",
    "fn csr_out_degree_profile",
];

/// Vocabulary a shared crate may never use, independent of the roster. These
/// are shading languages and vendor tooling rather than workspace members, so
/// no manifest would ever surface them.
const ALWAYS_BANNED: [&str; 8] = [
    "cuda", "nvidia", "wgsl", "hlsl", "glsl", "nvrtc", "cudarc", "opencl",
];

#[test]
fn device_module_names_no_concrete_backend() {
    let banned = banned_vocabulary();
    assert!(
        banned.contains("cuda") && banned.contains("wgpu"),
        "Fix: the banned vocabulary must be derived, not empty: {banned:?}"
    );

    let module = vyre_crate_directory(OWNER_CRATE).join(DEVICE_MODULE);
    let sources = rust_sources(&module);
    assert!(
        sources.len() >= 4,
        "Fix: expected the device module at {} to hold its contracts; found {} file(s). A moved \
         module must move this gate with it.",
        module.display(),
        sources.len()
    );

    let mut leaks = Vec::new();
    for source in &sources {
        let text = std::fs::read_to_string(source)
            .unwrap_or_else(|error| panic!("Fix: cannot read {}: {error}", source.display()));
        let lower = text.to_ascii_lowercase();
        for (number, line) in lower.lines().enumerate() {
            for term in &banned {
                if contains_word(line, term) {
                    leaks.push(format!("{}:{}: {term}", source.display(), number + 1));
                }
            }
        }
    }

    assert!(
        leaks.is_empty(),
        "Fix: {OWNER_CRATE} is a shared crate and must stay backend-neutral. Use primary text, \
         primary binary, secondary text, native module, backend, target, device or artifact, and \
         keep the concrete name inside the owning driver crate. Leaks:\n{}",
        leaks.join("\n")
    );
}

#[test]
fn the_resident_layout_is_declared_in_exactly_one_crate() {
    let root = vyre_workspace_root();
    let owner = vyre_crate_directory(OWNER_CRATE).join(OWNER_FILE);
    let owner_text = std::fs::read_to_string(&owner)
        .unwrap_or_else(|error| panic!("Fix: cannot read {}: {error}", owner.display()));
    for declaration in OWNED_DECLARATIONS {
        assert!(
            owner_text.contains(declaration),
            "Fix: `{declaration}` must be declared in {}. If the owner moved, move this gate.",
            owner.display()
        );
    }

    let mut restatements = Vec::new();
    for member in structure_gate::workspace_members(&root) {
        let directory = root.join(&member);
        if directory == vyre_crate_directory(OWNER_CRATE) {
            continue;
        }
        for source in rust_sources(&directory.join("src")) {
            let text = match std::fs::read_to_string(&source) {
                Ok(text) => text,
                Err(_) => continue,
            };
            for declaration in OWNED_DECLARATIONS {
                if text.contains(declaration) {
                    restatements.push(format!("{}: {declaration}", source.display()));
                }
            }
        }
    }

    assert!(
        restatements.is_empty(),
        "Fix: the device-resident token/fact graph layout has one owner, {}. A backend crate \
         converts it into its own scheduler types and issues its own device calls; it does not \
         redeclare the envelope or the out-degree profile. Restatements:\n{}",
        owner.display(),
        restatements.join("\n")
    );
}

/// Concrete backend names, derived from the workspace roster so a new driver or
/// emitter crate extends the ban without an edit here.
fn banned_vocabulary() -> BTreeSet<String> {
    let root = vyre_workspace_root();
    let mut banned: BTreeSet<String> = ALWAYS_BANNED.iter().map(|term| (*term).to_string()).collect();
    for member in structure_gate::workspace_members(&root) {
        let name = member.rsplit('/').next().unwrap_or(&member);
        for prefix in ["vyre-driver-", "vyre-emit-"] {
            if let Some(target) = name.strip_prefix(prefix) {
                if !target.is_empty() && target != "reference" {
                    banned.insert(target.to_ascii_lowercase());
                }
            }
        }
    }
    banned
}

/// Every `.rs` file under `directory`, recursively. An unreadable directory
/// yields nothing so a crate without `src` is simply skipped.
fn rust_sources(directory: &Path) -> Vec<PathBuf> {
    let mut sources = Vec::new();
    let Ok(entries) = std::fs::read_dir(directory) else {
        return sources;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            sources.extend(rust_sources(&path));
        } else if path.extension().is_some_and(|extension| extension == "rs") {
            sources.push(path);
        }
    }
    sources.sort();
    sources
}

/// Whether `line` contains `term` as a whole word.
///
/// Substring matching would flag `metal` inside `metallic` and, worse, would
/// never flag `cuda` written as `CUDA_` because the surrounding text differs.
/// Word boundaries are the identifier boundaries that matter in source.
fn contains_word(line: &str, term: &str) -> bool {
    let bytes = line.as_bytes();
    let needle = term.as_bytes();
    let mut start = 0;
    while let Some(offset) = line[start..].find(term) {
        let begin = start + offset;
        let end = begin + needle.len();
        let before_ok = begin == 0 || !is_word_byte(bytes[begin - 1]);
        let after_ok = end == bytes.len() || !is_word_byte(bytes[end]);
        if before_ok && after_ok {
            return true;
        }
        start = begin + 1;
    }
    false
}

fn is_word_byte(byte: u8) -> bool {
    byte.is_ascii_alphanumeric() || byte == b'_'
}
