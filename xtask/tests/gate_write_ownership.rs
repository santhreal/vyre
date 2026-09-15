//! No gate command drives another gate's writer.
//!
//! Every recorded artifact has one gate that owns it, and the ownership guard
//! in `artifact_gate` reports a gate that writes a file outside its declared
//! set. That guard fires only while the offending gate is running with
//! `--write`, which on the release path means after a full measurement on a
//! device. A gate that shells out to `xtask <other-gate> --write` writes the
//! other gate's artifacts under its own name, and the whole run is discarded
//! for a defect that is visible in the source.
//!
//! What this does not catch: a gate that writes another gate's artifact
//! through a shared writer function rather than a child process, and a child
//! process assembled from a name that is not a literal in the source. The
//! ownership guard remains the authority on what was actually written.

use std::path::Path;

use super::workspace_sources::{sources_under, workspace_member_src_dirs, workspace_root};

/// Characters allowed between the two literals for them to be one argument
/// list: separators and layout, nothing that could be another argument.
fn is_argument_separator(byte: u8) -> bool {
    byte.is_ascii_whitespace() || byte == b',' || byte == b'"'
}

/// How far back the runner's own name may sit from the gate name.
///
/// Wide enough to clear the cargo preamble a child command carries
/// (`run`, `--release`, `--bin`, `--quiet`, `--`), narrow enough that the name
/// has to belong to the same argument list.
const RUNNER_LOOKBEHIND_BYTES: usize = 200;

/// Whether `text` spawns `gate` with `--write` as an adjacent argument.
///
/// The shape this rejects is one argument list that names the runner binary,
/// then the gate, then `--write`. Both halves are required. Adjacency of the
/// name and the flag keeps help text that mentions each separately from
/// reading as an invocation; the runner name keeps a declarative table, such
/// as the release evidence census, from reading as one, because a census
/// records the command a maintainer runs and spawns nothing.
fn drives_writer(text: &str, gate: &str) -> bool {
    let needle = format!("\"{gate}\"");
    let bytes = text.as_bytes();
    let mut from = 0usize;
    while let Some(offset) = text[from..].find(&needle) {
        let name_at = from + offset;
        let mut cursor = name_at + needle.len();
        while cursor < bytes.len() && is_argument_separator(bytes[cursor]) {
            cursor += 1;
        }
        let flag_follows = text[cursor.min(text.len())..].starts_with("--write\"");
        let window = &text[name_at.saturating_sub(RUNNER_LOOKBEHIND_BYTES)..name_at];
        if flag_follows && window.contains("\"xtask\"") {
            return true;
        }
        from = name_at + needle.len();
    }
    false
}

/// Every production source file in the workspace, paired with its text.
fn workspace_sources(root: &Path) -> Vec<(String, String)> {
    let mut sources = Vec::new();
    for directory in workspace_member_src_dirs(root) {
        for path in sources_under(&directory, &["rs"]) {
            let Ok(text) = std::fs::read_to_string(&path) else {
                continue;
            };
            let name = path
                .strip_prefix(root)
                .unwrap_or(&path)
                .display()
                .to_string();
            sources.push((name, text));
        }
    }
    sources
}

/// WHY: closes the class where one gate records another gate's artifacts by
/// running its writer as a child process. The variant space is every
/// registered gate name, read from the descriptor table at run time, so a gate
/// added tomorrow is covered without editing this test.
#[test]
fn no_gate_command_runs_another_gates_writer() {
    let root = workspace_root();
    let sources = workspace_sources(&root);
    assert!(
        sources.len() > 1000,
        "Fix: the workspace scan found {} source files, so the contract judged almost nothing.",
        sources.len()
    );
    let mut invocations = Vec::new();
    for (path, text) in &sources {
        for descriptor in xtask::gate_metadata::GATE_METADATA {
            if drives_writer(text, descriptor.name) {
                invocations.push(format!("{path} runs `{}` with --write", descriptor.name));
            }
        }
    }
    assert!(
        invocations.is_empty(),
        "Fix: a gate must write only the artifacts its own descriptor declares. Delete the \
         nested invocation and let the owning gate run in the subset: {}",
        invocations.join(", ")
    );
}

/// WHY: a scan that cannot recognise the shape it forbids passes on a tree
/// that still contains it. This pins the recogniser against the exact argument
/// list that was removed from the release benchmark runner, and against the
/// three shapes that are not invocations.
#[test]
fn the_recogniser_reads_an_argument_list_and_not_a_mention() {
    let invocation = "&[\"run\", \"--bin\", \"xtask\", \"--\", \"optimization-matrix\", \
                      \"--write\",]";
    assert!(
        drives_writer(invocation, "optimization-matrix"),
        "Fix: the argument list that the ownership guard caught must read as an invocation."
    );
    let prose = "Regenerate the matrix by running `optimization-matrix` yourself; the writer \
                 behind it takes --write.";
    assert!(
        !drives_writer(prose, "optimization-matrix"),
        "Fix: help text naming a gate and the flag separately is not an invocation."
    );
    let other_gate = "&[\"xtask\", \"optimization-matrix\", \"--report\", \"--write\"]";
    assert!(
        !drives_writer(other_gate, "optimization-matrix"),
        "Fix: an argument between the name and the flag means the flag belongs to that argument."
    );
    let census = "EvidenceCommand::required(&[\"lego-duplicate-report\", \"--write\"]),";
    assert!(
        !drives_writer(census, "lego-duplicate-report"),
        "Fix: a table recording which command produces an artifact spawns nothing."
    );
}
