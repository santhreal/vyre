//! ONE-PLACE lock for the CSR neighbor-expansion edge-scan.
//!
//! The inner edge walk: "for a source node whose frontier bit is set, load its
//! `[edge_offsets[src], edge_offsets[src+1])` range, and for every edge passing the
//! kind-mask, atomic-OR the target bit into the frontier, marking the run changed on
//! a newly-set bit", was hand-written FIVE times across the graph module
//! (single-serial `body.rs`, grid-sync-per-lane `program_parallel.rs`, batched
//! `program_parallel_batch.rs`, batched-global `program_parallel_batch_global.rs`,
//! and the persistent-BFS batch step). Byte-identical copies DRIFT: the persistent_bfs
//! seed bug and the `{n,m}` lowering bugs both hid in near-duplicate paths.
//!
//! It is now owned in ONE place by `crate::graph::edge_scan`: `csr_edge_expand_nodes`
//! (the edge-walk alone, for snapshot callers) and `csr_edge_scan_nodes` (the inline
//! source-bit read wrapped around it). Every caller supplies only the two axes that
//! genuinely differ: how a bitset word maps into its frontier buffer, and what to write
//! when a new bit is discovered.
//!
//! This test fails if a SIXTH copy reappears: any file in the CSR-forward family
//! (`csr_forward_or_changed` + `persistent_bfs`) that loads edge-targets directly
//! instead of routing through the owner. The edge-target load. `NAME_EDGE_TARGETS`
//! (or its `"pg_edge_targets"` value), is the unforgeable signature of a hand-written
//! walk, since a caller of the shared builder never touches it. It also asserts the
//! owner defines both entry points and every migrated caller still references them.

use std::fs;
use std::path::{Path, PathBuf};

/// The edge-target load is the signature of a hand-written CSR edge walk: the shared
/// `edge_scan` builder is the only thing that should ever load `pg_edge_targets` within
/// the forward-expansion family, so any OTHER file here carrying the token is a copy.
const EDGE_TARGET_CONST: &str = "NAME_EDGE_TARGETS";
const EDGE_TARGET_LITERAL: &str = "\"pg_edge_targets\"";

/// The scope of the invariant: the CSR-forward-or-changed module and the persistent-BFS
/// driver. Every neighbor expansion in these two subsystems must go through the owner.
/// The owner `graph/edge_scan.rs` sits ABOVE this scope (a peer of both), so it is
/// naturally excluded, no allowlist needed. Sibling graph algorithms (motif, reachable,
/// dominator_tree, csr_backward, adaptive_traverse, tensor_flow) are DISTINCT edge walks,
/// not members of this family, and are deliberately out of scope.
const SCOPED_DIRS: &[&str] = &["graph/csr_forward_or_changed", "graph/persistent_bfs"];
const SCOPED_FILES: &[&str] = &["graph/csr_forward_or_changed.rs"];

/// Migrated callers and the owner symbol each MUST still reference (positive lock:
/// catches a future edit that re-inlines a walk while dropping the shared call, which
/// the negative scan alone would miss if the re-inline also avoided the const).
const MIGRATED: &[(&str, &str)] = &[
    (
        "graph/csr_forward_or_changed/body.rs",
        "csr_edge_scan_nodes",
    ),
    (
        "graph/csr_forward_or_changed/program_parallel_batch_global.rs",
        "csr_edge_scan_nodes",
    ),
    (
        "graph/csr_forward_or_changed/program_parallel.rs",
        "csr_edge_expand_nodes",
    ),
    ("graph/persistent_bfs/program.rs", "csr_edge_expand_nodes"),
    // The batch variant unifies one level up: it delegates the whole program to the
    // batch-global builder (which is what routes through the owner), so its lock is the
    // delegation, not a direct edge_scan reference.
    (
        "graph/csr_forward_or_changed/program_parallel_batch.rs",
        "csr_forward_or_changed_parallel_batch_global_indexed",
    ),
];

#[test]
fn csr_edge_scan_has_exactly_one_owner() {
    let src = Path::new(env!("CARGO_MANIFEST_DIR")).join("src");

    // 1. The owner exists and defines both entry points.
    let owner = fs::read_to_string(src.join("graph/edge_scan.rs"))
        .expect("src/graph/edge_scan.rs (the ONE-PLACE owner) must exist");
    assert!(
        owner.contains("fn csr_edge_expand_nodes"),
        "graph/edge_scan.rs must define the edge-walk-only builder `csr_edge_expand_nodes`"
    );
    assert!(
        owner.contains("fn csr_edge_scan_nodes"),
        "graph/edge_scan.rs must define the inline-read builder `csr_edge_scan_nodes`"
    );

    // 2. No hand-written edge walk anywhere in the CSR-forward family: no scoped file
    //    may load edge-targets directly. The owner is above the scope, so it is not scanned.
    let mut scoped = Vec::new();
    for dir in SCOPED_DIRS {
        collect_rust_files(&src.join(dir), &mut scoped);
    }
    for file in SCOPED_FILES {
        scoped.push(src.join(file));
    }
    assert!(
        !scoped.is_empty(),
        "scope resolved to zero files, the CSR-forward module layout moved; update SCOPED_*"
    );

    let mut offenders = Vec::new();
    for path in &scoped {
        let text = fs::read_to_string(path).expect("primitive source must be readable");
        if text.contains(EDGE_TARGET_CONST) || text.contains(EDGE_TARGET_LITERAL) {
            offenders.push(
                path.strip_prefix(&src)
                    .unwrap_or(path)
                    .display()
                    .to_string(),
            );
        }
    }
    assert!(
        offenders.is_empty(),
        "a hand-written CSR edge walk reappeared (loads pg_edge_targets directly), route the \
         neighbor expansion through crate::graph::edge_scan::{{csr_edge_expand_nodes, \
         csr_edge_scan_nodes}} instead. Offending files:\n{}",
        offenders.join("\n")
    );

    // 3. Every migrated caller still routes through the owner (or its delegation).
    for (rel, token) in MIGRATED {
        let text = fs::read_to_string(src.join(rel))
            .unwrap_or_else(|_| panic!("migrated file {rel} must exist"));
        assert!(
            text.contains(token),
            "{rel} was migrated onto the shared CSR edge-scan but no longer references `{token}` \
A re-inlined hand-written walk would silently reintroduce the duplication"
        );
    }
}

fn collect_rust_files(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(entries) = fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            collect_rust_files(&path, out);
        } else if path.extension().and_then(|e| e.to_str()) == Some("rs") {
            out.push(path);
        }
    }
}
