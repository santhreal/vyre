//! WHY: a device trap is only reported because the host reads the trap record
//! after the launch. That read lives in ONE place, `ModuleGlobalsLease`'s release,
//! and a dispatch path reaches it by taking a lease. A dispatch path that does not
//! is not a compile error and not a test failure on any program that does not trap:
//! it reports a trapped launch as successful and hands the caller whatever the
//! lanes wrote before the guard fired. That is the exact failure the trap sidecar
//! exists to remove, so a new dispatch path must not be able to reintroduce it
//! quietly.
//!
//! The class is "a launch whose trap nobody reads", so the scan is over every
//! dispatch path, not over the one FFI call. The crate funnels launches through
//! [`LAUNCH_HELPERS`]; a call to any of them from a function that is not itself a
//! helper IS a dispatch path, and each one needs a recorded decision. Both the
//! paths and the helper set are derived from the crate source at run time, so
//! adding a path, or renaming a helper so the scan stops matching, turns this RED.
//!
//! Three decisions are representable and each is a different reason a trap cannot
//! go unreported. `CrateAuthoredPtxDeclaresNoTrap` is the one that could rot
//! without anybody noticing, because it is a claim about generated text rather
//! than about control flow, so it is checked here against the text the builders
//! actually produce.
//!
//! What it does not catch: whether a `LeaseReadsTheRecord` row is true. A path
//! claiming lease coverage while calling the launch outside `launch_then_release`
//! passes here. That is what the lease's `self`-by-value consumption and its
//! private `release_after_launch` are for: the lease cannot be held without the
//! release running. It also does not check that the recorded lane and address are
//! the trapping ones.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

/// How one dispatch path accounts for the trap record.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum TrapAccounting {
    /// Runs its launches inside `ModuleGlobalsLease::launch_then_release`, whose
    /// release synchronizes the stream and reads the record.
    LeaseReadsTheRecord,
    /// Refuses a trap-declaring module before launching, because it cannot
    /// synchronize and so could never read the record back.
    RefusesTrapDeclaringModule,
    /// Launches PTX this crate writes itself, which declares no trap sidecar and
    /// therefore has no record to read. Checked below against the emitted text.
    CrateAuthoredPtxDeclaresNoTrap,
}

/// Every dispatch path and how it accounts for the trap record, keyed by source
/// file and enclosing function.
///
/// This is a ledger, not a description: a row is a claim someone made about a
/// launch path. A path missing from it fails this test, and a row naming a path
/// that no longer exists fails it too.
const RECORDED_DISPATCH_PATHS: &[(&str, &str, TrapAccounting)] = &[
    (
        "src/backend/cuda_graph.rs",
        "record_cuda_graph_borrowed",
        TrapAccounting::RefusesTrapDeclaringModule,
    ),
    (
        "src/backend/host_dispatch/mod.rs",
        "dispatch_borrowed_async_with_ptx_concrete",
        TrapAccounting::LeaseReadsTheRecord,
    ),
    (
        "src/backend/resident_dispatch/async_dispatch.rs",
        "dispatch_resident_async_concrete_with_ptx_key",
        TrapAccounting::LeaseReadsTheRecord,
    ),
    (
        "src/backend/resident_dispatch/batch.rs",
        "dispatch_resident_batch_async_concrete_with_ptx_key",
        TrapAccounting::LeaseReadsTheRecord,
    ),
    (
        "src/backend/resident_dispatch/sequence_fused.rs",
        "fill_upload_resident_many_repeated_sequence_read_ranges_borrowed_into",
        TrapAccounting::LeaseReadsTheRecord,
    ),
    (
        "src/egraph_kernel_plan/backend_rewrite.rs",
        "run_egraph_canonical_rewrite_kernel_inner",
        TrapAccounting::CrateAuthoredPtxDeclaresNoTrap,
    ),
    (
        "src/egraph_kernel_plan/backend_rewrite.rs",
        "run_egraph_signature_refresh_kernel_inner",
        TrapAccounting::CrateAuthoredPtxDeclaresNoTrap,
    ),
    (
        "src/egraph_kernel_plan/backend_structural.rs",
        "run_egraph_structural_equivalence_kernel_inner",
        TrapAccounting::CrateAuthoredPtxDeclaresNoTrap,
    ),
];

/// The launch funnel: the FFI boundary and every internal helper that forwards to
/// it. A call to one of these from a non-helper function is a dispatch path.
const LAUNCH_HELPERS: &[&str] = &[
    "launch_cuda_function",
    "launch_resolved_function",
    "launch_prevalidated_function",
    "replay_fixpoint_launches",
];

/// Any nonzero SM selects a PTX ISA version and leaves the body text unchanged, so
/// the sidecar check does not depend on which one. This is the host's.
const PTX_CHECK_TARGET_SM: u32 = 120;

fn crate_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

/// Rust source files under `src`, walked at run time so a new module is covered
/// without editing a list.
fn source_files(dir: &Path, out: &mut Vec<PathBuf>) {
    let entries = std::fs::read_dir(dir)
        .unwrap_or_else(|error| panic!("Fix: cannot read {}: {error}", dir.display()));
    for entry in entries {
        let entry =
            entry.unwrap_or_else(|error| panic!("Fix: cannot read {}: {error}", dir.display()));
        let path = entry.path();
        if path.is_dir() {
            source_files(&path, out);
        } else if path.extension().is_some_and(|ext| ext == "rs") {
            out.push(path);
        }
    }
}

/// Indentation of a source line.
fn indent_of(line: &str) -> usize {
    line.len() - line.trim_start().len()
}

/// The name of the nearest `fn` declared at or above `line_index`.
///
/// This is a scan, not a parser, because the question is only "which function body
/// is this call in", and every launch site in this crate sits directly in a
/// function body or in a closure inside one. A closure keeps the enclosing `fn`,
/// which is what the ledger names.
fn enclosing_function(lines: &[&str], line_index: usize) -> String {
    for line in lines[..=line_index].iter().rev() {
        let trimmed = line.trim_start();
        let rest = trimmed
            .strip_prefix("pub(crate) ")
            .or_else(|| trimmed.strip_prefix("pub(super) "))
            .or_else(|| trimmed.strip_prefix("pub "))
            .unwrap_or(trimmed);
        let rest = rest.strip_prefix("unsafe ").unwrap_or(rest);
        let Some(after) = rest.strip_prefix("fn ") else {
            continue;
        };
        let name: String = after
            .chars()
            .take_while(|c| c.is_alphanumeric() || *c == '_')
            .collect();
        if !name.is_empty() {
            return name;
        }
    }
    panic!(
        "Fix: no enclosing `fn` found above a launch call, so the scan cannot name the dispatch path."
    );
}

/// Line ranges covered by a `#[cfg(test)]` item.
///
/// A test region runs from the attribute to the first later line that is exactly a
/// closing brace at the attribute's own indentation, which in formatted source is
/// the item's closer and nothing else. Brace counting is deliberately avoided: the
/// crate's PTX templates are raw strings full of braces, and a counter that reads
/// them as code drifts by hundreds of lines.
///
/// This matters because a `#[cfg(test)]` module can sit ABOVE production code in
/// the same file. Treating "any earlier attribute" as "inside a test module" skips
/// every real launch below one, which reads as an empty scan rather than a failure.
fn test_regions(lines: &[&str]) -> Vec<(usize, usize)> {
    let mut regions = Vec::new();
    for (index, line) in lines.iter().enumerate() {
        if line.trim_start() != "#[cfg(test)]" {
            continue;
        }
        let indent = indent_of(line);
        let end = lines[index + 1..]
            .iter()
            .position(|candidate| {
                candidate.trim_end() == format!("{}}}", " ".repeat(indent))
            })
            .map_or(lines.len(), |offset| index + 1 + offset);
        regions.push((index, end));
    }
    regions
}

/// Whether `line_index` sits inside a `#[cfg(test)]` item.
///
/// Test launches deliberately pass null handles to exercise the pre-FFI guards.
/// They never reach a device, so they have no trap record to read and no decision
/// to record.
fn inside_test_module(regions: &[(usize, usize)], line_index: usize) -> bool {
    regions
        .iter()
        .any(|(start, end)| line_index > *start && line_index <= *end)
}

/// Dispatch paths found in the crate source, and the helper definitions the scan
/// relied on to find them.
fn scan_dispatch_paths() -> (BTreeSet<(String, String)>, BTreeSet<String>) {
    let root = crate_root();
    let mut files = Vec::new();
    source_files(&root.join("src"), &mut files);
    files.sort();
    assert!(
        !files.is_empty(),
        "Fix: no Rust sources found under {}. This test derives dispatch paths from source, so an empty walk means it is proving nothing.",
        root.join("src").display()
    );

    let mut paths = BTreeSet::new();
    let mut defined_helpers = BTreeSet::new();
    for file in &files {
        let text = std::fs::read_to_string(file)
            .unwrap_or_else(|error| panic!("Fix: cannot read {}: {error}", file.display()));
        let relative = file
            .strip_prefix(&root)
            .unwrap_or(file)
            .to_string_lossy()
            .replace('\\', "/");
        let lines: Vec<&str> = text.lines().collect();
        let regions = test_regions(&lines);
        for (index, line) in lines.iter().enumerate() {
            // A comment naming a helper is prose, not a launch.
            if line.trim_start().starts_with("//") {
                continue;
            }
            for helper in LAUNCH_HELPERS {
                if line.contains(&format!("fn {helper}(")) {
                    defined_helpers.insert((*helper).to_owned());
                }
                if !line.contains(&format!("{helper}(")) {
                    continue;
                }
                if inside_test_module(&regions, index) {
                    continue;
                }
                let function = enclosing_function(&lines, index);
                // A helper forwarding to another helper is not a decision point,
                // and neither is a helper's own definition line.
                if LAUNCH_HELPERS.contains(&function.as_str()) {
                    continue;
                }
                paths.insert((relative.clone(), function));
            }
        }
    }
    (paths, defined_helpers)
}

#[test]
fn every_cuda_dispatch_path_records_how_it_reports_a_device_trap() {
    let (found, defined_helpers) = scan_dispatch_paths();

    let missing_helpers: Vec<&&str> = LAUNCH_HELPERS
        .iter()
        .filter(|helper| !defined_helpers.contains(**helper))
        .collect();
    assert!(
        missing_helpers.is_empty(),
        "Fix: these launch helpers were never found as definitions: {missing_helpers:?}. The scan finds dispatch paths by their calls, so a renamed helper makes it match nothing and stop failing on a new path. Update LAUNCH_HELPERS to the current names."
    );

    let recorded_keys: BTreeSet<(String, String)> = RECORDED_DISPATCH_PATHS
        .iter()
        .map(|(file, function, _)| ((*file).to_owned(), (*function).to_owned()))
        .collect();
    assert_eq!(
        recorded_keys.len(),
        RECORDED_DISPATCH_PATHS.len(),
        "Fix: RECORDED_DISPATCH_PATHS names the same (file, function) twice, so one row's decision is silently discarded."
    );

    let unrecorded: Vec<&(String, String)> = found.difference(&recorded_keys).collect();
    assert!(
        unrecorded.is_empty(),
        "Fix: these CUDA dispatch paths have no recorded trap decision: {unrecorded:?}. A launch that neither runs under ModuleGlobalsLease::launch_then_release, nor refuses a trap-declaring module, nor launches crate-authored PTX without a trap, reports a trapped launch as successful. Give it one of those three properties, then add the path to RECORDED_DISPATCH_PATHS."
    );

    let stale: Vec<&(String, String)> = recorded_keys.difference(&found).collect();
    assert!(
        stale.is_empty(),
        "Fix: these recorded trap decisions name dispatch paths that no longer exist: {stale:?}. Delete the stale rows so the ledger keeps describing the code."
    );
}

/// WHY: `CrateAuthoredPtxDeclaresNoTrap` is the only decision in the ledger that
/// is a claim about generated text. Control-flow decisions break loudly when they
/// break; this one breaks by someone adding a trap to a hand-written PTX template,
/// after which the module declares a sidecar no host reads and a trapping launch
/// reports success. So the claim is checked against the text the builders produce.
#[test]
fn crate_authored_ptx_declares_no_trap_sidecar() {
    let claimed = RECORDED_DISPATCH_PATHS
        .iter()
        .any(|(_, _, accounting)| *accounting == TrapAccounting::CrateAuthoredPtxDeclaresNoTrap);
    assert!(
        claimed,
        "Fix: no dispatch path claims CrateAuthoredPtxDeclaresNoTrap any more, so this check has nothing to prove and must be deleted along with the decision."
    );

    let structural =
        vyre_driver_cuda::cuda_egraph_structural_equivalence_kernel_ptx(PTX_CHECK_TARGET_SM)
            .expect("Fix: the structural-equivalence kernel must emit PTX for a valid SM.");
    let rewrite = vyre_driver_cuda::cuda_egraph_canonical_rewrite_kernel_ptx(PTX_CHECK_TARGET_SM)
        .expect("Fix: the canonical-rewrite kernel must emit PTX for a valid SM.");
    let refresh = vyre_driver_cuda::cuda_egraph_signature_refresh_kernel_ptx(PTX_CHECK_TARGET_SM)
        .expect("Fix: the signature-refresh kernel must emit PTX for a valid SM.");

    for (name, source) in [
        ("structural equivalence", structural.source.as_str()),
        ("canonical rewrite", rewrite.source.as_str()),
        ("signature refresh", refresh.source.as_str()),
    ] {
        assert!(
            !source.contains(vyre_emit_ptx::TRAP_SIDECAR_SYMBOL),
            "Fix: the crate-authored {name} kernel now declares `{}`, but the paths that launch it take no ModuleGlobalsLease, so nothing reads the record and a trapping launch would report success. Either route that launch through a lease and change its ledger row to LeaseReadsTheRecord, or drop the trap from the template.",
            vyre_emit_ptx::TRAP_SIDECAR_SYMBOL
        );
    }
}
