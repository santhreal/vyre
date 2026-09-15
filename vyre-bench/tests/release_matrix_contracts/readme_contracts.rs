use super::*;

/// Performance documentation names the manifest and the generated evidence rather
/// than restating either.
///
/// WHY: this asserted the old performance document, which the mdbook deletion removed, so the test
/// failed on a missing file instead of judging a claim. The surviving surface is the
/// crate README plus the machine-readable contract data under docs/.
#[test]
fn benchmark_documentation_defers_to_the_manifest_and_generated_evidence() {
    let workspace = workspace_root();
    let readme = std::fs::read_to_string(workspace.join("vyre-bench/README.md"))
        .expect("Fix: vyre-bench/README.md must remain readable.");

    assert!(
        readme.contains("docs/optimization/BENCH_TARGETS.toml")
            && readme.contains("release/evidence/benchmarks/"),
        "Fix: benchmark documentation must name the benchmark-target manifest and the generated evidence directory as the authorities for thresholds and measurements."
    );
}

/// The release workload table names cases and owners; the threshold and the CPU
/// baseline belong to the manifest and the harness.
///
/// WHY: this table published `100× vs Hyperscan/ripgrep-class CPU bitmap
/// materialization` for a case the harness runs at a threshold the table did not
/// share, against a baseline that is an in-process scalar loop. Three owners for one
/// threshold, two of them wrong, and no test read the table. Rows are checked against
/// the registry and the macro roster at run time, so a stale id or a re-added
/// multiplier fails here.
///
/// Does not catch a workload family missing from the table entirely; the family
/// coverage contracts own that.
#[test]
fn release_workload_table_names_live_cases_and_restates_no_threshold() {
    let workspace = workspace_root();
    let readme = std::fs::read_to_string(workspace.join("vyre-bench/README.md"))
        .expect("Fix: vyre-bench/README.md must remain readable.");
    let table = readme
        .split("## Release Workloads")
        .nth(1)
        .and_then(|section| section.split("## ").next())
        .expect("Fix: vyre-bench/README.md must keep a Release Workloads section.");
    let registry = vyre_bench::registry::collect_all();
    let active = registry
        .iter()
        .filter(|case| case.active_in_suite(&SuiteKind::Release))
        .map(|case| case.id().0)
        .collect::<BTreeSet<_>>();
    let owner_by_id = registry
        .iter()
        .filter(|case| case.active_in_suite(&SuiteKind::Release))
        .map(|case| (case.id().0, case.metadata().owner_crate))
        .collect::<std::collections::BTreeMap<_, _>>();
    let mut rows = 0usize;
    let mut failures = Vec::new();
    for line in table.lines() {
        let cells = line
            .split('|')
            .map(str::trim)
            .filter(|cell| !cell.is_empty())
            .collect::<Vec<_>>();
        let [_family, case_cell, owner] = cells.as_slice() else {
            continue;
        };
        let Some(case_id) = case_cell
            .strip_prefix('`')
            .and_then(|id| id.strip_suffix('`'))
        else {
            continue;
        };
        rows += 1;
        if !active.contains(case_id) {
            failures.push(format!(
                "Fix: release workload table row names `{case_id}`, which is not an active release case."
            ));
            continue;
        }
        if let Some(registered) = owner_by_id.get(case_id) {
            if registered != owner {
                failures.push(format!(
                    "Fix: release workload table gives `{case_id}` owner `{owner}` where the case registers `{registered}`."
                ));
            }
        }
    }
    assert!(
        rows >= 12,
        "Fix: the release workload table parsed {rows} case row(s); the release suite must cover at least 12 families."
    );
    for line in table.lines() {
        let restates_threshold = line.split_whitespace().any(|word| {
            let trimmed = word.trim_end_matches(['×', 'x']);
            trimmed.len() < word.len()
                && !trimmed.is_empty()
                && trimmed.chars().all(|ch| ch.is_ascii_digit() || ch == '.')
        });
        if restates_threshold {
            failures.push(format!(
                "Fix: release workload documentation restates a speedup threshold in `{}`. The threshold lives in docs/optimization/BENCH_TARGETS.toml and is enforced by the harness contract.",
                line.trim()
            ));
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}
