//! Check 7: the composed fraction of an operation does not regress.
//!
//! The baseline is a committed table of one fraction per operation. A rewrite
//! that inlines work a registered child used to carry lowers the fraction
//! without changing any other measurement, so the fraction is pinned.

use super::*;

pub(super) const COMPOSITION_REGRESSION_EPSILON: f64 = 1.0e-9;

pub(super) fn composition_regressed(old_fraction: f64, new_fraction: f64) -> bool {
    new_fraction + COMPOSITION_REGRESSION_EPSILON < old_fraction
}

/// Operations whose `composed_fraction` stepped down on purpose, and whose
/// baseline therefore predates a deliberate shape change.
///
/// The trend check reads its baseline out of the previous release tag, which is
/// history and cannot be edited. When a codec that was a thin `vyre-libs`
/// wrapper around a registered `vyre-primitives` child collapses into one
/// module with one op id, the wrapper's child Region goes with it: the op now
/// emits its own IR instead of nesting a region that emitted it. Composition did
/// not regress, it stopped being counted, and there is no shape the fix line
/// asks for that would bring the number back without restoring the second
/// module.
///
/// Each row is held to the condition it suppresses: a row whose op no longer
/// regresses against the baseline is reported, so it cannot outlive the release
/// that earned it.
pub(super) const INTENDED_COMPOSITION_COLLAPSES: [&str; 10] = [
    "vyre-libs::decode::base64",
    "vyre-libs::decode::hex",
    "vyre-libs::decode::inflate_stored_block",
    "vyre-libs::hash::adler32",
    "vyre-libs::hash::crc32",
    "vyre-libs::hash::fnv1a32",
    "vyre-libs::hash::fnv1a64",
    "vyre-libs::hash::multi_hash",
    "vyre-libs::math::succinct::rank1_superblocks",
    "vyre-libs::parsing::core_delimiter_match",
];

/// Whether an operation has no own work left to hand to a child.
///
/// `composed_fraction` is `composed / (own + composed)`, and an operation that
/// composes everything below its own entry boundary still counts that boundary
/// as one own node. Its fraction is therefore `1 - 1/total`, which rises with
/// program size: shrinking such a program lowers the fraction while composing
/// exactly as much of it, and the fix line asks for a restoration that has
/// nothing to restore. A regression is only real when the operation retained
/// nodes a registered child used to carry, which needs own work above the
/// boundary.
pub(super) fn no_own_work_to_compose(own_nodes: usize) -> bool {
    own_nodes <= 1
}

pub(super) fn check_7_trend(report: &mut Report, ops: &[OpInfo]) -> usize {
    report.note("Composition trend (current composed_fraction must not regress from the latest available baseline)".to_string());
    let Some(root) = workspace_root() else {
        report.find(violation("  ✗ workspace root not reachable from xtask. Fix: run from the vyre workspace checkout.".to_string()));
        return 1;
    };
    let Some(tag) = previous_tag(&root) else {
        report.note("  ✓ no previous git tag found; trend check has no baseline".to_string());
        return 0;
    };
    let (previous, baseline_name) = if let Some(previous) =
        previous_composition_baseline(&root, &tag)
    {
        (previous, tag.clone())
    } else if let Some(current_baseline) = current_composition_baseline(&root) {
        report.note(format!("  • previous tag `{tag}` predates composition baselines; comparing against the checked-in bootstrap baseline"));
        (current_baseline, COMPOSITION_BASELINE_PATH.to_string())
    } else {
        report.find(violation(format!("  ✗ previous tag `{tag}` has no composition baseline and no bootstrap baseline is checked in. Fix: run `./cargo_full run --bin xtask -- lego-trend --write-baseline` and commit {COMPOSITION_BASELINE_PATH}.")));
        return 1;
    };

    let current = composition_fractions(ops);
    let own_work: BTreeMap<&str, usize> = ops
        .iter()
        .map(|op| (op.id.as_str(), op.own_nodes))
        .collect();
    let mut flagged = 0usize;
    let mut suppressed: Vec<&str> = Vec::new();
    for (op_id, old_fraction) in previous {
        let Some(new_fraction) = current.get(&op_id) else {
            continue;
        };
        if !composition_regressed(old_fraction, *new_fraction) {
            continue;
        }
        if let Some(row) = INTENDED_COMPOSITION_COLLAPSES
            .iter()
            .find(|row| **row == op_id)
        {
            suppressed.push(row);
            report.note(format!(
                "  • {op_id} composed_fraction stepped from {:.1}% to {:.1}% by design; the wrapper module and its child Region were collapsed into one op",
                old_fraction * 100.0,
                new_fraction * 100.0
            ));
            continue;
        }
        if own_work
            .get(op_id.as_str())
            .copied()
            .is_some_and(no_own_work_to_compose)
        {
            report.note(format!(
                "  • {op_id} composed_fraction moved from {:.1}% to {:.1}% with every node below its own entry boundary composed; the program got smaller, not less composed",
                old_fraction * 100.0,
                new_fraction * 100.0
            ));
            continue;
        }
        report.find(violation(format!("  ✗ {op_id} composed_fraction regressed from {:.1}% to {:.1}%. Fix: restore Region composition or extract shared work to Tier 2.5.",
            old_fraction * 100.0,
            new_fraction * 100.0)));
        flagged += 1;
    }
    for row in INTENDED_COMPOSITION_COLLAPSES {
        if !suppressed.contains(&row) {
            report.find(Finding::new(
                format!("the intended-collapse row `{row}` suppresses no composed_fraction regression against `{baseline_name}`"),
                DEAD_EXEMPTION_FIX,
            ));
            flagged += 1;
        }
    }
    if flagged == 0 {
        report.note(format!(
            "  ✓ no composed_fraction regressions against `{baseline_name}`"
        ));
    }
    flagged
}

pub(super) fn composition_fractions(ops: &[OpInfo]) -> BTreeMap<String, f64> {
    ops.iter()
        .map(|op| {
            let total = op.own_nodes + op.composed_nodes;
            let fraction = if total == 0 {
                1.0
            } else {
                op.composed_nodes as f64 / total as f64
            };
            (op.id.clone(), fraction)
        })
        .collect()
}

pub(super) use xtask::artifact_paths::LEGO_COMPOSITION_BASELINE as COMPOSITION_BASELINE_PATH;

pub(super) fn write_composition_baseline(root: &std::path::Path, ops: &[OpInfo]) -> io::Result<()> {
    let path = root.join(COMPOSITION_BASELINE_PATH);
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let mut rendered = String::from("# op_id\tcomposed_fraction\n");
    for (op_id, fraction) in composition_fractions(ops) {
        rendered.push_str(&format!("{op_id}\t{fraction:.12}\n"));
    }
    std::fs::write(&path, rendered)?;
    Ok(())
}

pub(super) fn current_composition_baseline(
    root: &std::path::Path,
) -> Option<BTreeMap<String, f64>> {
    let text = std::fs::read_to_string(root.join(COMPOSITION_BASELINE_PATH)).ok()?;
    parse_composition_baseline(&text)
}

pub(super) fn parse_composition_baseline(text: &str) -> Option<BTreeMap<String, f64>> {
    let mut out = BTreeMap::new();
    for line in text.lines().filter(|line| !line.starts_with('#')) {
        let mut cols = line.split('\t');
        let Some(op_id) = cols.next().filter(|op_id| !op_id.is_empty()) else {
            continue;
        };
        let Some(fraction) = cols.next().and_then(|raw| raw.parse::<f64>().ok()) else {
            continue;
        };
        if fraction.is_finite() && (0.0..=1.0).contains(&fraction) {
            out.insert(op_id.to_string(), fraction);
        }
    }
    (!out.is_empty()).then_some(out)
}

pub(super) fn previous_tag(root: &std::path::Path) -> Option<String> {
    let output = std::process::Command::new("git")
        .args(["describe", "--tags", "--abbrev=0", "HEAD^"])
        .current_dir(root)
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let tag = String::from_utf8(output.stdout).ok()?;
    let tag = tag.trim();
    (!tag.is_empty()).then(|| tag.to_string())
}

pub(super) fn previous_composition_baseline(
    root: &std::path::Path,
    tag: &str,
) -> Option<BTreeMap<String, f64>> {
    let output = std::process::Command::new("git")
        .args(["show", &format!("{tag}:audits/lego-composition.tsv")])
        .current_dir(root)
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let text = String::from_utf8(output.stdout).ok()?;
    parse_composition_baseline(&text)
}

// ============================================================
// Check 8: composability  -  flag islands.
// ============================================================
//
// An op O is an "island" when no other op composes it AND O composes
// nothing of its own. Islands fail the LEGO thesis: they are leaves
// with no upstream consumer, which means either (a) they were shipped
// on speculation and never wired in, or (b) they reinvent something a
// caller already has inline. Both cases want the user to look.
//
// Tier-2 intrinsics and Tier-2.5 primitives are terminal building blocks.
// Explicit Tier-3 leaves and tiny flat ops follow the same contract.

#[cfg(test)]
mod tests {
    use super::*;

    /// This adversarial parser test rejects malformed and out-of-range baseline rows while preserving exact valid fractions.
    #[test]
    fn composition_baseline_parser_accepts_only_bounded_finite_rows() {
        let parsed = parse_composition_baseline(
            "# op_id\tcomposed_fraction\nvalid::op\t0.25\nbad\tNaN\nhigh\t1.1\nmissing\n",
        )
        .expect("Fix: one valid composition baseline row must parse");
        assert_eq!(parsed.len(), 1);
        assert_eq!(parsed.get("valid::op"), Some(&0.25));
    }

    /// This numeric boundary test prevents baseline serialization rounding from becoming a false composition regression.
    #[test]
    fn composition_regression_tolerates_serialization_rounding_only() {
        assert!(!composition_regressed(0.913043478261, 21.0 / 23.0));
        assert!(composition_regressed(0.913043478261, 0.90));
    }

    /// WHY: an op whose composed_fraction dropped to zero on purpose composes
    /// nothing, so checks 6 and 8 flag it as a non-leaf with no child Region and
    /// as an island. Exempting it from the trend check alone moves the finding
    /// rather than closing it, and the next reader sees a collapse row and
    /// assumes the collapse is accounted for everywhere. The two tables must
    /// agree, which is the invariant a new collapse row goes red on.
    ///
    /// What this does not catch: a declared leaf that is not a collapse. That
    /// direction is legal, because an op can be a leaf without ever having had a
    /// higher baseline.
    #[test]
    fn every_intended_collapse_is_also_a_declared_leaf() {
        for row in INTENDED_COMPOSITION_COLLAPSES {
            assert!(
                is_declared_tier3_leaf(row),
                "`{row}` composes nothing by design, so it must also be a declared Tier-3 leaf or checks 6 and 8 flag it"
            );
        }
    }

    /// WHY: the trend check compares against a committed table and its fix line
    /// names a command that rewrites it. While no code called the writer, the
    /// table could only ever be edited by hand, and a ratchet whose baseline
    /// cannot be regenerated from the registry pins whatever a human typed.
    /// This holds the writer to the reader: what one records, the other reads
    /// back unchanged.
    ///
    /// What this does not catch: whether the fractions themselves are right.
    /// `composition_fractions` owns that and the checks above read it.
    #[test]
    fn a_written_baseline_reads_back_as_the_fractions_it_recorded() {
        let root = tempfile::tempdir().expect("Fix: create a temporary directory.");
        let mut composed =
            crate::gates::lego_audit::test_ops::op("vyre-libs::alpha::wide", Tier::T3, &[]);
        composed.own_nodes = 1;
        composed.composed_nodes = 3;
        let flat = crate::gates::lego_audit::test_ops::op("vyre-libs::alpha::flat", Tier::T3, &[]);
        let ops = vec![composed, flat];

        write_composition_baseline(root.path(), &ops).expect("Fix: write the baseline.");
        let read = current_composition_baseline(root.path())
            .expect("Fix: the writer must produce a table its own reader accepts.");

        assert_eq!(
            read,
            composition_fractions(&ops),
            "Fix: a baseline the reader cannot recover is a ratchet against a number nobody can regenerate."
        );
    }

    /// WHY: `composed_fraction` is a ratio over total nodes, and an operation
    /// that composes everything below its own entry boundary still counts that
    /// boundary as one own node. Its fraction is `1 - 1/total`, so shrinking
    /// the program lowers the number while composing every node of it, and the
    /// check reported a regression whose fix line asked for a restoration with
    /// nothing left to restore. The class is every fully composed operation
    /// whose program got smaller, so the assertion is over the predicate at the
    /// boundary rather than over one operation that hit it.
    ///
    /// What this does not catch: an operation that composes every node and also
    /// grew. That direction cannot lower the fraction, so it never reaches the
    /// report.
    #[test]
    fn a_shrinking_fully_composed_program_is_not_a_regression() {
        for total in 2..64usize {
            let own = 1usize;
            let composed = total - own;
            let fraction = composed as f64 / total as f64;
            let larger = (total * 2 - own) as f64 / (total * 2) as f64;
            assert!(
                composition_regressed(larger, fraction),
                "a fully composed program of {total} nodes reads as a regression against twice its size, which is the arithmetic this predicate exists to discount"
            );
            assert!(
                no_own_work_to_compose(own),
                "an operation whose only own node is its entry boundary has no work left to hand to a child"
            );
        }
    }

    /// WHY: discounting the fully composed case must not discount the case the
    /// check was built for. An operation that kept nodes a registered child
    /// used to carry has own work above its entry boundary, and that is exactly
    /// what separates it from the shrinking case above.
    #[test]
    fn an_operation_holding_own_work_above_its_boundary_still_regresses() {
        for own in 2..32usize {
            assert!(
                !no_own_work_to_compose(own),
                "{own} own nodes is work a child could carry, so the trend check must still judge it"
            );
        }
    }

    /// WHY: the predicate is read through a map keyed by op id, and an op id the
    /// baseline holds but the current registry does not must not be treated as
    /// fully composed by default. A missing key that read as "no own work"
    /// would silence every regression for a renamed operation.
    #[test]
    fn an_op_missing_from_the_current_walk_is_not_read_as_fully_composed() {
        let own_work: BTreeMap<&str, usize> = BTreeMap::new();
        assert!(
            !own_work
                .get("vyre-libs::alpha::gone")
                .copied()
                .is_some_and(no_own_work_to_compose),
            "Fix: an absent op id must not satisfy the fully composed discount."
        );
    }
}
