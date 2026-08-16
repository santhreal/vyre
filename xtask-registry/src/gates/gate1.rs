//! `cargo xtask gate1`  -  Gate 1 complexity-budget enforcement.
//!
//! The composition policy states the rule; this gate owns the half of it that
//! is countable. An operation is either small enough to read whole, or mostly
//! made of other registered operations. Nothing in between. Reuse count is
//! policy the author applies, not a number a gate can read off a program.
//!
//! For every registered operation, `composition_budget::measure` counts the
//! nodes, the loops and the share of nodes that are calls to another registered
//! operation, and the verdict is either the raw budget or that share. The
//! diagnostic lists the inline loops and uncomposed region bodies an author
//! would have to extract, so the report names the work rather than the number.
//!
//! `abstraction-gate` reads the same walk for a different question: whether
//! every child region names a building block that exists. The budget is
//! reported once, here.

use xtask::gate::{Finding, Gate, GateCtx, GateError, Report};

use crate::gates::composition_budget::{
    self, Counts, COMPOSED_FRACTION_THRESHOLD, LOOP_BUDGET, NODE_BUDGET,
};

/// Entry point for the `gate1` subcommand.
/// Enforces the Gate 1 complexity budget over every registered operation.
pub struct Gate1;

impl Gate for Gate1 {
    fn name(&self) -> &'static str {
        "gate1"
    }

    fn help(&self) -> &'static str {
        "Enforce the Gate 1 complexity budget"
    }

    fn run(&self, _ctx: &GateCtx) -> Result<Report, GateError> {
        let mut report = Report::clean();
        let ops = composition_budget::collect_ops(&mut report);
        let ids = composition_budget::registered_ids(&ops);
        let mut audited: Vec<(String, Counts)> = ops
            .iter()
            .map(|op| {
                (
                    op.id.clone(),
                    composition_budget::measure(&op.program, &ids, &mut |_| {}),
                )
            })
            .collect();
        audited.sort_by(|left, right| left.0.cmp(&right.0));

        report.note(format!(
            "budget: loops <= {LOOP_BUDGET} and nodes <= {NODE_BUDGET}, or composed_fraction >= {:.0}%",
            COMPOSED_FRACTION_THRESHOLD * 100.0
        ));
        report.note(format!("{} operation(s) audited", audited.len()));
        for (op_id, counts) in &audited {
            report.note(format!(
                "{op_id:<60}  loops={:<3} nodes={:<5} composed={:>5.1}%",
                counts.loops,
                counts.total_nodes,
                counts.composed_fraction_pct()
            ));
            if counts.passes() {
                continue;
            }
            report.find(over_budget(op_id, counts));
        }
        Ok(report)
    }
}

/// The finding for one operation that is neither small nor composed.
fn over_budget(op_id: &str, counts: &Counts) -> Finding {
    let fix = if counts.inline_hot_spots.is_empty() {
        "factor the inline work into a registered operation and call it through vyre_foundation::composition::wrap_child_region"
            .to_string()
    } else {
        format!(
            "extract each inline hot spot into a registered operation and call it through vyre_foundation::composition::wrap_child_region: {}",
            counts.inline_hot_spots.join(", ")
        )
    };
    Finding::new(
        format!(
            "operation `{op_id}` is over the Gate 1 budget: loops={} (budget {LOOP_BUDGET}), nodes={} (budget {NODE_BUDGET}), composed={:.1}% (need {:.0}%)",
            counts.loops,
            counts.total_nodes,
            counts.composed_fraction_pct(),
            COMPOSED_FRACTION_THRESHOLD * 100.0
        ),
        fix,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    /// WHY: the finding an author reads has to name the inline work, not only
    /// the share it produced, or the only actionable half of the diagnostic is
    /// the one the walk collected and dropped.
    #[test]
    fn the_finding_names_the_inline_work_it_collected() {
        let counts = Counts {
            total_nodes: 300,
            loops: 9,
            composed_nodes: 0,
            inline_hot_spots: vec!["inline loop with 4 body nodes".to_string()],
        };

        let finding = over_budget("owner::op", &counts);

        assert!(finding.message.contains("loops=9"), "{finding:?}");
        assert!(finding.message.contains("composed=0.0%"), "{finding:?}");
        assert!(
            finding.fix.contains("inline loop with 4 body nodes"),
            "{finding:?}"
        );
    }

    /// WHY: an operation whose bulk is inline but whose loops all sit inside a
    /// composed subtree collects no hot spot, and the fix still has to say what
    /// to do.
    #[test]
    fn a_finding_with_no_hot_spot_still_states_the_action() {
        let counts = Counts {
            total_nodes: 300,
            loops: 9,
            composed_nodes: 0,
            inline_hot_spots: Vec::new(),
        };

        let finding = over_budget("owner::op", &counts);

        assert!(finding.fix.contains("wrap_child_region"), "{finding:?}");
    }
}
