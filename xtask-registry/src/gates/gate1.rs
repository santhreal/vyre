//! `cargo xtask gate1`  -  Gate 1 complexity-budget enforcement.
//!
//! The composition policy states the rule; this gate enforces the half of it
//! that is countable. An op is either small enough to read whole, or mostly
//! made of other registered ops. Nothing in between. Reuse count is policy the
//! author applies, not a number this gate can read off a program.
//!
//! For every registered op (vyre-libs + vyre-primitives inventories):
//!
//! 1. Build the op's `Program`.
//! 2. Walk the entry-body Node tree:
//!    - `total_nodes`  -  recursive node count.
//!    - `loops`  -  count of `Node::Loop`.
//!    - `composed_nodes`  -  count of nodes that live inside a
//!      `Node::Region { source_region: Some(_), .. }` (i.e. the Region
//!      was constructed by composing another registered op rather than
//!      being an anonymous local wrapper).
//! 3. Pass if EITHER:
//!    - Under raw budget: `loops <= 4 AND total_nodes <= 200`, OR
//!    - Adequate composition: `composed_nodes / total_nodes >= 0.6`.
//!
//! On fail, the diagnostic lists the inline sub-blocks (the loops /
//! large Block / If branches that aren't wrapped in a child Region)
//! so an author can see exactly what should have been a primitive
//! call.
//!
//! Exit code 0 = all ops pass. Exit code 1 = ≥ 1 op fails (CI signal).

use vyre::ir::{Node, Program};
use xtask::gate::{Finding, Gate, GateCtx, GateError, Report};

const LOOP_BUDGET: usize = 4;
const NODE_BUDGET: usize = 200;
const COMPOSED_FRACTION_THRESHOLD: f64 = 0.6;

/// Per-op gate-1 verdict.
#[derive(Debug)]
struct Verdict {
    op_id: String,
    total_nodes: usize,
    loops: usize,
    composed_nodes: usize,
    inline_hot_spots: Vec<String>,
}

impl Verdict {
    fn passes(&self) -> bool {
        if self.loops <= LOOP_BUDGET && self.total_nodes <= NODE_BUDGET {
            return true;
        }
        if self.total_nodes == 0 {
            return true;
        }
        let composed_fraction = self.composed_nodes as f64 / self.total_nodes as f64;
        composed_fraction >= COMPOSED_FRACTION_THRESHOLD
    }

    fn composed_fraction_pct(&self) -> f64 {
        if self.total_nodes == 0 {
            return 100.0;
        }
        100.0 * self.composed_nodes as f64 / self.total_nodes as f64
    }
}

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
        let mut verdicts: Vec<Verdict> = Vec::new();
        for entry in vyre_registry_link::operation::live_operation_registry().iter() {
            let Some(program) = entry.program() else {
                report.find(Finding::new(
                    format!(
                        "registered operation `{}` provides no neutral builder, so its complexity cannot be judged",
                        entry.id
                    ),
                    "register a neutral builder for it, or withdraw the registration",
                ));
                continue;
            };
            verdicts.push(verdict_for(entry.id, &program));
        }
        verdicts.sort_by(|left, right| left.op_id.cmp(&right.op_id));

        report.note(format!(
            "budget: loops <= {LOOP_BUDGET} and nodes <= {NODE_BUDGET}, or composed_fraction >= {:.0}%",
            COMPOSED_FRACTION_THRESHOLD * 100.0
        ));
        report.note(format!("{} operation(s) audited", verdicts.len()));
        for verdict in &verdicts {
            report.note(format!(
                "{:<60}  loops={:<3} nodes={:<5} composed={:>5.1}%",
                verdict.op_id,
                verdict.loops,
                verdict.total_nodes,
                verdict.composed_fraction_pct()
            ));
            if verdict.passes() {
                continue;
            }
            let fix = if verdict.inline_hot_spots.is_empty() {
                "factor the inline work into a registered primitive call through region::wrap_child"
                    .to_string()
            } else {
                format!(
                    "extract each inline hot spot into a vyre-primitives operation and call it through region::wrap_child: {}",
                    verdict.inline_hot_spots.join(", ")
                )
            };
            report.find(Finding::new(
                format!(
                    "operation `{}` is over the Gate 1 budget: loops={} (budget {LOOP_BUDGET}), nodes={} (budget {NODE_BUDGET}), composed={:.1}% (need {:.0}%)",
                    verdict.op_id,
                    verdict.loops,
                    verdict.total_nodes,
                    verdict.composed_fraction_pct(),
                    COMPOSED_FRACTION_THRESHOLD * 100.0
                ),
                fix,
            ));
        }
        Ok(report)
    }
}

fn verdict_for(op_id: &'static str, program: &Program) -> Verdict {
    let mut state = WalkState::default();
    for node in program.entry() {
        walk(node, false, &mut state);
    }
    Verdict {
        op_id: op_id.to_string(),
        total_nodes: state.total_nodes,
        loops: state.loops,
        composed_nodes: state.composed_nodes,
        inline_hot_spots: state.inline_hot_spots,
    }
}

#[derive(Default)]
struct WalkState {
    total_nodes: usize,
    loops: usize,
    composed_nodes: usize,
    inline_hot_spots: Vec<String>,
}

/// Walk a node, counting it and recursing.
///
/// `inside_composed_region` propagates downward: once we enter a
/// `Region { source_region: Some(_), .. }`, every node beneath counts
/// toward `composed_nodes`. Anonymous regions (`source_region: None`)
/// do NOT promote their children to composed  -  they're local wrappers,
/// not composition.
fn walk(node: &Node, inside_composed_region: bool, state: &mut WalkState) {
    state.total_nodes += 1;
    if inside_composed_region {
        state.composed_nodes += 1;
    }

    match node {
        Node::Region {
            source_region,
            body,
            generator,
        } => {
            let now_composed = inside_composed_region || source_region.is_some();
            for child in body.iter() {
                walk(child, now_composed, state);
            }
            // Hot spot: an anonymous Region with > 50 inline nodes  -
            // either factor the body into a registered primitive or
            // mark the source_region.
            if !inside_composed_region && source_region.is_none() && body.len() > 50 {
                state.inline_hot_spots.push(format!(
                    "anonymous Region `{}` with {} top-level body nodes",
                    generator.as_str(),
                    body.len()
                ));
            }
        }
        Node::Loop { body, .. } => {
            state.loops += 1;
            for child in body {
                walk(child, inside_composed_region, state);
            }
            if !inside_composed_region {
                state.inline_hot_spots.push(format!(
                    "inline `Node::Loop` with {} body nodes",
                    body.len()
                ));
            }
        }
        Node::Block(children) => {
            for child in children {
                walk(child, inside_composed_region, state);
            }
        }
        Node::If {
            then, otherwise, ..
        } => {
            for child in then {
                walk(child, inside_composed_region, state);
            }
            for child in otherwise {
                walk(child, inside_composed_region, state);
            }
        }
        // Leaves  -  Let, Assign, Store, Return, Barrier, IndirectDispatch,
        // AsyncLoad, AsyncWait, Opaque  -  count themselves and stop.
        _ => {}
    }
}
