//! `print-composition` - walk the Region decomposition chain of every
//! registered operation.
//!
//! The chain is what makes a public operation a composition of registered
//! primitives rather than a hand-written body, so an operation that cannot be
//! walked has no chain to audit. The gate resolves the Program of every
//! registered operation and recurses into every `Node::Region` in its entry
//! body. A registered operation whose builder yields no Program is a finding,
//! and so is a Region that composes nothing, because an empty composition
//! boundary claims a decomposition step that does not exist. `--op-id ID`
//! narrows the corpus to one operation. The tree itself is context, so it is
//! reported as notes and never counted.

use vyre::ir::Node;
use xtask::gate::{Finding, Gate, GateCtx, GateError, Report};

/// Walks the composition chain of the registered operations.
pub struct PrintComposition;

impl Gate for PrintComposition {
    fn name(&self) -> &'static str {
        "print-composition"
    }

    fn help(&self) -> &'static str {
        "Walk the decomposition chain of every registered operation; --op-id ID narrows to one"
    }

    fn run(&self, ctx: &GateCtx) -> Result<Report, GateError> {
        let selected = ctx.flag("--op-id");
        let mut report = Report::clean();
        let mut walked = 0usize;
        let mut matched = false;
        for entry in vyre_registry_link::operation::live_operation_registry().iter() {
            if let Some(op_id) = selected {
                if entry.id != op_id {
                    continue;
                }
            }
            matched = true;
            let Some(program) = entry.program() else {
                report.find(Finding::new(
                    format!(
                        "registered operation `{}` provides no neutral builder, so its composition chain cannot be walked",
                        entry.id
                    ),
                    "give the canonical registration a neutral builder, or withdraw the registration",
                ));
                continue;
            };
            walked += 1;
            report.note(format!(
                "{}  [{} top-level Nodes]",
                entry.id,
                program.entry().len()
            ));
            for node in program.entry() {
                walk(entry.id, node, 1, &mut report);
            }
        }
        if let Some(op_id) = selected {
            if !matched {
                return Err(GateError::new(
                    format!("op id `{op_id}` is not registered"),
                    "run the gate without --op-id to list every registered operation",
                ));
            }
        }
        report.note(format!("{walked} composition chain(s) walked"));
        Ok(report)
    }
}

fn walk(op_id: &str, node: &Node, depth: usize, report: &mut Report) {
    let indent = "  ".repeat(depth);
    match node {
        Node::Region {
            generator, body, ..
        } => {
            report.note(format!(
                "{indent}{}  [{} Nodes]",
                generator.as_str(),
                body.len()
            ));
            if body.is_empty() {
                report.find(Finding::new(
                    format!(
                        "operation `{op_id}` opens region `{}` with an empty body",
                        generator.as_str()
                    ),
                    "emit the region body from its generator, or drop the region so the chain reports what it really composes",
                ));
            }
            for child in body.iter() {
                walk(op_id, child, depth + 1, report);
            }
        }
        Node::Block(children) => {
            for child in children {
                walk(op_id, child, depth, report);
            }
        }
        Node::If {
            then, otherwise, ..
        } => {
            for child in then {
                walk(op_id, child, depth, report);
            }
            for child in otherwise {
                walk(op_id, child, depth, report);
            }
        }
        Node::Loop { body, .. } => {
            for child in body {
                walk(op_id, child, depth, report);
            }
        }
        _ => {}
    }
}
