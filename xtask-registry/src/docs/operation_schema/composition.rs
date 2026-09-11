//! The nested-operation chain a built Program spells out, and its shape rules.

use std::collections::BTreeSet;

use vyre::ir::Node;

use super::schema::CompositionStep;

pub(super) fn collect_composition(
    node: &Node,
    depth: usize,
    all_ids: &BTreeSet<&str>,
    out: &mut Vec<CompositionStep>,
) {
    match node {
        Node::Region {
            generator, body, ..
        } => {
            let operation = generator.as_str().to_string();
            out.push(CompositionStep {
                depth,
                registered: all_ids.contains(operation.as_str()),
                operation,
            });
            for child in body.iter() {
                collect_composition(child, depth + 1, all_ids, out);
            }
        }
        Node::Block(children) => {
            for child in children {
                collect_composition(child, depth, all_ids, out);
            }
        }
        Node::If {
            then, otherwise, ..
        } => {
            for child in then {
                collect_composition(child, depth, all_ids, out);
            }
            for child in otherwise {
                collect_composition(child, depth, all_ids, out);
            }
        }
        Node::Loop { body, .. } => {
            for child in body {
                collect_composition(child, depth, all_ids, out);
            }
        }
        _ => {}
    }
}

pub(super) fn validate_composition(id: &str, chain: &[CompositionStep], errors: &mut Vec<String>) {
    let mut previous_depth = 0;
    for (index, step) in chain.iter().enumerate() {
        if step.operation.trim().is_empty() {
            errors.push(format!("operation `{id}` has an empty composition step"));
        }
        if index == 0 && step.depth != 0 {
            errors.push(format!(
                "operation `{id}` composition chain starts at depth {} instead of 0",
                step.depth
            ));
        } else if index > 0 && step.depth > previous_depth + 1 {
            errors.push(format!(
                "operation `{id}` composition depth jumps from {previous_depth} to {}",
                step.depth
            ));
        }
        previous_depth = step.depth;
    }
}
