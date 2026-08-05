//! Composition metadata helpers for fused programs.
//!
//! Some generated regions are intentionally not composable with another
//! instance of themselves inside the same fused kernel. We encode that
//! contract in the region generator string so both validation and
//! optimization passes can enforce it without needing a new IR field.

use std::sync::Arc;

use crate::ir::model::expr::{GeneratorRef, Ident};
use crate::ir::{Node, Program};
use rustc_hash::FxHashMap;

/// Generator suffix marking a region as non-composable with itself.
pub const SELF_EXCLUSIVE_REGION_SUFFIX: &str = "#self-exclusive";

/// Append the self-exclusive marker to a generator id.
#[must_use]
pub fn mark_self_exclusive_region(generator: &str) -> String {
    format!("{generator}{SELF_EXCLUSIVE_REGION_SUFFIX}")
}
/// Wrap nodes in a named, substrate-neutral composition region.
#[must_use]
pub fn wrap_region(
    generator: &str,
    body: Vec<Node>,
    source_region: Option<GeneratorRef>,
) -> Node {
    Node::Region {
        generator: Ident::from(generator),
        source_region,
        body: Arc::new(body),
    }
}

/// Wrap nodes in a composition region without source metadata.
#[must_use]
pub fn wrap_anonymous_region(generator: &str, body: Vec<Node>) -> Node {
    wrap_region(generator, body, None)
}

/// Wrap nodes in a composition region attributed to a parent generator.
#[must_use]
pub fn wrap_child_region(generator: &str, parent: GeneratorRef, body: Vec<Node>) -> Node {
    wrap_region(generator, body, Some(parent))
}

/// Clone a program's entry regions and attach them to a composing parent.
#[must_use]
pub fn reparent_program_children(program: &Program, parent_op_id: &str) -> Vec<Node> {
    let parent = GeneratorRef {
        name: parent_op_id.to_string(),
    };
    program
        .entry()
        .iter()
        .cloned()
        .map(|node| reparent_entry_node(node, &parent))
        .collect()
}

/// Wrap an existing program under one canonical parent composition boundary.
#[must_use]
pub fn tag_program(parent_op_id: &str, program: Program) -> Program {
    let generator = if program.is_non_composable_with_self() {
        mark_self_exclusive_region(parent_op_id)
    } else {
        parent_op_id.to_string()
    };
    let parent = GeneratorRef {
        name: parent_op_id.to_string(),
    };
    program.map_entry(|entry| {
        let children = entry
            .into_iter()
            .map(|node| reparent_entry_node(node, &parent))
            .collect();
        vec![Node::Region {
            generator: Ident::from(generator),
            source_region: None,
            body: Arc::new(children),
        }]
    })
}

fn reparent_entry_node(node: Node, parent: &GeneratorRef) -> Node {
    match node {
        Node::Region {
            generator, body, ..
        } => Node::Region {
            generator: if generator.as_ref() == Program::ROOT_REGION_GENERATOR {
                Ident::from(format!("inline::{}", parent.name))
            } else {
                generator
            },
            source_region: Some(parent.clone()),
            body,
        },
        other => Node::Region {
            generator: Ident::from(format!("inline::{}", parent.name)),
            source_region: Some(parent.clone()),
            body: Arc::new(vec![other]),
        },
    }
}

/// Return the base generator id when this region is self-exclusive.
#[must_use]
pub fn self_exclusive_region_key(generator: &str) -> Option<&str> {
    generator.strip_suffix(SELF_EXCLUSIVE_REGION_SUFFIX)
}

/// Return duplicate self-exclusive generators present in one program.
#[must_use]
pub fn duplicate_self_exclusive_regions(nodes: &[Node]) -> Vec<String> {
    let mut counts = FxHashMap::<&str, usize>::default();
    collect_self_exclusive_regions(nodes, &mut counts);
    let mut duplicates = counts
        .into_iter()
        .filter_map(|(generator, count)| (count > 1).then_some(generator.to_string()))
        .collect::<Vec<_>>();
    duplicates.sort();
    duplicates
}

fn collect_self_exclusive_regions<'a>(nodes: &'a [Node], counts: &mut FxHashMap<&'a str, usize>) {
    for node in nodes {
        match node {
            Node::If {
                then, otherwise, ..
            } => {
                collect_self_exclusive_regions(then, counts);
                collect_self_exclusive_regions(otherwise, counts);
            }
            Node::Loop { body, .. } | Node::Block(body) => {
                collect_self_exclusive_regions(body, counts);
            }
            Node::Region {
                generator, body, ..
            } => {
                if let Some(base) = self_exclusive_region_key(generator.as_str()) {
                    *counts.entry(base).or_insert(0) += 1;
                }
                collect_self_exclusive_regions(body, counts);
            }
            Node::Let { .. }
            | Node::Assign { .. }
            | Node::Store { .. }
            | Node::AllReduce { .. }
            | Node::AllGather { .. }
            | Node::ReduceScatter { .. }
            | Node::Broadcast { .. }
            | Node::Return
            | Node::Barrier { .. }
            | Node::IndirectDispatch { .. }
            | Node::AsyncLoad { .. }
            | Node::AsyncStore { .. }
            | Node::AsyncWait { .. }
            | Node::Trap { .. }
            | Node::Resume { .. }
            | Node::Opaque(_) => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{Node, Program};
    use std::sync::Arc;

    #[test]
    fn duplicate_self_exclusive_regions_are_reported() {
        let generator = mark_self_exclusive_region("vyre.test.parser");
        let program = Program::wrapped(
            Vec::new(),
            [1, 1, 1],
            vec![
                Node::Region {
                    generator: generator.clone().into(),
                    source_region: None,
                    body: Arc::new(vec![Node::Return]),
                },
                Node::Region {
                    generator: generator.into(),
                    source_region: None,
                    body: Arc::new(vec![Node::Return]),
                },
                Node::Region {
                    generator: "plain.region".into(),
                    source_region: None,
                    body: Arc::new(vec![Node::Return]),
                },
            ],
        );
        assert_eq!(
            duplicate_self_exclusive_regions(program.entry()),
            vec!["vyre.test.parser".to_string()]
        );
    }
}
