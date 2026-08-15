//! Composition metadata helpers for fused programs.
//!
//! Some generated regions are intentionally not composable with another
//! instance of themselves inside the same fused kernel. We encode that
//! contract in the region generator string so both validation and
//! optimization passes can enforce it without needing a new IR field.

use std::sync::Arc;

use crate::ir::model::expr::{Expr, GeneratorRef, Ident};
use crate::ir::{BufferDecl, DataType, Node, Program};
use rustc_hash::FxHashMap;

/// Generator suffix marking a region as non-composable with itself.
pub const SELF_EXCLUSIVE_REGION_SUFFIX: &str = "#self-exclusive";

/// Append the self-exclusive marker to a generator id.
#[must_use]
pub fn mark_self_exclusive_region(generator: &str) -> String {
    format!("{generator}{SELF_EXCLUSIVE_REGION_SUFFIX}")
}

/// Generator prefixes that state there is no operation behind a region.
///
/// Two producers mint them and they mean the same thing:
///
/// - `inline::<parent>` comes from [`reparent_entry_node`] below, for a body
///   the composer reparented onto its caller because it had no region of its
///   own.
/// - `anonymous::<label>` is written by a builder that needs a named phase
///   boundary inside one operation and has no operation to name it with.
///
/// The distinction matters to anything that reads a generator as an operation
/// id. Composition itself stamps `source_region` onto EVERY entry region it
/// reparents, anonymous ones included, so `source_region.is_some()` does not
/// mean the author declared an edge to a registered building block. The
/// prefix is what says the generator was never an operation id, and a
/// consumer that knows only one of the two prefixes demands a registration
/// for an operation that must not exist.
pub const ANONYMOUS_GENERATOR_PREFIXES: [&str; 2] = ["inline::", "anonymous::"];

/// True when `generator` names no operation, by [`ANONYMOUS_GENERATOR_PREFIXES`].
///
/// Prefix, not substring: `foo::not_inline::bar` is an ordinary generator.
#[must_use]
pub fn is_anonymous_generator(generator: &str) -> bool {
    ANONYMOUS_GENERATOR_PREFIXES
        .iter()
        .any(|prefix| generator.starts_with(prefix))
}

/// Wrap nodes in a named, substrate-neutral composition region.
#[must_use]
pub fn wrap_region(generator: &str, body: Vec<Node>, source_region: Option<GeneratorRef>) -> Node {
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

/// Guard nodes so that only invocation zero on axis zero runs them.
#[must_use]
pub fn single_invocation(body: Vec<Node>) -> Vec<Node> {
    vec![Node::if_then(
        Expr::eq(Expr::InvocationId { axis: 0 }, Expr::u32(0)),
        body,
    )]
}

/// The entry of a serial kernel: one anonymous region that runs on invocation zero.
#[must_use]
pub fn single_invocation_region(op_id: &str, body: Vec<Node>) -> Vec<Node> {
    vec![wrap_anonymous_region(op_id, single_invocation(body))]
}

/// The program a builder returns when its inputs cannot produce a valid one.
///
/// Every primitive and composition builder is infallible, so an invalid shape
/// becomes an IR trap the backend refuses rather than a host panic. `output`
/// declares the single result buffer a caller expects to bind, and is `None`
/// for a builder whose signature promises no output.
#[must_use]
pub fn trap_program(
    op_id: &str,
    output: Option<(&str, DataType)>,
    message: impl Into<String>,
) -> Program {
    let buffers = match output {
        Some((name, element)) => vec![BufferDecl::output(name, 0, element).with_count(1)],
        None => Vec::new(),
    };
    Program::wrapped(
        buffers,
        [1, 1, 1],
        vec![wrap_anonymous_region(
            op_id,
            vec![Node::trap(Expr::u32(0), message.into())],
        )],
    )
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

/// `inline::<parent>`, the generator for a body with no region of its own.
fn inline_generator(parent: &GeneratorRef) -> Ident {
    Ident::from(format!(
        "{}{}",
        ANONYMOUS_GENERATOR_PREFIXES[0], parent.name
    ))
}

fn reparent_entry_node(node: Node, parent: &GeneratorRef) -> Node {
    match node {
        Node::Region {
            generator, body, ..
        } => Node::Region {
            generator: if generator.as_ref() == Program::ROOT_REGION_GENERATOR {
                inline_generator(parent)
            } else {
                generator
            },
            source_region: Some(parent.clone()),
            body,
        },
        other => Node::Region {
            generator: inline_generator(parent),
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
        if let Node::Region { generator, .. } = node {
            if let Some(base) = self_exclusive_region_key(generator.as_str()) {
                *counts.entry(base).or_insert(0) += 1;
            }
        }
        for body in crate::transform::visit::child_bodies(node) {
            collect_self_exclusive_regions(body, counts);
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

    /// The trap body and the declared output are the whole contract callers rely on.
    #[test]
    fn a_trap_program_declares_only_the_output_it_was_asked_for() {
        let with_output = trap_program(
            "vyre.test.reduce",
            Some(("result", DataType::F32)),
            "shape has zero elements",
        );
        assert_eq!(with_output.buffers().len(), 1);
        let buffer = &with_output.buffers()[0];
        assert_eq!(&*buffer.name, "result");
        assert_eq!(buffer.element, DataType::F32);
        assert_eq!(buffer.count, 1);

        let Some(Node::Region { generator, body, .. }) = with_output.entry().first() else {
            panic!("a trap program's entry is one region: {:?}", with_output.entry());
        };
        assert_eq!(generator.as_str(), "vyre.test.reduce");
        assert!(
            matches!(body.as_slice(), [Node::Trap { tag, .. }] if tag.as_str() == "shape has zero elements"),
            "the region body carries the message as a trap: {body:?}"
        );

        let without_output = trap_program("vyre.test.barrier", None, "unsupported scope");
        assert!(without_output.buffers().is_empty());
    }

    /// A serial kernel runs its body once, on invocation zero of axis zero.
    #[test]
    fn a_single_invocation_region_guards_the_body_it_was_given() {
        let entry = single_invocation_region("vyre.test.scan", vec![Node::Return]);
        let [Node::Region {
            generator,
            source_region,
            body,
        }] = entry.as_slice()
        else {
            panic!("a serial kernel entry is one anonymous region: {entry:?}");
        };
        assert_eq!(generator.as_str(), "vyre.test.scan");
        assert_eq!(*source_region, None);
        assert_eq!(body.as_slice(), single_invocation(vec![Node::Return]));

        let [Node::If { cond, then, otherwise }] = body.as_slice() else {
            panic!("the region body is one guarded branch: {body:?}");
        };
        assert_eq!(
            *cond,
            Expr::eq(Expr::InvocationId { axis: 0 }, Expr::u32(0))
        );
        assert_eq!(then.as_slice(), [Node::Return]);
        assert!(otherwise.is_empty(), "a serial guard has no else arm");
    }
}
