//! Rebuilding part of a program keeps the rest of it.
//!
//! `Program::wrapped` is the constructor for a NEW program: it starts the
//! metadata from scratch and wraps the entry in a root Region. Using it to
//! rebuild an EXISTING program therefore drops `entry_op_id` and
//! `non_composable_with_self` and adds a Region layer, both silently. Four
//! shipped sites did exactly that, in a test harness, in the program fuser, and
//! in two optimizer paths, which says the shape is easy to reach for and the
//! consequence is invisible.
//!
//! `with_rewritten_buffers`, `with_rewritten_entry`, `with_rewritten_wrapped_entry`
//! and `map_entry` are the rebuild forms. This suite is what makes a fifth site
//! impossible to add unnoticed.

use vyre_foundation::ir::{BufferDecl, DataType, Expr, Node, Program};

const OP_ID: &str = "vyre.test.rebuild";

/// A program carrying every piece of metadata a rebuild could drop.
fn tagged_program() -> Program {
    let buffers = vec![
        BufferDecl::read("in", 0, DataType::U32).with_count(4),
        BufferDecl::read_write("out", 1, DataType::U32).with_count(4),
    ];
    let body = vec![Node::store(
        "out",
        Expr::gid_x(),
        Expr::load("in", Expr::gid_x()),
    )];
    Program::wrapped(buffers, [64, 1, 1], body)
        .with_entry_op_id(OP_ID)
        .with_non_composable_with_self(true)
}

fn assert_metadata_survived(rebuilt: &Program, form: &str) {
    assert!(
        rebuilt.is_non_composable_with_self(),
        "{form} dropped non_composable_with_self, so a body that cannot be duplicated now looks safe to fuse twice"
    );
    assert_eq!(
        rebuilt.entry_op_id().map(ToString::to_string),
        Some(OP_ID.to_string()),
        "{form} dropped entry_op_id, so the self-aliasing check loses the key it dedups on"
    );
    assert_eq!(
        rebuilt.workgroup_size(),
        [64, 1, 1],
        "{form} changed the workgroup size, which no rebuild form is entitled to do"
    );
}

/// Replacing the buffer table keeps the entry and the metadata. This is the
/// form the two optimizer paths needed: they add or drop one declaration.
#[test]
fn with_rewritten_buffers_keeps_the_entry_and_the_metadata() {
    let program = tagged_program();
    let mut buffers = program.buffers().to_vec();
    buffers.push(BufferDecl::workgroup("scratch", 4, DataType::U32));
    let rebuilt = program.with_rewritten_buffers(buffers);

    assert_metadata_survived(&rebuilt, "with_rewritten_buffers");
    assert_eq!(rebuilt.buffers().len(), 3, "the new declaration is present");
    assert_eq!(
        rebuilt.entry(),
        program.entry(),
        "the entry is untouched, including its Region nesting"
    );
}

/// Replacing the entry keeps the buffers and the metadata.
#[test]
fn with_rewritten_entry_keeps_the_buffers_and_the_metadata() {
    let program = tagged_program();
    let rebuilt = program.with_rewritten_entry(program.entry().to_vec());

    assert_metadata_survived(&rebuilt, "with_rewritten_entry");
    assert_eq!(rebuilt.buffers().len(), program.buffers().len());
}

/// The wrapping form adds exactly one Region, the runnable-root contract, and
/// does not stack a second one on an entry that already has it.
#[test]
fn with_rewritten_wrapped_entry_wraps_once_and_keeps_the_metadata() {
    let program = tagged_program();
    let inner = vec![Node::store("out", Expr::u32(0), Expr::u32(7))];
    let rebuilt = program.with_rewritten_wrapped_entry(inner);

    assert_metadata_survived(&rebuilt, "with_rewritten_wrapped_entry");
    assert!(
        matches!(rebuilt.entry(), [Node::Region { .. }]),
        "the runnable root is a single Region"
    );
    let Some(Node::Region { body, .. }) = rebuilt.entry().first() else {
        panic!("the entry must be one Region");
    };
    assert!(
        !matches!(body.as_slice(), [Node::Region { .. }]),
        "the body given to this form is placed inside the root, not wrapped again"
    );
}

/// The consuming form keeps the metadata too. It exists to reuse the entry Arc
/// under the optimizer fixpoint, which is the hottest rebuild path there is.
#[test]
fn map_entry_keeps_the_metadata() {
    let rebuilt = tagged_program().map_entry(|entry| entry);
    assert_metadata_survived(&rebuilt, "map_entry");
}

/// The contrast that motivates all of the above: the constructor does NOT
/// preserve anything, and that is correct for a new program. Asserting it here
/// keeps the distinction visible rather than leaving it a footgun.
#[test]
fn the_constructor_deliberately_starts_the_metadata_fresh() {
    let program = tagged_program();
    let fresh = Program::wrapped(
        program.buffers().to_vec(),
        program.workgroup_size(),
        program.entry().to_vec(),
    );
    assert!(
        !fresh.is_non_composable_with_self(),
        "Program::wrapped builds a new program; if this ever starts preserving metadata, the rebuild forms are what should change, not this"
    );
    assert!(fresh.entry_op_id().is_none());
    assert_eq!(
        fresh.entry(),
        program.entry(),
        "the entry survives unchanged: an already-rooted entry is not wrapped a second time, so Regions do not accumulate. The loss is the metadata, and only the metadata"
    );
}
