//! Shared-nothing analysis must see a buffer read that only a subgroup operand
//! performs.
//!
//! `collect_expr_reads` classified `SubgroupBallot`, `SubgroupShuffle`, and
//! `SubgroupReduce` as leaves, so a `load` nested inside one never entered the
//! statement's read set. `detect_parallelism` then compared an empty read set
//! against the next statement's writes, found no conflict, and put a read of
//! `acc` in the same concurrent dispatch group as the write that clobbers it.
//! Two dispatches in one group have no ordering between them, so the read is
//! free to observe the post-write value: a read-after-write dependency dropped.

use vyre_foundation::ir::{BufferAccess, BufferDecl, DataType, Expr, Node, Program};
use vyre_foundation::transform::parallelism::{detect_parallelism, DispatchGroup};
use vyre_foundation::validate::{validate_with_options, BackendCapabilities, ValidationOptions};

/// ```text
/// out[0] = subgroup_add(acc[0]);   // reads acc
/// acc[0] = 1;                      // writes acc
/// ```
fn nodes_reading_acc_through_a_subgroup_operand() -> Vec<Node> {
    vec![
        Node::store(
            "out",
            Expr::u32(0),
            Expr::subgroup_add(Expr::load("acc", Expr::u32(0))),
        ),
        Node::store("acc", Expr::u32(0), Expr::u32(1)),
    ]
}

/// The read-after-write pair must not share a dispatch group.
#[test]
fn a_read_inside_a_subgroup_operand_forces_a_serial_boundary() {
    let groups = detect_parallelism(&nodes_reading_acc_through_a_subgroup_operand());

    assert_eq!(
        groups,
        vec![
            DispatchGroup::Serial { node_index: 0 },
            DispatchGroup::Serial { node_index: 1 },
        ],
    );
}

/// The same read written without the subgroup wrapper already serialised, so the
/// wrapper must not change the grouping. This pins the two forms together: a
/// future walk that loses the operand again makes these two disagree.
#[test]
fn the_subgroup_wrapper_does_not_change_the_grouping() {
    let bare = vec![
        Node::store("out", Expr::u32(0), Expr::load("acc", Expr::u32(0))),
        Node::store("acc", Expr::u32(0), Expr::u32(1)),
    ];

    assert_eq!(
        detect_parallelism(&nodes_reading_acc_through_a_subgroup_operand()),
        detect_parallelism(&bare),
    );
}

/// Negative control: without the conflicting write the pair still fuses, so the
/// serial boundary above comes from the dependency and not from the subgroup op
/// being treated as an unconditional barrier.
#[test]
fn independent_statements_still_share_one_parallel_group() {
    let nodes = vec![
        Node::store(
            "out",
            Expr::u32(0),
            Expr::subgroup_add(Expr::load("acc", Expr::u32(0))),
        ),
        Node::store("other", Expr::u32(0), Expr::u32(1)),
    ];

    assert_eq!(
        detect_parallelism(&nodes),
        vec![DispatchGroup::Parallel {
            node_indices: vec![0, 1],
        }],
    );
}
/// The fixture is legal IR, not a shape the validator would have rejected, so
/// the serial boundary above describes a program a backend can actually be
/// asked to dispatch.
#[test]
fn the_fixture_statements_form_a_valid_program() {
    let program = Program::wrapped(
        vec![
            BufferDecl::storage("acc", 0, BufferAccess::ReadWrite, DataType::U32).with_count(1),
            BufferDecl::output("out", 1, DataType::U32).with_count(1),
        ],
        [1, 1, 1],
        nodes_reading_acc_through_a_subgroup_operand(),
    );
    let options = ValidationOptions::default().with_backend_capabilities(BackendCapabilities {
        supports_subgroup_ops: true,
        ..BackendCapabilities::default()
    });

    let report = validate_with_options(&program, options);

    assert!(
        report.errors.is_empty() && report.warnings.is_empty(),
        "Fix: the fixture program must validate cleanly, got {:?} / {:?}",
        report.errors,
        report.warnings,
    );
}
