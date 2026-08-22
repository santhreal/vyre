//! Fusion must not widen the workgroup of an arm that reasons about its own.
//!
//! The fused launch geometry is the axis-wise maximum over the arms. That is a
//! pure launch-size change for an arm whose invocations are independent, and a
//! semantic change for an arm that synchronizes its workgroup or keeps state in
//! workgroup memory. Such an arm guards its body for its own width, so under a
//! wider workgroup the extra invocations skip the guarded body and never reach
//! the barrier the working invocations wait on. A workgroup barrier not reached
//! by every invocation in the workgroup is undefined.
//!
//! The bug this suite locks out: an inclusive prefix scan built for 4 elements
//! (`vyre-libs::math::scan_prefix_sum`, workgroup 4) fused behind a 256-wide
//! elementwise arm returned the wrong final lane on roughly one dispatch in
//! ten. It was intermittent, so it read as flakiness rather than as the
//! unsound fusion it was. See BACKLOG.md R73.

use std::sync::Arc;

use vyre_foundation::execution_plan::fusion::{fuse_programs, FusionError};
use vyre_foundation::ir::MemoryOrdering;
use vyre_foundation::ir::{BufferDecl, DataType, Expr, Node, Program};

/// An arm with the given workgroup size whose invocations are independent.
///
/// No workgroup memory, no barrier, so widening its workgroup only launches
/// more invocations of the same independent body.
fn independent_arm(name: &str, workgroup: u32) -> Program {
    let input = format!("{name}_in");
    let output = format!("{name}_out");
    let buffers = vec![
        BufferDecl::read(&input, 0, DataType::U32).with_count(4),
        BufferDecl::read_write(&output, 1, DataType::U32).with_count(4),
    ];
    let body = vec![Node::store(
        output.as_str(),
        Expr::gid_x(),
        Expr::load(input.as_str(), Expr::gid_x()),
    )];
    Program::wrapped(buffers, [workgroup, 1, 1], body)
}

/// An arm that synchronizes its workgroup, with no workgroup memory.
fn barrier_arm(name: &str, workgroup: u32) -> Program {
    let input = format!("{name}_in");
    let output = format!("{name}_out");
    let buffers = vec![
        BufferDecl::read(&input, 0, DataType::U32).with_count(4),
        BufferDecl::read_write(&output, 1, DataType::U32).with_count(4),
    ];
    let body = vec![
        Node::store(
            output.as_str(),
            Expr::gid_x(),
            Expr::load(input.as_str(), Expr::gid_x()),
        ),
        Node::barrier_with_ordering(MemoryOrdering::SeqCst),
    ];
    Program::wrapped(buffers, [workgroup, 1, 1], body)
}

/// An arm that keeps state in workgroup memory, with no barrier.
fn workgroup_memory_arm(name: &str, workgroup: u32) -> Program {
    let input = format!("{name}_in");
    let output = format!("{name}_out");
    let scratch = format!("{name}_scratch");
    let buffers = vec![
        BufferDecl::read(&input, 0, DataType::U32).with_count(4),
        BufferDecl::read_write(&output, 1, DataType::U32).with_count(4),
        BufferDecl::workgroup(&scratch, 4, DataType::U32),
    ];
    let body = vec![
        Node::store(
            scratch.as_str(),
            Expr::gid_x(),
            Expr::load(input.as_str(), Expr::gid_x()),
        ),
        Node::store(
            output.as_str(),
            Expr::gid_x(),
            Expr::load(scratch.as_str(), Expr::gid_x()),
        ),
    ];
    Program::wrapped(buffers, [workgroup, 1, 1], body)
}

/// An arm whose serial body is admitted by its workgroup identity alone.
///
/// No workgroup memory and no barrier: the running result lives in read-write
/// storage, and `workgroup_id.x == 0` is exact only while the workgroup holds
/// one invocation.
fn serial_workgroup_guard_arm(name: &str, workgroup: u32) -> Program {
    serial_arm(name, workgroup, Expr::is_first_workgroup())
}

/// The same arm, admitted by its invocation instead.
///
/// The same lane in a one-wide geometry and the only lane in any other, so this
/// one is safe to widen.
fn serial_invocation_guard_arm(name: &str, workgroup: u32) -> Program {
    serial_arm(name, workgroup, Expr::is_first_invocation())
}

fn serial_arm(name: &str, workgroup: u32, guard: Expr) -> Program {
    let input = format!("{name}_in");
    let output = format!("{name}_out");
    let buffers = vec![
        BufferDecl::read(&input, 0, DataType::U32).with_count(4),
        BufferDecl::read_write(&output, 1, DataType::U32).with_count(4),
    ];
    let body = vec![Node::if_then(
        guard,
        vec![Node::loop_for(
            "idx",
            Expr::u32(0),
            Expr::u32(4),
            vec![Node::store(
                output.as_str(),
                Expr::u32(0),
                Expr::add(
                    Expr::load(output.as_str(), Expr::u32(0)),
                    Expr::load(input.as_str(), Expr::var("idx")),
                ),
            )],
        )],
    )];
    Program::wrapped(buffers, [workgroup, 1, 1], body)
}


fn geometry_error(result: Result<Program, FusionError>) -> String {
    match result {
        Ok(program) => panic!(
            "fusion accepted a workgroup widening and produced a program with workgroup {:?}",
            program.workgroup_size()
        ),
        Err(FusionError::WorkgroupGeometry(error)) => error.to_string(),
        Err(other) => panic!("expected a workgroup-geometry refusal, got: {other}"),
    }
}

/// The exact shape of the reported bug: a narrow synchronizing arm fused
/// behind a wide independent one. Before the fix this produced a program whose
/// workgroup was 256 while the scan arm expected 4.
#[test]
fn a_narrow_synchronizing_arm_is_not_widened_by_a_wide_neighbour() {
    let message = geometry_error(fuse_programs(&[
        independent_arm("wide", 256),
        barrier_arm("narrow", 4),
    ]));
    assert!(
        message.contains("arm 1"),
        "the refusal must name the offending arm: {message}"
    );
    assert!(
        message.contains("[256, 1, 1]") && message.contains("[4, 1, 1]"),
        "the refusal must state both geometries so the caller can see the widening: {message}"
    );
    assert!(
        message.contains("synchronizes its workgroup"),
        "the refusal must say what makes the widening unsafe: {message}"
    );
    assert!(
        message.contains("dispatch this arm separately"),
        "the refusal must be actionable: {message}"
    );
}

/// Arm order must not matter. The hazard is the geometry mismatch, not which
/// side of the pipe the narrow arm sits on.
#[test]
fn the_refusal_does_not_depend_on_which_arm_is_narrow() {
    let message = geometry_error(fuse_programs(&[
        barrier_arm("narrow", 4),
        independent_arm("wide", 256),
    ]));
    assert!(
        message.contains("arm 0"),
        "the narrow arm is arm 0 here: {message}"
    );
}

/// Workgroup memory alone is enough. The buffer is sized for the arm's own
/// width, so a wider workgroup has invocations indexing past it even with no
/// barrier anywhere in the arm.
#[test]
fn workgroup_memory_alone_blocks_the_widening() {
    let message = geometry_error(fuse_programs(&[
        independent_arm("wide", 256),
        workgroup_memory_arm("narrow", 4),
    ]));
    assert!(
        message.contains("keeps state in workgroup memory"),
        "the refusal must name workgroup memory as the reason: {message}"
    );
    assert!(
        !message.contains("synchronizes"),
        "this arm has no barrier, so the reason must not claim one: {message}"
    );
}

/// Both together are reported together, so the caller fixing one does not
/// discover the other on the next attempt.
#[test]
fn an_arm_that_both_synchronizes_and_uses_workgroup_memory_reports_both() {
    let narrow = workgroup_memory_arm("narrow", 4).with_rewritten_entry(vec![Node::Region {
        generator: "test.narrow".into(),
        source_region: None,
        body: Arc::new(vec![
            Node::store(
                "narrow_scratch",
                Expr::gid_x(),
                Expr::load("narrow_in", Expr::gid_x()),
            ),
            Node::barrier_with_ordering(MemoryOrdering::SeqCst),
            Node::store(
                "narrow_out",
                Expr::gid_x(),
                Expr::load("narrow_scratch", Expr::gid_x()),
            ),
        ]),
    }]);
    let message = geometry_error(fuse_programs(&[independent_arm("wide", 256), narrow]));
    assert!(
        message.contains("keeps state in workgroup memory and synchronizes its workgroup"),
        "both reasons belong in one message: {message}"
    );
}

/// The check is about a CHANGE in geometry, not about barriers as such. Two
/// arms that already agree on their workgroup fuse normally even when both
/// synchronize.
#[test]
fn matching_workgroups_fuse_even_when_both_arms_synchronize() {
    let fused = fuse_programs(&[barrier_arm("first", 64), barrier_arm("second", 64)])
        .expect("arms that agree on their workgroup have no geometry hazard");
    assert_eq!(
        fused.workgroup_size(),
        [64, 1, 1],
        "the fused geometry is the shared one, unchanged"
    );
}

/// An arm with no workgroup-scoped reasoning is still widened, because for it
/// the fused geometry is only a launch-size change. Refusing here would cost
/// the fuser its whole purpose.
#[test]
fn independent_arms_are_still_widened() {
    let fused = fuse_programs(&[independent_arm("wide", 256), independent_arm("narrow", 4)])
        .expect("independent arms carry no workgroup-scoped assumption");
    assert_eq!(
        fused.workgroup_size(),
        [256, 1, 1],
        "the fused geometry is the axis-wise maximum"
    );
}

/// A single program is returned verbatim, so there is no geometry to reconcile
/// and no refusal to make.
#[test]
fn a_lone_arm_is_never_refused() {
    let fused = fuse_programs(&[barrier_arm("only", 4)]).expect("one arm cannot mismatch itself");
    assert_eq!(fused.workgroup_size(), [4, 1, 1]);
}

/// The widening is only unsafe on the axis that actually changes, but the
/// check is deliberately conservative across all three: an arm built for
/// [4, 1, 1] reasoning about its workgroup is equally broken under [4, 8, 1],
/// where its workgroup holds 32 invocations rather than 4.
#[test]
fn a_widening_on_a_different_axis_is_refused_too() {
    let wide = independent_arm("wide", 4);
    let wide = Program::wrapped(wide.buffers().to_vec(), [4, 8, 1], wide.entry().to_vec());
    let message = geometry_error(fuse_programs(&[wide, barrier_arm("narrow", 4)]));
    assert!(
        message.contains("[4, 8, 1]"),
        "the fused geometry on the changed axis belongs in the message: {message}"
    );
}

/// A serial body admitted by workgroup identity is the third unsafe shape.
///
/// Measured divergence this locks out: `vyre-libs::nn::top_k`, workgroup
/// [1, 1, 1], fused behind a 256-wide elementwise arm returned index 7 where
/// the reference returned index 6. Every invocation of workgroup 0 passed the
/// guard and ran the same insertion scan over the same scratch, so the result
/// named one input lane twice. Nothing in the arm declares workgroup memory or
/// a barrier, so the first two shapes do not see it.
#[test]
fn a_serial_arm_guarded_on_its_workgroup_is_not_widened() {
    let message = geometry_error(fuse_programs(&[
        independent_arm("wide", 256),
        serial_workgroup_guard_arm("serial", 1),
    ]));
    assert!(
        message.contains("arm 1") && message.contains("[256, 1, 1]") && message.contains("[1, 1, 1]"),
        "the refusal must name the arm and both geometries: {message}"
    );
    assert!(
        message.contains("read-write storage") && message.contains("workgroup identity"),
        "the refusal must say what makes the widening unsafe: {message}"
    );
}

/// The same arm guarded on its invocation is widened.
///
/// This is the fix the refusal asks for, so it has to be accepted. A rule that
/// refused both forms would make every serial arm permanently unfusable and
/// leave a caller no way out.
#[test]
fn a_serial_arm_guarded_on_its_invocation_is_still_widened() {
    let fused = fuse_programs(&[
        independent_arm("wide", 256),
        serial_invocation_guard_arm("serial", 1),
    ])
    .expect("an invocation-exact serial arm is safe under any workgroup");
    assert_eq!(
        fused.workgroup_size(),
        [256, 1, 1],
        "the fused launch is the axis-wise maximum over the arms"
    );
}

/// Read-write storage alone is not the hazard.
///
/// `independent_arm` writes read-write storage at its own global index. Every
/// invocation addressing its own element is exactly what a wider workgroup is
/// for, so treating the buffer access as the hazard would refuse every fusion
/// this fuser exists to make.
#[test]
fn read_write_storage_without_an_identity_guard_is_widened() {
    let fused = fuse_programs(&[independent_arm("wide", 256), independent_arm("narrow", 4)])
        .expect("independent arms fuse regardless of buffer access");
    assert_eq!(fused.workgroup_size(), [256, 1, 1]);
}
