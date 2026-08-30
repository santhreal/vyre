//! Smoke tests for target-neutral `vyre_aot::compile` behavior.

mod fixture_target;

use vyre_aot::{compile, emit_launcher_rust, CompileError, LauncherError, LauncherOpts, TargetId};
use vyre_foundation::ir::{BufferDecl, DataType, Expr, Node, Program};

fn trivial_xor_program() -> Program {
    Program::wrapped(
        vec![
            BufferDecl::read("a", 0, DataType::U32).with_count(1),
            BufferDecl::read("b", 1, DataType::U32).with_count(1),
            BufferDecl::read_write("out", 2, DataType::U32).with_count(1),
        ],
        [1, 1, 1],
        vec![
            Node::let_bind("idx", Expr::u32(0)),
            Node::store(
                "out",
                Expr::var("idx"),
                Expr::bitxor(
                    Expr::load("a", Expr::var("idx")),
                    Expr::load("b", Expr::var("idx")),
                ),
            ),
        ],
    )
}

#[test]
fn compile_requires_linked_target_compiler() {
    let p = trivial_xor_program();
    let target = TargetId::expect_valid("unlinked-fixture-target");
    let err = compile(&p, target.clone())
        .expect_err("Fix: vyre-aot must not emit target bytes without a linked target compiler.");
    assert!(
        matches!(&err, CompileError::TargetNotEnabled(id) if id == &target),
        "Fix: missing target compiler must report target-not-enabled, got {err:?}."
    );
}

#[test]
fn launcher_requires_linked_target_emitter() {
    let artifact = minimal_ptx_artifact_for_template_test();
    let opts = LauncherOpts::default();
    let target = TargetId::expect_valid("unlinked-fixture-target");
    let err = emit_launcher_rust(&artifact, target.clone(), &opts)
        .expect_err("Fix: target launcher files must come from linked driver crates.");
    assert!(
        matches!(&err, LauncherError::TargetNotEnabled(id) if id == &target),
        "Fix: missing launcher emitter must report target-not-enabled, got {err:?}."
    );
}

fn minimal_ptx_artifact_for_template_test() -> vyre_aot::ArtifactEnvelope {
    fixture_target::compiled_artifact()
}

/// A program declaring workgroup-scoped scratch, which the shared-memory
/// capability arm of the admission gate reads.
fn workgroup_scratch_program() -> Program {
    Program::wrapped(
        vec![
            BufferDecl::read_write("out", 0, DataType::U32).with_count(1),
            BufferDecl::workgroup("tile", 8, DataType::U32),
        ],
        [32, 1, 1],
        vec![
            Node::store("tile", Expr::u32(0), Expr::u32(1)),
            Node::store("out", Expr::u32(0), Expr::u32(2)),
        ],
    )
}

/// WHY: 130. The neutral half of every artifact is compiled against
/// `DeviceFacts::unknown`, which states no capability snapshot. Reading that
/// absence as a device that grants nothing refused every program declaring
/// workgroup scratch here with `MKC001_INVALID_PROGRAM`, before any target had
/// been selected, while the artifact identity this path produces must stay
/// device-neutral and so cannot hold a real device's facts instead.
///
/// Against the previous behaviour this failed at the `neutral-request` stage.
#[test]
fn a_neutral_compile_admits_workgroup_scratch_when_no_snapshot_is_stated() {
    compile(
        &workgroup_scratch_program(),
        fixture_target::fixture_target(),
    )
    .expect("Fix: a device-neutral compile must not judge an unstated capability");
}

/// The same program against a target that is not linked, so the neutral stage is
/// isolated from every target decision: the only refusal left is the missing
/// target compiler, which `compile` reaches only after the neutral artifact.
#[test]
fn the_neutral_stage_admits_workgroup_scratch_before_any_target_is_resolved() {
    let target = TargetId::expect_valid("unlinked-fixture-target");
    let error = compile(&workgroup_scratch_program(), target.clone())
        .expect_err("an unlinked target cannot emit bytes");
    assert!(
        matches!(&error, CompileError::TargetNotEnabled(id) if id == &target),
        "Fix: the neutral artifact must be built before the target is resolved, got {error:?}."
    );
}
