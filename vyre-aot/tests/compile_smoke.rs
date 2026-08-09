//! Smoke tests for target-neutral `vyre_aot::compile` behavior.

mod common;

use vyre_aot::artifact::Target;
use vyre_aot::{compile, emit_launcher_rust, CompileError, LauncherError, LauncherOpts};
use vyre_foundation::ir::{BufferDecl, DataType, Expr, Node, Program};

fn trivial_xor_program() -> Program {
    Program::wrapped(
        vec![
            BufferDecl::read("a", 0, DataType::U32),
            BufferDecl::read("b", 1, DataType::U32),
            BufferDecl::read_write("out", 2, DataType::U32),
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
    let err = compile(&p, Target::Ptx)
        .expect_err("Fix: vyre-aot must not emit target bytes without a linked target compiler.");
    assert!(
        matches!(err, CompileError::TargetNotEnabled(Target::Ptx)),
        "Fix: missing target compiler must report target-not-enabled, got {err:?}."
    );
}

#[test]
fn launcher_requires_linked_target_emitter() {
    let artifact = minimal_ptx_artifact_for_template_test();
    let opts = LauncherOpts::default();
    let err = emit_launcher_rust(&artifact, Target::Ptx, &opts)
        .expect_err("Fix: target launcher files must come from linked driver crates.");
    assert!(
        matches!(err, LauncherError::TargetNotEnabled("secondary_text")),
        "Fix: missing launcher emitter must report target-not-enabled, got {err:?}."
    );
}

fn minimal_ptx_artifact_for_template_test() -> vyre_aot::ArtifactEnvelope {
    common::compiled_artifact()
}
