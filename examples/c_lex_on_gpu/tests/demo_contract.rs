//! Contract tests for the standalone `c_lex_on_gpu` demo crate.

use std::process::Command;

#[test]
fn demo_runs_on_gpu_and_reports_ast_coverage() {
    let output = Command::new(env!("CARGO_BIN_EXE_c_lex_on_gpu"))
        .arg("1")
        .output()
        .expect("c_lex_on_gpu demo binary should launch");

    assert!(
        output.status.success(),
        "c_lex_on_gpu should parse its default C translation unit on GPU. stderr:\n{}",
        String::from_utf8_lossy(&output.stderr)
    );

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("backend:          cuda"),
        "demo must use the CUDA release backend on this GPU fleet. stdout:\n{stdout}"
    );
    assert!(
        stdout.contains("AST coverage:"),
        "demo must report AST coverage evidence. stdout:\n{stdout}"
    );
    assert!(
        stdout.contains("There is no CPU fallback path."),
        "demo must state the GPU-only execution contract. stdout:\n{stdout}"
    );
}
