use std::process::Command;

#[test]
fn demo_reports_backend_neutral_source_to_ir_evidence() {
    let output = Command::new(env!("CARGO_BIN_EXE_c_source_to_ir"))
        .arg("4")
        .output()
        .expect("C source-to-IR example binary should launch");

    assert!(
        output.status.success(),
        "C source-to-IR example should lower its translation unit. stderr:\n{}",
        String::from_utf8_lossy(&output.stderr)
    );

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("source bytes:"), "stdout:\n{stdout}");
    assert!(stdout.contains("syntax nodes:"), "stdout:\n{stdout}");
    assert!(stdout.contains("IR statements: 1"), "stdout:\n{stdout}");
    assert!(stdout.contains("source-to-IR:"), "stdout:\n{stdout}");
    assert!(
        !stdout.contains("backend:"),
        "frontend example must not select an execution backend. stdout:\n{stdout}"
    );
}
