use external_backend_extension::{build_probe_program, manifest, probe_wire};
use vyre::ir::Program;

#[test]
fn external_backend_probe_builds_public_ir_and_wire() {
    let manifest = manifest();
    assert_eq!(manifest.id, "example.external.backend");
    assert_eq!(manifest.version, env!("CARGO_PKG_VERSION"));

    let wire = probe_wire().expect("external backend probe program must serialize");
    let decoded = Program::from_wire(&wire).expect("external backend probe wire must decode");

    assert_eq!(decoded.buffers().len(), 1);
    assert_eq!(decoded.entry().len(), 1);
}

#[test]
fn probe_program_keeps_single_output_contract() {
    let program = build_probe_program();
    let buffers = program.buffers();

    assert_eq!(buffers.len(), 1);
    assert_eq!(buffers[0].name(), "out");
    assert_eq!(buffers[0].count(), 1);
}
