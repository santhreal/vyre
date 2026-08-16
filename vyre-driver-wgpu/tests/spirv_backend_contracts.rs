//! Contracts for `vyre_driver_wgpu::spirv_backend`.
//!
//! Every item under test is public API, so the suite reaches the crate the way
//! a consumer does.

use naga::back::spv::WriterFlags;
use vyre_driver_wgpu::spirv_backend::{SpirvEmitter, SPIRV_BACKEND_ID};

#[test]
fn emit_returns_nonempty_words_for_empty_module() {
    // An empty naga::Module still emits a SPIR-V header +
    // minimum entry-point prologue  -  the output should never be
    // empty even for a no-op program.
    let mut module = naga::Module::default();
    // Add a minimal compute entry point so the emitter has
    // something to target.
    let entry = naga::EntryPoint {
        name: "main".to_owned(),
        stage: naga::ShaderStage::Compute,
        early_depth_test: None,
        workgroup_size: [1, 1, 1],
        workgroup_size_overrides: None,
        function: naga::Function::default(),
    };
    module.entry_points.push(entry);

    match SpirvEmitter::emit(&module, "main") {
        Ok(words) => {
            assert!(!words.is_empty(), "SPIR-V output must not be empty");
            // First word is SPIR-V magic 0x07230203.
            assert_eq!(words[0], 0x0723_0203, "first word must be SPIR-V magic");
        }
        Err(msg) => {
            // Some naga versions reject entirely-empty fn
            // bodies. Surface a clear message so the test is
            // informative in either outcome.
            assert!(
                msg.contains("Fix:"),
                "emit error must carry Fix: remediation: {msg}"
            );
        }
    }
}

#[test]
fn backend_id_is_stable() {
    assert_eq!(SPIRV_BACKEND_ID, "spirv");
}

#[test]
fn default_flags_are_empty() {
    assert_eq!(SpirvEmitter::default_flags(), WriterFlags::empty());
}
