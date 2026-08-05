//! End-to-end proof that C frontend IR executes only after a driver consumes it.

use vyre_driver::{DispatchConfig, VyreBackend};
use vyre_driver_reference::CpuRefBackend;
use vyre_frontend_c::lower_source;

#[test]
fn scalar_c_ir_executes_through_reference_driver() {
    let program = lower_source("unsigned int kernel(void) { return 6u * 7u; }")
        .expect("frontend must lower supported C without selecting a backend");

    let outputs = CpuRefBackend
        .dispatch(&program, &[], &DispatchConfig::default())
        .expect("upper driver harness must execute frontend IR");

    assert_eq!(outputs, vec![42u32.to_le_bytes().to_vec()]);
}

#[test]
fn buffer_c_ir_executes_with_driver_owned_inputs() {
    let program = lower_source(
        "void kernel(const unsigned int *input, unsigned int *output) { output[0] = input[0] * 2u; }",
    )
    .expect("frontend must preserve the typed input/output contract");

    let inputs = [21u32.to_le_bytes().to_vec()];
    let outputs = CpuRefBackend
        .dispatch(&program, &inputs, &DispatchConfig::default())
        .expect("driver must bind inputs and execute separately");

    assert_eq!(outputs, vec![42u32.to_le_bytes().to_vec()]);
}
