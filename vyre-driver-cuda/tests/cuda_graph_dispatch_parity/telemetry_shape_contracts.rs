#![cfg(feature = "device-tests")]

use super::*;

#[test]
fn cuda_graph_recording_accounts_raw_device_allocations() {
    let backend = Arc::new(
        CudaBackend::acquire()
            .expect("Fix: CUDA backend acquisition must succeed on the GPU-required test host."),
    );

    let program = add_one_program();
    let inputs = vec![u32_bytes(&[0, 1, 2, 3, 4, 5, 6, 7])];
    let config = DispatchConfig::default();
    backend.reset_telemetry();

    let _cached = backend
        .record_cuda_graph(&program, &inputs, &config)
        .expect("Fix: cudaGraph recording must succeed for the add-one telemetry contract.");

    let telemetry = backend.telemetry_snapshot();
    assert!(
        telemetry.transient_allocation_bytes_requested >= 64,
        "Fix: cudaGraph recording allocates raw input/output device buffers outside the \
         transient pool; telemetry must include at least the 32-byte input and 32-byte output \
         buffers instead of underreporting CUDA memory pressure. observed={}",
        telemetry.transient_allocation_bytes_requested
    );
    assert_eq!(
        telemetry.sync_points, 1,
        "Fix: cudaGraph recording must account for the parameter-initialization stream synchronization exactly once."
    );
}

#[test]
fn cuda_graph_rejects_input_shape_mismatch() {
    let backend = Arc::new(
        CudaBackend::acquire()
            .expect("Fix: CUDA backend acquisition must succeed on the GPU-required test host."),
    );

    let program = add_one_program();
    let inputs = vec![u32_bytes(&[0; 8])];
    let config = DispatchConfig::default();

    let mut cached = backend
        .record_cuda_graph(&program, &inputs, &config)
        .expect("record must succeed");

    // Try replay with WRONG-LENGTH input.
    let bad_inputs = [u32_bytes(&[0; 4])]; // half the recorded size
    let bad_refs: Vec<&[u8]> = bad_inputs.iter().map(Vec::as_slice).collect();
    match backend.dispatch_via_cuda_graph(&mut cached, &bad_refs) {
        Err(BackendError::InvalidProgram { fix }) => {
            assert!(
                fix.contains("re-record") || fix.contains("expects"),
                "rejection error must mention the size mismatch + tell the user to re-record \
                 the graph; got: {fix}"
            );
        }
        Ok(_) => panic!(
            "cuda_graph dispatch must NOT silently accept inputs of the wrong byte length; \
             expected BackendError::InvalidProgram with a structured fix string"
        ),
        Err(other) => panic!(
            "cuda_graph dispatch with mismatched input size must return InvalidProgram, \
             not {other:?}"
        ),
    }
}
