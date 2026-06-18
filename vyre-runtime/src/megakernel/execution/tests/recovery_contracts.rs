use super::*;

#[test]
fn recovery_retry_preserves_caller_output_slots() {
    let mut outputs = vec![Vec::with_capacity(8)];
    let outputs_addr = outputs.as_ptr() as usize;
    let slot_addr = outputs[0].as_ptr() as usize;
    let dispatch_calls = Arc::new(AtomicUsize::new(0));
    let backend = Arc::new(RecoveringBackend {
        dispatch_calls: Arc::clone(&dispatch_calls),
        expected_outputs_addr: outputs_addr,
        expected_slot_addr: slot_addr,
    });
    let kernel = Megakernel::bootstrap_sharded(backend, 1, 1, Vec::new())
        .expect("Fix: recovering backend must compile megakernel");
    let control = Megakernel::try_encode_control(false, 1, 0).unwrap();
    let ring = Megakernel::try_encode_empty_ring(1).unwrap();
    let debug = Megakernel::try_encode_empty_debug_log(protocol::debug::RECORD_CAPACITY).unwrap();
    let io_queue = io::try_encode_empty_io_queue(io::IO_SLOT_COUNT).unwrap();

    let stats = kernel
        .dispatch_with_io_queue_borrowed_into(&control, &ring, &debug, &io_queue, &mut outputs)
        .expect("Fix: recovery retry must reuse caller-owned output storage");

    assert!(stats.recovered_after_device_loss);
    assert_eq!(dispatch_calls.load(Ordering::SeqCst), 2);
    assert_eq!(outputs, vec![vec![9, 8, 7, 6]]);
    assert_eq!(outputs.as_ptr() as usize, outputs_addr);
    assert_eq!(outputs[0].as_ptr() as usize, slot_addr);
}

