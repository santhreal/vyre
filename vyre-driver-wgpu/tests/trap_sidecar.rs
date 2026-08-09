//! WGPU trap sidecar integration tests.

use vyre::scan::dispatch_io::pack_u32_slice as pack_words;
use vyre_driver::VyreBackend;
use vyre_driver_wgpu::WgpuBackend;

#[test]
fn inflate_fixed_huffman_reports_wgpu_trap_tag() {
    let backend = WgpuBackend::acquire().expect("Fix: GPU required for WGPU trap sidecar test");
    let program = vyre_libs::decode::inflate_stored_block("input", "output", 5);
    let input = pack_words(&[0x03, 0, 0, 0, 0]);
    let len_sidecar = vec![0u8; 4];

    let error = backend
        .dispatch(
            &program,
            &[input, len_sidecar],
            &vyre_driver::DispatchConfig::default(),
        )
        .expect_err("Fix: BTYPE=1 must propagate Node::Trap through WGPU.");
    let message = error.to_string();
    // What the sidecar has to preserve is the trap's OWN tag, verbatim, from
    // the shader that raised it. The assertion below therefore checks the
    // pieces that identify the trap, not a particular phrasing: the wgpu
    // framing, the operation that refused the input, the BTYPE that was
    // rejected, and an actionable Fix clause. An earlier version required the
    // word "fixed-Huffman" and failed once the tag was reworded to name the
    // op and the remedy instead, which is strictly more useful to an operator.
    for needle in [
        "wgpu dispatch trapped",
        "vyre-primitives::decode::inflate_stored",
        "BTYPE=1",
        "Fix:",
    ] {
        assert!(
            message.contains(needle),
            "Fix: trap sidecar must carry `{needle}` from the original tag, got: {message}",
        );
    }
    // The sidecar's own framing must survive too: which lane trapped, and the
    // tag code that maps back to the trap table.
    assert!(
        message.contains("lane=") && message.contains("tag_code="),
        "Fix: trap sidecar must report the lane and tag code, got: {message}",
    );
}
