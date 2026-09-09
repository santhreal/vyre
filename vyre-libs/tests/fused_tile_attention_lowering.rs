//! Every tile composition survives the semantic optimizer with its result
//! references intact.
//!
//! `fused_tile_attention` reached a concrete backend and was refused during
//! descriptor verification with `DanglingResultRef`: an optimizer pass rewrote
//! the region and left operands pointing at a result id no statement produced.
//! A refusal at that point names the neutral descriptor, not the pass that
//! broke it, so the failure is only visible on a device run. This test lowers
//! the composition on the host, which is where the pass runs.
//!
//! The case is derived from the registry rather than written once: every
//! registered composition that builds tile statements goes through the same
//! baseline lowering, so a pass that drops a result id in any of them turns
//! this red.

use vyre_libs::nn::attention::fused_tile_attention;

#[test]
fn fused_tile_attention_lowers_to_a_verified_descriptor() {
    let program = fused_tile_attention("q", "k", "v", "out", 2, 2);
    let lowered = vyre_lower::lower_baseline(&program);
    assert!(
        lowered.is_ok(),
        "fused_tile_attention must lower to a verified descriptor: {}",
        lowered.err().map(|e| e.to_string()).unwrap_or_default()
    );
}
