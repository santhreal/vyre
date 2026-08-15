use super::*;

/// Every semantic-gap case, on the GPU, through all four stages.
///
/// One test rather than one per case and per stage. The case list is
/// `semantic_gap_constructs::CASES`, which the CPU arm in `vyre-libs/tests`
/// iterates as well, so a construct cannot reach one arm and miss the other. The
/// per-case `#[test]` functions this replaces named four of the six cases for the
/// classifier and only two for the property-graph lowerer, and named
/// `inner_typedef_shadows_outer` nowhere: the one case in this family whose whole
/// point is scope-dependent typedef visibility went unproven on the GPU.
#[test]
fn gpu_parity_semantic_gap_cases() {
    assert_family_parity(&GpuArm, semantic_gap_constructs::CASES);
}
