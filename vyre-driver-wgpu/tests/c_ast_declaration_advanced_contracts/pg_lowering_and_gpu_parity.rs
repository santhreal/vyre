use super::*;

/// Every advanced-declaration case, on the GPU, through all four stages.
///
/// One test rather than one per case and per stage. The case list is
/// `declaration_advanced_constructs::CASES`, which the CPU arm in
/// `vyre-libs/tests` iterates as well, so a construct cannot reach one arm and
/// miss the other. The per-case `#[test]` functions this replaces named all nine
/// cases for the builder, annotator and classifier but only eight for the
/// property-graph lowerer: `anonymous_struct_union` never reached it.
#[test]
fn gpu_parity_declaration_advanced_cases() {
    assert_family_parity(&GpuArm, declaration_advanced_constructs::CASES);
}
