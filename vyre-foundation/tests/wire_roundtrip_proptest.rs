//! Wire-format roundtrip proptest. Shared generators and invariant cases stay
//! in one flat scope because the support file opens the `proptest!` block used
//! by the included cases.
#![allow(dead_code)]
mod wire_roundtrip_proptest_suite {
    include!("contract_cases/wire_roundtrip_proptest_support__extension_kind.rs");
    include!("contract_cases/wire_roundtrip_proptest_support__arb_node.rs");
    include!(
        "contract_cases/wire_roundtrip_proptest__program_wire_roundtrip_preserves_structure.rs"
    );
    include!("contract_cases/wire_roundtrip_proptest__every_expression_variant_roundtrips_in_one_program.rs");
}
