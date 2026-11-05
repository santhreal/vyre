//! Generated adversarial contract tests for vyre-primitives.

// Adversarial tests for bitset::contains

#![allow(
    unused_imports,
    unused_variables,
    dead_code,
    unused_mut,
    unused_macros,
    clippy::identity_op,
    clippy::assertions_on_constants
)]

use crate::gate_fixtures;

use vyre_libs_bitset::bitset::contains::*;

fn cpu_ref(input: &[u32], index: u32) -> u32 {
    vyre_reference::composition_witness::bitset_test_bit_witness(input, index)
}

vyre_test_support::adversarial_cpu_ref_cases! {
    test_bitset_contains_adv_0: &[0u32; 0], 0u32 => 0u32, "FINDING-ADV-BITSET-CONTAINS-0: Exact bit output mismatch";
    test_bitset_contains_adv_1: &[0u32; 0], 4294967295u32 => 0u32, "FINDING-ADV-BITSET-CONTAINS-1: Exact bit output mismatch";
    test_bitset_contains_adv_2: &[0u32; 0], 31u32 => 0u32, "FINDING-ADV-BITSET-CONTAINS-2: Exact bit output mismatch";
    test_bitset_contains_adv_3: &[0u32; 0], 0u32 => 0u32, "FINDING-ADV-BITSET-CONTAINS-3: Exact bit output mismatch";
    test_bitset_contains_adv_4: &[0u32; 0], 4294967295u32 => 0u32, "FINDING-ADV-BITSET-CONTAINS-4: Exact bit output mismatch";
    test_bitset_contains_adv_5: &[0u32; 0], 31u32 => 0u32, "FINDING-ADV-BITSET-CONTAINS-5: Exact bit output mismatch";
    test_bitset_contains_adv_6: &[0u32; 0], 0u32 => 0u32, "FINDING-ADV-BITSET-CONTAINS-6: Exact bit output mismatch";
    test_bitset_contains_adv_7: &[0u32; 0], 4294967295u32 => 0u32, "FINDING-ADV-BITSET-CONTAINS-7: Exact bit output mismatch";
    test_bitset_contains_adv_8: &[0u32; 0], 31u32 => 0u32, "FINDING-ADV-BITSET-CONTAINS-8: Exact bit output mismatch";
    test_bitset_contains_adv_9: &[0u32; 0], 0u32 => 0u32, "FINDING-ADV-BITSET-CONTAINS-9: Exact bit output mismatch";
    test_bitset_contains_adv_10: &[0u32; 0], 4294967295u32 => 0u32, "FINDING-ADV-BITSET-CONTAINS-10: Exact bit output mismatch";
    test_bitset_contains_adv_11: &[0u32; 0], 31u32 => 0u32, "FINDING-ADV-BITSET-CONTAINS-11: Exact bit output mismatch";
    test_bitset_contains_adv_12: &[0u32; 0], 0u32 => 0u32, "FINDING-ADV-BITSET-CONTAINS-12: Exact bit output mismatch";
    test_bitset_contains_adv_13: &[0u32; 0], 4294967295u32 => 0u32, "FINDING-ADV-BITSET-CONTAINS-13: Exact bit output mismatch";
    test_bitset_contains_adv_14: &[0u32; 0], 31u32 => 0u32, "FINDING-ADV-BITSET-CONTAINS-14: Exact bit output mismatch";
    test_bitset_contains_adv_15: &[0u32; 0], 0u32 => 0u32, "FINDING-ADV-BITSET-CONTAINS-15: Exact bit output mismatch";
    test_bitset_contains_adv_16: &[0u32; 0], 4294967295u32 => 0u32, "FINDING-ADV-BITSET-CONTAINS-16: Exact bit output mismatch";
    test_bitset_contains_adv_17: &[0u32; 0], 31u32 => 0u32, "FINDING-ADV-BITSET-CONTAINS-17: Exact bit output mismatch";
    test_bitset_contains_adv_18: &[0u32; 0], 0u32 => 0u32, "FINDING-ADV-BITSET-CONTAINS-18: Exact bit output mismatch";
    test_bitset_contains_adv_19: &[0u32; 0], 4294967295u32 => 0u32, "FINDING-ADV-BITSET-CONTAINS-19: Exact bit output mismatch";
    test_bitset_contains_adv_20: &[0u32; 0], 31u32 => 0u32, "FINDING-ADV-BITSET-CONTAINS-20: Exact bit output mismatch";
    test_bitset_contains_adv_21: &[0u32; 0], 0u32 => 0u32, "FINDING-ADV-BITSET-CONTAINS-21: Exact bit output mismatch";
    test_bitset_contains_adv_22: &[0u32; 0], 4294967295u32 => 0u32, "FINDING-ADV-BITSET-CONTAINS-22: Exact bit output mismatch";
    test_bitset_contains_adv_23: &[0u32; 0], 31u32 => 0u32, "FINDING-ADV-BITSET-CONTAINS-23: Exact bit output mismatch";
    test_bitset_contains_adv_24: &[0u32; 0], 0u32 => 0u32, "FINDING-ADV-BITSET-CONTAINS-24: Exact bit output mismatch";
    test_bitset_contains_adv_25: &[0u32; 0], 4294967295u32 => 0u32, "FINDING-ADV-BITSET-CONTAINS-25: Exact bit output mismatch";
    test_bitset_contains_adv_26: &[0u32; 0], 31u32 => 0u32, "FINDING-ADV-BITSET-CONTAINS-26: Exact bit output mismatch";
    test_bitset_contains_adv_27: &[0u32; 0], 0u32 => 0u32, "FINDING-ADV-BITSET-CONTAINS-27: Exact bit output mismatch";
    test_bitset_contains_adv_28: &[0u32; 0], 4294967295u32 => 0u32, "FINDING-ADV-BITSET-CONTAINS-28: Exact bit output mismatch";
    test_bitset_contains_adv_29: &[0u32; 0], 31u32 => 0u32, "FINDING-ADV-BITSET-CONTAINS-29: Exact bit output mismatch";
    test_bitset_contains_adv_30: &[0u32; 0], 0u32 => 0u32, "FINDING-ADV-BITSET-CONTAINS-30: Exact bit output mismatch";
    test_bitset_contains_adv_31: &[0u32; 0], 4294967295u32 => 0u32, "FINDING-ADV-BITSET-CONTAINS-31: Exact bit output mismatch";
    test_bitset_contains_adv_32: &[0u32; 0], 31u32 => 0u32, "FINDING-ADV-BITSET-CONTAINS-32: Exact bit output mismatch";
    test_bitset_contains_adv_33: &[0u32; 0], 0u32 => 0u32, "FINDING-ADV-BITSET-CONTAINS-33: Exact bit output mismatch";
    test_bitset_contains_adv_34: &[0u32; 0], 4294967295u32 => 0u32, "FINDING-ADV-BITSET-CONTAINS-34: Exact bit output mismatch";
    test_bitset_contains_adv_35: &[0u32; 0], 31u32 => 0u32, "FINDING-ADV-BITSET-CONTAINS-35: Exact bit output mismatch";
    test_bitset_contains_adv_36: &[0u32; 0], 0u32 => 0u32, "FINDING-ADV-BITSET-CONTAINS-36: Exact bit output mismatch";
    test_bitset_contains_adv_37: &[0u32; 0], 4294967295u32 => 0u32, "FINDING-ADV-BITSET-CONTAINS-37: Exact bit output mismatch";
    test_bitset_contains_adv_38: &[0u32; 0], 31u32 => 0u32, "FINDING-ADV-BITSET-CONTAINS-38: Exact bit output mismatch";
    test_bitset_contains_adv_39: &[0u32; 0], 0u32 => 0u32, "FINDING-ADV-BITSET-CONTAINS-39: Exact bit output mismatch";
    test_bitset_contains_adv_40: &[0u32; 0], 4294967295u32 => 0u32, "FINDING-ADV-BITSET-CONTAINS-40: Exact bit output mismatch";
    test_bitset_contains_adv_41: &[0u32; 0], 31u32 => 0u32, "FINDING-ADV-BITSET-CONTAINS-41: Exact bit output mismatch";
    test_bitset_contains_adv_42: &[0u32; 0], 0u32 => 0u32, "FINDING-ADV-BITSET-CONTAINS-42: Exact bit output mismatch";
    test_bitset_contains_adv_43: &[0u32; 0], 4294967295u32 => 0u32, "FINDING-ADV-BITSET-CONTAINS-43: Exact bit output mismatch";
    test_bitset_contains_adv_44: &[0u32; 0], 31u32 => 0u32, "FINDING-ADV-BITSET-CONTAINS-44: Exact bit output mismatch";
    test_bitset_contains_adv_45: &[1u32; 0], 0u32 => 0u32, "FINDING-ADV-BITSET-CONTAINS-45: Exact bit output mismatch";
    test_bitset_contains_adv_46: &[1u32; 0], 4294967295u32 => 0u32, "FINDING-ADV-BITSET-CONTAINS-46: Exact bit output mismatch";
    test_bitset_contains_adv_47: &[1u32; 0], 31u32 => 0u32, "FINDING-ADV-BITSET-CONTAINS-47: Exact bit output mismatch";
    test_bitset_contains_adv_48: &[1u32; 0], 0u32 => 0u32, "FINDING-ADV-BITSET-CONTAINS-48: Exact bit output mismatch";
    test_bitset_contains_adv_49: &[1u32; 0], 4294967295u32 => 0u32, "FINDING-ADV-BITSET-CONTAINS-49: Exact bit output mismatch";
    test_bitset_contains_adv_50: &[1u32; 0], 31u32 => 0u32, "FINDING-ADV-BITSET-CONTAINS-50: Exact bit output mismatch";
    test_bitset_contains_adv_51: &[1u32; 0], 0u32 => 0u32, "FINDING-ADV-BITSET-CONTAINS-51: Exact bit output mismatch";
    test_bitset_contains_adv_52: &[1u32; 0], 4294967295u32 => 0u32, "FINDING-ADV-BITSET-CONTAINS-52: Exact bit output mismatch";
    test_bitset_contains_adv_53: &[1u32; 0], 31u32 => 0u32, "FINDING-ADV-BITSET-CONTAINS-53: Exact bit output mismatch";
    test_bitset_contains_adv_54: &[1u32; 0], 0u32 => 0u32, "FINDING-ADV-BITSET-CONTAINS-54: Exact bit output mismatch";
    test_bitset_contains_adv_55: &[1u32; 0], 4294967295u32 => 0u32, "FINDING-ADV-BITSET-CONTAINS-55: Exact bit output mismatch";
    test_bitset_contains_adv_56: &[1u32; 0], 31u32 => 0u32, "FINDING-ADV-BITSET-CONTAINS-56: Exact bit output mismatch";
    test_bitset_contains_adv_57: &[1u32; 0], 0u32 => 0u32, "FINDING-ADV-BITSET-CONTAINS-57: Exact bit output mismatch";
    test_bitset_contains_adv_58: &[1u32; 0], 4294967295u32 => 0u32, "FINDING-ADV-BITSET-CONTAINS-58: Exact bit output mismatch";
    test_bitset_contains_adv_59: &[1u32; 0], 31u32 => 0u32, "FINDING-ADV-BITSET-CONTAINS-59: Exact bit output mismatch";
    test_bitset_contains_adv_60: &[1u32; 0], 0u32 => 0u32, "FINDING-ADV-BITSET-CONTAINS-60: Exact bit output mismatch";
    test_bitset_contains_adv_61: &[1u32; 0], 4294967295u32 => 0u32, "FINDING-ADV-BITSET-CONTAINS-61: Exact bit output mismatch";
    test_bitset_contains_adv_62: &[1u32; 0], 31u32 => 0u32, "FINDING-ADV-BITSET-CONTAINS-62: Exact bit output mismatch";
    test_bitset_contains_adv_63: &[1u32; 0], 0u32 => 0u32, "FINDING-ADV-BITSET-CONTAINS-63: Exact bit output mismatch";
    test_bitset_contains_adv_64: &[1u32; 0], 4294967295u32 => 0u32, "FINDING-ADV-BITSET-CONTAINS-64: Exact bit output mismatch";
    test_bitset_contains_adv_65: &[1u32; 0], 31u32 => 0u32, "FINDING-ADV-BITSET-CONTAINS-65: Exact bit output mismatch";
    test_bitset_contains_adv_66: &[1u32; 0], 0u32 => 0u32, "FINDING-ADV-BITSET-CONTAINS-66: Exact bit output mismatch";
    test_bitset_contains_adv_67: &[1u32; 0], 4294967295u32 => 0u32, "FINDING-ADV-BITSET-CONTAINS-67: Exact bit output mismatch";
    test_bitset_contains_adv_68: &[1u32; 0], 31u32 => 0u32, "FINDING-ADV-BITSET-CONTAINS-68: Exact bit output mismatch";
    test_bitset_contains_adv_69: &[1u32; 0], 0u32 => 0u32, "FINDING-ADV-BITSET-CONTAINS-69: Exact bit output mismatch";
}
