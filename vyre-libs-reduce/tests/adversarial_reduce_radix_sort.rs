//! Generated adversarial contract tests for vyre-primitives.

// Adversarial tests for reduce::radix_sort

#![allow(
    unused_imports,
    unused_variables,
    dead_code,
    unused_mut,
    unused_macros,
    clippy::identity_op,
    clippy::assertions_on_constants
)]


use vyre_libs_reduce::reduce::radix_sort::*;
use vyre_reference::composition_witness::radix_sort_masked_witness as cpu_ref;

vyre_test_support::adversarial_cpu_ref_cases! {
    test_reduce_radix_sort_adv_0: &[0u32; 0], 0u32 => Vec::<u32>::new(), "FINDING-ADV-REDUCE-RADIX_SORT-0: Exact bit output mismatch";
    test_reduce_radix_sort_adv_1: &[0u32; 0], 0u32 => Vec::<u32>::new(), "FINDING-ADV-REDUCE-RADIX_SORT-1: Exact bit output mismatch";
    test_reduce_radix_sort_adv_2: &[0u32; 0], 0u32 => Vec::<u32>::new(), "FINDING-ADV-REDUCE-RADIX_SORT-2: Exact bit output mismatch";
    test_reduce_radix_sort_adv_3: &[0u32; 0], 0u32 => Vec::<u32>::new(), "FINDING-ADV-REDUCE-RADIX_SORT-3: Exact bit output mismatch";
    test_reduce_radix_sort_adv_4: &[0u32; 0], 0u32 => Vec::<u32>::new(), "FINDING-ADV-REDUCE-RADIX_SORT-4: Exact bit output mismatch";
    test_reduce_radix_sort_adv_5: &[0u32; 0], 0u32 => Vec::<u32>::new(), "FINDING-ADV-REDUCE-RADIX_SORT-5: Exact bit output mismatch";
    test_reduce_radix_sort_adv_6: &[0u32; 0], 0u32 => Vec::<u32>::new(), "FINDING-ADV-REDUCE-RADIX_SORT-6: Exact bit output mismatch";
    test_reduce_radix_sort_adv_7: &[0u32; 0], 0u32 => Vec::<u32>::new(), "FINDING-ADV-REDUCE-RADIX_SORT-7: Exact bit output mismatch";
    test_reduce_radix_sort_adv_8: &[0u32; 0], 0u32 => Vec::<u32>::new(), "FINDING-ADV-REDUCE-RADIX_SORT-8: Exact bit output mismatch";
    test_reduce_radix_sort_adv_9: &[0u32; 0], 0u32 => Vec::<u32>::new(), "FINDING-ADV-REDUCE-RADIX_SORT-9: Exact bit output mismatch";
    test_reduce_radix_sort_adv_10: &[0u32; 0], 0u32 => Vec::<u32>::new(), "FINDING-ADV-REDUCE-RADIX_SORT-10: Exact bit output mismatch";
    test_reduce_radix_sort_adv_11: &[0u32; 0], 0u32 => Vec::<u32>::new(), "FINDING-ADV-REDUCE-RADIX_SORT-11: Exact bit output mismatch";
    test_reduce_radix_sort_adv_12: &[0u32; 0], 0u32 => Vec::<u32>::new(), "FINDING-ADV-REDUCE-RADIX_SORT-12: Exact bit output mismatch";
    test_reduce_radix_sort_adv_13: &[0u32; 0], 0u32 => Vec::<u32>::new(), "FINDING-ADV-REDUCE-RADIX_SORT-13: Exact bit output mismatch";
    test_reduce_radix_sort_adv_14: &[0u32; 0], 0u32 => Vec::<u32>::new(), "FINDING-ADV-REDUCE-RADIX_SORT-14: Exact bit output mismatch";
    test_reduce_radix_sort_adv_15: &[0u32; 0], 0u32 => Vec::<u32>::new(), "FINDING-ADV-REDUCE-RADIX_SORT-15: Exact bit output mismatch";
    test_reduce_radix_sort_adv_16: &[0u32; 0], 0u32 => Vec::<u32>::new(), "FINDING-ADV-REDUCE-RADIX_SORT-16: Exact bit output mismatch";
    test_reduce_radix_sort_adv_17: &[0u32; 0], 0u32 => Vec::<u32>::new(), "FINDING-ADV-REDUCE-RADIX_SORT-17: Exact bit output mismatch";
    test_reduce_radix_sort_adv_18: &[0u32; 0], 0u32 => Vec::<u32>::new(), "FINDING-ADV-REDUCE-RADIX_SORT-18: Exact bit output mismatch";
    test_reduce_radix_sort_adv_19: &[0u32; 0], 0u32 => Vec::<u32>::new(), "FINDING-ADV-REDUCE-RADIX_SORT-19: Exact bit output mismatch";
    test_reduce_radix_sort_adv_20: &[0u32; 0], 0u32 => Vec::<u32>::new(), "FINDING-ADV-REDUCE-RADIX_SORT-20: Exact bit output mismatch";
    test_reduce_radix_sort_adv_21: &[0u32; 0], 0u32 => Vec::<u32>::new(), "FINDING-ADV-REDUCE-RADIX_SORT-21: Exact bit output mismatch";
    test_reduce_radix_sort_adv_22: &[0u32; 0], 0u32 => Vec::<u32>::new(), "FINDING-ADV-REDUCE-RADIX_SORT-22: Exact bit output mismatch";
    test_reduce_radix_sort_adv_23: &[0u32; 0], 0u32 => Vec::<u32>::new(), "FINDING-ADV-REDUCE-RADIX_SORT-23: Exact bit output mismatch";
    test_reduce_radix_sort_adv_24: &[0u32; 0], 0u32 => Vec::<u32>::new(), "FINDING-ADV-REDUCE-RADIX_SORT-24: Exact bit output mismatch";
    test_reduce_radix_sort_adv_25: &[0u32; 0], 0u32 => Vec::<u32>::new(), "FINDING-ADV-REDUCE-RADIX_SORT-25: Exact bit output mismatch";
    test_reduce_radix_sort_adv_26: &[0u32; 0], 0u32 => Vec::<u32>::new(), "FINDING-ADV-REDUCE-RADIX_SORT-26: Exact bit output mismatch";
    test_reduce_radix_sort_adv_27: &[0u32; 0], 0u32 => Vec::<u32>::new(), "FINDING-ADV-REDUCE-RADIX_SORT-27: Exact bit output mismatch";
    test_reduce_radix_sort_adv_28: &[0u32; 0], 0u32 => Vec::<u32>::new(), "FINDING-ADV-REDUCE-RADIX_SORT-28: Exact bit output mismatch";
    test_reduce_radix_sort_adv_29: &[0u32; 0], 0u32 => Vec::<u32>::new(), "FINDING-ADV-REDUCE-RADIX_SORT-29: Exact bit output mismatch";
    test_reduce_radix_sort_adv_30: &[0u32; 0], 0u32 => Vec::<u32>::new(), "FINDING-ADV-REDUCE-RADIX_SORT-30: Exact bit output mismatch";
    test_reduce_radix_sort_adv_31: &[0u32; 0], 0u32 => Vec::<u32>::new(), "FINDING-ADV-REDUCE-RADIX_SORT-31: Exact bit output mismatch";
    test_reduce_radix_sort_adv_32: &[0u32; 0], 0u32 => Vec::<u32>::new(), "FINDING-ADV-REDUCE-RADIX_SORT-32: Exact bit output mismatch";
    test_reduce_radix_sort_adv_33: &[0u32; 0], 0u32 => Vec::<u32>::new(), "FINDING-ADV-REDUCE-RADIX_SORT-33: Exact bit output mismatch";
    test_reduce_radix_sort_adv_34: &[0u32; 0], 0u32 => Vec::<u32>::new(), "FINDING-ADV-REDUCE-RADIX_SORT-34: Exact bit output mismatch";
    test_reduce_radix_sort_adv_35: &[0u32; 0], 0u32 => Vec::<u32>::new(), "FINDING-ADV-REDUCE-RADIX_SORT-35: Exact bit output mismatch";
    test_reduce_radix_sort_adv_36: &[0u32; 0], 0u32 => Vec::<u32>::new(), "FINDING-ADV-REDUCE-RADIX_SORT-36: Exact bit output mismatch";
    test_reduce_radix_sort_adv_37: &[0u32; 0], 0u32 => Vec::<u32>::new(), "FINDING-ADV-REDUCE-RADIX_SORT-37: Exact bit output mismatch";
    test_reduce_radix_sort_adv_38: &[0u32; 0], 0u32 => Vec::<u32>::new(), "FINDING-ADV-REDUCE-RADIX_SORT-38: Exact bit output mismatch";
    test_reduce_radix_sort_adv_39: &[0u32; 0], 0u32 => Vec::<u32>::new(), "FINDING-ADV-REDUCE-RADIX_SORT-39: Exact bit output mismatch";
    test_reduce_radix_sort_adv_40: &[0u32; 0], 0u32 => Vec::<u32>::new(), "FINDING-ADV-REDUCE-RADIX_SORT-40: Exact bit output mismatch";
    test_reduce_radix_sort_adv_41: &[0u32; 0], 0u32 => Vec::<u32>::new(), "FINDING-ADV-REDUCE-RADIX_SORT-41: Exact bit output mismatch";
    test_reduce_radix_sort_adv_42: &[0u32; 0], 0u32 => Vec::<u32>::new(), "FINDING-ADV-REDUCE-RADIX_SORT-42: Exact bit output mismatch";
    test_reduce_radix_sort_adv_43: &[0u32; 0], 0u32 => Vec::<u32>::new(), "FINDING-ADV-REDUCE-RADIX_SORT-43: Exact bit output mismatch";
    test_reduce_radix_sort_adv_44: &[0u32; 0], 0u32 => Vec::<u32>::new(), "FINDING-ADV-REDUCE-RADIX_SORT-44: Exact bit output mismatch";
    test_reduce_radix_sort_adv_45: &[1u32; 0], 1u32 => Vec::<u32>::new(), "FINDING-ADV-REDUCE-RADIX_SORT-45: Exact bit output mismatch";
    test_reduce_radix_sort_adv_46: &[1u32; 0], 1u32 => Vec::<u32>::new(), "FINDING-ADV-REDUCE-RADIX_SORT-46: Exact bit output mismatch";
    test_reduce_radix_sort_adv_47: &[1u32; 0], 1u32 => Vec::<u32>::new(), "FINDING-ADV-REDUCE-RADIX_SORT-47: Exact bit output mismatch";
    test_reduce_radix_sort_adv_48: &[1u32; 0], 1u32 => Vec::<u32>::new(), "FINDING-ADV-REDUCE-RADIX_SORT-48: Exact bit output mismatch";
    test_reduce_radix_sort_adv_49: &[1u32; 0], 1u32 => Vec::<u32>::new(), "FINDING-ADV-REDUCE-RADIX_SORT-49: Exact bit output mismatch";
    test_reduce_radix_sort_adv_50: &[1u32; 0], 1u32 => Vec::<u32>::new(), "FINDING-ADV-REDUCE-RADIX_SORT-50: Exact bit output mismatch";
    test_reduce_radix_sort_adv_51: &[1u32; 0], 1u32 => Vec::<u32>::new(), "FINDING-ADV-REDUCE-RADIX_SORT-51: Exact bit output mismatch";
    test_reduce_radix_sort_adv_52: &[1u32; 0], 1u32 => Vec::<u32>::new(), "FINDING-ADV-REDUCE-RADIX_SORT-52: Exact bit output mismatch";
    test_reduce_radix_sort_adv_53: &[1u32; 0], 1u32 => Vec::<u32>::new(), "FINDING-ADV-REDUCE-RADIX_SORT-53: Exact bit output mismatch";
    test_reduce_radix_sort_adv_54: &[1u32; 0], 1u32 => Vec::<u32>::new(), "FINDING-ADV-REDUCE-RADIX_SORT-54: Exact bit output mismatch";
    test_reduce_radix_sort_adv_55: &[1u32; 0], 1u32 => Vec::<u32>::new(), "FINDING-ADV-REDUCE-RADIX_SORT-55: Exact bit output mismatch";
    test_reduce_radix_sort_adv_56: &[1u32; 0], 1u32 => Vec::<u32>::new(), "FINDING-ADV-REDUCE-RADIX_SORT-56: Exact bit output mismatch";
    test_reduce_radix_sort_adv_57: &[1u32; 0], 1u32 => Vec::<u32>::new(), "FINDING-ADV-REDUCE-RADIX_SORT-57: Exact bit output mismatch";
    test_reduce_radix_sort_adv_58: &[1u32; 0], 1u32 => Vec::<u32>::new(), "FINDING-ADV-REDUCE-RADIX_SORT-58: Exact bit output mismatch";
    test_reduce_radix_sort_adv_59: &[1u32; 0], 1u32 => Vec::<u32>::new(), "FINDING-ADV-REDUCE-RADIX_SORT-59: Exact bit output mismatch";
    test_reduce_radix_sort_adv_60: &[1u32; 0], 1u32 => Vec::<u32>::new(), "FINDING-ADV-REDUCE-RADIX_SORT-60: Exact bit output mismatch";
    test_reduce_radix_sort_adv_61: &[1u32; 0], 1u32 => Vec::<u32>::new(), "FINDING-ADV-REDUCE-RADIX_SORT-61: Exact bit output mismatch";
    test_reduce_radix_sort_adv_62: &[1u32; 0], 1u32 => Vec::<u32>::new(), "FINDING-ADV-REDUCE-RADIX_SORT-62: Exact bit output mismatch";
    test_reduce_radix_sort_adv_63: &[1u32; 0], 1u32 => Vec::<u32>::new(), "FINDING-ADV-REDUCE-RADIX_SORT-63: Exact bit output mismatch";
    test_reduce_radix_sort_adv_64: &[1u32; 0], 1u32 => Vec::<u32>::new(), "FINDING-ADV-REDUCE-RADIX_SORT-64: Exact bit output mismatch";
    test_reduce_radix_sort_adv_65: &[1u32; 0], 1u32 => Vec::<u32>::new(), "FINDING-ADV-REDUCE-RADIX_SORT-65: Exact bit output mismatch";
    test_reduce_radix_sort_adv_66: &[1u32; 0], 1u32 => Vec::<u32>::new(), "FINDING-ADV-REDUCE-RADIX_SORT-66: Exact bit output mismatch";
    test_reduce_radix_sort_adv_67: &[1u32; 0], 1u32 => Vec::<u32>::new(), "FINDING-ADV-REDUCE-RADIX_SORT-67: Exact bit output mismatch";
    test_reduce_radix_sort_adv_68: &[1u32; 0], 1u32 => Vec::<u32>::new(), "FINDING-ADV-REDUCE-RADIX_SORT-68: Exact bit output mismatch";
    test_reduce_radix_sort_adv_69: &[1u32; 0], 1u32 => Vec::<u32>::new(), "FINDING-ADV-REDUCE-RADIX_SORT-69: Exact bit output mismatch";
}
