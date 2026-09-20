//! Generated adversarial contract tests for vyre-primitives.

// Adversarial tests for reduce::gather

#![allow(
    unused_imports,
    unused_variables,
    dead_code,
    unused_mut,
    unused_macros,
    clippy::identity_op,
    clippy::assertions_on_constants
)]


use vyre_libs_reduce::reduce::gather::*;
use vyre_reference::composition_witness::gather_witness as cpu_ref;

vyre_test_support::adversarial_cpu_ref_cases! {
    test_reduce_gather_adv_0: &[0u32; 0], &[0u32; 0] => Vec::<u32>::new(), "FINDING-ADV-REDUCE-GATHER-0: Exact bit output mismatch";
    test_reduce_gather_adv_1: &[0u32; 0], &[0u32; 0] => Vec::<u32>::new(), "FINDING-ADV-REDUCE-GATHER-1: Exact bit output mismatch";
    test_reduce_gather_adv_2: &[0u32; 0], &[0u32; 0] => Vec::<u32>::new(), "FINDING-ADV-REDUCE-GATHER-2: Exact bit output mismatch";
    test_reduce_gather_adv_3: &[0u32; 0], &[4294967295u32; 0] => Vec::<u32>::new(), "FINDING-ADV-REDUCE-GATHER-3: Exact bit output mismatch";
    test_reduce_gather_adv_4: &[0u32; 0], &[4294967295u32; 0] => Vec::<u32>::new(), "FINDING-ADV-REDUCE-GATHER-4: Exact bit output mismatch";
    test_reduce_gather_adv_5: &[0u32; 0], &[4294967295u32; 0] => Vec::<u32>::new(), "FINDING-ADV-REDUCE-GATHER-5: Exact bit output mismatch";
    test_reduce_gather_adv_6: &[0u32; 0], &[2143289344u32; 0] => Vec::<u32>::new(), "FINDING-ADV-REDUCE-GATHER-6: Exact bit output mismatch";
    test_reduce_gather_adv_7: &[0u32; 0], &[2143289344u32; 0] => Vec::<u32>::new(), "FINDING-ADV-REDUCE-GATHER-7: Exact bit output mismatch";
    test_reduce_gather_adv_8: &[0u32; 0], &[2143289344u32; 0] => Vec::<u32>::new(), "FINDING-ADV-REDUCE-GATHER-8: Exact bit output mismatch";
    test_reduce_gather_adv_9: &[0u32; 0], &[0u32; 1] => vec![0], "FINDING-ADV-REDUCE-GATHER-9: Exact bit output mismatch";
    test_reduce_gather_adv_10: &[0u32; 0], &[0u32; 1] => vec![0], "FINDING-ADV-REDUCE-GATHER-10: Exact bit output mismatch";
    test_reduce_gather_adv_11: &[0u32; 0], &[0u32; 1] => vec![0], "FINDING-ADV-REDUCE-GATHER-11: Exact bit output mismatch";
    test_reduce_gather_adv_12: &[0u32; 0], &[4294967295u32; 1] => vec![0], "FINDING-ADV-REDUCE-GATHER-12: Exact bit output mismatch";
    test_reduce_gather_adv_13: &[0u32; 0], &[4294967295u32; 1] => vec![0], "FINDING-ADV-REDUCE-GATHER-13: Exact bit output mismatch";
    test_reduce_gather_adv_14: &[0u32; 0], &[4294967295u32; 1] => vec![0], "FINDING-ADV-REDUCE-GATHER-14: Exact bit output mismatch";
    test_reduce_gather_adv_15: &[0u32; 0], &[2143289344u32; 1] => vec![0], "FINDING-ADV-REDUCE-GATHER-15: Exact bit output mismatch";
    test_reduce_gather_adv_16: &[0u32; 0], &[2143289344u32; 1] => vec![0], "FINDING-ADV-REDUCE-GATHER-16: Exact bit output mismatch";
    test_reduce_gather_adv_17: &[0u32; 0], &[2143289344u32; 1] => vec![0], "FINDING-ADV-REDUCE-GATHER-17: Exact bit output mismatch";
    test_reduce_gather_adv_18: &[0u32; 0], &[0u32; 31] => vec![0; 31], "FINDING-ADV-REDUCE-GATHER-18: Exact bit output mismatch";
    test_reduce_gather_adv_19: &[0u32; 0], &[0u32; 31] => vec![0; 31], "FINDING-ADV-REDUCE-GATHER-19: Exact bit output mismatch";
    test_reduce_gather_adv_20: &[0u32; 0], &[0u32; 31] => vec![0; 31], "FINDING-ADV-REDUCE-GATHER-20: Exact bit output mismatch";
    test_reduce_gather_adv_21: &[0u32; 0], &[4294967295u32; 31] => vec![0; 31], "FINDING-ADV-REDUCE-GATHER-21: Exact bit output mismatch";
    test_reduce_gather_adv_22: &[0u32; 0], &[4294967295u32; 31] => vec![0; 31], "FINDING-ADV-REDUCE-GATHER-22: Exact bit output mismatch";
    test_reduce_gather_adv_23: &[0u32; 0], &[4294967295u32; 31] => vec![0; 31], "FINDING-ADV-REDUCE-GATHER-23: Exact bit output mismatch";
    test_reduce_gather_adv_24: &[0u32; 0], &[2143289344u32; 31] => vec![0; 31], "FINDING-ADV-REDUCE-GATHER-24: Exact bit output mismatch";
    test_reduce_gather_adv_25: &[0u32; 0], &[2143289344u32; 31] => vec![0; 31], "FINDING-ADV-REDUCE-GATHER-25: Exact bit output mismatch";
    test_reduce_gather_adv_26: &[0u32; 0], &[2143289344u32; 31] => vec![0; 31], "FINDING-ADV-REDUCE-GATHER-26: Exact bit output mismatch";
    test_reduce_gather_adv_27: &[0u32; 0], &[0u32; 32] => vec![0; 32], "FINDING-ADV-REDUCE-GATHER-27: Exact bit output mismatch";
    test_reduce_gather_adv_28: &[0u32; 0], &[0u32; 32] => vec![0; 32], "FINDING-ADV-REDUCE-GATHER-28: Exact bit output mismatch";
    test_reduce_gather_adv_29: &[0u32; 0], &[0u32; 32] => vec![0; 32], "FINDING-ADV-REDUCE-GATHER-29: Exact bit output mismatch";
    test_reduce_gather_adv_30: &[0u32; 0], &[4294967295u32; 32] => vec![0; 32], "FINDING-ADV-REDUCE-GATHER-30: Exact bit output mismatch";
    test_reduce_gather_adv_31: &[0u32; 0], &[4294967295u32; 32] => vec![0; 32], "FINDING-ADV-REDUCE-GATHER-31: Exact bit output mismatch";
    test_reduce_gather_adv_32: &[0u32; 0], &[4294967295u32; 32] => vec![0; 32], "FINDING-ADV-REDUCE-GATHER-32: Exact bit output mismatch";
    test_reduce_gather_adv_33: &[0u32; 0], &[2143289344u32; 32] => vec![0; 32], "FINDING-ADV-REDUCE-GATHER-33: Exact bit output mismatch";
    test_reduce_gather_adv_34: &[0u32; 0], &[2143289344u32; 32] => vec![0; 32], "FINDING-ADV-REDUCE-GATHER-34: Exact bit output mismatch";
    test_reduce_gather_adv_35: &[0u32; 0], &[2143289344u32; 32] => vec![0; 32], "FINDING-ADV-REDUCE-GATHER-35: Exact bit output mismatch";
    test_reduce_gather_adv_36: &[0u32; 0], &[0u32; 1024] => vec![0; 1024], "FINDING-ADV-REDUCE-GATHER-36: Exact bit output mismatch";
    test_reduce_gather_adv_37: &[0u32; 0], &[0u32; 1024] => vec![0; 1024], "FINDING-ADV-REDUCE-GATHER-37: Exact bit output mismatch";
    test_reduce_gather_adv_38: &[0u32; 0], &[0u32; 1024] => vec![0; 1024], "FINDING-ADV-REDUCE-GATHER-38: Exact bit output mismatch";
    test_reduce_gather_adv_39: &[0u32; 0], &[4294967295u32; 1024] => vec![0; 1024], "FINDING-ADV-REDUCE-GATHER-39: Exact bit output mismatch";
    test_reduce_gather_adv_40: &[0u32; 0], &[4294967295u32; 1024] => vec![0; 1024], "FINDING-ADV-REDUCE-GATHER-40: Exact bit output mismatch";
    test_reduce_gather_adv_41: &[0u32; 0], &[4294967295u32; 1024] => vec![0; 1024], "FINDING-ADV-REDUCE-GATHER-41: Exact bit output mismatch";
    test_reduce_gather_adv_42: &[0u32; 0], &[2143289344u32; 1024] => vec![0; 1024], "FINDING-ADV-REDUCE-GATHER-42: Exact bit output mismatch";
    test_reduce_gather_adv_43: &[0u32; 0], &[2143289344u32; 1024] => vec![0; 1024], "FINDING-ADV-REDUCE-GATHER-43: Exact bit output mismatch";
    test_reduce_gather_adv_44: &[0u32; 0], &[2143289344u32; 1024] => vec![0; 1024], "FINDING-ADV-REDUCE-GATHER-44: Exact bit output mismatch";
    test_reduce_gather_adv_45: &[1u32; 0], &[0u32; 0] => Vec::<u32>::new(), "FINDING-ADV-REDUCE-GATHER-45: Exact bit output mismatch";
    test_reduce_gather_adv_46: &[1u32; 0], &[0u32; 0] => Vec::<u32>::new(), "FINDING-ADV-REDUCE-GATHER-46: Exact bit output mismatch";
    test_reduce_gather_adv_47: &[1u32; 0], &[0u32; 0] => Vec::<u32>::new(), "FINDING-ADV-REDUCE-GATHER-47: Exact bit output mismatch";
    test_reduce_gather_adv_48: &[1u32; 0], &[4294967295u32; 0] => Vec::<u32>::new(), "FINDING-ADV-REDUCE-GATHER-48: Exact bit output mismatch";
    test_reduce_gather_adv_49: &[1u32; 0], &[4294967295u32; 0] => Vec::<u32>::new(), "FINDING-ADV-REDUCE-GATHER-49: Exact bit output mismatch";
    test_reduce_gather_adv_50: &[1u32; 0], &[4294967295u32; 0] => Vec::<u32>::new(), "FINDING-ADV-REDUCE-GATHER-50: Exact bit output mismatch";
    test_reduce_gather_adv_51: &[1u32; 0], &[2143289344u32; 0] => Vec::<u32>::new(), "FINDING-ADV-REDUCE-GATHER-51: Exact bit output mismatch";
    test_reduce_gather_adv_52: &[1u32; 0], &[2143289344u32; 0] => Vec::<u32>::new(), "FINDING-ADV-REDUCE-GATHER-52: Exact bit output mismatch";
    test_reduce_gather_adv_53: &[1u32; 0], &[2143289344u32; 0] => Vec::<u32>::new(), "FINDING-ADV-REDUCE-GATHER-53: Exact bit output mismatch";
    test_reduce_gather_adv_54: &[1u32; 0], &[0u32; 1] => vec![0], "FINDING-ADV-REDUCE-GATHER-54: Exact bit output mismatch";
    test_reduce_gather_adv_55: &[1u32; 0], &[0u32; 1] => vec![0], "FINDING-ADV-REDUCE-GATHER-55: Exact bit output mismatch";
    test_reduce_gather_adv_56: &[1u32; 0], &[0u32; 1] => vec![0], "FINDING-ADV-REDUCE-GATHER-56: Exact bit output mismatch";
    test_reduce_gather_adv_57: &[1u32; 0], &[4294967295u32; 1] => vec![0], "FINDING-ADV-REDUCE-GATHER-57: Exact bit output mismatch";
    test_reduce_gather_adv_58: &[1u32; 0], &[4294967295u32; 1] => vec![0], "FINDING-ADV-REDUCE-GATHER-58: Exact bit output mismatch";
    test_reduce_gather_adv_59: &[1u32; 0], &[4294967295u32; 1] => vec![0], "FINDING-ADV-REDUCE-GATHER-59: Exact bit output mismatch";
    test_reduce_gather_adv_60: &[1u32; 0], &[2143289344u32; 1] => vec![0], "FINDING-ADV-REDUCE-GATHER-60: Exact bit output mismatch";
    test_reduce_gather_adv_61: &[1u32; 0], &[2143289344u32; 1] => vec![0], "FINDING-ADV-REDUCE-GATHER-61: Exact bit output mismatch";
    test_reduce_gather_adv_62: &[1u32; 0], &[2143289344u32; 1] => vec![0], "FINDING-ADV-REDUCE-GATHER-62: Exact bit output mismatch";
    test_reduce_gather_adv_63: &[1u32; 0], &[0u32; 31] => vec![0; 31], "FINDING-ADV-REDUCE-GATHER-63: Exact bit output mismatch";
    test_reduce_gather_adv_64: &[1u32; 0], &[0u32; 31] => vec![0; 31], "FINDING-ADV-REDUCE-GATHER-64: Exact bit output mismatch";
    test_reduce_gather_adv_65: &[1u32; 0], &[0u32; 31] => vec![0; 31], "FINDING-ADV-REDUCE-GATHER-65: Exact bit output mismatch";
    test_reduce_gather_adv_66: &[1u32; 0], &[4294967295u32; 31] => vec![0; 31], "FINDING-ADV-REDUCE-GATHER-66: Exact bit output mismatch";
    test_reduce_gather_adv_67: &[1u32; 0], &[4294967295u32; 31] => vec![0; 31], "FINDING-ADV-REDUCE-GATHER-67: Exact bit output mismatch";
    test_reduce_gather_adv_68: &[1u32; 0], &[4294967295u32; 31] => vec![0; 31], "FINDING-ADV-REDUCE-GATHER-68: Exact bit output mismatch";
    test_reduce_gather_adv_69: &[1u32; 0], &[2143289344u32; 31] => vec![0; 31], "FINDING-ADV-REDUCE-GATHER-69: Exact bit output mismatch";
}
