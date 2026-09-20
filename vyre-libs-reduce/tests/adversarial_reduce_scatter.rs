//! Generated adversarial contract tests for vyre-primitives.

// Adversarial tests for reduce::scatter

#![allow(
    unused_imports,
    unused_variables,
    dead_code,
    unused_mut,
    unused_macros,
    clippy::identity_op,
    clippy::assertions_on_constants
)]


use vyre_libs_reduce::reduce::scatter::*;
use vyre_reference::composition_witness::scatter_witness as cpu_ref;

vyre_test_support::adversarial_cpu_ref_cases! {
    test_reduce_scatter_adv_0: &[0u32; 0], &[0u32; 0], 0usize => Vec::<u32>::new(), "FINDING-ADV-REDUCE-SCATTER-0: Exact bit output mismatch";
    test_reduce_scatter_adv_1: &[0u32; 0], &[0u32; 0], 0usize => Vec::<u32>::new(), "FINDING-ADV-REDUCE-SCATTER-1: Exact bit output mismatch";
    test_reduce_scatter_adv_2: &[0u32; 0], &[0u32; 0], 0usize => Vec::<u32>::new(), "FINDING-ADV-REDUCE-SCATTER-2: Exact bit output mismatch";
    test_reduce_scatter_adv_3: &[0u32; 0], &[4294967295u32; 0], 0usize => Vec::<u32>::new(), "FINDING-ADV-REDUCE-SCATTER-3: Exact bit output mismatch";
    test_reduce_scatter_adv_4: &[0u32; 0], &[4294967295u32; 0], 0usize => Vec::<u32>::new(), "FINDING-ADV-REDUCE-SCATTER-4: Exact bit output mismatch";
    test_reduce_scatter_adv_5: &[0u32; 0], &[4294967295u32; 0], 0usize => Vec::<u32>::new(), "FINDING-ADV-REDUCE-SCATTER-5: Exact bit output mismatch";
    test_reduce_scatter_adv_6: &[0u32; 0], &[2143289344u32; 0], 0usize => Vec::<u32>::new(), "FINDING-ADV-REDUCE-SCATTER-6: Exact bit output mismatch";
    test_reduce_scatter_adv_7: &[0u32; 0], &[2143289344u32; 0], 0usize => Vec::<u32>::new(), "FINDING-ADV-REDUCE-SCATTER-7: Exact bit output mismatch";
    test_reduce_scatter_adv_8: &[0u32; 0], &[2143289344u32; 0], 0usize => Vec::<u32>::new(), "FINDING-ADV-REDUCE-SCATTER-8: Exact bit output mismatch";
    test_reduce_scatter_adv_9: &[0u32; 0], &[0u32; 1], 1usize => vec![0], "FINDING-ADV-REDUCE-SCATTER-9: Exact bit output mismatch";
    test_reduce_scatter_adv_10: &[0u32; 0], &[0u32; 1], 1usize => vec![0], "FINDING-ADV-REDUCE-SCATTER-10: Exact bit output mismatch";
    test_reduce_scatter_adv_11: &[0u32; 0], &[0u32; 1], 1usize => vec![0], "FINDING-ADV-REDUCE-SCATTER-11: Exact bit output mismatch";
    test_reduce_scatter_adv_12: &[0u32; 0], &[4294967295u32; 1], 1usize => vec![0], "FINDING-ADV-REDUCE-SCATTER-12: Exact bit output mismatch";
    test_reduce_scatter_adv_13: &[0u32; 0], &[4294967295u32; 1], 1usize => vec![0], "FINDING-ADV-REDUCE-SCATTER-13: Exact bit output mismatch";
    test_reduce_scatter_adv_14: &[0u32; 0], &[4294967295u32; 1], 1usize => vec![0], "FINDING-ADV-REDUCE-SCATTER-14: Exact bit output mismatch";
    test_reduce_scatter_adv_15: &[0u32; 0], &[2143289344u32; 1], 1usize => vec![0], "FINDING-ADV-REDUCE-SCATTER-15: Exact bit output mismatch";
    test_reduce_scatter_adv_16: &[0u32; 0], &[2143289344u32; 1], 1usize => vec![0], "FINDING-ADV-REDUCE-SCATTER-16: Exact bit output mismatch";
    test_reduce_scatter_adv_17: &[0u32; 0], &[2143289344u32; 1], 1usize => vec![0], "FINDING-ADV-REDUCE-SCATTER-17: Exact bit output mismatch";
    test_reduce_scatter_adv_18: &[0u32; 0], &[0u32; 31], 31usize => vec![0; 31], "FINDING-ADV-REDUCE-SCATTER-18: Exact bit output mismatch";
    test_reduce_scatter_adv_19: &[0u32; 0], &[0u32; 31], 31usize => vec![0; 31], "FINDING-ADV-REDUCE-SCATTER-19: Exact bit output mismatch";
    test_reduce_scatter_adv_20: &[0u32; 0], &[0u32; 31], 31usize => vec![0; 31], "FINDING-ADV-REDUCE-SCATTER-20: Exact bit output mismatch";
    test_reduce_scatter_adv_21: &[0u32; 0], &[4294967295u32; 31], 31usize => vec![0; 31], "FINDING-ADV-REDUCE-SCATTER-21: Exact bit output mismatch";
    test_reduce_scatter_adv_22: &[0u32; 0], &[4294967295u32; 31], 31usize => vec![0; 31], "FINDING-ADV-REDUCE-SCATTER-22: Exact bit output mismatch";
    test_reduce_scatter_adv_23: &[0u32; 0], &[4294967295u32; 31], 31usize => vec![0; 31], "FINDING-ADV-REDUCE-SCATTER-23: Exact bit output mismatch";
    test_reduce_scatter_adv_24: &[0u32; 0], &[2143289344u32; 31], 31usize => vec![0; 31], "FINDING-ADV-REDUCE-SCATTER-24: Exact bit output mismatch";
    test_reduce_scatter_adv_25: &[0u32; 0], &[2143289344u32; 31], 31usize => vec![0; 31], "FINDING-ADV-REDUCE-SCATTER-25: Exact bit output mismatch";
    test_reduce_scatter_adv_26: &[0u32; 0], &[2143289344u32; 31], 31usize => vec![0; 31], "FINDING-ADV-REDUCE-SCATTER-26: Exact bit output mismatch";
    test_reduce_scatter_adv_27: &[0u32; 0], &[0u32; 32], 32usize => vec![0; 32], "FINDING-ADV-REDUCE-SCATTER-27: Exact bit output mismatch";
    test_reduce_scatter_adv_28: &[0u32; 0], &[0u32; 32], 32usize => vec![0; 32], "FINDING-ADV-REDUCE-SCATTER-28: Exact bit output mismatch";
    test_reduce_scatter_adv_29: &[0u32; 0], &[0u32; 32], 32usize => vec![0; 32], "FINDING-ADV-REDUCE-SCATTER-29: Exact bit output mismatch";
    test_reduce_scatter_adv_30: &[0u32; 0], &[4294967295u32; 32], 32usize => vec![0; 32], "FINDING-ADV-REDUCE-SCATTER-30: Exact bit output mismatch";
    test_reduce_scatter_adv_31: &[0u32; 0], &[4294967295u32; 32], 32usize => vec![0; 32], "FINDING-ADV-REDUCE-SCATTER-31: Exact bit output mismatch";
    test_reduce_scatter_adv_32: &[0u32; 0], &[4294967295u32; 32], 32usize => vec![0; 32], "FINDING-ADV-REDUCE-SCATTER-32: Exact bit output mismatch";
    test_reduce_scatter_adv_33: &[0u32; 0], &[2143289344u32; 32], 32usize => vec![0; 32], "FINDING-ADV-REDUCE-SCATTER-33: Exact bit output mismatch";
    test_reduce_scatter_adv_34: &[0u32; 0], &[2143289344u32; 32], 32usize => vec![0; 32], "FINDING-ADV-REDUCE-SCATTER-34: Exact bit output mismatch";
    test_reduce_scatter_adv_35: &[0u32; 0], &[2143289344u32; 32], 32usize => vec![0; 32], "FINDING-ADV-REDUCE-SCATTER-35: Exact bit output mismatch";
    test_reduce_scatter_adv_36: &[0u32; 0], &[0u32; 1024], 1024usize => vec![0; 1024], "FINDING-ADV-REDUCE-SCATTER-36: Exact bit output mismatch";
    test_reduce_scatter_adv_37: &[0u32; 0], &[0u32; 1024], 1024usize => vec![0; 1024], "FINDING-ADV-REDUCE-SCATTER-37: Exact bit output mismatch";
    test_reduce_scatter_adv_38: &[0u32; 0], &[0u32; 1024], 1024usize => vec![0; 1024], "FINDING-ADV-REDUCE-SCATTER-38: Exact bit output mismatch";
    test_reduce_scatter_adv_39: &[0u32; 0], &[4294967295u32; 1024], 1024usize => vec![0; 1024], "FINDING-ADV-REDUCE-SCATTER-39: Exact bit output mismatch";
    test_reduce_scatter_adv_40: &[0u32; 0], &[4294967295u32; 1024], 1024usize => vec![0; 1024], "FINDING-ADV-REDUCE-SCATTER-40: Exact bit output mismatch";
    test_reduce_scatter_adv_41: &[0u32; 0], &[4294967295u32; 1024], 1024usize => vec![0; 1024], "FINDING-ADV-REDUCE-SCATTER-41: Exact bit output mismatch";
    test_reduce_scatter_adv_42: &[0u32; 0], &[2143289344u32; 1024], 1024usize => vec![0; 1024], "FINDING-ADV-REDUCE-SCATTER-42: Exact bit output mismatch";
    test_reduce_scatter_adv_43: &[0u32; 0], &[2143289344u32; 1024], 1024usize => vec![0; 1024], "FINDING-ADV-REDUCE-SCATTER-43: Exact bit output mismatch";
    test_reduce_scatter_adv_44: &[0u32; 0], &[2143289344u32; 1024], 1024usize => vec![0; 1024], "FINDING-ADV-REDUCE-SCATTER-44: Exact bit output mismatch";
    test_reduce_scatter_adv_45: &[1u32; 0], &[0u32; 0], 0usize => Vec::<u32>::new(), "FINDING-ADV-REDUCE-SCATTER-45: Exact bit output mismatch";
    test_reduce_scatter_adv_46: &[1u32; 0], &[0u32; 0], 0usize => Vec::<u32>::new(), "FINDING-ADV-REDUCE-SCATTER-46: Exact bit output mismatch";
    test_reduce_scatter_adv_47: &[1u32; 0], &[0u32; 0], 0usize => Vec::<u32>::new(), "FINDING-ADV-REDUCE-SCATTER-47: Exact bit output mismatch";
    test_reduce_scatter_adv_48: &[1u32; 0], &[4294967295u32; 0], 0usize => Vec::<u32>::new(), "FINDING-ADV-REDUCE-SCATTER-48: Exact bit output mismatch";
    test_reduce_scatter_adv_49: &[1u32; 0], &[4294967295u32; 0], 0usize => Vec::<u32>::new(), "FINDING-ADV-REDUCE-SCATTER-49: Exact bit output mismatch";
    test_reduce_scatter_adv_50: &[1u32; 0], &[4294967295u32; 0], 0usize => Vec::<u32>::new(), "FINDING-ADV-REDUCE-SCATTER-50: Exact bit output mismatch";
    test_reduce_scatter_adv_51: &[1u32; 0], &[2143289344u32; 0], 0usize => Vec::<u32>::new(), "FINDING-ADV-REDUCE-SCATTER-51: Exact bit output mismatch";
    test_reduce_scatter_adv_52: &[1u32; 0], &[2143289344u32; 0], 0usize => Vec::<u32>::new(), "FINDING-ADV-REDUCE-SCATTER-52: Exact bit output mismatch";
    test_reduce_scatter_adv_53: &[1u32; 0], &[2143289344u32; 0], 0usize => Vec::<u32>::new(), "FINDING-ADV-REDUCE-SCATTER-53: Exact bit output mismatch";
    test_reduce_scatter_adv_54: &[1u32; 0], &[0u32; 1], 1usize => vec![0], "FINDING-ADV-REDUCE-SCATTER-54: Exact bit output mismatch";
    test_reduce_scatter_adv_55: &[1u32; 0], &[0u32; 1], 1usize => vec![0], "FINDING-ADV-REDUCE-SCATTER-55: Exact bit output mismatch";
    test_reduce_scatter_adv_56: &[1u32; 0], &[0u32; 1], 1usize => vec![0], "FINDING-ADV-REDUCE-SCATTER-56: Exact bit output mismatch";
    test_reduce_scatter_adv_57: &[1u32; 0], &[4294967295u32; 1], 1usize => vec![0], "FINDING-ADV-REDUCE-SCATTER-57: Exact bit output mismatch";
    test_reduce_scatter_adv_58: &[1u32; 0], &[4294967295u32; 1], 1usize => vec![0], "FINDING-ADV-REDUCE-SCATTER-58: Exact bit output mismatch";
    test_reduce_scatter_adv_59: &[1u32; 0], &[4294967295u32; 1], 1usize => vec![0], "FINDING-ADV-REDUCE-SCATTER-59: Exact bit output mismatch";
    test_reduce_scatter_adv_60: &[1u32; 0], &[2143289344u32; 1], 1usize => vec![0], "FINDING-ADV-REDUCE-SCATTER-60: Exact bit output mismatch";
    test_reduce_scatter_adv_61: &[1u32; 0], &[2143289344u32; 1], 1usize => vec![0], "FINDING-ADV-REDUCE-SCATTER-61: Exact bit output mismatch";
    test_reduce_scatter_adv_62: &[1u32; 0], &[2143289344u32; 1], 1usize => vec![0], "FINDING-ADV-REDUCE-SCATTER-62: Exact bit output mismatch";
    test_reduce_scatter_adv_63: &[1u32; 0], &[0u32; 31], 31usize => vec![0; 31], "FINDING-ADV-REDUCE-SCATTER-63: Exact bit output mismatch";
    test_reduce_scatter_adv_64: &[1u32; 0], &[0u32; 31], 31usize => vec![0; 31], "FINDING-ADV-REDUCE-SCATTER-64: Exact bit output mismatch";
    test_reduce_scatter_adv_65: &[1u32; 0], &[0u32; 31], 31usize => vec![0; 31], "FINDING-ADV-REDUCE-SCATTER-65: Exact bit output mismatch";
    test_reduce_scatter_adv_66: &[1u32; 0], &[4294967295u32; 31], 31usize => vec![0; 31], "FINDING-ADV-REDUCE-SCATTER-66: Exact bit output mismatch";
    test_reduce_scatter_adv_67: &[1u32; 0], &[4294967295u32; 31], 31usize => vec![0; 31], "FINDING-ADV-REDUCE-SCATTER-67: Exact bit output mismatch";
    test_reduce_scatter_adv_68: &[1u32; 0], &[4294967295u32; 31], 31usize => vec![0; 31], "FINDING-ADV-REDUCE-SCATTER-68: Exact bit output mismatch";
    test_reduce_scatter_adv_69: &[1u32; 0], &[2143289344u32; 31], 31usize => vec![0; 31], "FINDING-ADV-REDUCE-SCATTER-69: Exact bit output mismatch";
}
