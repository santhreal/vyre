//! Generated adversarial contract tests for vyre-primitives.

// Adversarial tests for reduce::histogram

#![allow(
    unused_imports,
    unused_variables,
    dead_code,
    unused_mut,
    unused_macros,
    clippy::identity_op,
    clippy::assertions_on_constants
)]

use vyre_libs_reduce::reduce::histogram::*;
use vyre_reference::composition_witness::histogram_witness as cpu_ref;

vyre_test_support::adversarial_cpu_ref_cases! {
    test_reduce_histogram_adv_0: &[0u32; 0], 1u32 => vec![0], "FINDING-ADV-REDUCE-HISTOGRAM-0: Exact bit output mismatch";
    test_reduce_histogram_adv_1: &[0u32; 0], 1u32 => vec![0], "FINDING-ADV-REDUCE-HISTOGRAM-1: Exact bit output mismatch";
    test_reduce_histogram_adv_2: &[0u32; 0], 1u32 => vec![0], "FINDING-ADV-REDUCE-HISTOGRAM-2: Exact bit output mismatch";
    test_reduce_histogram_adv_3: &[0u32; 0], 1u32 => vec![0], "FINDING-ADV-REDUCE-HISTOGRAM-3: Exact bit output mismatch";
    test_reduce_histogram_adv_4: &[0u32; 0], 1u32 => vec![0], "FINDING-ADV-REDUCE-HISTOGRAM-4: Exact bit output mismatch";
    test_reduce_histogram_adv_5: &[0u32; 0], 1u32 => vec![0], "FINDING-ADV-REDUCE-HISTOGRAM-5: Exact bit output mismatch";
    test_reduce_histogram_adv_6: &[0u32; 0], 1u32 => vec![0], "FINDING-ADV-REDUCE-HISTOGRAM-6: Exact bit output mismatch";
    test_reduce_histogram_adv_7: &[0u32; 0], 1u32 => vec![0], "FINDING-ADV-REDUCE-HISTOGRAM-7: Exact bit output mismatch";
    test_reduce_histogram_adv_8: &[0u32; 0], 1u32 => vec![0], "FINDING-ADV-REDUCE-HISTOGRAM-8: Exact bit output mismatch";
    test_reduce_histogram_adv_9: &[0u32; 0], 1u32 => vec![0], "FINDING-ADV-REDUCE-HISTOGRAM-9: Exact bit output mismatch";
    test_reduce_histogram_adv_10: &[0u32; 0], 1u32 => vec![0], "FINDING-ADV-REDUCE-HISTOGRAM-10: Exact bit output mismatch";
    test_reduce_histogram_adv_11: &[0u32; 0], 1u32 => vec![0], "FINDING-ADV-REDUCE-HISTOGRAM-11: Exact bit output mismatch";
    test_reduce_histogram_adv_12: &[0u32; 0], 1u32 => vec![0], "FINDING-ADV-REDUCE-HISTOGRAM-12: Exact bit output mismatch";
    test_reduce_histogram_adv_13: &[0u32; 0], 1u32 => vec![0], "FINDING-ADV-REDUCE-HISTOGRAM-13: Exact bit output mismatch";
    test_reduce_histogram_adv_14: &[0u32; 0], 1u32 => vec![0], "FINDING-ADV-REDUCE-HISTOGRAM-14: Exact bit output mismatch";
    test_reduce_histogram_adv_15: &[0u32; 0], 1u32 => vec![0], "FINDING-ADV-REDUCE-HISTOGRAM-15: Exact bit output mismatch";
    test_reduce_histogram_adv_16: &[0u32; 0], 1u32 => vec![0], "FINDING-ADV-REDUCE-HISTOGRAM-16: Exact bit output mismatch";
    test_reduce_histogram_adv_17: &[0u32; 0], 1u32 => vec![0], "FINDING-ADV-REDUCE-HISTOGRAM-17: Exact bit output mismatch";
    test_reduce_histogram_adv_18: &[0u32; 0], 31u32 => vec![0; 31], "FINDING-ADV-REDUCE-HISTOGRAM-18: Exact bit output mismatch";
    test_reduce_histogram_adv_19: &[0u32; 0], 31u32 => vec![0; 31], "FINDING-ADV-REDUCE-HISTOGRAM-19: Exact bit output mismatch";
    test_reduce_histogram_adv_20: &[0u32; 0], 31u32 => vec![0; 31], "FINDING-ADV-REDUCE-HISTOGRAM-20: Exact bit output mismatch";
    test_reduce_histogram_adv_21: &[0u32; 0], 31u32 => vec![0; 31], "FINDING-ADV-REDUCE-HISTOGRAM-21: Exact bit output mismatch";
    test_reduce_histogram_adv_22: &[0u32; 0], 31u32 => vec![0; 31], "FINDING-ADV-REDUCE-HISTOGRAM-22: Exact bit output mismatch";
    test_reduce_histogram_adv_23: &[0u32; 0], 31u32 => vec![0; 31], "FINDING-ADV-REDUCE-HISTOGRAM-23: Exact bit output mismatch";
    test_reduce_histogram_adv_24: &[0u32; 0], 31u32 => vec![0; 31], "FINDING-ADV-REDUCE-HISTOGRAM-24: Exact bit output mismatch";
    test_reduce_histogram_adv_25: &[0u32; 0], 31u32 => vec![0; 31], "FINDING-ADV-REDUCE-HISTOGRAM-25: Exact bit output mismatch";
    test_reduce_histogram_adv_26: &[0u32; 0], 31u32 => vec![0; 31], "FINDING-ADV-REDUCE-HISTOGRAM-26: Exact bit output mismatch";
    test_reduce_histogram_adv_27: &[0u32; 0], 32u32 => vec![0; 32], "FINDING-ADV-REDUCE-HISTOGRAM-27: Exact bit output mismatch";
    test_reduce_histogram_adv_28: &[0u32; 0], 32u32 => vec![0; 32], "FINDING-ADV-REDUCE-HISTOGRAM-28: Exact bit output mismatch";
    test_reduce_histogram_adv_29: &[0u32; 0], 32u32 => vec![0; 32], "FINDING-ADV-REDUCE-HISTOGRAM-29: Exact bit output mismatch";
    test_reduce_histogram_adv_30: &[0u32; 0], 32u32 => vec![0; 32], "FINDING-ADV-REDUCE-HISTOGRAM-30: Exact bit output mismatch";
    test_reduce_histogram_adv_31: &[0u32; 0], 32u32 => vec![0; 32], "FINDING-ADV-REDUCE-HISTOGRAM-31: Exact bit output mismatch";
    test_reduce_histogram_adv_32: &[0u32; 0], 32u32 => vec![0; 32], "FINDING-ADV-REDUCE-HISTOGRAM-32: Exact bit output mismatch";
    test_reduce_histogram_adv_33: &[0u32; 0], 32u32 => vec![0; 32], "FINDING-ADV-REDUCE-HISTOGRAM-33: Exact bit output mismatch";
    test_reduce_histogram_adv_34: &[0u32; 0], 32u32 => vec![0; 32], "FINDING-ADV-REDUCE-HISTOGRAM-34: Exact bit output mismatch";
    test_reduce_histogram_adv_35: &[0u32; 0], 32u32 => vec![0; 32], "FINDING-ADV-REDUCE-HISTOGRAM-35: Exact bit output mismatch";
    test_reduce_histogram_adv_36: &[0u32; 0], 1024u32 => vec![0; 1024], "FINDING-ADV-REDUCE-HISTOGRAM-36: Exact bit output mismatch";
    test_reduce_histogram_adv_37: &[0u32; 0], 1024u32 => vec![0; 1024], "FINDING-ADV-REDUCE-HISTOGRAM-37: Exact bit output mismatch";
    test_reduce_histogram_adv_38: &[0u32; 0], 1024u32 => vec![0; 1024], "FINDING-ADV-REDUCE-HISTOGRAM-38: Exact bit output mismatch";
    test_reduce_histogram_adv_39: &[0u32; 0], 1024u32 => vec![0; 1024], "FINDING-ADV-REDUCE-HISTOGRAM-39: Exact bit output mismatch";
    test_reduce_histogram_adv_40: &[0u32; 0], 1024u32 => vec![0; 1024], "FINDING-ADV-REDUCE-HISTOGRAM-40: Exact bit output mismatch";
    test_reduce_histogram_adv_41: &[0u32; 0], 1024u32 => vec![0; 1024], "FINDING-ADV-REDUCE-HISTOGRAM-41: Exact bit output mismatch";
    test_reduce_histogram_adv_42: &[0u32; 0], 1024u32 => vec![0; 1024], "FINDING-ADV-REDUCE-HISTOGRAM-42: Exact bit output mismatch";
    test_reduce_histogram_adv_43: &[0u32; 0], 1024u32 => vec![0; 1024], "FINDING-ADV-REDUCE-HISTOGRAM-43: Exact bit output mismatch";
    test_reduce_histogram_adv_44: &[0u32; 0], 1024u32 => vec![0; 1024], "FINDING-ADV-REDUCE-HISTOGRAM-44: Exact bit output mismatch";
    test_reduce_histogram_adv_45: &[1u32; 0], 1u32 => vec![0], "FINDING-ADV-REDUCE-HISTOGRAM-45: Exact bit output mismatch";
    test_reduce_histogram_adv_46: &[1u32; 0], 1u32 => vec![0], "FINDING-ADV-REDUCE-HISTOGRAM-46: Exact bit output mismatch";
    test_reduce_histogram_adv_47: &[1u32; 0], 1u32 => vec![0], "FINDING-ADV-REDUCE-HISTOGRAM-47: Exact bit output mismatch";
    test_reduce_histogram_adv_48: &[1u32; 0], 1u32 => vec![0], "FINDING-ADV-REDUCE-HISTOGRAM-48: Exact bit output mismatch";
    test_reduce_histogram_adv_49: &[1u32; 0], 1u32 => vec![0], "FINDING-ADV-REDUCE-HISTOGRAM-49: Exact bit output mismatch";
    test_reduce_histogram_adv_50: &[1u32; 0], 1u32 => vec![0], "FINDING-ADV-REDUCE-HISTOGRAM-50: Exact bit output mismatch";
    test_reduce_histogram_adv_51: &[1u32; 0], 1u32 => vec![0], "FINDING-ADV-REDUCE-HISTOGRAM-51: Exact bit output mismatch";
    test_reduce_histogram_adv_52: &[1u32; 0], 1u32 => vec![0], "FINDING-ADV-REDUCE-HISTOGRAM-52: Exact bit output mismatch";
    test_reduce_histogram_adv_53: &[1u32; 0], 1u32 => vec![0], "FINDING-ADV-REDUCE-HISTOGRAM-53: Exact bit output mismatch";
    test_reduce_histogram_adv_54: &[1u32; 0], 1u32 => vec![0], "FINDING-ADV-REDUCE-HISTOGRAM-54: Exact bit output mismatch";
    test_reduce_histogram_adv_55: &[1u32; 0], 1u32 => vec![0], "FINDING-ADV-REDUCE-HISTOGRAM-55: Exact bit output mismatch";
    test_reduce_histogram_adv_56: &[1u32; 0], 1u32 => vec![0], "FINDING-ADV-REDUCE-HISTOGRAM-56: Exact bit output mismatch";
    test_reduce_histogram_adv_57: &[1u32; 0], 1u32 => vec![0], "FINDING-ADV-REDUCE-HISTOGRAM-57: Exact bit output mismatch";
    test_reduce_histogram_adv_58: &[1u32; 0], 1u32 => vec![0], "FINDING-ADV-REDUCE-HISTOGRAM-58: Exact bit output mismatch";
    test_reduce_histogram_adv_59: &[1u32; 0], 1u32 => vec![0], "FINDING-ADV-REDUCE-HISTOGRAM-59: Exact bit output mismatch";
    test_reduce_histogram_adv_60: &[1u32; 0], 1u32 => vec![0], "FINDING-ADV-REDUCE-HISTOGRAM-60: Exact bit output mismatch";
    test_reduce_histogram_adv_61: &[1u32; 0], 1u32 => vec![0], "FINDING-ADV-REDUCE-HISTOGRAM-61: Exact bit output mismatch";
    test_reduce_histogram_adv_62: &[1u32; 0], 1u32 => vec![0], "FINDING-ADV-REDUCE-HISTOGRAM-62: Exact bit output mismatch";
    test_reduce_histogram_adv_63: &[1u32; 0], 31u32 => vec![0; 31], "FINDING-ADV-REDUCE-HISTOGRAM-63: Exact bit output mismatch";
    test_reduce_histogram_adv_64: &[1u32; 0], 31u32 => vec![0; 31], "FINDING-ADV-REDUCE-HISTOGRAM-64: Exact bit output mismatch";
    test_reduce_histogram_adv_65: &[1u32; 0], 31u32 => vec![0; 31], "FINDING-ADV-REDUCE-HISTOGRAM-65: Exact bit output mismatch";
    test_reduce_histogram_adv_66: &[1u32; 0], 31u32 => vec![0; 31], "FINDING-ADV-REDUCE-HISTOGRAM-66: Exact bit output mismatch";
    test_reduce_histogram_adv_67: &[1u32; 0], 31u32 => vec![0; 31], "FINDING-ADV-REDUCE-HISTOGRAM-67: Exact bit output mismatch";
    test_reduce_histogram_adv_68: &[1u32; 0], 31u32 => vec![0; 31], "FINDING-ADV-REDUCE-HISTOGRAM-68: Exact bit output mismatch";
    test_reduce_histogram_adv_69: &[1u32; 0], 31u32 => vec![0; 31], "FINDING-ADV-REDUCE-HISTOGRAM-69: Exact bit output mismatch";
}
