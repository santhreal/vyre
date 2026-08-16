mod wire_words;
use wire_words::{alternating, lcg_u32 as lcg, ramp};

//! Oracle matrix for every generated-volume reduce sweep.
//!
//! One shared hostile case list feeds every reducer, and each call shape has one
//! assertion body driven by a `const` row list of `(op name, production cpu_ref,
//! independent oracle)`. The oracle for a row is always built a structurally
//! different way from the production reference (Kernighan popcount against
//! `count_ones`, widened u64 accumulation against wrapping u32 folds, per-bit
//! "any" against an OR fold), so a row proves the reference rather than
//! restating it.
//!
//! Volume testing.volume - do NOT weaken to shape-only asserts.

#![forbid(unsafe_code)]
#![cfg(all(feature = "reduce", feature = "cpu-parity"))]

use vyre_libs::reduce::{
    all, any, count, count_non_zero, gather, histogram, max, min, multi_block_prefix_scan,
    range_counts, scatter, sum, workgroup_any,
};

type ScalarReduce = fn(&[u32]) -> u32;
type VectorReduce = fn(&[u32]) -> Vec<u32>;

/// Reducers collapsing a u32 slice to one u32.
const SCALAR: &[(&str, ScalarReduce, ScalarReduce)] = &[
    ("reduce_any", any::cpu_ref, |input| {
        u32::from(input.iter().any(|value| *value != 0))
    }),
    ("reduce_all", all::cpu_ref, |input| {
        u32::from(input.iter().all(|value| *value != 0))
    }),
    ("reduce_sum", sum::cpu_ref, |input| {
        // Widened accumulation truncated once, versus the production per-element
        // wrapping u32 fold.
        let wide: u64 = input.iter().map(|value| u64::from(*value)).sum();
        wide as u32
    }),
    ("reduce_min", min::cpu_ref, |input| {
        input.iter().copied().min().unwrap_or(u32::MAX)
    }),
    ("reduce_max", max::cpu_ref, |input| {
        input.iter().copied().max().unwrap_or(0)
    }),
    ("reduce_count", count::cpu_ref, |input| {
        // Kernighan clear-lowest-set-bit loop, versus the production `count_ones`.
        input.iter().copied().map(kernighan_popcount).sum()
    }),
    ("reduce_count_non_zero", count_non_zero::cpu_ref, |input| {
        // Complement of the zero count, versus the production non-zero filter.
        let zeros = input.iter().filter(|value| **value == 0).count();
        (input.len() - zeros) as u32
    }),
    ("reduce_workgroup_any", workgroup_any::cpu_ref, |input| {
        // `workgroup_any_u32` reduces with `Expr::bitor`, so the result is the
        // bitwise OR of every value (`[0,0,7,0] -> 7`), not a boolean 0/1.
        // Computed here as the halving tree the GPU actually emits, versus the
        // production left-to-right fold.
        tree_or(input)
    }),
];

/// Reducers producing one u32 per input element.
const VECTOR: &[(&str, VectorReduce, VectorReduce)] = &[
    (
        "multi_block_prefix_scan_inclusive",
        multi_block_prefix_scan::cpu_ref,
        |input| {
            let mut acc = 0u32;
            input
                .iter()
                .map(|value| {
                    acc = acc.wrapping_add(*value);
                    acc
                })
                .collect()
        },
    ),
    (
        "multi_block_prefix_scan_exclusive",
        multi_block_prefix_scan::cpu_ref_exclusive,
        |input| {
            let mut acc = 0u32;
            input
                .iter()
                .map(|value| {
                    let carried = acc;
                    acc = acc.wrapping_add(*value);
                    carried
                })
                .collect()
        },
    ),
];

#[test]
fn scalar_reducers_match_independent_oracles() {
    for (name, actual, expected) in SCALAR {
        for (case_idx, input) in reduce_inputs().enumerate() {
            assert_eq!(
                actual(&input),
                expected(&input),
                "Fix: {name} case {case_idx} len={} must match the independent oracle.",
                input.len()
            );
        }
    }
}

#[test]
fn vector_reducers_match_independent_oracles() {
    for (name, actual, expected) in VECTOR {
        for (case_idx, input) in reduce_inputs().enumerate() {
            let expected_out = expected(&input);
            assert_eq!(
                actual(&input),
                expected_out,
                "Fix: {name} case {case_idx} len={} must match the independent oracle.",
                input.len()
            );
        }
    }
}

#[test]
fn multi_block_prefix_scan_into_overwrites_stale_output() {
    for (case_idx, input) in reduce_inputs().enumerate() {
        let mut out = vec![0xDEAD_BEEFu32; input.len() + 7];
        multi_block_prefix_scan::cpu_ref_into(&input, &mut out);
        assert_eq!(
            out,
            multi_block_prefix_scan::cpu_ref(&input),
            "Fix: multi_block_prefix_scan cpu_ref_into case {case_idx} len={} must overwrite every stale word and resize the buffer.",
            input.len()
        );
    }
}

#[test]
fn gather_matches_independent_oracle() {
    for (case_idx, (src, indices)) in index_pairs().enumerate() {
        let expected: Vec<u32> = indices
            .iter()
            .map(|index| src.get(*index as usize).copied().unwrap_or(0))
            .collect();
        assert_eq!(
            gather::cpu_ref(&src, &indices),
            expected,
            "Fix: reduce_gather case {case_idx} src_len={} indices_len={} must match the independent oracle.",
            src.len(),
            indices.len()
        );
        let mut out = vec![0xDEAD_BEEFu32; indices.len() + 5];
        gather::cpu_ref_into(&src, &indices, &mut out);
        assert_eq!(
            out, expected,
            "Fix: reduce_gather cpu_ref_into case {case_idx} must overwrite every stale word and resize the buffer."
        );
    }
}

#[test]
fn scatter_matches_independent_oracle() {
    for (case_idx, (src, indices)) in index_pairs().enumerate() {
        let dst_len = 1 + (case_idx % 64);
        let mut expected = vec![0u32; dst_len];
        for (value, index) in src.iter().zip(&indices) {
            if let Some(slot) = expected.get_mut(*index as usize) {
                *slot = *value;
            }
        }
        assert_eq!(
            scatter::cpu_ref(&src, &indices, dst_len),
            expected,
            "Fix: reduce_scatter case {case_idx} src_len={} dst_len={dst_len} must match the independent oracle.",
            src.len()
        );
        let mut out = vec![0xDEAD_BEEFu32; dst_len + 5];
        scatter::cpu_ref_into(&src, &indices, dst_len, &mut out);
        assert_eq!(
            out, expected,
            "Fix: reduce_scatter cpu_ref_into case {case_idx} must overwrite every stale word and resize the buffer."
        );
    }
}

#[test]
fn histogram_matches_independent_oracle() {
    for (case_idx, (input, num_bins)) in histogram_inputs().enumerate() {
        // Per-bin scan of the whole input, versus the production per-element
        // increment into an accumulator.
        let expected: Vec<u32> = (0..num_bins)
            .map(|bin| input.iter().filter(|value| **value == bin).count() as u32)
            .collect();
        assert_eq!(
            histogram::cpu_ref(&input, num_bins),
            expected,
            "Fix: reduce_histogram case {case_idx} len={} num_bins={num_bins} must match the independent oracle.",
            input.len()
        );
        let mut out = vec![0xDEAD_BEEFu32; num_bins as usize + 5];
        histogram::cpu_ref_into(&input, num_bins, &mut out);
        assert_eq!(
            out, expected,
            "Fix: reduce_histogram cpu_ref_into case {case_idx} must overwrite every stale bin and resize the buffer."
        );
    }
}

#[test]
fn range_counts_matches_independent_oracle() {
    for (case_idx, input) in reduce_inputs().enumerate() {
        let start = (case_idx % 8) as u32;
        let end = start + 1 + ((case_idx >> 3) % 8) as u32;
        // WRAPPING sum: the GPU IR (and `range_counts::cpu_ref`) accumulate u32
        // with two's-complement wraparound, so a widened oracle must truncate
        // the same way rather than reject the full-range inputs this sweep feeds.
        let clamped_end = (end as usize).min(input.len());
        let expected = if (start as usize) >= clamped_end {
            0
        } else {
            let wide: u64 = input[start as usize..clamped_end]
                .iter()
                .map(|value| u64::from(*value))
                .sum();
            wide as u32
        };
        assert_eq!(
            range_counts::cpu_ref(&input, start, end),
            expected,
            "Fix: reduce_range_counts case {case_idx} len={} start={start} end={end} must match the independent oracle.",
            input.len()
        );
    }
}

fn kernighan_popcount(mut word: u32) -> u32 {
    let mut set = 0u32;
    while word != 0 {
        word &= word - 1;
        set += 1;
    }
    set
}

fn tree_or(values: &[u32]) -> u32 {
    match values.len() {
        0 => 0,
        1 => values[0],
        len => tree_or(&values[..len / 2]) | tree_or(&values[len / 2..]),
    }
}

/// Input/bin-count pairs where most values land inside the bin range, so both
/// the in-range increment and the `bin < num_bins` drop are exercised on every
/// case. A uniform u32 population would leave every bin at zero.
fn histogram_inputs() -> impl Iterator<Item = (Vec<u32>, u32)> {
    (0..CASES).map(|case| {
        let num_bins = (4 + (case % 32)) as u32;
        let input = lcg(case as u32, case % 97)
            .into_iter()
            .enumerate()
            .map(|(slot, raw)| {
                if slot % 4 == 3 {
                    raw
                } else {
                    raw % (num_bins * 2)
                }
            })
            .collect();
        (input, num_bins)
    })
}

/// Hostile fixed shapes plus three independent generated populations. Every
/// reducer sees the whole list, so no op is swept over a narrower population
/// than another.
fn reduce_inputs() -> impl Iterator<Item = Vec<u32>> {
    let fixed_lengths = [0usize, 1, 2, 31, 32, 33, 64, 65, 127, 128, 255, 256, 1024];
    let fixed = fixed_lengths.into_iter().flat_map(|len| {
        [
            vec![0u32; len],
            vec![u32::MAX; len],
            ramp(len, 0x1357_9BDF),
            alternating(len, 0x5555_5555, 0xAAAA_AAAA),
        ]
    });
    let short = (0..CASES).map(|case| {
        let len = match case % 20 {
            0 => 0,
            1 => 1,
            2 => 32,
            3 => 257,
            _ => 1 + (case % 130),
        };
        lcg(case as u32 ^ 0xAED0_CE00, len)
    });
    let wide = (0..CASES).map(|case| lcg(case as u32, 1 + (case % 200)));
    let dense = (0..CASES).map(|case| lcg(case as u32, 1 + (case % 129)));
    fixed.chain(short).chain(wide).chain(dense)
}

/// Value/index pairs for the indexed-move shapes. Indices are drawn from the
/// same generator as values, so most land out of range and exercise the
/// bounds gate the GPU IR applies.
fn index_pairs() -> impl Iterator<Item = (Vec<u32>, Vec<u32>)> {
    (0..CASES).map(|case| {
        let values = lcg(case as u32, 1 + (case % 64));
        let indices = lcg(case as u32 ^ 0x1DEF_0001, 1 + (case % 64))
            .into_iter()
            .enumerate()
            .map(|(slot, raw)| {
                // Half in range, half hostile, so both the copied and the
                // dropped branch are swept on every case.
                if slot % 2 == 0 {
                    raw % 64
                } else {
                    raw
                }
            })
            .collect();
        (values, indices)
    })
}

const CASES: usize = 16384;



