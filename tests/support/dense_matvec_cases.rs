//! The declared dense byte-tile Four-Russians matvec case table, and the naive
//! boolean-semiring oracle every arm is measured against.
//!
//! Two suites asserted the same contract over the same corpus generator: the
//! primitive `dense_matvec_*` family owned by `vyre_primitives::bitset::four_russians`
//! and the substrate transform family in `vyre_libs::encoding::bitset_transform_pipeline`.
//! The two files were identical apart from their sweep bounds, and the bounds
//! disagreed in both directions: the primitive suite swept 0..=18 tiles by
//! 1..=5 destination words, the substrate suite swept 0..=24 by 1..=4, so 24
//! tiles at 5 destination words was exercised by neither. A corpus copied per
//! crate is a corpus whose holes nobody can see, because the two copies look
//! the same at a glance.
//!
//! This file owns which cases exist and what the answer is. What a crate
//! asserts about a case stays in that crate: an arm names the API it pins.
//! Consumers include this file with `#[path]`, the same way
//! `tests/support/sweep_rng.rs` is shared.
//!
//! [`arm_coverage`] is why a case group cannot be declared and then quietly
//! skipped: each arm records the groups it asserted and the ledger reads this
//! table back at run time, so a group added below with no arm in a crate turns
//! that crate's suite red instead of widening the table for nobody.

#![allow(dead_code)]

use vyre_test_support::case_table::ArmCoverage;

use vyre_primitives::bitset::four_russians::{
    frontier_words_for_byte_tiles, BYTE_TILE_STATES, BYTE_TILE_WIDTH,
};
use vyre_primitives::wire::pack_u32_slice;
use vyre_reference::value::Value;

/// Minimum declared group count, the floor [`arm_coverage`] enforces.
///
/// The table is enumerated by a function, so its failure mode is returning
/// almost nothing: an arm that covers two groups out of two is trivially
/// complete. The floor makes a broken table fail instead of reporting a clean
/// sweep of an empty set.
const MIN_DECLARED_GROUPS: usize = 4;

/// Minimum cases one arm must actually assert, across all groups.
///
/// Both original suites carried a `checked >= 6_000` floor for the same reason:
/// a sweep whose bounds collapse to zero iterations asserts nothing and passes.
const MIN_ASSERTED_CASES: usize = 6_000;

/// How a case fills its frontier words.
#[derive(Clone, Copy, Debug)]
pub(crate) enum FrontierShape {
    /// Pseudorandom words, masked down to the tiles the count actually covers.
    Pseudorandom,
    /// Exactly one tile active, carrying `active_byte`; every other tile clear.
    SingleTile { tile: u32, active_byte: u32 },
    /// Every covered tile fully active. The LUT row selected is the last one.
    Saturated,
}

/// One declared dense-matvec input.
#[derive(Clone, Copy, Debug)]
pub(crate) struct DenseMatvecCase {
    /// Byte tiles of source columns, eight source bits each.
    pub(crate) tile_count: u32,
    /// Destination words produced per tile.
    pub(crate) dst_words: u32,
    /// Seed the column corpus is drawn from.
    pub(crate) seed: u32,
    /// Frontier fill for this case.
    pub(crate) frontier_shape: FrontierShape,
}

/// One declared case group. `name` is the coverage key an arm records.
pub(crate) struct CaseGroup {
    /// Coverage key. Stable: an arm matches on it.
    pub(crate) name: &'static str,
    /// Every case in the group.
    pub(crate) cases: Vec<DenseMatvecCase>,
}

/// Active bytes swept by the single-tile group.
///
/// Empty, one bit, two adjacent bits, one bit high, and three widths of
/// low-run: the LUT row index is the active byte, so these pin row selection at
/// 0, at the powers of two either side of a byte boundary, and at the last row.
const SWEPT_ACTIVE_BYTES: [u32; 8] = [0, 1, 2, 3, 7, 31, 127, 255];

/// Every declared dense-matvec case group.
///
/// Bounds are the union of what the two copied suites swept, so no arm loses a
/// case to the merge, plus a saturated-frontier group neither had: both drew
/// pseudorandom frontiers, which reach the all-ones LUT row of a tile with
/// probability 2^-8 per tile and never reach it for every tile at once.
pub(crate) fn declared_groups() -> Vec<CaseGroup> {
    let mut groups = Vec::new();

    let mut frontier_sweep = Vec::new();
    for tile_count in 0..=24u32 {
        for dst_words in 1..=5u32 {
            for seed in 0..64u32 {
                frontier_sweep.push(DenseMatvecCase {
                    tile_count,
                    dst_words,
                    seed: seed ^ 0x1357_2468,
                    frontier_shape: FrontierShape::Pseudorandom,
                });
            }
        }
    }
    groups.push(CaseGroup {
        name: "frontier_sweep",
        cases: frontier_sweep,
    });

    let mut single_tile = Vec::new();
    for tile_count in 1..=32u32 {
        for tile in 0..tile_count {
            for active_byte in SWEPT_ACTIVE_BYTES {
                single_tile.push(DenseMatvecCase {
                    tile_count,
                    dst_words: 3,
                    seed: tile_count ^ 0xA5A5,
                    frontier_shape: FrontierShape::SingleTile { tile, active_byte },
                });
            }
        }
    }
    groups.push(CaseGroup {
        name: "single_tile_active_byte",
        cases: single_tile,
    });

    let mut saturated = Vec::new();
    for tile_count in 1..=16u32 {
        for dst_words in 1..=4u32 {
            saturated.push(DenseMatvecCase {
                tile_count,
                dst_words,
                seed: mix(tile_count.wrapping_mul(0x1000_0001) ^ dst_words),
                frontier_shape: FrontierShape::Saturated,
            });
        }
    }
    groups.push(CaseGroup {
        name: "saturated_frontier",
        cases: saturated,
    });

    groups.push(CaseGroup {
        name: "dirty_output_overwrite",
        cases: vec![
            DenseMatvecCase {
                tile_count: 5,
                dst_words: 3,
                seed: 0xC0DE_5EED,
                frontier_shape: FrontierShape::Pseudorandom,
            },
            DenseMatvecCase {
                tile_count: 3,
                dst_words: 2,
                seed: 0xC001_CAFE,
                frontier_shape: FrontierShape::Pseudorandom,
            },
        ],
    });

    groups
}

impl DenseMatvecCase {
    /// Source columns for this case: `tile_count * BYTE_TILE_WIDTH * dst_words`
    /// words, one per (source bit, destination word).
    pub(crate) fn columns(&self) -> Vec<u32> {
        let len = self.tile_count as usize * BYTE_TILE_WIDTH as usize * self.dst_words as usize;
        (0..len)
            .map(|idx| mix(self.seed ^ (idx as u32).wrapping_mul(0x9E37_79B9)))
            .collect()
    }

    /// Frontier words for this case, sized by
    /// [`frontier_words_for_byte_tiles`] and filled per [`FrontierShape`].
    pub(crate) fn frontier(&self) -> Vec<u32> {
        let len = frontier_words_for_byte_tiles(self.tile_count) as usize;
        match self.frontier_shape {
            FrontierShape::Pseudorandom => (0..len)
                .map(|idx| mix(self.seed ^ idx as u32) & self.covered_mask(idx))
                .collect(),
            FrontierShape::Saturated => (0..len).map(|idx| self.covered_mask(idx)).collect(),
            FrontierShape::SingleTile { tile, active_byte } => {
                let mut words = vec![0u32; len];
                words[(tile / 4) as usize] = active_byte << ((tile % 4) * 8);
                words
            }
        }
    }

    /// Boolean-semiring matvec computed straight from the source columns, with
    /// no LUT: the independent oracle both arms are measured against.
    pub(crate) fn naive(&self, columns: &[u32], frontier: &[u32]) -> Vec<u32> {
        let mut out = vec![0u32; self.dst_words as usize];
        for tile in 0..self.tile_count {
            let active_byte = if frontier.is_empty() {
                0
            } else {
                (frontier[(tile / 4) as usize] >> ((tile % 4) * 8)) & (BYTE_TILE_STATES - 1)
            };
            for source_bit in 0..BYTE_TILE_WIDTH {
                if (active_byte & (1 << source_bit)) == 0 {
                    continue;
                }
                for dst_word in 0..self.dst_words {
                    let column_idx = ((tile * BYTE_TILE_WIDTH + source_bit) * self.dst_words
                        + dst_word) as usize;
                    out[dst_word as usize] |= columns[column_idx];
                }
            }
        }
        out
    }

    /// Case identity for a failure message.
    pub(crate) fn label(&self) -> String {
        format!(
            "tile_count={}, dst_words={}, seed={:#010x}, frontier={:?}",
            self.tile_count, self.dst_words, self.seed, self.frontier_shape
        )
    }

    /// Bits of frontier word `idx` that name a tile this case actually has.
    ///
    /// The trailing word of a frontier is partially covered whenever the tile
    /// count is not a multiple of four, and a bit past the last tile is not a
    /// source the LUT has a column for, so leaving it set would ask the oracle
    /// and the LUT to agree about a tile that does not exist.
    fn covered_mask(&self, idx: usize) -> u32 {
        let covered_tiles = self.tile_count.saturating_sub((idx as u32) * 4).min(4);
        let covered_bits = covered_tiles * BYTE_TILE_WIDTH;
        if covered_bits == 0 {
            0
        } else if covered_bits >= 32 {
            u32::MAX
        } else {
            u32::MAX >> (32 - covered_bits)
        }
    }
}

/// One LUT per column-corpus shape.
///
/// Building the LUT is the whole cost of a sweep: `single_tile_active_byte`
/// declares 4224 cases over 32 distinct corpora, so a rebuild per case would
/// spend 132 times the work for the same assertions. The key is the corpus
/// identity, not the frontier, because the frontier is what the group varies.
pub(crate) struct LutCache {
    key: Option<(u32, u32, u32)>,
    columns: Vec<u32>,
    lut: Vec<u32>,
}

impl LutCache {
    /// An empty cache; the first case builds.
    pub(crate) fn new() -> Self {
        Self {
            key: None,
            columns: Vec::new(),
            lut: Vec::new(),
        }
    }

    /// Columns and LUT for `case`, building through `build` only when the
    /// corpus shape changed.
    pub(crate) fn get(
        &mut self,
        case: &DenseMatvecCase,
        build: impl FnOnce(&[u32], u32, u32) -> Vec<u32>,
    ) -> (&[u32], &[u32]) {
        let key = (case.tile_count, case.dst_words, case.seed);
        if self.key != Some(key) {
            self.columns = case.columns();
            self.lut = build(&self.columns, case.tile_count, case.dst_words);
            self.key = Some(key);
        }
        (&self.columns, &self.lut)
    }
}

/// A dense-matvec byte-LUT builder: columns, tile count, destination words.
pub(crate) type LutBuilder = fn(&[u32], u32, u32) -> Vec<u32>;

/// A dense-matvec dispatch Program builder: frontier, LUT and output buffer
/// names, then the tile count and destination words the shape is pinned to.
pub(crate) type MatvecProgramBuilder =
    fn(&str, &str, &str, u32, u32) -> vyre_foundation::ir::Program;

/// Assert `program` overwrites a dirty output buffer with the boolean-semiring
/// result rather than accumulating into it, for every case.
///
/// The dirty buffer is all-ones, so an accumulating implementation returns
/// all-ones and a correct one returns the oracle. Both arms drove the reference
/// interpreter with the same three buffers in the same order, differing only in
/// which builders they passed, so `arm` names the crate under test in the
/// failure message and the builders are arguments.
pub(crate) fn assert_program_overwrites_dirty_output(
    arm: &str,
    cases: &[DenseMatvecCase],
    lut_of: LutBuilder,
    program_of: MatvecProgramBuilder,
) {
    for case in cases {
        let columns = case.columns();
        let lut = lut_of(&columns, case.tile_count, case.dst_words);
        let frontier = case.frontier();
        let expected = case.naive(&columns, &frontier);
        let program = program_of(
            "frontier",
            "tile_lut",
            "out",
            case.tile_count,
            case.dst_words,
        );
        let outputs = vyre_reference::reference_eval(
            &program,
            &[
                Value::from(pack_u32_slice(&frontier)),
                Value::from(pack_u32_slice(&lut)),
                Value::from(pack_u32_slice(&vec![u32::MAX; case.dst_words as usize])),
            ],
        )
        .unwrap_or_else(|err| {
            panic!(
                "Fix: the {arm} dense matvec Program must execute in the reference oracle: {err}"
            )
        });

        assert_eq!(
            outputs[0].to_bytes(),
            pack_u32_slice(&expected),
            "Fix: the {arm} dense matvec Program must overwrite dirty output with the exact boolean-semiring result for {}.",
            case.label()
        );
    }
}

/// This crate's ledger over the declared dense-matvec groups.
///
/// The declared set is read from [`declared_groups`] on each call, so it is
/// whatever the table says on this run.
pub(crate) fn arm_coverage() -> ArmCoverage {
    ArmCoverage::new(
        "dense-matvec",
        "tests/support/dense_matvec_cases.rs",
        declared_groups().iter().map(|group| group.name).collect(),
        MIN_DECLARED_GROUPS,
        MIN_ASSERTED_CASES,
    )
}

/// Deterministic 32-bit mix, the corpus generator both arms draw from.
fn mix(mut value: u32) -> u32 {
    value ^= value >> 16;
    value = value.wrapping_mul(0x7FEB_352D);
    value ^= value >> 15;
    value = value.wrapping_mul(0x846C_A68B);
    value ^ (value >> 16)
}
