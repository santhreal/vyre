//! Canonical megakernel wave-policy corpora, shared by the neutral planner
//! tests and every backend parity gate.
//!
//! Barrier placement, wave/topology selection, and frontier scheduling are one
//! backend-neutral policy that this crate owns. Proving a backend did not fork
//! that policy means driving both entry points with the *same* inputs, so the
//! inputs cannot be a copy: a corpus edited on one side turns a parity gate into
//! two suites that agree about nothing while still passing.
//!
//! Each item below is therefore the one definition of a shape the policy has to
//! decide, named after the decision it forces rather than after its numbers.
//! Every value is expressed against [`crate::megakernel_barrier`] and
//! [`crate::megakernel_frontier`], so nothing here names a target, dialect or
//! driver.
//!
//! Enabled by the `test-fixtures` feature: it is scaffolding, not product code,
//! and a published build should not carry it.

use crate::megakernel_barrier::MegakernelWaveDependency;
use crate::megakernel_frontier::MegakernelFrontierWave;

/// A three-wave serial chain: every wave depends on the previous one, so no two
/// can share a barrier-free group.
pub const CHAIN_DEPENDENCIES: &[MegakernelWaveDependency] = &[
    MegakernelWaveDependency {
        before: 0,
        after: 1,
    },
    MegakernelWaveDependency {
        before: 1,
        after: 2,
    },
];

/// A four-wave chain, one barrier per edge.
pub const LONG_CHAIN_DEPENDENCIES: &[MegakernelWaveDependency] = &[
    MegakernelWaveDependency {
        before: 0,
        after: 1,
    },
    MegakernelWaveDependency {
        before: 1,
        after: 2,
    },
    MegakernelWaveDependency {
        before: 2,
        after: 3,
    },
];

/// A diamond: wave 0 fans out to 1 and 2, which both feed wave 3.
///
/// This is the shape that distinguishes a planner that fuses independent middle
/// waves from one that serializes them, so it is the corpus every barrier and
/// frontier decision is checked against.
pub const DIAMOND_DEPENDENCIES: &[MegakernelWaveDependency] = &[
    MegakernelWaveDependency {
        before: 0,
        after: 1,
    },
    MegakernelWaveDependency {
        before: 0,
        after: 2,
    },
    MegakernelWaveDependency {
        before: 1,
        after: 3,
    },
    MegakernelWaveDependency {
        before: 2,
        after: 3,
    },
];

/// A two-wave cycle, which no schedule can satisfy.
pub const CYCLE_DEPENDENCIES: &[MegakernelWaveDependency] = &[
    MegakernelWaveDependency {
        before: 0,
        after: 1,
    },
    MegakernelWaveDependency {
        before: 1,
        after: 0,
    },
];

/// One wave whose fused budget is a third of the group budget the scenarios use.
pub const ONE_WAVE: &[MegakernelFrontierWave] = &[MegakernelFrontierWave {
    frontier_bytes: 40,
    scratch_bytes: 15,
    output_bytes: 0,
}];

/// Three waves whose fused budgets sum to exactly one group budget, which is the
/// boundary between fitting and splitting.
pub const THREE_EQUAL_WAVES: &[MegakernelFrontierWave] = &[
    MegakernelFrontierWave {
        frontier_bytes: 40,
        scratch_bytes: 15,
        output_bytes: 0,
    },
    MegakernelFrontierWave {
        frontier_bytes: 40,
        scratch_bytes: 15,
        output_bytes: 0,
    },
    MegakernelFrontierWave {
        frontier_bytes: 40,
        scratch_bytes: 15,
        output_bytes: 0,
    },
];

/// Four waves of strictly growing footprint, matched to
/// [`DIAMOND_DEPENDENCIES`].
///
/// The growth is what makes peak accounting observable: the peak of the fused
/// middle group is wave 3's footprint alone, not the sum of all four.
pub const DIAMOND_WAVES: &[MegakernelFrontierWave] = &[
    MegakernelFrontierWave {
        frontier_bytes: 1_024,
        scratch_bytes: 512,
        output_bytes: 256,
    },
    MegakernelFrontierWave {
        frontier_bytes: 2_048,
        scratch_bytes: 1_024,
        output_bytes: 512,
    },
    MegakernelFrontierWave {
        frontier_bytes: 4_096,
        scratch_bytes: 2_048,
        output_bytes: 1_024,
    },
    MegakernelFrontierWave {
        frontier_bytes: 8_192,
        scratch_bytes: 4_096,
        output_bytes: 2_048,
    },
];

/// Two independent waves whose static output volume exceeds any plausible
/// measured readback, so the plan must amortize against the static figure.
pub const OUTPUT_HEAVY_WAVES: &[MegakernelFrontierWave] = &[
    MegakernelFrontierWave {
        frontier_bytes: 1_024,
        scratch_bytes: 512,
        output_bytes: 3_072,
    },
    MegakernelFrontierWave {
        frontier_bytes: 1_024,
        scratch_bytes: 512,
        output_bytes: 3_072,
    },
];

/// Three identical small waves, for checking that an independent layer splits
/// into budget-sized groups rather than one oversized fused group.
pub const THREE_SMALL_WAVES: &[MegakernelFrontierWave] = &[
    MegakernelFrontierWave {
        frontier_bytes: 10,
        scratch_bytes: 10,
        output_bytes: 10,
    },
    MegakernelFrontierWave {
        frontier_bytes: 10,
        scratch_bytes: 10,
        output_bytes: 10,
    },
    MegakernelFrontierWave {
        frontier_bytes: 10,
        scratch_bytes: 10,
        output_bytes: 10,
    },
];

/// Two waves whose combined frontier bytes overflow `u64`.
///
/// The first wave alone is `u64::MAX`, so any accumulation across the pair must
/// be rejected before launch planning instead of wrapping.
pub const OVERFLOW_WAVES: &[MegakernelFrontierWave] = &[
    MegakernelFrontierWave {
        frontier_bytes: u64::MAX,
        scratch_bytes: 1,
        output_bytes: 1,
    },
    MegakernelFrontierWave {
        frontier_bytes: 1,
        scratch_bytes: 1,
        output_bytes: 1,
    },
];
