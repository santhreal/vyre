//! `grid_stride_tree_sum_u32` sums the whole input at every shape it admits.
//!
//! WHY: the builder chooses between a single-block form and a strided two-pass
//! form, and the single-block form reads one tile and nothing else. It used to
//! be selected whenever the caller asked for one block, so at `count > tile` it
//! returned a program that summed a prefix and reported it as the total. On a
//! backend whose device profile reports no compute units the block count is one,
//! which made a 1M-element release reduction return the sum of its first 256
//! elements: 32640 against 133693440, a factor of exactly `count / tile`.
//!
//! The shapes below are a cross product computed here rather than a list of
//! recorded answers: every expected value is the sum of the input the case
//! builds, so a shape whose program covers less than the input fails on the
//! arithmetic instead of on a pinned constant. A new admitted shape is covered
//! by adding its extent to one of the arrays.
//!
//! This also exercises the launch width. `reference_eval` infers its own grid
//! from the program, and so does a compiled artifact, so the two-pass form runs
//! under the grid its own buffer table implies. Its partial store is still
//! guarded on the block index: a caller that over-fires the launch discards the
//! extra blocks instead of writing past the partial buffer.
#![cfg(feature = "reduce")]

use vyre_libs_reduce::reduce::grid_stride_tree::{
    grid_stride_tree_sum_u32, grid_stride_tree_sum_u32_blocks,
};
use vyre_primitives::wire::pack_u32_slice as pack_u32;
use vyre_reference::value::Value;

/// The input the reduction is measured over: bounded per lane so the total
/// cannot wrap, and not constant so a program that reads one tile cannot
/// coincide with one that reads every element.
fn values(count: u32) -> Vec<u32> {
    (0..count)
        .map(|index| index.wrapping_mul(17).wrapping_add(3) & 0xff)
        .collect()
}

fn reduced(count: u32, tile: u32) -> u32 {
    let input = values(count);
    let program = grid_stride_tree_sum_u32("values", "out", count, tile);
    // `out` is a backend-allocated output, so the dispatch carries one input.
    let outputs =
        vyre_reference::ReferenceRequest::standard(&program, &[Value::from(pack_u32(&input))])
            .outputs()
            .unwrap_or_else(|error| {
                panic!("Fix: count={count} tile={tile} must evaluate: {error}")
            });
    let bytes = outputs
        .last()
        .unwrap_or_else(|| panic!("Fix: count={count} tile={tile} must report an output buffer"))
        .to_bytes();
    assert!(
        bytes.len() >= 4,
        "Fix: count={count} tile={tile} reported {} output byte(s)",
        bytes.len()
    );
    u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]])
}

#[test]
fn every_admitted_shape_sums_every_element() {
    let mut covered_multi_pass = 0usize;
    let mut covered_single_block = 0usize;
    // 513 and 1025 exceed one reducing workgroup's span at the 16- and
    // 32-wide tiles without being a multiple of it, which is the only shape
    // where a span count that truncates instead of rounding up drops the tail
    // of the input. Every smaller count fits one span, so the whole sweep
    // agreed with a builder that covered part of its input.
    for count in [1u32, 31, 32, 33, 256, 257, 512, 513, 1024, 1025] {
        let expected = values(count)
            .into_iter()
            .fold(0u32, |total, value| total.wrapping_add(value));
        for tile in [16u32, 32, 64, 256] {
            let blocks = grid_stride_tree_sum_u32_blocks(count, tile);
            if count <= tile {
                covered_single_block += 1;
            } else {
                covered_multi_pass += 1;
            }
            assert_eq!(
                reduced(count, tile),
                expected,
                "Fix: count={count} tile={tile} (blocks {blocks}) summed a subset of its input"
            );
        }
    }
    assert!(
        covered_single_block > 0 && covered_multi_pass > 0,
        "Fix: the sweep covered {covered_single_block} single-block and {covered_multi_pass} two-pass shape(s); both forms must be reached or the sweep proves one of them"
    );
}

/// WHY: this is the shape the release reduction hit, and the one the builder
/// answered with a prefix. It is stated on its own so a future narrowing of the
/// sweep above cannot drop it silently. The extents stay small because the
/// reference interpreter walks every lane of every workgroup: the defect is a
/// function of `count > tile`, not of the size.
#[test]
fn many_tiles_are_not_summed_as_a_prefix() {
    let count = 1024u32;
    let tile = 256;
    let expected = values(count)
        .into_iter()
        .fold(0u32, |total, value| total.wrapping_add(value));
    let prefix = values(tile)
        .into_iter()
        .fold(0u32, |total, value| total.wrapping_add(value));
    assert_ne!(
        expected, prefix,
        "Fix: the fixture must make a prefix sum distinguishable from the total"
    );
    assert_eq!(
        reduced(count, tile),
        expected,
        "Fix: {} tiles returned the first tile's sum",
        count / tile
    );
}

/// WHY: a caller that over-fires the launch runs blocks the partial buffer has
/// no slot for, so an unguarded store walks past it. The interpreter absorbs an
/// out-of-bounds store as a no-op, which is exactly the masking a real backend
/// does not do, so this asserts the tally rather than the value alone.
#[test]
fn a_launch_wider_than_the_built_grid_stays_in_bounds() {
    let count = 1024u32;
    let expected = values(count)
        .into_iter()
        .fold(0u32, |total, value| total.wrapping_add(value));
    let input = values(count);
    for tile in [16u32, 64, 256] {
        let program = grid_stride_tree_sum_u32("values", "out", count, tile);
        let blocks = grid_stride_tree_sum_u32_blocks(count, tile);
        // Four times the lanes the built grid covers, so every launch here fires
        // blocks the program has no partial slot for.
        let over_fire = blocks * tile * 4;
        let over_fire_inputs = [Value::from(pack_u32(&input))];
        let request = vyre_reference::ReferenceRequest::standard(&program, &over_fire_inputs)
            .with_min_dispatch_elements(over_fire);
        let oob = request
            .execute_permissive()
            .unwrap_or_else(|error| panic!("Fix: tile={tile} must evaluate over-fired: {error}"))
            .oob_report;
        assert_eq!(
            oob.total(),
            0,
            "Fix: tile={tile} (blocks {blocks}) indexed past a buffer under a {over_fire}-lane launch: {oob:?}"
        );
        let outputs = request
            .outputs()
            .unwrap_or_else(|error| panic!("Fix: tile={tile} must evaluate over-fired: {error}"));
        let bytes = outputs
            .last()
            .unwrap_or_else(|| panic!("Fix: tile={tile} must report an output buffer"))
            .to_bytes();
        assert_eq!(
            u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]),
            expected,
            "Fix: tile={tile} (blocks {blocks}) changed its answer under a {over_fire}-lane launch"
        );
    }
}
