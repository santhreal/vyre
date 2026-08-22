//! The multi-block prefix scan's answer on a real device, across the sizes that
//! change how it is scheduled.
//!
//! WHY: the scan's shape was reviewed at the IR level and its reference answer
//! was covered on the host interpreter, but no test had ever asked a device what
//! it computed for an input large enough to need more than one block. Two
//! separate things hide there. The chain recurses once per level of block
//! totals, so the number of levels is a function of the input size and every
//! level is a place the offsets can be applied to the wrong elements. And above
//! the device's cooperative residency the whole program stops being one launch
//! and becomes a host-orchestrated sequence, so the answer depends on state
//! surviving between launches rather than on a barrier inside one.
//!
//! Sizes here straddle both transitions: one block, several blocks inside one
//! level of recursion, a size that forces a second level, and a size past the
//! cooperative residency bound of any current device. A single size proves
//! whichever of those it happens to land in and nothing else, which is how a
//! wrong answer at a megabyte survived a green suite.

#![cfg(feature = "device-tests")]

use vyre_driver::DispatchConfig;
use vyre_driver_cuda::CudaBackend;
use vyre_libs::math::scan::scan_prefix_sum;

/// The same deterministic fill the CUB baseline benchmark uses: every value is
/// nonzero, so a dropped element changes the sum, and the pattern repeats with a
/// period coprime to every block width in play, so a misapplied block offset
/// cannot coincidentally land on an equal value.
fn host_input(elements: u32) -> Vec<u32> {
    (0..elements).map(|index| (index % 7) + 1).collect()
}

fn host_inclusive_scan(input: &[u32]) -> Vec<u32> {
    let mut running = 0_u32;
    input
        .iter()
        .map(|value| {
            running = running.wrapping_add(*value);
            running
        })
        .collect()
}

fn unpack_u32(bytes: &[u8]) -> Vec<u32> {
    bytes
        .chunks_exact(4)
        .map(|word| u32::from_le_bytes([word[0], word[1], word[2], word[3]]))
        .collect()
}

/// Report the first divergence rather than the count, and say what the index
/// means: which block it lands in is the difference between a wrong local scan
/// and a wrong block offset.
fn first_divergence(actual: &[u32], expected: &[u32]) -> Option<String> {
    if actual.len() != expected.len() {
        return Some(format!(
            "length {} but expected {}",
            actual.len(),
            expected.len()
        ));
    }
    actual
        .iter()
        .zip(expected)
        .enumerate()
        .find(|(_, (got, want))| got != want)
        .map(|(index, (got, want))| {
            format!(
                "index {index} (block {} of 256 lanes, lane {}): got {got}, expected {want}",
                index / 256,
                index % 256
            )
        })
}

#[test]
fn the_multi_block_scan_answers_correctly_at_every_scheduling_size() {
    let backend = CudaBackend::acquire()
        .expect("Fix: CUDA backend acquisition must succeed on the GPU-required test host.");

    // 256: one block, no chain at all.
    // 1024: several blocks, one level of block-total recursion.
    // 65_536: block totals themselves exceed one block, forcing a second level.
    // 1_048_576: 4096 blocks, past the cooperative residency of every current
    //            device, so the program runs as a host-orchestrated split.
    for elements in [256_u32, 1024, 65_536, 1_048_576] {
        let input = host_input(elements);
        let expected = host_inclusive_scan(&input);
        let program = scan_prefix_sum("input", "output", elements);
        let inputs = vec![vyre_primitives::wire::pack_u32_slice(&input)];

        let outputs = backend
            .dispatch(&program, &inputs, &DispatchConfig::default())
            .unwrap_or_else(|error| {
                panic!("Fix: the {elements}-element scan must dispatch on this device: {error}")
            });
        let actual = unpack_u32(
            outputs
                .last()
                .expect("Fix: the scan program declares one output buffer."),
        );

        assert!(
            first_divergence(&actual, &expected).is_none(),
            "Fix: the {elements}-element inclusive scan is wrong on device at {}. A divergence \
             whose index is a multiple of the block width means the per-block offset is applied \
             to the wrong block; one inside a single block means the local scan is wrong.",
            first_divergence(&actual, &expected).unwrap_or_default()
        );
    }
}
