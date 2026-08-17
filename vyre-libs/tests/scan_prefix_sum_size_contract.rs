//! `scan_prefix_sum` is the one size contract over the two scan bodies.
//!
//! The builder picks the compact workgroup scan at or under 1024 elements and
//! the multi-block chain above it. Those are different algorithms with
//! different launch geometry, and the boundary between them is the seam where a
//! scan silently returns a prefix of the right answer. This runs the emitted IR
//! on the reference interpreter on both sides of the seam and compares against
//! a host wrapping scan, so a builder that traps, truncates, or drops a block
//! carry at the boundary fails here rather than in a consumer.

#![cfg(feature = "math-scan")]

use vyre_foundation::ir::BufferAccess;
use vyre_libs::math::scan::scan_prefix_sum;
use vyre_reference::value::Value;

/// Both sides of the single-block limit, both sides of the workgroup cap, and
/// two lengths that divide by neither.
const SIZES: [u32; 10] = [1, 2, 255, 256, 257, 1023, 1024, 1025, 4095, 12_289];

#[test]
fn scan_matches_the_host_oracle_across_the_algorithm_boundary() {
    for n in SIZES {
        let input: Vec<u32> = (0..n).map(|i| (i % 251) + 1).collect();
        assert_eq!(
            run_scan(n, &input),
            wrapping_scan(&input),
            "n={n}: the emitted scan diverged from the host oracle"
        );
    }
}

#[test]
fn scan_wraps_modulo_two_to_the_thirty_second_on_both_paths() {
    for n in [1024_u32, 1025, 3000] {
        let mut input = vec![1_u32; n as usize];
        input[(n / 3) as usize] = u32::MAX;
        input[(n - 1) as usize] = u32::MAX;
        assert_eq!(
            run_scan(n, &input),
            wrapping_scan(&input),
            "n={n}: the scan must wrap modulo 2^32 rather than saturate"
        );
    }
}

#[test]
fn scan_traps_on_an_empty_input() {
    let program = scan_prefix_sum("input", "output", 0);
    assert!(
        program.stats().trap(),
        "n=0 must trap: an empty scan has no output surface to define"
    );
}

/// Run the emitted Program on the reference interpreter and return `output`.
///
/// The multi-block path fuses in scratch buffers, so this seeds a zero slot for
/// every non-workgroup buffer and locates `output` among the writable ones
/// rather than assuming a binding index.
fn run_scan(n: u32, input: &[u32]) -> Vec<u32> {
    let program = scan_prefix_sum("input", "output", n);
    let mut values = Vec::new();
    let mut output_slot = None;
    let mut writable = 0_usize;
    for buffer in program.buffers() {
        if buffer.access() == BufferAccess::Workgroup {
            continue;
        }
        let bytes = if buffer.name() == "input" {
            input.iter().flat_map(|word| word.to_le_bytes()).collect()
        } else {
            vec![0_u8; (buffer.count() as usize) * 4]
        };
        values.push(Value::from(bytes));
        if buffer.access() == BufferAccess::ReadWrite {
            if buffer.name() == "output" {
                output_slot = Some(writable);
            }
            writable += 1;
        }
    }
    let outputs = vyre_reference::reference_eval(&program, &values)
        .expect("Fix: scan_prefix_sum must execute on the reference interpreter");
    let slot = output_slot.expect("Fix: scan_prefix_sum must declare a writable output buffer");
    outputs[slot]
        .to_bytes()
        .chunks_exact(4)
        .map(|word| u32::from_le_bytes([word[0], word[1], word[2], word[3]]))
        .collect()
}

fn wrapping_scan(input: &[u32]) -> Vec<u32> {
    let mut running = 0_u32;
    input
        .iter()
        .map(|value| {
            running = running.wrapping_add(*value);
            running
        })
        .collect()
}
