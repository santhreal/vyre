//! GPU-IR fixpoint parity for `graph::csr_backward_or_changed_parallel`.
//!
//! The registration fixture (registry_oob_clean) proves the reverse-or-changed PROGRAM
//! on one tiny hand-checked graph; the oracle matrix proves the `cpu_ref_closure` ORACLE
//! against an independent reverse-BFS model at scale. This closes the last dimension:
//! the actual node-parallel IR PROGRAM, iterated to a fixed point through `reference_eval`,
//! must converge to the same reverse-reachable set as `cpu_ref_closure` across many
//! generated CSR shapes. A single node-parallel pass reads the LIVE accumulator and is
//! order-dependent for multi-hop chains, but the CONVERGED set is unique, that is the
//! op's contract, and iterating the real IR is the only way to pin it end to end.
#![cfg(all(feature = "graph", feature = "bitset", feature = "cpu-parity"))]

use vyre_foundation::ir::Program;
use vyre_primitives::graph::csr_backward_or_changed::{
    cpu_ref_closure, csr_backward_or_changed_parallel,
};
use vyre_primitives::graph::program_graph::ProgramGraphShape;
use vyre_primitives::wire::{decode_u32_le_bytes_all as unpack, pack_u32_slice as pack};
use vyre_reference::value::Value;

fn bitset_words(node_count: u32) -> usize {
    node_count.div_ceil(32) as usize
}

fn next_u32(state: &mut u32) -> u32 {
    *state ^= *state << 13;
    *state ^= *state >> 17;
    *state ^= *state << 5;
    *state
}

/// Deterministic CSR + a seed frontier masked to valid node bits (no padding bits, so the
/// IR output and the CPU closure, both of which monotonically retain the seed, compare
/// exactly with no out-of-domain-bit ambiguity).
fn generated(seed: u32) -> (u32, Vec<u32>, Vec<u32>, Vec<u32>, Vec<u32>, u32) {
    let node_count = 1 + (seed % 64);
    let mut state = seed ^ 0x1234_ABCD;
    let mut offsets = vec![0u32];
    let mut targets = Vec::new();
    let mut masks = Vec::new();
    for _ in 0..node_count {
        let degree = next_u32(&mut state) % 4;
        for _ in 0..degree {
            targets.push(next_u32(&mut state) % node_count);
            masks.push(1u32 << (next_u32(&mut state) % 4));
        }
        offsets.push(targets.len() as u32);
    }
    let words = bitset_words(node_count);
    let mut frontier = vec![0u32; words];
    for node in 0..node_count {
        if next_u32(&mut state) & 0b11 == 0 {
            frontier[(node / 32) as usize] |= 1u32 << (node % 32);
        }
    }
    if frontier.iter().all(|&w| w == 0) {
        let node = seed % node_count;
        frontier[(node / 32) as usize] |= 1u32 << (node % 32);
    }
    // Drop any bits at or above node_count in the final word.
    if node_count % 32 != 0 {
        let mask = (1u32 << (node_count % 32)) - 1;
        let last = frontier.len() - 1;
        frontier[last] &= mask;
    }
    let allow = match seed % 5 {
        0 => 0b0001,
        1 => 0b0011,
        2 => 0b1010,
        3 => 0b0101,
        _ => 0xFFFF_FFFF,
    };
    (node_count, offsets, targets, masks, frontier, allow)
}

fn out_by_name(program: &Program, outputs: &[Value], name: &str) -> Vec<u32> {
    let index = vyre_reference::output_index(program, name)
        .unwrap_or_else(|| panic!("Fix: backward program must declare output `{name}`"));
    unpack(&outputs[index].to_bytes())
}

/// Iterate the real IR one reverse-or-changed pass at a time, feeding each pass's frontier
/// back as the next seed, until the program's own `changed` flag reports no new bit, then
/// return the converged frontier.
fn ir_fixpoint(
    node_count: u32,
    offsets: &[u32],
    targets: &[u32],
    masks: &[u32],
    seed: &[u32],
    allow: u32,
) -> Vec<u32> {
    let edge_count = targets.len() as u32;
    let shape = ProgramGraphShape::new(node_count, edge_count);
    let program = csr_backward_or_changed_parallel(shape, "frontier", "changed", allow);
    let nodes = vec![0u32; node_count as usize];
    let tags = vec![0u32; node_count as usize];
    let mut frontier = seed.to_vec();
    // A monotone reverse closure adds at most one hop's worth of nodes per pass in the worst
    // (single-hop) ordering, so node_count + 1 passes always reach the fixed point.
    for _ in 0..node_count + 1 {
        let outputs = vyre_reference::reference_eval(
            &program,
            &[
                Value::from(pack(&nodes)),
                Value::from(pack(offsets)),
                Value::from(pack(targets)),
                Value::from(pack(masks)),
                Value::from(pack(&tags)),
                Value::from(pack(&frontier)),
                Value::from(pack(&[0u32])),
            ],
        )
        .expect("backward reverse-or-changed reference evaluation must succeed");
        frontier = out_by_name(&program, &outputs, "frontier");
        if out_by_name(&program, &outputs, "changed")[0] == 0 {
            break;
        }
    }
    frontier
}

#[test]
fn ir_program_fixpoint_matches_cpu_ref_closure_across_generated_shapes() {
    for seed in 1..3072u32 {
        let (node_count, offsets, targets, masks, frontier, allow) = generated(seed);
        let edge_count = targets.len();
        // Skip degenerate empty-edge graphs where the shape's edge buffers collapse to the
        // `.max(1)` floor (the closure is a no-op and adds no signal).
        if edge_count == 0 {
            continue;
        }

        let ir = ir_fixpoint(node_count, &offsets, &targets, &masks, &frontier, allow);
        let (oracle, _changed) = cpu_ref_closure(
            node_count,
            &offsets,
            &targets,
            &masks,
            &frontier,
            allow,
            node_count + 1,
        );

        assert_eq!(
            ir, oracle,
            "seed {seed}: IR reverse-or-changed fixpoint (node_count={node_count}, edges={edge_count}, \
             allow={allow:#x}) diverged from cpu_ref_closure"
        );
    }
}
