//! Contracts for `vyre_runtime::resident_work_queue::rule_catalog`.
//!
//! Every item under test is public API, so the suite reaches the crate the way
//! a consumer does.
#![cfg(feature = "megakernel-batch")]

use vyre_runtime::resident_work_queue::rule_catalog::{
    accepted_rule_fingerprints_into, pack_rule_catalog, try_pack_u16_transitions_into,
    unpack_u16_transition, BatchRuleProgram, PackedRuleCatalog, ALPHABET_SIZE,
};

/// Resolve the next state the COMPRESSED packed catalog yields for
/// `(rule, state, byte)`: mirrors the GPU kernel's index math exactly so
/// the parity tests can prove byte-for-byte equivalence to the dense table.
fn packed_next_state(packed: &PackedRuleCatalog, meta_index: usize, state: u32, byte: u8) -> u32 {
    let meta = packed.rule_meta[meta_index];
    let class = packed.class_maps[meta.class_map_base as usize + byte as usize];
    let idx =
        meta.transition_base as usize + state as usize * meta.num_classes as usize + class as usize;
    packed.transitions[idx]
}

#[test]
fn duplicate_dfas_share_catalog_storage() {
    let first = BatchRuleProgram::new(0, vec![0; 256], vec![0], 1).unwrap();
    let second = BatchRuleProgram::new(1, vec![0; 256], vec![0], 1).unwrap();
    let packed = pack_rule_catalog(&[first, second]).unwrap();
    // Identical DFAs share compressed transition, accept AND class-map storage.
    assert_eq!(
        packed.rule_meta[0].transition_base,
        packed.rule_meta[1].transition_base
    );
    assert_eq!(
        packed.rule_meta[0].accept_base,
        packed.rule_meta[1].accept_base
    );
    assert_eq!(
        packed.rule_meta[0].class_map_base,
        packed.rule_meta[1].class_map_base
    );
    assert_eq!(
        packed.rule_meta[0].num_classes,
        packed.rule_meta[1].num_classes
    );
    // An all-zero 1-state DFA collapses to a SINGLE byte class (every byte
    // self-loops to state 0), so its compressed row is exactly one word, not
    // 256. transition_base points just past the 1-word inert row.
    assert_eq!(packed.rule_meta[0].num_classes, 1);
    assert_eq!(packed.rule_meta[0].transition_base, 1);
    assert_eq!(
        packed.transitions.len(),
        packed.rule_meta[0].transition_base as usize + 1
    );
    assert_eq!(
        packed.accept.len(),
        packed.rule_meta[0].accept_base as usize + 1
    );
    assert!(packed.rejected_rules.is_empty());
}

#[test]
fn u16_pack_round_trips_losslessly_including_odd_tail() {
    // Odd element count exercises the lone-remainder path; 65535 is the max
    // legal u16 target.
    let compressed: Vec<u32> = vec![0, 1, 2, 65_535, 100, 0, 42];
    let mut packed = Vec::new();
    try_pack_u16_transitions_into(&compressed, &mut packed).expect("all targets fit u16");
    // Two u16 per u32 word, rounding up for the odd tail.
    assert_eq!(packed.len(), compressed.len().div_ceil(2));
    // Every flat index unpacks to EXACTLY the original target, proving the
    // pack→kernel-unpack round-trip changes no transition (Law 6).
    for (idx, &original) in compressed.iter().enumerate() {
        assert_eq!(
            unpack_u16_transition(&packed, idx),
            original,
            "u16 round-trip diverged at flat index {idx}",
        );
    }
}

#[test]
fn u16_pack_fails_closed_on_target_exceeding_u16() {
    // 70000 > u16::MAX: packing MUST refuse, never silently `& 0xFFFF` it to a
    // wrong next-state (Law 10 (that truncation is an invisible recall loss)).
    let compressed: Vec<u32> = vec![0, 1, 70_000, 3];
    let mut out = Vec::new();
    let err = try_pack_u16_transitions_into(&compressed, &mut out)
        .expect_err("a target above u16::MAX must be refused");
    let msg = err.to_string();
    assert!(
        msg.contains("70000") && msg.contains("index 2") && msg.contains("u16"),
        "error must name the offending target/index and the u16 cause: {msg}",
    );
}

/// Regression for P2 decoration test: the structural shared-storage checks
/// above were not sufficient, a refactor could share the WRONG compressed
/// block and still pass the field-equality assertion. This test packs TWO
/// identical copies of the non-trivial 3-class DFA from
/// `byte_class_compression_is_lossless` and then calls `packed_next_state`
/// on BOTH meta indices for every (state, byte) pair, asserting both return
/// the same value AND that value matches the dense source table.
#[test]
fn duplicate_dfas_shared_storage_both_rules_fire_correctly() {
    // 3-state, 3-class DFA (same fixture as byte_class_compression_is_lossless).
    let states = 3usize;
    let mut dense = vec![0u32; states * 256];
    dense[0 * 256 + 0x41] = 1; // state 0: 'A' -> 1
    dense[1 * 256 + 0x41] = 2; // state 1: 'A' -> 2
    dense[1 * 256 + 0x42] = 2; // state 1: 'B' -> 2
    dense[2 * 256 + 0x41] = 2; // state 2: 'A' -> 2
    let accept = vec![0u32, 0, 1];

    let rule0 = BatchRuleProgram::new(0, dense.clone(), accept.clone(), states as u32).unwrap();
    let rule1 = BatchRuleProgram::new(1, dense.clone(), accept.clone(), states as u32).unwrap();
    let packed = pack_rule_catalog(&[rule0, rule1]).unwrap();

    assert!(packed.rejected_rules.is_empty());
    // Both rules must share storage.
    assert_eq!(
        packed.rule_meta[0].transition_base, packed.rule_meta[1].transition_base,
        "Fix: duplicate DFAs must share transition storage"
    );
    assert_eq!(
        packed.rule_meta[0].accept_base, packed.rule_meta[1].accept_base,
        "Fix: duplicate DFAs must share accept storage"
    );

    // Critical: verify BOTH meta indices yield the correct DFA output for
    // every (state, byte), structural field sharing is necessary but not
    // sufficient; the shared block must actually encode the right DFA.
    for state in 0..states as u32 {
        for byte in 0u16..256 {
            let byte = byte as u8;
            let expected = dense[state as usize * 256 + byte as usize];
            let got0 = packed_next_state(&packed, 0, state, byte);
            let got1 = packed_next_state(&packed, 1, state, byte);
            assert_eq!(
                got0, expected,
                "Fix: rule0 compressed transition mismatch at state {state} byte {byte:#x}: expected {expected} got {got0}"
            );
            assert_eq!(
                got1, expected,
                "Fix: rule1 compressed transition mismatch at state {state} byte {byte:#x}: expected {expected} got {got1}"
            );
        }
    }
}

#[test]
fn duplicate_dfas_do_not_reserve_raw_duplicate_storage() {
    let rules = (0..32)
        .map(|rule_idx| BatchRuleProgram::new(rule_idx, vec![0; 256], vec![0], 1).unwrap())
        .collect::<Vec<_>>();

    let packed = pack_rule_catalog(&rules).unwrap();

    // 1-word inert row + 1-word shared compressed row for all 32 duplicates.
    assert_eq!(packed.transitions.len(), 2);
    assert!(
        packed.transitions.capacity() < ALPHABET_SIZE as usize * rules.len(),
        "Fix: duplicate DFA catalogs must not reserve memory as if every rule had unique transition storage."
    );
    assert_eq!(packed.accept.len(), 2);
    assert!(
        packed.accept.capacity() < rules.len(),
        "Fix: duplicate DFA catalogs must not reserve accept storage for every duplicate rule."
    );
    // One inert + one shared class map, not 32.
    assert_eq!(packed.class_maps.len(), ALPHABET_SIZE as usize * 2);
}

/// The compressed catalog yields byte-for-byte identical next-states to the
/// dense `state * 256 + byte` table for EVERY (state, byte) of a non-trivial
/// multi-class DFA (the lossless parity contract the GPU kernel depends on).
#[test]
fn byte_class_compression_is_lossless() {
    // 3-state DFA. byte 0x41 ('A') advances 0->1->2->2; byte 0x42 ('B')
    // advances 1->2 only; all other bytes reset to 0. This forces THREE
    // distinct byte classes (A, B, everything-else) so num_classes < 256
    // and the compression is exercised, not a degenerate single class.
    let states = 3usize;
    let mut dense = vec![0u32; states * 256];
    // state 0: 'A' -> 1, else -> 0
    dense[0 * 256 + 0x41] = 1;
    // state 1: 'A' -> 2, 'B' -> 2, else -> 0
    dense[1 * 256 + 0x41] = 2;
    dense[1 * 256 + 0x42] = 2;
    // state 2: 'A' -> 2, else -> 0
    dense[2 * 256 + 0x41] = 2;
    let accept = vec![0u32, 0, 1];
    let rule = BatchRuleProgram::new(0, dense.clone(), accept, states as u32).unwrap();
    let packed = pack_rule_catalog(&[rule]).unwrap();

    assert_eq!(packed.rejected_rules.len(), 0);
    // 'A', 'B', and the rest are three behaviourally-distinct columns.
    assert_eq!(packed.rule_meta[0].num_classes, 3);
    assert!(
        packed.transitions.len() < 1 + states * 256,
        "compressed transitions must be smaller than the dense table"
    );

    for state in 0..states as u32 {
        for byte in 0u16..256 {
            let byte = byte as u8;
            let expected = dense[state as usize * 256 + byte as usize];
            let got = packed_next_state(&packed, 0, state, byte);
            assert_eq!(
                got, expected,
                "compressed transition mismatch at state {state} byte {byte:#x}: dense={expected} packed={got}"
            );
        }
    }
}

/// A DFA whose every byte transitions differently in some state must NOT be
/// over-compressed: it keeps all 256 classes and still round-trips losslessly.
#[test]
fn full_alphabet_dfa_keeps_all_classes_and_is_lossless() {
    // 2-state DFA where state 0 sends byte b -> (b as state is impossible
    // with 2 states), so instead: state 0 sends EVERY byte to a distinct
    // value by using state 1 vs 0 based on parity, that only yields 2
    // classes. To force 256 classes we need 256 distinct columns, which
    // needs >=256 states. Use a 256-state identity: state s, byte b -> b.
    let states = 256usize;
    let mut dense = vec![0u32; states * 256];
    for s in 0..states {
        for b in 0..256 {
            dense[s * 256 + b] = b as u32; // column for byte b is constant = b across all states
        }
    }
    // Every byte's column is the constant vector [b; 256], all distinct, so
    // 256 classes.
    let accept = vec![0u32; states];
    let rule = BatchRuleProgram::new(0, dense.clone(), accept, states as u32).unwrap();
    let packed = pack_rule_catalog(&[rule]).unwrap();
    assert_eq!(packed.rule_meta[0].num_classes, 256);
    for state in 0..states as u32 {
        for byte in 0u16..256 {
            let byte = byte as u8;
            let expected = dense[state as usize * 256 + byte as usize];
            assert_eq!(packed_next_state(&packed, 0, state, byte), expected);
        }
    }
}

#[test]
fn accepted_rule_fingerprints_into_returns_rejections_and_reuses_caller_storage() {
    let rules = (0..8)
        .map(|rule_idx| BatchRuleProgram::new(rule_idx, vec![0; 256], vec![0], 1).unwrap())
        .collect::<Vec<_>>();
    let mut fingerprints = Vec::with_capacity(16);
    let mut occupied = Vec::with_capacity(16);
    let mut addressed = Vec::with_capacity(16);
    let fingerprint_ptr = fingerprints.as_ptr();
    let occupied_ptr = occupied.as_ptr();
    let addressed_ptr = addressed.as_ptr();

    let rejections =
        accepted_rule_fingerprints_into(&rules, &mut fingerprints, &mut occupied, &mut addressed);

    assert!(rejections.is_empty());
    assert_eq!(fingerprints.len(), rules.len());
    assert_eq!(fingerprints.as_ptr(), fingerprint_ptr);
    assert_eq!(occupied.as_ptr(), occupied_ptr);
    assert_eq!(addressed.as_ptr(), addressed_ptr);
}

#[test]
fn invalid_rules_are_isolated_to_inert_catalog_entries() {
    let valid = BatchRuleProgram::new(0, vec![0; 256], vec![1], 1).unwrap();
    let invalid = BatchRuleProgram {
        rule_idx: 1,
        transitions: vec![0; 8],
        accept: vec![0],
        state_count: 1,
    };

    let packed = pack_rule_catalog(&[valid, invalid]).unwrap();
    assert_eq!(packed.rejected_rules.len(), 1);
    assert_eq!(packed.rejected_rules[0].rule_idx, Some(1));
    // Valid rule (slot 0) points at a REAL compressed block past the inert
    // row; the inert/rejected slot 1 points back at the inert row 0.
    assert_eq!(packed.rule_meta[0].state_count, 1);
    assert!(packed.rule_meta[0].transition_base >= 1);
    assert_eq!(packed.rule_meta[1].transition_base, 0);
    assert_eq!(packed.rule_meta[1].accept_base, 0);
    assert_eq!(packed.rule_meta[1].state_count, 1);
    assert_eq!(packed.rule_meta[1].class_map_base, 0);
    assert_eq!(packed.rule_meta[1].num_classes, 1);
    // Inert row 0: a single self-loop word and an all-zero 256-entry class
    // map (the rejected slot reads a well-formed no-match DFA).
    assert_eq!(packed.transitions[0], 0);
    assert_eq!(packed.accept[0], 0);
    assert_eq!(
        &packed.class_maps[..ALPHABET_SIZE as usize],
        &vec![0; ALPHABET_SIZE as usize]
    );
    // Regression for P2 decoration test: a single-byte spot check is not
    // sufficient, a corrupt inert row could have non-zero entries at other
    // bytes or at the accept table while still passing b'X'. This loop
    // proves the inert slot self-loops to state 0 on EVERY byte value and
    // that the accept entry for the inert slot is zero (can never match).
    for byte in 0u16..256 {
        let byte = byte as u8;
        assert_eq!(
            packed_next_state(&packed, 1, 0, byte),
            0,
            "Fix: inert slot must self-loop to state 0 for every byte, failed at byte {byte:#x}"
        );
    }
    // Accept entry for the inert slot at state 0 must be zero (no match).
    assert_eq!(
        packed.accept[packed.rule_meta[1].accept_base as usize], 0,
        "Fix: inert slot accept entry at state 0 must be 0, the inert DFA must never produce a match"
    );
}
