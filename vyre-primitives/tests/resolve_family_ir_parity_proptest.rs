//! Tier 3 - Property: differential proptest driving the ACTUAL `label::resolve_family` IR through
//! `reference_eval` vs `cpu_ref`. `resolve_family` reuses the shared `nodeset_filter` kernel but
//! selects the `Intersects(mask)` predicate (`node_tags[v] & family_mask != 0` -> set bit `v`), a
//! DISTINCT code path from the `Eq` predicate exercised by node_kind_eq_ir_parity: the IR condition is
//! `ne(bitand(tag, mask), 0)` rather than `eq(tag, kind)`. Its shipped tests are all cpu_ref-only
//! (`grep reference_eval` = 0), so the packed-NodeSet atomic_or scatter under the intersect predicate
//! is never run through a faithful executor.
//!
//! The sweep randomizes node_tags as BITMASKS (each tag a small OR of bit flags) and family_mask from
//! the same flag alphabet, so hits and misses are both dense and multiple matching lanes atomic_or the
//! same output word concurrently. Asserts the full packed NodeSet bit-exact vs `cpu_ref`, plus
//! word-boundary + all-match / no-match (mask=0) / match-all (mask=!0) anchors.
#![cfg(feature = "cpu-parity")]

use proptest::prelude::*;
use vyre_reference::value::Value;

use vyre_primitives::label::resolve_family::{cpu_ref, resolve_family};

fn run_ir(node_tags: &[u32], family_mask: u32) -> Vec<u32> {
    let node_count = node_tags.len() as u32;
    let words = node_count.div_ceil(32).max(1) as usize;
    let program = resolve_family("tags", "nodeset", node_count, family_mask);
    let pack = |d: &[u32]| Value::from(vyre_primitives::wire::pack_u32_slice(d));
    let outputs =
        vyre_reference::reference_eval(&program, &[pack(node_tags), pack(&vec![0u32; words])])
            .expect("resolve_family reference evaluation must succeed");
    outputs[0]
        .to_bytes()
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(2000))]

    #[test]
    fn resolve_family_ir_matches_cpu_ref(
        // Tags are OR-combinations of the low 8 flag bits so intersect hits/misses are both common.
        node_tags in prop::collection::vec(0u32..256, 1..=256),
        family_mask in 0u32..256,
    ) {
        let got = run_ir(&node_tags, family_mask);
        let want = cpu_ref(&node_tags, family_mask);
        prop_assert_eq!(&got, &want, "family_mask={:#b} node_count={}", family_mask, node_tags.len());
    }
}

#[test]
fn resolve_family_ir_intersect_edges_and_word_boundary() {
    for &n in &[1usize, 31, 32, 33, 64, 65, 200] {
        let words = n.div_ceil(32);
        // Every tag carries bit 1 (0b0010); mask 0b0010 → all match.
        let all: Vec<u32> = vec![0b0010u32; n];
        let got = run_ir(&all, 0b0010);
        assert_eq!(got, cpu_ref(&all, 0b0010), "all-intersect n={n}");
        let mut expected = vec![0u32; words];
        for v in 0..n {
            expected[v / 32] |= 1u32 << (v % 32);
        }
        assert_eq!(got, expected, "all-intersect packed pattern n={n}");

        // mask = 0 → intersect impossible → empty NodeSet.
        assert_eq!(run_ir(&all, 0), vec![0u32; words], "mask=0 empties n={n}");

        // Disjoint bits: tag 0b0100, mask 0b0010 → no intersection.
        let disjoint = vec![0b0100u32; n];
        assert_eq!(
            run_ir(&disjoint, 0b0010),
            vec![0u32; words],
            "disjoint n={n}"
        );

        // mask = !0 → any nonzero tag matches.
        let mixed: Vec<u32> = (0..n as u32).map(|i| i & 0xFF).collect();
        assert_eq!(
            run_ir(&mixed, u32::MAX),
            cpu_ref(&mixed, u32::MAX),
            "mask=all n={n}"
        );
    }
}
