//! Combined-AC segmented megakernel: differential conservation against the
//! `classic_ac_scan` CPU oracle, plus 8 MiB throughput.
//!
//! The combined path compiles a whole catalog into ONE Aho-Corasick automaton
//! and runs it ONCE per segment (`queue_len = segment_count`), emitting the SET
//! of pattern ids accepting at each state. This test pins the two contracts:
//!
//!   1. CONSERVATION (differential vs oracle, Law 10 — no silent drop). For
//!      EVERY segmentation geometry the GPU's `(pattern_id, end_offset)` set
//!      must EXACTLY equal `classic_ac_scan` over the whole buffer — no miss,
//!      no duplicate, no fabricated hit, `dropped_hits == 0`. `classic_ac_scan`
//!      is the same linear oracle the `segmentation.rs` `combined_segmented_scan`
//!      proptest is proven equal to, so this closes the loop on the REAL GPU.
//!   2. THROUGHPUT. The fastest conserving geometry must beat Hyperscan's
//!      ~1.5 GB/s warm device floor on a single 8 MiB input — and it does so
//!      with the per-rule `rule_count` queue multiplier GONE (one transition
//!      read per byte regardless of catalog size).
//!
//! Run: cargo test -p vyre-driver-wgpu --features megakernel-batch,wgpu \
//!        --test megakernel_combined_scan -- --ignored --nocapture
#![cfg(feature = "megakernel-batch")]

use std::collections::BTreeSet;
use std::time::Duration;

use vyre_driver_wgpu::megakernel::{
    BatchDispatchConfig, BatchFile, CombinedBatch, CombinedDispatcher, HitRecord,
};
use vyre_driver_wgpu::WgpuBackend;
use vyre_libs::scan::classic_ac::{classic_ac_compile, classic_ac_scan};

const FILE_LEN: usize = 8 * 1024 * 1024; // 8 MiB
const HS_FLOOR_GBPS: f64 = 1.5;

/// A varied-length literal catalog: short and long patterns so the warm-up
/// `overlap = max_pattern_len` is genuinely exercised at window boundaries.
fn catalog() -> Vec<&'static [u8]> {
    vec![
        b"alpha", b"bravo", b"charlie", b"delta", b"echo", b"foxtrot", b"golf",
        b"hotel", b"india", b"juliet", b"kilo", b"lima", b"mike", b"november",
        b"oscar", b"papa", b"quebec", b"romeo", b"sierra", b"tango", b"uniform",
        b"victor", b"whiskey", b"xray", b"yankee", b"zulu", b"AB", b"ABCDEFGH",
        b"needle", b"haystack", b"0123456789", b"the-quick-brown-fox",
    ]
}

/// Plant catalog patterns at a fixed spread of offsets — including offsets that
/// straddle the 512 / 1024 / 4096 segment boundaries the geometry sweep uses —
/// so warm-up + emit-guard ownership is tested at the seams. Offsets are
/// geometry-invariant so the oracle ground truth is the same for every seg_len.
fn build_haystack(patterns: &[&[u8]]) -> Vec<u8> {
    let mut buf = vec![b'.'; FILE_LEN];
    let boundary_seeds = [500usize, 1020, 4090, 65530];
    for (pattern_index, pattern) in patterns.iter().enumerate() {
        // A uniform spread across the file...
        let mut offset = 4096 + pattern_index * 211;
        while offset + pattern.len() <= FILE_LEN {
            buf[offset..offset + pattern.len()].copy_from_slice(pattern);
            offset += 131_072; // 128 KiB stride
        }
        // ...plus deliberate boundary-straddling placements.
        for &seed in &boundary_seeds {
            let at = seed + pattern_index;
            if at + pattern.len() <= FILE_LEN {
                buf[at..at + pattern.len()].copy_from_slice(pattern);
            }
        }
    }
    buf
}

struct GeomResult {
    seg_len: u32,
    found: BTreeSet<(u32, u32)>,
    dropped: u32,
    gbps: f64,
    under_claimed: bool,
}

fn run_geometry(
    backend: &WgpuBackend,
    buf: &[u8],
    transitions: &[u32],
    output_offsets: &[u32],
    output_records: &[u32],
    state_count: u32,
    max_pattern_len: u32,
    seg_len: u32,
) -> GeomResult {
    let hit_capacity: u32 = 1 << 20;
    let config = BatchDispatchConfig {
        workgroup_size_x: 64,
        worker_groups: 2048,
        hit_capacity,
        timeout: Duration::from_secs(120),
        ..Default::default()
    };
    let files = vec![BatchFile::new(0xC0FFEE, 0, buf.to_vec())];
    let mut batch = CombinedBatch::upload(
        backend.device_queue(),
        &files,
        transitions,
        output_offsets,
        output_records,
        state_count,
        max_pattern_len,
        seg_len,
        hit_capacity,
    )
    .expect("Fix: CombinedBatch upload must succeed for a valid automaton");
    let mut dispatcher = CombinedDispatcher::new(backend.clone(), config);

    // Warm once (compile out of the timing). A loud under-claim is acceptable.
    let mut hits: Vec<HitRecord> = Vec::new();
    let _ = dispatcher.dispatch_into(&batch, &mut hits);

    let mut best = Duration::from_secs(3600);
    let mut found: BTreeSet<(u32, u32)> = BTreeSet::new();
    let mut dropped = 0u32;
    let mut under_claimed = false;
    let _ = &mut batch;
    for _ in 0..3 {
        hits.clear();
        match dispatcher.dispatch_into(&batch, &mut hits) {
            Ok(summary) => {
                best = best.min(summary.wall_time);
                found = hits
                    .iter()
                    .map(|h| (h.rule_idx, h.match_offset))
                    .collect::<BTreeSet<_>>();
                dropped = summary.dropped_hits;
            }
            Err(e) if e.to_string().contains("drain incomplete") => {
                under_claimed = true;
                break;
            }
            Err(e) => panic!("unexpected combined dispatch error at seg_len={seg_len}: {e}"),
        }
    }
    GeomResult {
        seg_len,
        found,
        dropped,
        gbps: if best == Duration::from_secs(3600) {
            0.0
        } else {
            FILE_LEN as f64 / best.as_secs_f64() / 1e9
        },
        under_claimed,
    }
}

#[test]
#[ignore = "live GPU; run with --ignored --nocapture"]
fn combined_scan_conserves_every_match_and_beats_hyperscan() {
    let backend = WgpuBackend::new()
        .expect("Fix: live GPU required (missing GPU is a configuration bug, not a fallback)");

    let patterns = catalog();
    let max_pattern_len = patterns
        .iter()
        .map(|p| p.len())
        .max()
        .expect("non-empty catalog") as u32;
    let buf = build_haystack(&patterns);

    // Ground truth: the linear classic-AC oracle over the WHOLE buffer. The GPU
    // must reproduce this set exactly for every geometry.
    let ac = classic_ac_compile(&patterns);
    let oracle: BTreeSet<(u32, u32)> = classic_ac_scan(&ac, &buf).into_iter().collect();
    assert!(
        !oracle.is_empty(),
        "fixture must plant matches; oracle found none"
    );

    let transitions = &ac.dfa.transitions;
    let output_offsets = &ac.dfa.output_offsets;
    let output_records = &ac.dfa.output_records;
    let state_count = ac.dfa.state_count;

    // The combined kernel byte-class compresses this dense table. Confirm the
    // compression is REAL (fewer than 256 classes) for this catalog, so the
    // conservation result below actually exercises the compressed transition
    // path, not a 256-class identity passthrough.
    let mut class_map = Vec::new();
    let num_classes = vyre_runtime::megakernel::rule_catalog::build_byte_class_map_for_table(
        transitions,
        state_count as usize,
        &mut class_map,
    );
    assert!(
        num_classes < 256,
        "expected byte-class compression to collapse the alphabet (<256 classes) for this \
         catalog; got {num_classes} — the compressed-transition path would be untested"
    );
    eprintln!(
        "combined automaton: {state_count} states, {num_classes} byte-classes (dense 256 → compressed {num_classes})",
    );

    // seg_len = u32::MAX is one segment per file (whole-file, no segmentation);
    // the rest tile the 8 MiB file into many windows to saturate the device.
    let geometries = [u32::MAX, 65_536u32, 16_384, 4_096, 1_024, 512];

    eprintln!(
        "8 MiB / {} patterns — combined-AC conservation + throughput (oracle has {} matches, Hyperscan {HS_FLOOR_GBPS} GB/s):",
        patterns.len(),
        oracle.len()
    );
    eprintln!(
        "  {:>9}  {:>8}  {:>8}  {:>8}  {:>6}  {:>9}",
        "seg_len", "found", "dropped", "GB/s", "vs HS", "conserves?"
    );

    let mut results = Vec::new();
    for seg_len in geometries {
        let r = run_geometry(
            &backend,
            &buf,
            transitions,
            output_offsets,
            output_records,
            state_count,
            max_pattern_len,
            seg_len,
        );
        let conserves = r.found == oracle && r.dropped == 0;
        let status = if conserves {
            "conserves"
        } else if r.under_claimed {
            "LOUD-underclaim"
        } else {
            "DIVERGES"
        };
        eprintln!(
            "  {:>9}  {:>8}  {:>8}  {:>8.3}  {:>5.2}x  {:>15}",
            if r.seg_len == u32::MAX {
                "whole".to_string()
            } else {
                r.seg_len.to_string()
            },
            r.found.len(),
            r.dropped,
            r.gbps,
            r.gbps / HS_FLOOR_GBPS,
            status,
        );
        results.push(r);
    }

    // Contract 1 — exact conservation vs the oracle for every NON-under-claimed
    // geometry. A geometry that returned an Ok dispatch but a different hit set
    // is a silent recall bug (miss, dup, or fabrication).
    for r in &results {
        if r.under_claimed {
            continue;
        }
        assert_eq!(
            r.found.len(),
            oracle.len(),
            "seg_len={} found {} matches, oracle has {} — combined scan is not conserving (dropped={})",
            r.seg_len,
            r.found.len(),
            oracle.len(),
            r.dropped
        );
        assert_eq!(
            r.found, oracle,
            "seg_len={} hit set diverged from the classic_ac_scan oracle (same cardinality, different pairs)",
            r.seg_len
        );
        assert_eq!(
            r.dropped, 0,
            "seg_len={} dropped {} hits — the ring overflowed and the result is incomplete",
            r.seg_len, r.dropped
        );
    }

    // Contract 2 — throughput. Fastest conserving geometry beats Hyperscan.
    let best_ok_gbps = results
        .iter()
        .filter(|r| r.found == oracle && r.dropped == 0)
        .map(|r| r.gbps)
        .fold(0.0f64, f64::max);
    eprintln!(
        "best conserving throughput: {best_ok_gbps:.3} GB/s ⇒ combined-AC {} Hyperscan",
        if best_ok_gbps > HS_FLOOR_GBPS {
            "BEATS"
        } else {
            "does NOT beat"
        }
    );
    assert!(
        best_ok_gbps > HS_FLOOR_GBPS,
        "no conserving combined-AC geometry beat Hyperscan: best {best_ok_gbps:.3} GB/s <= {HS_FLOOR_GBPS} GB/s"
    );
}
