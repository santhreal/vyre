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

/// ~2048 distinct secret-shaped literals approximating keyhog's many-literal
/// catalog (the real workload is ~6000): varied real-detector prefixes + base62
/// bodies of varied length, so the combined Aho-Corasick has thousands of states
/// and a realistically compressed byte alphabet. Deterministic (no RNG) so the
/// oracle ground truth is stable across runs.
fn large_catalog() -> Vec<Vec<u8>> {
    let prefixes: &[&[u8]] = &[
        b"AKIA", b"ghp_", b"gho_", b"xoxb-", b"xoxp-", b"AIza", b"sk-", b"pk_",
        b"rk_", b"glpat-", b"ya29.", b"ASIA", b"SG.", b"shpat_", b"AccountKey=",
        b"eyJ", b"-----BEGIN", b"npm_", b"dop_v1_", b"sk_live_",
    ];
    let body: &[u8] = b"0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ";
    let mut out: Vec<Vec<u8>> = Vec::with_capacity(2048);
    let mut seen = std::collections::HashSet::new();
    let mut i = 0usize;
    while out.len() < 2048 {
        let prefix = prefixes[i % prefixes.len()];
        let len = 6 + (i % 27); // bodies 6..=32 bytes
        let mut p = prefix.to_vec();
        for k in 0..len {
            // Deterministic, well-spread index into the base62 alphabet.
            let idx = i
                .wrapping_mul(2_654_435_761)
                .wrapping_add(k.wrapping_mul(40_503))
                % body.len();
            p.push(body[idx]);
        }
        if seen.insert(p.clone()) {
            out.push(p);
        }
        i += 1;
    }
    out
}

#[test]
#[ignore = "live GPU; run with --ignored --nocapture"]
fn combined_scan_beats_hyperscan_at_keyhog_catalog_scale() {
    // The KEYSTONE the smaller test does not cover: keyhog's catalog is thousands
    // of literals ("few files × MANY literals"), not the 32 above. The combined-AC
    // geometry collapse is O(input) and INDEPENDENT of rule count, so the 8 MiB
    // win must HOLD as the automaton grows to thousands of states. This proves it
    // on a realistic-scale catalog: still 0 dropped, still beats Hyperscan.
    let backend = WgpuBackend::new()
        .expect("Fix: live GPU required (missing GPU is a configuration bug, not a fallback)");

    let owned = large_catalog();
    let patterns: Vec<&[u8]> = owned.iter().map(|p| p.as_slice()).collect();
    let max_pattern_len = patterns
        .iter()
        .map(|p| p.len())
        .max()
        .expect("non-empty catalog") as u32;
    let buf = build_haystack(&patterns);

    let ac = classic_ac_compile(&patterns);
    let oracle: BTreeSet<(u32, u32)> = classic_ac_scan(&ac, &buf).into_iter().collect();
    assert!(
        oracle.len() > 10_000,
        "a {}-literal catalog planted across 8 MiB must yield a large oracle; got {}",
        patterns.len(),
        oracle.len()
    );

    let transitions = &ac.dfa.transitions;
    let output_offsets = &ac.dfa.output_offsets;
    let output_records = &ac.dfa.output_records;
    let state_count = ac.dfa.state_count;

    let mut class_map = Vec::new();
    let num_classes = vyre_runtime::megakernel::rule_catalog::build_byte_class_map_for_table(
        transitions,
        state_count as usize,
        &mut class_map,
    );
    eprintln!(
        "keyhog-scale combined automaton: {} patterns, {state_count} states, {num_classes} byte-classes",
        patterns.len()
    );
    assert!(
        state_count > 5_000,
        "a {}-literal catalog must build a large automaton (>5000 states) to exercise scale; got {state_count}",
        patterns.len()
    );

    // Segmenting geometries only (no whole-file: at this scale the unsegmented
    // pass is the slow path the segmentation exists to beat). Swept down to very
    // fine windows because a large automaton is memory-bound — more, smaller
    // segments trade warm-up overhead for parallelism, and the sweep finds where
    // that balance peaks for a thousands-of-states catalog.
    let geometries = [16_384u32, 4_096, 1_024, 512, 256, 128];
    eprintln!(
        "8 MiB / {} patterns — combined-AC conservation + throughput (oracle {} matches, Hyperscan {HS_FLOOR_GBPS} GB/s):",
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
            r.seg_len, r.found.len(), r.dropped, r.gbps, r.gbps / HS_FLOOR_GBPS, status,
        );
        results.push(r);
    }

    // Conservation: every non-under-claimed geometry must reproduce the oracle
    // EXACTLY at scale — no miss, dup, or fabrication when the automaton is large.
    for r in &results {
        if r.under_claimed {
            continue;
        }
        assert_eq!(
            r.found, oracle,
            "seg_len={} diverged from the oracle at keyhog scale (found {}, oracle {}, dropped {})",
            r.seg_len,
            r.found.len(),
            oracle.len(),
            r.dropped
        );
        assert_eq!(r.dropped, 0, "seg_len={} dropped {} hits", r.seg_len, r.dropped);
    }

    let best_ok_gbps = results
        .iter()
        .filter(|r| r.found == oracle && r.dropped == 0)
        .map(|r| r.gbps)
        .fold(0.0f64, f64::max);
    eprintln!(
        "best conserving throughput at {}-literal scale: {best_ok_gbps:.3} GB/s ⇒ combined-AC {} Hyperscan",
        patterns.len(),
        if best_ok_gbps > HS_FLOOR_GBPS { "BEATS" } else { "does NOT beat" }
    );
    assert!(
        best_ok_gbps > HS_FLOOR_GBPS,
        "the combined-AC win did NOT hold at {}-literal keyhog scale: best {best_ok_gbps:.3} GB/s <= {HS_FLOOR_GBPS} GB/s",
        patterns.len()
    );
}

/// PROFILE-FIRST measurement for the megakernel's open optimization lane
/// (`docs/GPU_OOM_SEGMENTATION.md`: the throughput degradation at catalog scale
/// is an L1 working-set / memory-transaction effect, NOT L2 capacity). Two
/// candidate levers narrow each transition read: (1) **row deduplication** —
/// merge identical compressed transition rows behind a `state → row`
/// indirection, shrinking the count of DISTINCT physical rows a warp's 32 lanes
/// touch; (2) **u16 targets** — halve every transition word when
/// `state_count <= 65535`. The doc says both are profile-gated for the
/// *throughput* claim, but their *ceilings* are exact CPU facts that decide
/// whether either is worth a kernel change at all. This test pins those facts on
/// the real keyhog-scale combined automaton (CPU-only — no GPU), so the lever
/// choice is grounded in measured structure, not a guess.
///
/// It also PROVES the row→indirection transform is LOSSLESS: every state's
/// deduplicated row must be byte-identical to its compressed row, so adopting it
/// cannot change a single firing (Law 6/9). The distinct-row count is asserted
/// EXACTLY (the catalog is deterministic) — a real regression value, not a shape
/// check.
#[test]
fn row_dedup_and_u16_ceiling_on_keyhog_scale_combined_automaton() {
    use std::collections::HashMap;
    use vyre_runtime::megakernel::rule_catalog::{
        build_byte_class_map_for_table, compress_dense_transitions_into,
    };

    let owned = large_catalog();
    let patterns: Vec<&[u8]> = owned.iter().map(|p| p.as_slice()).collect();
    let ac = classic_ac_compile(&patterns);
    let state_count = ac.dfa.state_count as usize;

    // Compress the dense state*256 table to state*num_classes (the form the
    // kernel actually indexes), via the SHARED primitive — no fork.
    let mut class_map = Vec::new();
    let num_classes =
        build_byte_class_map_for_table(&ac.dfa.transitions, state_count, &mut class_map) as usize;
    let mut compressed = Vec::with_capacity(state_count * num_classes);
    compress_dense_transitions_into(
        &ac.dfa.transitions,
        state_count,
        &class_map,
        num_classes as u32,
        &mut compressed,
    );
    assert_eq!(compressed.len(), state_count * num_classes);

    // (1) Row-dedup ceiling: assign each DISTINCT compressed row a physical id,
    // build the state→row indirection, and count distinct rows.
    let mut row_to_id: HashMap<&[u32], u32> = HashMap::with_capacity(state_count);
    let mut row_of: Vec<u32> = Vec::with_capacity(state_count);
    let mut distinct_rows: Vec<&[u32]> = Vec::new();
    for s in 0..state_count {
        let row = &compressed[s * num_classes..(s + 1) * num_classes];
        let id = *row_to_id.entry(row).or_insert_with(|| {
            let id = distinct_rows.len() as u32;
            distinct_rows.push(row);
            id
        });
        row_of.push(id);
    }
    let distinct = distinct_rows.len();

    // LOSSLESS: reconstruct each state's row from the deduped table + indirection
    // and assert byte-identity with the compressed row. A deduped table that
    // changed any transition would silently drop/forge matches — refuse it.
    for s in 0..state_count {
        let original = &compressed[s * num_classes..(s + 1) * num_classes];
        let deduped = distinct_rows[row_of[s] as usize];
        assert_eq!(
            original, deduped,
            "state {s}: deduped row diverged from compressed row — the indirection is NOT lossless",
        );
    }

    // (2) u16 viability: every transition target is a state index < state_count;
    // u16 packing is sound iff state_count fits u16.
    let max_target = ac.dfa.transitions.iter().copied().max().unwrap_or(0);
    let u16_viable = state_count <= u16::MAX as usize + 1 && max_target <= u16::MAX as u32;

    // Byte accounting (u32 words today).
    let dense_bytes = state_count * 256 * 4;
    let compressed_bytes = state_count * num_classes * 4;
    let deduped_bytes = distinct * num_classes * 4 + state_count * 4; // table + indirection
    let u16_compressed_bytes = state_count * num_classes * 2;

    eprintln!("keyhog-scale combined automaton (CPU profile-first):");
    eprintln!("  patterns            = {}", patterns.len());
    eprintln!("  states              = {state_count}");
    eprintln!("  byte-classes        = {num_classes}");
    eprintln!(
        "  dense table         = {} KiB ({state_count} x 256 x 4B)",
        dense_bytes / 1024
    );
    eprintln!(
        "  byte-class table    = {} KiB ({state_count} x {num_classes} x 4B)  [SHIPS today]",
        compressed_bytes / 1024
    );
    eprintln!(
        "  distinct rows       = {distinct}  ({:.1}% of states)  ratio {:.3}x",
        100.0 * distinct as f64 / state_count as f64,
        state_count as f64 / distinct as f64
    );
    eprintln!(
        "  + row-dedup table   = {} KiB (table {} KiB + indirection {} KiB)  => {:.3}x vs byte-class",
        deduped_bytes / 1024,
        distinct * num_classes * 4 / 1024,
        state_count * 4 / 1024,
        compressed_bytes as f64 / deduped_bytes as f64,
    );
    eprintln!(
        "  + u16 table         = {} KiB  => {:.3}x vs byte-class  (viable: {u16_viable}, max_target {max_target})",
        u16_compressed_bytes / 1024,
        compressed_bytes as f64 / u16_compressed_bytes as f64,
    );

    // --- Pinned regression facts (deterministic catalog ⇒ exact values).
    // Measured once on CPU; a build-side change (alphabet, DFA construction) that
    // shifts these is caught. Update deliberately with a note, never to paper over
    // an unexplained drift. ---
    assert_eq!(
        patterns.len(),
        2048,
        "large_catalog() must yield 2048 literals"
    );
    assert_eq!(
        state_count, 13_199,
        "combined automaton state count drifted"
    );
    assert_eq!(num_classes, 67, "byte-class count drifted");
    assert_eq!(
        distinct, 12_579,
        "row-dedup ceiling (distinct compressed rows) drifted"
    );

    // DECISION ENCODED — row-dedup is REFUTED at this scale, u16 is the lever:
    //
    // 95.3% of compressed rows are already distinct (ratio 1.049x), so a
    // `state -> row` indirection shrinks the byte-class table by only ~3% while
    // ADDING a per-byte `row_of[state]` load to the hot loop and barely reducing
    // the distinct-rows-per-warp working set that is the named L1 limiter. That is
    // a net pessimization, not a win — do NOT build it. If a future build ever
    // makes rows highly redundant (dedup ratio >= 1.5x), this assert fires so the
    // decision is revisited with fresh evidence rather than silently stale.
    let dedup_ratio = state_count as f64 / distinct as f64;
    assert!(
        dedup_ratio < 1.5,
        "row-dedup was refuted at {dedup_ratio:.3}x; a >=1.5x ratio means rows are now \
         redundant enough to reconsider the state->row indirection lever",
    );

    // u16 is the structurally stronger lever: viable at this scale (13k states <<
    // 65535) and HALVES the byte-class table with NO extra indirection load. The
    // only open question is the WGSL u16-unpack ALU cost (GPU-profile-gated).
    assert!(
        u16_viable,
        "u16 targets must be viable at keyhog scale ({state_count} states, max_target {max_target})",
    );
    assert_eq!(
        compressed_bytes,
        2 * u16_compressed_bytes,
        "u16 packing must halve the byte-class transition table exactly",
    );
}
