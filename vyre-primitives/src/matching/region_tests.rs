use super::region::*;

fn cluster_metadata_for_sorted(input: &[RegionTriple]) -> (Vec<u32>, Vec<u32>) {
    let mut survivors = vec![0u32; input.len()];
    let mut merged_ends = input.iter().map(|region| region.end).collect::<Vec<_>>();

    for i in 0..input.len() {
        let current = input[i];
        let has_prev_overlap = input[..i]
            .iter()
            .any(|prior| prior.pid == current.pid && prior.end >= current.start);
        if has_prev_overlap {
            continue;
        }

        survivors[i] = 1;
        let mut merged_end = current.end;
        for next in &input[i + 1..] {
            if next.pid != current.pid || next.start > merged_end {
                break;
            }
            merged_end = merged_end.max(next.end);
        }
        merged_ends[i] = merged_end;
    }

    (survivors, merged_ends)
}

fn compact_cluster_metadata(
    sorted: &[RegionTriple],
    survivors: &[u32],
    merged_ends: &[u32],
) -> Vec<RegionTriple> {
    sorted
        .iter()
        .zip(survivors.iter())
        .zip(merged_ends.iter())
        .filter_map(|((&region, &survivor), &merged_end)| {
            (survivor != 0).then(|| RegionTriple::new(region.pid, region.start, merged_end))
        })
        .collect()
}

#[test]
fn empty_input() {
    assert!(dedup_regions_cpu(vec![]).is_empty());
}

#[test]
fn single_pass_through() {
    let r = RegionTriple::new(0, 5, 10);
    assert_eq!(dedup_regions_cpu(vec![r]), vec![r]);
}

#[test]
fn exact_duplicate_collapses() {
    let r = RegionTriple::new(0, 5, 10);
    assert_eq!(dedup_regions_cpu(vec![r, r]), vec![r]);
}

#[test]
fn overlapping_same_pid_merges() {
    let a = RegionTriple::new(0, 5, 10);
    let b = RegionTriple::new(0, 7, 12);
    assert_eq!(
        dedup_regions_cpu(vec![a, b]),
        vec![RegionTriple::new(0, 5, 12)]
    );
}

#[test]
fn touching_same_pid_merges() {
    let a = RegionTriple::new(0, 5, 10);
    let b = RegionTriple::new(0, 10, 15);
    assert_eq!(
        dedup_regions_cpu(vec![a, b]),
        vec![RegionTriple::new(0, 5, 15)]
    );
}

#[test]
fn different_pids_never_merge() {
    let a = RegionTriple::new(0, 5, 10);
    let b = RegionTriple::new(1, 5, 10);
    let mut got = dedup_regions_cpu(vec![a, b]);
    got.sort_unstable();
    assert_eq!(got, vec![a, b]);
}

#[test]
fn unsorted_input_handled() {
    let a = RegionTriple::new(0, 5, 10);
    let b = RegionTriple::new(0, 7, 12);
    let c = RegionTriple::new(1, 3, 4);
    let got = dedup_regions_cpu(vec![b, a, c]);
    assert_eq!(got, vec![RegionTriple::new(0, 5, 12), c]);
}

#[test]
fn cluster_of_three_merges() {
    let a = RegionTriple::new(0, 1, 3);
    let b = RegionTriple::new(0, 2, 5);
    let c = RegionTriple::new(0, 4, 8);
    assert_eq!(
        dedup_regions_cpu(vec![a, b, c]),
        vec![RegionTriple::new(0, 1, 8)]
    );
}

#[test]
fn zero_width_matches_preserved() {
    let a = RegionTriple::new(0, 5, 5);
    let b = RegionTriple::new(1, 5, 5);
    let mut got = dedup_regions_cpu(vec![a, b]);
    got.sort_unstable();
    assert_eq!(got, vec![a, b]);
}

#[test]
fn cluster_metadata_handles_nested_short_previous_span() {
    let sorted = vec![
        RegionTriple::new(7, 0, 10),
        RegionTriple::new(7, 2, 3),
        RegionTriple::new(7, 9, 12),
        RegionTriple::new(7, 20, 25),
    ];
    let (survivors, merged_ends) = cluster_metadata_for_sorted(&sorted);

    assert_eq!(survivors, vec![1, 0, 0, 1]);
    assert_eq!(
        compact_cluster_metadata(&sorted, &survivors, &merged_ends),
        vec![RegionTriple::new(7, 0, 12), RegionTriple::new(7, 20, 25)]
    );
}

#[test]
fn generated_cluster_metadata_matches_cpu_dedup() {
    let mut state = 0xC013_CADE_u32;
    for case in 0..4096u32 {
        state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        let count = (state % 96) as usize;
        let mut input = Vec::with_capacity(count);
        for index in 0..count {
            state = state.rotate_left(5) ^ (index as u32).wrapping_mul(0x9E37_79B9);
            let pid = state % 5;
            state = state.rotate_left(7).wrapping_add(case);
            let start = state % 160;
            state = state.rotate_left(11) ^ 0x85EB_CA6B;
            let width = state % 24;
            input.push(RegionTriple::new(pid, start, start.saturating_add(width)));
        }

        let expected = dedup_regions_cpu(input.clone());
        let mut sorted = input;
        sort_regions_cpu(&mut sorted);
        let (survivors, merged_ends) = cluster_metadata_for_sorted(&sorted);
        let actual = compact_cluster_metadata(&sorted, &survivors, &merged_ends);

        assert_eq!(actual, expected, "generated region cluster case {case}");
    }
}

#[test]
fn sort_regions_cpu_matches_ord_impl() {
    let mut a = vec![
        RegionTriple::new(2, 0, 1),
        RegionTriple::new(0, 5, 10),
        RegionTriple::new(1, 3, 4),
        RegionTriple::new(0, 5, 8),
        RegionTriple::new(0, 5, 10),
    ];
    sort_regions_cpu(&mut a);
    assert_eq!(
        a,
        vec![
            RegionTriple::new(0, 5, 8),
            RegionTriple::new(0, 5, 10),
            RegionTriple::new(0, 5, 10),
            RegionTriple::new(1, 3, 4),
            RegionTriple::new(2, 0, 1),
        ]
    );
}

#[test]
fn sort_regions_cpu_is_stable_for_equal_triples() {
    let mut a = vec![
        RegionTriple::new(0, 5, 10),
        RegionTriple::new(0, 5, 10),
        RegionTriple::new(0, 5, 10),
    ];
    sort_regions_cpu(&mut a);
    assert_eq!(a.len(), 3);
    for r in &a {
        assert_eq!(*r, RegionTriple::new(0, 5, 10));
    }
}

#[test]
fn region_dedup_dispatch_grid_packs_large_match_buffers() {
    assert_eq!(region_dedup_dispatch_grid(0), [1, 1, 1]);
    assert_eq!(region_dedup_dispatch_grid(1), [1, 1, 1]);
    assert_eq!(region_dedup_dispatch_grid(256), [1, 1, 1]);
    assert_eq!(region_dedup_dispatch_grid(257), [2, 1, 1]);
    assert_eq!(region_dedup_dispatch_grid(513), [3, 1, 1]);
}

#[test]
fn dedup_regions_flag_program_emits_expected_buffers() {
    let p = dedup_regions_flag_program("pids", "starts", "ends", "survivors", 513);
    assert_eq!(p.workgroup_size, REGION_DEDUP_WORKGROUP_SIZE);
    let names: Vec<&str> = p.buffers.iter().map(|b| b.name()).collect();
    assert_eq!(names, vec!["pids", "starts", "ends", "survivors"]);
    for buf in p.buffers.iter() {
        assert_eq!(buf.count(), 513);
    }
}

#[test]
fn dedup_regions_cluster_program_emits_survivor_and_merged_end_outputs() {
    let p = dedup_regions_cluster_program("pids", "starts", "ends", "survivors", "merged", 64);
    assert_eq!(p.workgroup_size, REGION_DEDUP_WORKGROUP_SIZE);
    let names: Vec<&str> = p.buffers.iter().map(|b| b.name()).collect();
    assert_eq!(names, vec!["pids", "starts", "ends", "survivors", "merged"]);
    assert_eq!(
        p.buffers[3].access(),
        vyre_foundation::ir::BufferAccess::WriteOnly
    );
    assert_eq!(
        p.buffers[4].access(),
        vyre_foundation::ir::BufferAccess::WriteOnly
    );
}

#[test]
fn region_sort_program_emits_expected_buffers() {
    let p = region_sort_program("pi", "si", "ei", "po", "so", "eo", 64);
    assert_eq!(p.workgroup_size, [256, 1, 1]);
    let names: Vec<&str> = p.buffers.iter().map(|b| b.name()).collect();
    assert_eq!(names, vec!["pi", "si", "ei", "po", "so", "eo"]);
    for buf in p.buffers.iter() {
        assert_eq!(buf.count(), 64);
    }
}

#[test]
fn cap_regions_per_pattern_flag_program_emits_expected_buffers() {
    let p = cap_regions_per_pattern_flag_program("pids", "survivors", 3, 128);
    assert_eq!(p.workgroup_size, REGION_DEDUP_WORKGROUP_SIZE);
    let names: Vec<&str> = p.buffers.iter().map(|b| b.name()).collect();
    assert_eq!(names, vec!["pids", "survivors"]);
    assert_eq!(
        p.buffers[0].access(),
        vyre_foundation::ir::BufferAccess::ReadOnly
    );
    assert_eq!(
        p.buffers[1].access(),
        vyre_foundation::ir::BufferAccess::WriteOnly
    );
    for buf in p.buffers.iter() {
        assert_eq!(buf.count(), 128);
    }
}

/// Run the actual cap kernel IR on the reference interpreter and return the
/// survivor flags it writes (the real device program, not a host mirror).
fn eval_cap_survivors(pids: &[u32], k: u32) -> Vec<u32> {
    use std::sync::Arc;
    use vyre_reference::reference_eval;
    use vyre_reference::value::Value;

    let count = pids.len() as u32;
    let program = cap_regions_per_pattern_flag_program("pids", "survivors", k, count);
    let to_value = |data: &[u32]| Value::Bytes(Arc::from(crate::wire::pack_u32_slice(data)));
    // Binding order: pids (in), survivors (out, seeded zero).
    let inputs = vec![to_value(pids), to_value(&vec![0u32; pids.len()])];
    let results = reference_eval(&program, &inputs).expect("Fix: cap kernel interpreter failed");
    // The interpreter returns only the writable buffer(s); `survivors` is the
    // single output, so it is `results[0]`.
    results[0]
        .to_bytes()
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

#[test]
fn cap_kernel_matches_cpu_oracle_over_random_pid_streams() {
    // Deterministic LCG (no Date/rand in primitives tests).
    let mut state = 0x2545_F491_4F6C_DD1Du64;
    let mut next = || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (state >> 33) as u32
    };
    for case in 0..400 {
        let n = (next() % 60) as usize; // include the empty-buffer case
                                        // Small pid alphabet so several matches share a pid (caps actually bite).
        let pids: Vec<u32> = (0..n).map(|_| next() % 6).collect();
        let k = next() % 5; // includes k == 0 (cap everything to nothing)

        if pids.is_empty() {
            // count == 0 yields an empty program; skip the interpreter (no buffers
            // to bind) but assert the oracle agrees it is empty.
            assert!(cap_regions_per_pattern_survivors_cpu(&pids, k).is_empty());
            continue;
        }

        let kernel = eval_cap_survivors(&pids, k);
        let oracle = cap_regions_per_pattern_survivors_cpu(&pids, k);
        assert_eq!(
            kernel, oracle,
            "case {case}: cap kernel survivor flags must equal the running-count oracle\n\
             pids={pids:?} k={k}"
        );

        // Independent property: each pid keeps exactly min(group_size, k) survivors.
        use std::collections::HashMap;
        let mut group: HashMap<u32, u32> = HashMap::new();
        for &p in &pids {
            *group.entry(p).or_insert(0) += 1;
        }
        let mut kept: HashMap<u32, u32> = HashMap::new();
        for (&p, &flag) in pids.iter().zip(kernel.iter()) {
            *kept.entry(p).or_insert(0) += flag;
        }
        for (pid, total) in group {
            assert_eq!(
                kept.get(&pid).copied().unwrap_or(0),
                total.min(k),
                "case {case}: pid {pid} must keep min(group={total}, k={k}) survivors"
            );
        }
    }
}

#[test]
fn cap_kernel_edge_k_zero_and_k_above_group() {
    // k == 0 drops everything; a k above every group keeps everything.
    let pids = [2u32, 2, 5, 2, 5, 9];
    assert_eq!(eval_cap_survivors(&pids, 0), vec![0, 0, 0, 0, 0, 0]);
    assert_eq!(eval_cap_survivors(&pids, 100), vec![1, 1, 1, 1, 1, 1]);
    // k == 2: pid 2 (3 occurrences) keeps its first two; pid 5 (2) keeps both.
    assert_eq!(eval_cap_survivors(&pids, 2), vec![1, 1, 1, 0, 1, 1]);
}

#[test]
fn region_sort_program_zero_count_traps() {
    let p = region_sort_program("pi", "si", "ei", "po", "so", "eo", 0);
    assert!(p.stats().trap());
}

/// Run one Program on the reference interpreter, returning every writable
/// buffer it produces (in binding order) decoded back to `u32`. Inputs are one
/// byte-packed value per numbered storage/output buffer, in binding order.
#[cfg(test)]
fn run_u32_program(program: &vyre_foundation::ir::Program, inputs: &[&[u32]]) -> Vec<Vec<u32>> {
    use std::sync::Arc;
    use vyre_reference::reference_eval;
    use vyre_reference::value::Value;
    let values: Vec<Value> = inputs
        .iter()
        .map(|data| Value::Bytes(Arc::from(crate::wire::pack_u32_slice(data))))
        .collect();
    let results = reference_eval(program, &values).expect("Fix: interpreter failed");
    results
        .iter()
        .map(|value| {
            value
                .to_bytes()
                .chunks_exact(4)
                .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
                .collect()
        })
        .collect()
}

/// End-to-end operator path for the per-pattern cap: the device pipeline
/// `cap flags → exclusive prefix-scan → stream-compact` must produce exactly the
/// first-`k`-per-pattern matches, compacted, with the right live count, the
/// device-side post-processing W2-5 gives consumers in place of a host pass over
/// a full readback. Executes every stage on the reference interpreter (no host
/// mirror) and checks the compacted survivors against a plain host reference.
#[test]
fn cap_pipeline_compacts_first_k_per_pattern_on_device() {
    use crate::math::prefix_scan::{prefix_scan, ScanKind};
    use crate::math::stream_compact::stream_compact;

    // Input already sorted by (pid, start, end), the order region_sort_program
    // produces (so "first k in array order" is "k earliest-start per pattern").
    let triples: &[(u32, u32, u32)] = &[
        (1, 0, 4),
        (1, 10, 14),
        (1, 20, 24),
        (1, 30, 34),
        (3, 5, 9),
        (3, 15, 19),
        (7, 1, 2),
    ];
    let pids: Vec<u32> = triples.iter().map(|t| t.0).collect();
    let starts: Vec<u32> = triples.iter().map(|t| t.1).collect();
    let n = pids.len() as u32;
    let k = 2u32;

    // Stage 1 (cap flags on device).
    let survivors = run_u32_program(
        &cap_regions_per_pattern_flag_program("pids", "survivors", k, n),
        &[&pids, &vec![0u32; pids.len()]],
    )
    .remove(0);

    // Stage 2 (exclusive prefix scan of the flags (what stream_compact wants)).
    let offsets = run_u32_program(
        &prefix_scan("flags", "offsets", n, ScanKind::ExclusiveSum),
        &[&survivors, &vec![0u32; pids.len()]],
    )
    .remove(0);

    // Stage 3 (compact the pid AND start columns on the shared flags/offsets).
    let compact_pids = run_u32_program(
        &stream_compact("payloads", "flags", "offsets", "out", "live", n),
        &[
            &pids,
            &survivors,
            &offsets,
            &vec![0u32; pids.len()],
            &[0u32],
        ],
    );
    let live = compact_pids[1][0] as usize;
    let out_pids = &compact_pids[0][..live];

    let compact_starts = run_u32_program(
        &stream_compact("payloads", "flags", "offsets", "out", "live", n),
        &[
            &starts,
            &survivors,
            &offsets,
            &vec![0u32; pids.len()],
            &[0u32],
        ],
    )
    .remove(0);
    let out_starts = &compact_starts[..live];

    // Host reference: keep the first k matches of each pid in array order.
    use std::collections::HashMap;
    let mut seen: HashMap<u32, u32> = HashMap::new();
    let mut ref_pids = Vec::new();
    let mut ref_starts = Vec::new();
    for &(pid, start, _end) in triples {
        let count = seen.entry(pid).or_insert(0);
        if *count < k {
            ref_pids.push(pid);
            ref_starts.push(start);
        }
        *count += 1;
    }

    assert_eq!(
        live,
        ref_pids.len(),
        "live count must match the capped survivor count"
    );
    assert_eq!(
        out_pids,
        ref_pids.as_slice(),
        "compacted pids must be first-k-per-pattern"
    );
    assert_eq!(
        out_starts,
        ref_starts.as_slice(),
        "compacted starts must line up with their pids through the shared flags/offsets"
    );
    // pid 1 (4 matches) capped to 2, pid 3 (2) kept, pid 7 (1) kept -> 2+2+1 = 5.
    assert_eq!(live, 5);
}

#[test]
fn compact_first_per_region_pattern_flag_program_emits_expected_buffers() {
    let p = compact_first_per_region_pattern_flag_program("regions", "pids", "survivors", 128);
    assert_eq!(p.workgroup_size, REGION_DEDUP_WORKGROUP_SIZE);
    let names: Vec<&str> = p.buffers.iter().map(|b| b.name()).collect();
    assert_eq!(names, vec!["regions", "pids", "survivors"]);
    assert_eq!(
        p.buffers[0].access(),
        vyre_foundation::ir::BufferAccess::ReadOnly
    );
    assert_eq!(
        p.buffers[1].access(),
        vyre_foundation::ir::BufferAccess::ReadOnly
    );
    assert_eq!(
        p.buffers[2].access(),
        vyre_foundation::ir::BufferAccess::WriteOnly
    );
    for buf in p.buffers.iter() {
        assert_eq!(buf.count(), 128);
    }
}

/// Run the actual per-region compaction kernel IR on the reference interpreter
/// and return the survivor flags it writes (the real device program).
fn eval_compact_survivors(regions: &[u32], pids: &[u32]) -> Vec<u32> {
    use std::sync::Arc;
    use vyre_reference::reference_eval;
    use vyre_reference::value::Value;

    let count = regions.len() as u32;
    let program =
        compact_first_per_region_pattern_flag_program("regions", "pids", "survivors", count);
    let to_value = |data: &[u32]| Value::Bytes(Arc::from(crate::wire::pack_u32_slice(data)));
    // Binding order: regions (in), pids (in), survivors (out, seeded zero).
    let inputs = vec![
        to_value(regions),
        to_value(pids),
        to_value(&vec![0u32; regions.len()]),
    ];
    let results =
        reference_eval(&program, &inputs).expect("Fix: compaction kernel interpreter failed");
    // `survivors` is the single writable buffer, so it is `results[0]`.
    results[0]
        .to_bytes()
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

#[test]
fn compact_kernel_matches_cpu_oracle_over_random_region_pid_streams() {
    // Deterministic LCG (no Date/rand in primitives tests).
    let mut state = 0x1357_9BDF_2468_ACE0u64;
    let mut next = || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (state >> 33) as u32
    };
    for case in 0..400 {
        let n = (next() % 60) as usize; // include the empty-buffer case
                                        // Small region/pid alphabets so pairs recur (compaction actually bites).
        let regions: Vec<u32> = (0..n).map(|_| next() % 5).collect();
        let pids: Vec<u32> = (0..n).map(|_| next() % 6).collect();

        if regions.is_empty() {
            // count == 0 yields an empty program; assert the oracle agrees.
            assert!(compact_first_per_region_pattern_survivors_cpu(&regions, &pids).is_empty());
            continue;
        }

        let kernel = eval_compact_survivors(&regions, &pids);
        let oracle = compact_first_per_region_pattern_survivors_cpu(&regions, &pids);
        assert_eq!(
            kernel, oracle,
            "case {case}: compaction survivor flags must equal the first-occurrence oracle\n\
             regions={regions:?} pids={pids:?}"
        );

        // Independent property: each distinct (region, pid) pair keeps exactly one.
        use std::collections::HashMap;
        let mut kept: HashMap<(u32, u32), u32> = HashMap::new();
        for ((&r, &p), &flag) in regions.iter().zip(pids.iter()).zip(kernel.iter()) {
            *kept.entry((r, p)).or_insert(0) += flag;
        }
        for (pair, total) in kept {
            assert_eq!(
                total, 1,
                "case {case}: pair {pair:?} must keep exactly one positioned representative"
            );
        }
    }
}

#[test]
fn compact_kernel_edge_first_occurrence_only() {
    // Same pid in different regions is NOT a duplicate (keyed on the pair).
    let regions = [0u32, 0, 1, 0, 1, 1];
    let pids = [7u32, 7, 7, 9, 9, 9];
    // (0,7) first @0; (0,7) again @1 drop; (1,7) first @2; (0,9) first @3;
    // (1,9) first @4; (1,9) again @5 drop.
    assert_eq!(
        eval_compact_survivors(&regions, &pids),
        vec![1, 0, 1, 1, 1, 0]
    );
    // All-distinct pairs keep everything; all-identical pair keeps only the first.
    assert_eq!(
        eval_compact_survivors(&[0, 1, 2], &[0, 1, 2]),
        vec![1, 1, 1]
    );
    assert_eq!(
        eval_compact_survivors(&[4, 4, 4], &[3, 3, 3]),
        vec![1, 0, 0]
    );
}

/// End-to-end operator path for per-region compaction: the device pipeline
/// `compact flags → exclusive prefix-scan → stream-compact` must produce exactly
/// one positioned representative per `(region, pid)` pair, the positioned form
/// of the presence-by-region bitmap, computed on device with no host group-by.
#[test]
fn compact_pipeline_first_per_region_pattern_on_device() {
    use crate::math::prefix_scan::{prefix_scan, ScanKind};
    use crate::math::stream_compact::stream_compact;

    // (region, pid, start) tuples in array order. Pairs (0,1) and (2,1) recur.
    let tuples: &[(u32, u32, u32)] = &[
        (0, 1, 4),
        (0, 1, 10), // dup of (0,1), dropped
        (0, 3, 20),
        (2, 1, 5),
        (2, 1, 15), // dup of (2,1), dropped
        (2, 3, 25),
    ];
    let regions: Vec<u32> = tuples.iter().map(|t| t.0).collect();
    let pids: Vec<u32> = tuples.iter().map(|t| t.1).collect();
    let starts: Vec<u32> = tuples.iter().map(|t| t.2).collect();
    let n = regions.len() as u32;
    let seed = vec![0u32; regions.len()];

    // Stage 1 (compaction flags on device).
    let survivors = run_u32_program(
        &compact_first_per_region_pattern_flag_program("regions", "pids", "survivors", n),
        &[&regions, &pids, &seed],
    )
    .remove(0);

    // Stage 2 (exclusive prefix scan of the flags (offsets for stream_compact)).
    let offsets = run_u32_program(
        &prefix_scan("flags", "offsets", n, ScanKind::ExclusiveSum),
        &[&survivors, &seed],
    )
    .remove(0);

    // Stage 3 (compact the start column on the shared flags/offsets).
    let compact_starts = run_u32_program(
        &stream_compact("payloads", "flags", "offsets", "out", "live", n),
        &[&starts, &survivors, &offsets, &seed, &[0u32]],
    );
    let live = compact_starts[1][0] as usize;
    let out_starts = &compact_starts[0][..live];

    // Host reference: keep the first occurrence of each (region, pid) pair.
    use std::collections::HashSet;
    let mut seen: HashSet<(u32, u32)> = HashSet::new();
    let mut ref_starts = Vec::new();
    for &(region, pid, start) in tuples {
        if seen.insert((region, pid)) {
            ref_starts.push(start);
        }
    }

    assert_eq!(
        live,
        ref_starts.len(),
        "live count must match the distinct-pair count"
    );
    assert_eq!(
        out_starts,
        ref_starts.as_slice(),
        "compacted starts must be the first-per-(region,pid) positions"
    );
    // 4 distinct pairs: (0,1) (0,3) (2,1) (2,3) -> starts 4, 20, 5, 25.
    assert_eq!(out_starts, &[4, 20, 5, 25]);
}

#[test]
fn region_sort_program_pipeline_composes_with_dedup_cluster_metadata() {
    let sort_p = region_sort_program("pi", "si", "ei", "ps", "ss", "es", 32);
    let cluster_p = dedup_regions_cluster_program("ps", "ss", "es", "flags", "merged", 32);
    let sort_outputs: Vec<&str> = sort_p
        .buffers
        .iter()
        .filter(|b| b.access() == vyre_foundation::ir::BufferAccess::ReadWrite)
        .map(|b| b.name())
        .collect();
    assert_eq!(sort_outputs, vec!["ps", "ss", "es"]);
    let cluster_inputs: Vec<&str> = cluster_p
        .buffers
        .iter()
        .filter(|b| b.access() == vyre_foundation::ir::BufferAccess::ReadOnly)
        .map(|b| b.name())
        .collect();
    assert_eq!(cluster_inputs, vec!["ps", "ss", "es"]);
}
