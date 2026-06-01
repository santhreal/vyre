use super::super::baseline::cpu_aho_overlapping_matches;
use super::super::count;
use super::super::PATTERNS;
use vyre_libs::scan::classic_ac::{
    classic_ac_candidate_end_byte_mask_words, classic_ac_candidate_suffix2_mask_words,
    classic_ac_compile, classic_ac_scan_counts,
};
use vyre_libs::scan::pack_u32_slice;

#[test]
fn count_prefilter_mask_keeps_all_generated_overlapping_hits_and_skips_noise() {
    const CASES: u32 = 10_000;

    let ac = classic_ac_compile(PATTERNS);
    let candidate_mask = classic_ac_candidate_end_byte_mask_words(&ac.dfa);
    let suffix2_mask = classic_ac_candidate_suffix2_mask_words(&ac.dfa);
    let mut total_lanes = 0_u64;
    let mut total_candidate_lanes = 0_u64;
    let mut total_suffix2_lanes = 0_u64;
    let mut total_matches = 0_u64;
    let mut unaligned_matches = 0_u64;

    for case in 0..CASES {
        let len = 257 + (mix32(case ^ 0xA5C0_0001) % 3_840) as usize;
        let haystack = generated_scan_haystack(case, len);
        let aho_matches = cpu_aho_overlapping_matches(PATTERNS, &haystack)
            .unwrap_or_else(|error| panic!("generated AC baseline case {case} failed: {error}"));
        let count = classic_ac_scan_counts(&ac, &haystack)
            .into_iter()
            .sum::<u32>();
        let candidate_lanes = count::candidate_end_lane_count(&haystack, &candidate_mask);
        let suffix2_lanes =
            count::candidate_suffix2_lane_count(&haystack, &candidate_mask, &suffix2_mask);
        let inputs =
            count::scan_ac_count_inputs_with_masks(&ac, &haystack, &candidate_mask, &suffix2_mask);

        assert_eq!(
            count as usize,
            aho_matches.len(),
            "bounded count must match overlapping Aho-Corasick case {case}"
        );
        assert_eq!(
            inputs.len(),
            7,
            "prefiltered count input layout case {case}"
        );
        assert_eq!(
            inputs[3],
            pack_u32_slice(&candidate_mask),
            "candidate mask input case {case}"
        );
        assert_eq!(
            inputs[4],
            pack_u32_slice(&suffix2_mask),
            "suffix2 candidate mask input case {case}"
        );
        assert_eq!(
            inputs[5],
            pack_u32_slice(&[haystack.len() as u32]),
            "haystack length input case {case}"
        );
        for hit in &aho_matches {
            let end = hit.end as usize;
            assert!(
                end > 0,
                "generated AC hit must have nonzero end case {case}"
            );
            let end_byte = haystack[end - 1];
            assert!(
                count::byte_is_candidate_end(end_byte, &candidate_mask),
                "prefilter candidate mask rejected real overlapping hit case={case} hit={hit:?}"
            );
            if end > 1 {
                assert!(
                    count::suffix2_pair_is_candidate(haystack[end - 2], end_byte, &suffix2_mask),
                    "suffix2 prefilter rejected real overlapping hit case={case} hit={hit:?}"
                );
            }
            unaligned_matches += u64::from(hit.start % 4 != 0);
        }

        total_lanes += haystack.len() as u64;
        total_candidate_lanes += u64::from(candidate_lanes);
        total_suffix2_lanes += u64::from(suffix2_lanes);
        total_matches += aho_matches.len() as u64;
    }

    assert!(
        total_matches > u64::from(CASES),
        "generated scan corpus must exercise dense overlapping matches"
    );
    assert!(
        unaligned_matches > total_matches / 2,
        "generated scan corpus must hit mostly unaligned match starts"
    );
    assert!(
        total_candidate_lanes * 4 < total_lanes,
        "candidate-end prefilter should skip at least 75% of noisy lanes"
    );
    assert!(
        total_suffix2_lanes <= total_candidate_lanes,
        "suffix2 prefilter must never admit more lanes than the byte-end prefilter"
    );
    assert!(
        total_suffix2_lanes * 2 < total_candidate_lanes,
        "suffix2 prefilter should cut byte-end replay lanes by more than half"
    );
}

fn generated_scan_haystack(case: u32, len: usize) -> Vec<u8> {
    let mut haystack = vec![0_u8; len];
    for (index, byte) in haystack.iter_mut().enumerate() {
        let mixed = mix32(case ^ (index as u32).wrapping_mul(0x9E37_79B9));
        *byte = 33 + (mixed % 90) as u8;
    }

    for (pattern_index, pattern) in PATTERNS.iter().enumerate() {
        let stride = 31 + ((mix32(case ^ pattern_index as u32) % 127) as usize);
        let mut offset = ((mix32(case ^ (pattern_index as u32).wrapping_mul(0x45D9_F3B)) as usize)
            % stride)
            + pattern_index;
        while offset + pattern.len() <= haystack.len() {
            if (offset & 31) != 0 {
                haystack[offset..offset + pattern.len()].copy_from_slice(pattern);
            }
            offset = offset.saturating_add(stride + (pattern_index % 5));
        }
    }

    for overlap in 0..16 {
        let left = PATTERNS[overlap % PATTERNS.len()];
        let right = PATTERNS[(overlap * 7 + case as usize) % PATTERNS.len()];
        let offset = (mix32(case ^ overlap as u32) as usize) % len.saturating_sub(64).max(1);
        if offset + left.len() + right.len() <= haystack.len() {
            haystack[offset..offset + left.len()].copy_from_slice(left);
            let right_start = offset + left.len().saturating_sub(1).min(3);
            if right_start + right.len() <= haystack.len() {
                haystack[right_start..right_start + right.len()].copy_from_slice(right);
            }
        }
    }

    haystack
}

fn mix32(mut value: u32) -> u32 {
    value ^= value >> 16;
    value = value.wrapping_mul(0x7FEB_352D);
    value ^= value >> 15;
    value = value.wrapping_mul(0x846C_A68B);
    value ^ (value >> 16)
}
