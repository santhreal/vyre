//! Shared INDEPENDENT ground-truth oracle for region-presence conformance tests.
//!
//! The single source of truth for "what per-region presence bitmap must the
//! literal-set scan produce", used by both the CPU-reference gate
//! (`literal_set_presence_by_region_ground_truth`) and the real-GPU gate
//! (`literal_set_presence_by_region_gpu_ground_truth`). Keeping the oracle in one
//! place means the two backends are checked against the EXACT same specification,
//! so a wgpu/cuda-only under-fire cannot hide behind a reference-only oracle.
//!
//! The oracle walks the compiled DFA byte-by-byte in plain Rust (mirroring the
//! crate-internal `classic_ac_scan`, inlined so the gates stay always-on without
//! the `cpu-parity` feature). It touches NONE of the GPU program's machinery, no
//! suffix3 cascade, no candidate masks, no region binary search, no packed-byte
//! extraction (so any divergence is a real recall bug at the source).

// Each integration-test binary that `mod presence_oracle;`s this file uses a
// subset of the helpers; silence the unused-in-this-binary warnings. `dead_code`
// covers helpers a binary never calls; `unreachable_pub` covers the `pub` on a
// shared helper that a given binary does not re-export (the `pub` exists so the
// OTHER gate can import it).
#![allow(dead_code, unreachable_pub)]

use vyre_libs::scan::classic_ac::{
    classic_ac_compile, presence_by_region_words, ClassicAcAutomaton,
};

/// Deterministic LCG so failures reproduce from the case index alone.
pub struct Lcg(pub u64);

impl Lcg {
    #[must_use]
    pub fn new(seed: u64) -> Self {
        Self(seed)
    }

    pub fn next_u32(&mut self) -> u32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 33) as u32
    }

    pub fn below(&mut self, n: u32) -> u32 {
        if n == 0 {
            0
        } else {
            self.next_u32() % n
        }
    }
}

/// `region = largest r with region_starts[r] <= end_pos`. `region_starts` is
/// ascending and starts at 0, so this is `upper_bound(end_pos) - 1`.
#[must_use]
pub fn region_of(region_starts: &[u32], end_pos: u32) -> usize {
    let mut region = 0usize;
    for (r, &start) in region_starts.iter().enumerate() {
        if start <= end_pos {
            region = r;
        } else {
            break;
        }
    }
    region
}

/// Independent DFA walk (plain Rust, no packed bytes / prefilter / region
/// search): emit every `(pattern_id, end_pos)`.
#[must_use]
pub fn dfa_scan(ac: &ClassicAcAutomaton, haystack: &[u8]) -> Vec<(u32, u32)> {
    let dfa = &ac.dfa;
    let mut state = 0u32;
    let mut out = Vec::new();
    for (pos, &b) in haystack.iter().enumerate() {
        state = dfa.transitions[(state as usize) * 256 + (b as usize)];
        let begin = dfa.output_offsets[state as usize] as usize;
        let end = dfa.output_offsets[state as usize + 1] as usize;
        for &pattern_id in &dfa.output_records[begin..end] {
            out.push((pattern_id, pos as u32));
        }
    }
    out
}

/// Independent oracle presence bitmap: walk the DFA, attribute each match by its
/// end position. `presence_words` per region, `pattern_id`-indexed bits, the
/// exact layout `scan_presence_by_region` emits.
#[must_use]
pub fn oracle_presence(
    ac: &ClassicAcAutomaton,
    haystack: &[u8],
    region_starts: &[u32],
    pattern_count: u32,
) -> Vec<u32> {
    let presence_words = presence_by_region_words(pattern_count, 1) as usize;
    let total = presence_words * region_starts.len();
    let mut bits = vec![0u32; total];
    for (pattern_id, end_pos) in dfa_scan(ac, haystack) {
        let region = region_of(region_starts, end_pos);
        let word = region * presence_words + (pattern_id as usize) / 32;
        bits[word] |= 1u32 << (pattern_id % 32);
    }
    bits
}

/// Compare a backend-produced presence bitmap against the oracle; on mismatch
/// panic with the first divergent `(region, pattern_id)`, its direction
/// (UNDER-FIRE = missed real match, OVER-FIRE = spurious), and the full case.
pub fn assert_presence_matches(
    literals: &[Vec<u8>],
    haystack: &[u8],
    region_starts: &[u32],
    produced: &[u32],
    label: &str,
) {
    let pattern_refs: Vec<&[u8]> = literals.iter().map(Vec::as_slice).collect();
    let ac = classic_ac_compile(&pattern_refs);
    let pattern_count = literals.len() as u32;
    let oracle = oracle_presence(&ac, haystack, region_starts, pattern_count);
    assert_eq!(
        produced.len(),
        oracle.len(),
        "[{label}] presence word count mismatch: produced {} vs oracle {}",
        produced.len(),
        oracle.len()
    );
    let presence_words = presence_by_region_words(pattern_count, 1) as usize;
    for (word_idx, (&prod_word, &oracle_word)) in produced.iter().zip(oracle.iter()).enumerate() {
        if prod_word == oracle_word {
            continue;
        }
        let region = word_idx / presence_words;
        let base_pattern = (word_idx % presence_words) * 32;
        let underfire = oracle_word & !prod_word; // oracle set, produced clear
        let overfire = prod_word & !oracle_word; // produced set, oracle clear
        let mut detail = String::new();
        if underfire != 0 {
            let pid = base_pattern + underfire.trailing_zeros() as usize;
            detail.push_str(&format!(
                " UNDER-FIRE region {region} pattern {pid} (`{}`) missed;",
                String::from_utf8_lossy(&literals[pid])
            ));
        }
        if overfire != 0 {
            let pid = base_pattern + overfire.trailing_zeros() as usize;
            detail.push_str(&format!(
                " OVER-FIRE region {region} pattern {pid} (`{}`) spurious;",
                String::from_utf8_lossy(&literals[pid])
            ));
        }
        panic!(
            "[{label}] region-presence bitmap diverges from independent DFA oracle:{detail}\n\
             literals={:?}\n\
             haystack={:?}\n\
             region_starts={region_starts:?}\n\
             produced_word[{word_idx}]=0x{prod_word:08x} oracle_word[{word_idx}]=0x{oracle_word:08x}",
            literals
                .iter()
                .map(|l| String::from_utf8_lossy(l).into_owned())
                .collect::<Vec<_>>(),
            String::from_utf8_lossy(haystack),
        );
    }
}

/// Small alphabet so literals collide and the DFA / prefilter actually exercise
/// shared prefixes, suffix2/suffix3 candidate gating, and overlapping matches.
pub const ALPHABET: &[u8] = b"abcAB_0/-";

#[must_use]
pub fn random_literals(rng: &mut Lcg) -> Vec<Vec<u8>> {
    use std::collections::BTreeSet;
    let count = 1 + rng.below(8);
    let mut set: BTreeSet<Vec<u8>> = BTreeSet::new();
    for _ in 0..count {
        let len = 1 + rng.below(6);
        let mut lit = Vec::with_capacity(len as usize);
        for _ in 0..len {
            lit.push(ALPHABET[rng.below(ALPHABET.len() as u32) as usize]);
        }
        set.insert(lit);
    }
    set.into_iter().collect()
}

#[must_use]
pub fn random_haystack(rng: &mut Lcg) -> Vec<u8> {
    let len = 8 + rng.below(160);
    (0..len)
        .map(|_| ALPHABET[rng.below(ALPHABET.len() as u32) as usize])
        .collect()
}

#[must_use]
pub fn random_region_starts(rng: &mut Lcg, haystack_len: usize) -> Vec<u32> {
    use std::collections::BTreeSet;
    let region_count = 1 + rng.below(4);
    let mut starts: BTreeSet<u32> = BTreeSet::new();
    starts.insert(0);
    if haystack_len > 1 {
        for _ in 1..region_count {
            starts.insert(1 + rng.below(haystack_len as u32 - 1));
        }
    }
    starts.into_iter().collect()
}

/// Scaled random literal set toward keyhog's real shape: up to `max_count`
/// distinct patterns (so presence rows span multiple 32-bit words when
/// `max_count > 32`), lengths up to `max_len`, over either the small collision
/// alphabet or the FULL byte range (`full_byte`). Multi-word presence rows and
/// full-byte patterns are coverage the small edge cases never reach.
#[must_use]
pub fn random_literals_scaled(
    rng: &mut Lcg,
    min_count: u32,
    max_count: u32,
    max_len: u32,
    full_byte: bool,
) -> Vec<Vec<u8>> {
    use std::collections::BTreeSet;
    let span = (max_count - min_count).max(1);
    let count = min_count + rng.below(span);
    let mut set: BTreeSet<Vec<u8>> = BTreeSet::new();
    // Bounded attempts so a saturated small-alphabet space still terminates.
    let mut attempts = 0u32;
    while (set.len() as u32) < count && attempts < count * 8 {
        attempts += 1;
        let len = 1 + rng.below(max_len);
        let mut lit = Vec::with_capacity(len as usize);
        for _ in 0..len {
            let byte = if full_byte {
                rng.below(256) as u8
            } else {
                ALPHABET[rng.below(ALPHABET.len() as u32) as usize]
            };
            lit.push(byte);
        }
        set.insert(lit);
    }
    set.into_iter().collect()
}

/// Scaled haystack: `len` bytes over the small alphabet (dense matches) or the
/// full byte range.
#[must_use]
pub fn random_haystack_scaled(rng: &mut Lcg, len: usize, full_byte: bool) -> Vec<u8> {
    (0..len)
        .map(|_| {
            if full_byte {
                rng.below(256) as u8
            } else {
                ALPHABET[rng.below(ALPHABET.len() as u32) as usize]
            }
        })
        .collect()
}

/// `region_count` ascending region starts over `haystack_len` bytes, always
/// beginning at 0. Stresses the kernel's `ceil_log2(region_count)` binary search
/// at width the small edge cases never reach.
#[must_use]
pub fn many_region_starts(rng: &mut Lcg, haystack_len: usize, region_count: u32) -> Vec<u32> {
    use std::collections::BTreeSet;
    let mut starts: BTreeSet<u32> = BTreeSet::new();
    starts.insert(0);
    if haystack_len > 1 {
        while (starts.len() as u32) < region_count {
            starts.insert(1 + rng.below(haystack_len as u32 - 1));
        }
    }
    starts.into_iter().collect()
}

/// keyhog-shaped scale fixtures: multi-word presence rows (>32 and >64 patterns),
/// many small regions (stressing the region binary-search width), and full
/// byte-range patterns. These are the shapes W6-1 names as the consumer spec.
/// Deterministic (fixed seed) so a failure reproduces exactly.
#[must_use]
pub fn scale_cases() -> Vec<(String, Vec<Vec<u8>>, Vec<u8>, Vec<u32>)> {
    let mut cases: Vec<(String, Vec<Vec<u8>>, Vec<u8>, Vec<u32>)> = Vec::new();
    let mut rng = Lcg::new(0x5343_414c_455fu64);

    // 1. Multi-word presence: ~100 patterns (4 presence words), dense small-alphabet
    //    matches, a handful of regions. Catches a `pattern_id >> 5` word-indexing bug.
    {
        let literals = random_literals_scaled(&mut rng, 64, 128, 8, false);
        let haystack = random_haystack_scaled(&mut rng, 2048, false);
        let region_starts = many_region_starts(&mut rng, haystack.len(), 6);
        cases.push((
            format!("multiword-presence {} patterns", literals.len()),
            literals,
            haystack,
            region_starts,
        ));
    }

    // 2. Many small regions: 1000 regions over 8 KiB, few patterns (dense fire),
    //    exercising ceil_log2(1000)=10 binary-search iterations across every hit.
    {
        let literals = vec![
            b"ab".to_vec(),
            b"bc".to_vec(),
            b"c0".to_vec(),
            b"_a".to_vec(),
            b"AB".to_vec(),
        ];
        let haystack = random_haystack_scaled(&mut rng, 8192, false);
        let region_starts = many_region_starts(&mut rng, haystack.len(), 1000);
        cases.push((
            format!("many-regions {} regions", region_starts.len()),
            literals,
            haystack,
            region_starts,
        ));
    }

    // 3. Full byte-range patterns + haystack (0..=255): high bytes must not corrupt
    //    packed-byte extraction, the end/suffix masks, or the suffix3 bloom.
    {
        let literals = random_literals_scaled(&mut rng, 40, 80, 6, true);
        let haystack = random_haystack_scaled(&mut rng, 4096, true);
        let region_starts = many_region_starts(&mut rng, haystack.len(), 32);
        cases.push((
            format!("full-byte {} patterns", literals.len()),
            literals,
            haystack,
            region_starts,
        ));
    }

    // 4. Combined keyhog-ish: 200 patterns (7 presence words), 300 regions, 16 KiB.
    {
        let literals = random_literals_scaled(&mut rng, 150, 220, 12, false);
        let haystack = random_haystack_scaled(&mut rng, 16384, false);
        let region_starts = many_region_starts(&mut rng, haystack.len(), 300);
        cases.push((
            format!(
                "keyhog-ish {} patterns {} regions",
                literals.len(),
                region_starts.len()
            ),
            literals,
            haystack,
            region_starts,
        ));
    }

    // 5. keyhog-MAGNITUDE pattern count: ~2000 distinct patterns → 63 presence
    //    words per region, exercising `pattern_id >> 5` word indexing and
    //    `output_records` iteration at a scale the ~200-pattern cases never reach.
    //    (keyhog runs ~6k patterns / 920 detectors; 2000 is the largest that stays
    //    tractable on the CPU reference interpreter, a scaled proxy for that shape,
    //    not a downscale of the contract. The haystack is kept small so the
    //    interpreter's per-position walk stays short while the DFA/presence width
    //    is the thing under stress.) Small alphabet so patterns actually fire.
    {
        let literals = random_literals_scaled(&mut rng, 1800, 2000, 6, false);
        let haystack = random_haystack_scaled(&mut rng, 2048, false);
        let region_starts = many_region_starts(&mut rng, haystack.len(), 8);
        cases.push((
            format!("many-patterns {} patterns", literals.len()),
            literals,
            haystack,
            region_starts,
        ));
    }

    // 6. THOUSANDS of tiny regions: 4000 regions over 16 KiB (~4 bytes each) 
    //    the max `ceil_log2(4000)=12` binary-search width, dense region boundaries
    //    so a match's region attribution is stressed at every boundary. keyhog's
    //    "tens of thousands of small files coalesced" shape. (Requires the grid-aware
    //    reference eval, buffer-shape inference under-covers this haystack.)
    {
        let literals = vec![
            b"ab".to_vec(),
            b"bc".to_vec(),
            b"c9".to_vec(),
            b"_x".to_vec(),
            b"xa".to_vec(),
            b"9_".to_vec(),
        ];
        let haystack = random_haystack_scaled(&mut rng, 16384, false);
        let region_starts = many_region_starts(&mut rng, haystack.len(), 4000);
        cases.push((
            format!("many-tiny-regions {} regions", region_starts.len()),
            literals,
            haystack,
            region_starts,
        ));
    }

    // 7. FEW HUGE regions: 3 regions over 48 KiB (~16 KiB each), the opposite
    //    extreme. The binary search is shallow (`ceil_log2(3)=2`) but a match deep
    //    inside a huge region must still attribute to the correct region (stresses
    //    the region_base boundary arithmetic, not the search width). keyhog's
    //    "few huge files" shape.
    {
        let literals = random_literals_scaled(&mut rng, 16, 32, 8, false);
        let haystack = random_haystack_scaled(&mut rng, 49152, false);
        let region_starts = vec![0u32, 16384, 32768];
        cases.push((
            format!(
                "few-huge-regions {} regions {} KiB",
                region_starts.len(),
                haystack.len() / 1024
            ),
            literals,
            haystack,
            region_starts,
        ));
    }

    cases
}

/// GPU-ONLY large-scale conformance classes: the FULL keyhog literal-set
/// magnitude (~6,000 patterns) at ~1,000 coalesced regions, the shape
/// [`scale_cases`]'s `many-patterns` note calls out as beyond the CPU reference
/// INTERPRETER's tractable range (the interpreter re-walks the IR per position,
/// so 6k patterns is minutes/case there). The real GPU dispatches it in
/// milliseconds and the independent oracle is a SINGLE Aho–Corasick walk
/// (`dfa_scan`, O(haystack + matches), independent of pattern count), so the
/// GPU gate CAN prove this magnitude where the reference gate cannot.
///
/// This is a SYNTHETIC scale proxy (no keyhog data): ~6k distinct small-alphabet
/// patterns over a dense haystack so many actually fire, attributed across ~1,000
/// regions (a `ceil_log2(1000)=10`-deep binary search) with presence rows spanning
/// ~188 `u32` words, the widest `pattern_id >> 5` word-indexing and deepest
/// `output_records` iteration in the suite. Kept out of [`scale_cases`] precisely
/// so the always-run CPU-ref gate stays fast; only the GPU gate calls this.
#[must_use]
pub fn gpu_only_large_scale_cases() -> Vec<(String, Vec<Vec<u8>>, Vec<u8>, Vec<u32>)> {
    let mut rng = Lcg::new(0x6770_755f_3661u64);
    let mut cases = Vec::new();

    // ~6,000 distinct patterns, dense 24 KiB haystack, ~1,000 regions.
    {
        let literals = random_literals_scaled(&mut rng, 5800, 6000, 6, false);
        let haystack = random_haystack_scaled(&mut rng, 24576, false);
        let region_starts = many_region_starts(&mut rng, haystack.len(), 1000);
        cases.push((
            format!(
                "gpu-large {} patterns {} regions {} KiB",
                literals.len(),
                region_starts.len(),
                haystack.len() / 1024
            ),
            literals,
            haystack,
            region_starts,
        ));
    }

    cases
}

/// The targeted W1-1 edge classes as `(label, literals, haystack, region_starts)`
/// tuples, shared by the reference and GPU gates so both prove the SAME fixtures:
/// literals straddling every offset of a region boundary, matches at region
/// start/end, `\0` separator adjacency, dense saturation, non-ASCII adjacency,
/// and the single-byte early-offset (i==0/i==1) cascade boundaries.
#[must_use]
pub fn edge_cases() -> Vec<(String, Vec<Vec<u8>>, Vec<u8>, Vec<u32>)> {
    let mut cases: Vec<(String, Vec<Vec<u8>>, Vec<u8>, Vec<u32>)> = Vec::new();

    // 1. Match END straddling a region boundary at every offset.
    {
        let literals = vec![b"abcd".to_vec()];
        let haystack = b"xxabcdxx".to_vec(); // match ends at 5
        for boundary in 1..haystack.len() as u32 {
            cases.push((
                format!("boundary-straddle @ {boundary}"),
                literals.clone(),
                haystack.clone(),
                vec![0, boundary],
            ));
        }
    }

    // 2. Match at very start and very end.
    {
        let literals = vec![b"ab".to_vec(), b"yz".to_vec()];
        let haystack = b"ab......yz".to_vec();
        cases.push((
            "match-at-start-and-end".into(),
            literals.clone(),
            haystack.clone(),
            vec![0, 5],
        ));
        cases.push((
            "single-region-start-end".into(),
            literals,
            haystack,
            vec![0],
        ));
    }

    // 3. `\0` separator adjacency (coalesced-file layout).
    {
        let literals = vec![b"key".to_vec(), b"val".to_vec()];
        let mut haystack = Vec::new();
        haystack.extend_from_slice(b"key");
        haystack.push(0);
        haystack.extend_from_slice(b"val");
        cases.push((
            "nul-separator-adjacency".into(),
            literals,
            haystack,
            vec![0, 4],
        ));
    }

    // 4. Dense saturation: every position ends a match.
    {
        let literals = vec![b"a".to_vec(), b"aa".to_vec(), b"aaa".to_vec()];
        let haystack = vec![b'a'; 64];
        cases.push((
            "dense-saturation".into(),
            literals,
            haystack,
            vec![0, 16, 32, 48],
        ));
    }

    // 5. Non-ASCII bytes adjacent to matches.
    {
        let literals = vec![b"tok".to_vec(), vec![0xC3, 0xA9]];
        let mut haystack = Vec::new();
        haystack.push(0xFFu8);
        haystack.extend_from_slice(b"tok");
        haystack.push(0x80);
        haystack.extend_from_slice(&[0xC3, 0xA9]);
        haystack.push(0xFE);
        cases.push(("non-ascii-adjacent".into(), literals, haystack, vec![0, 4]));
    }

    // 6. Single-byte literal ending at offsets 0,1,2,3 (i==0 / i==1 cascade
    //    early-replay boundaries).
    {
        let literals = vec![b"z".to_vec()];
        for haystack in [
            b"z".to_vec(),
            b"az".to_vec(),
            b"abz".to_vec(),
            b"abcz".to_vec(),
        ] {
            cases.push((
                format!("single-byte-early-offset len {}", haystack.len()),
                literals.clone(),
                haystack,
                vec![0],
            ));
        }
    }

    cases
}
