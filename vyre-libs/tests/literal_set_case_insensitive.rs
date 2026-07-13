//! W2-1 gate: `GpuLiteralSet::compile_case_insensitive` matches ASCII case
//! interchangeably, IN KERNEL, with parity to the host-folded reference it
//! replaces.
//!
//! The correctness contract the plan names: a case-insensitive scan of the RAW
//! mixed-case haystack must equal a case-SENSITIVE scan of the host-lowercased
//! haystack with lowercased patterns, the exact `to_ascii_lowercase` pass this
//! feature removes from the consumer. Proven on the CPU reference backend
//! (randomized differential) AND the real wgpu GPU, plus a wire round-trip that
//! shows the case-insensitive flag survives the cache (else the lazily-rebuilt
//! prefilter masks would silently revert to case-sensitive and under-fire on
//! uppercase input: Law 10).

use vyre_driver_reference::CpuRefBackend;
use vyre_libs::scan::{GpuLiteralSet, LiteralMatch};

struct Lcg(u64);
impl Lcg {
    fn next_u32(&mut self) -> u32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 33) as u32
    }
    fn below(&mut self, n: u32) -> u32 {
        if n == 0 {
            0
        } else {
            self.next_u32() % n
        }
    }
}

fn sorted_triples(matches: &[LiteralMatch]) -> Vec<(u32, u32, u32)> {
    let mut v: Vec<(u32, u32, u32)> = matches
        .iter()
        .map(|m| (m.pattern_id, m.start, m.end))
        .collect();
    v.sort_unstable();
    v
}

/// Mixed-case alphabet so folding actually matters, plus a non-letter so digits
/// and separators are exercised (they must NOT be folded).
const ALPHABET: &[u8] = b"aAbBkKtT_9/";

fn random_patterns(rng: &mut Lcg) -> Vec<Vec<u8>> {
    use std::collections::BTreeSet;
    let count = 1 + rng.below(6);
    let mut set: BTreeSet<Vec<u8>> = BTreeSet::new();
    for _ in 0..count {
        let len = 1 + rng.below(6);
        let lit: Vec<u8> = (0..len)
            .map(|_| ALPHABET[rng.below(ALPHABET.len() as u32) as usize])
            .collect();
        set.insert(lit);
    }
    set.into_iter().collect()
}

fn random_haystack(rng: &mut Lcg) -> Vec<u8> {
    let len = 4 + rng.below(120);
    (0..len)
        .map(|_| ALPHABET[rng.below(ALPHABET.len() as u32) as usize])
        .collect()
}

fn lowered(bytes: &[u8]) -> Vec<u8> {
    bytes.iter().map(|b| b.to_ascii_lowercase()).collect()
}

/// The invariant, on a given backend: case-insensitive scan of the raw haystack
/// equals the host-folded case-sensitive scan.
fn assert_ci_equals_host_fold<B: vyre::VyreBackend + ?Sized>(
    backend: &B,
    patterns: &[Vec<u8>],
    haystack: &[u8],
    label: &str,
) {
    let refs: Vec<&[u8]> = patterns.iter().map(Vec::as_slice).collect();
    let ci = GpuLiteralSet::compile_case_insensitive(&refs);
    let ci_hits = ci
        .scan_all(backend, haystack)
        .unwrap_or_else(|e| panic!("[{label}] ci scan_all failed: {e}"));

    let lowered_pats: Vec<Vec<u8>> = patterns.iter().map(|p| lowered(p)).collect();
    let lowered_refs: Vec<&[u8]> = lowered_pats.iter().map(Vec::as_slice).collect();
    let reference = GpuLiteralSet::compile(&lowered_refs);
    let ref_hits = reference
        .scan_all(backend, &lowered(haystack))
        .unwrap_or_else(|e| panic!("[{label}] host-folded reference scan_all failed: {e}"));

    assert_eq!(
        sorted_triples(&ci_hits),
        sorted_triples(&ref_hits),
        "[{label}] case-insensitive scan of raw haystack must equal host-folded case-sensitive scan\n\
         patterns={:?}\n haystack={:?}",
        patterns
            .iter()
            .map(|p| String::from_utf8_lossy(p).into_owned())
            .collect::<Vec<_>>(),
        String::from_utf8_lossy(haystack),
    );
}

#[test]
fn ci_equals_host_fold_reference_high_volume_cpu() {
    let backend = CpuRefBackend;
    let cases: usize = std::env::var("VYRE_CI_CASES")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(1500);
    let mut rng = Lcg(0x6361_7365_5f69u64);
    for case in 0..cases {
        let patterns = random_patterns(&mut rng);
        let haystack = random_haystack(&mut rng);
        assert_ci_equals_host_fold(&backend, &patterns, &haystack, &format!("cpu case {case}"));
    }
}

#[test]
fn ci_matches_every_case_variant_and_leaves_non_letters_exact() {
    let backend = CpuRefBackend;
    let matcher = GpuLiteralSet::compile_case_insensitive(&[b"Key9".as_slice()]);
    // Every case variant of the letters matches; the digit '9' stays exact.
    for variant in [b"key9", b"KEY9", b"kEy9", b"KeY9"] {
        let hits = matcher.scan_all(&backend, variant).expect("ci scan_all");
        assert_eq!(
            hits.len(),
            1,
            "variant {:?} must match once",
            std::str::from_utf8(variant).unwrap()
        );
    }
    // A different digit must NOT match (non-letters are not folded).
    let miss = matcher.scan_all(&backend, b"key8").expect("ci scan_all");
    assert!(
        miss.is_empty(),
        "digit mismatch must not match under ASCII case folding"
    );
}

#[test]
fn case_insensitive_survives_wire_round_trip() {
    let backend = CpuRefBackend;
    let matcher = GpuLiteralSet::compile_case_insensitive(&[b"TOKEN".as_slice(), b"AKIA"]);
    let blob = matcher.to_bytes().expect("serialize ci matcher");
    let restored = GpuLiteralSet::from_bytes(&blob).expect("deserialize ci matcher");

    // The restored matcher must STILL be case-insensitive: its lazily-rebuilt
    // prefilter masks must admit uppercase-input candidates. A haystack in a
    // different case than the patterns is the discriminator.
    let haystack = b"__token__akia__ToKeN__";
    let restored_hits = restored
        .scan_all(&backend, haystack)
        .expect("restored scan");
    let fresh_hits = matcher.scan_all(&backend, haystack).expect("fresh scan");
    assert_eq!(
        sorted_triples(&restored_hits),
        sorted_triples(&fresh_hits),
        "case-insensitive flag must survive the wire round-trip (rebuilt masks stay folded)"
    );
    assert!(
        !restored_hits.is_empty(),
        "restored ci matcher must find the lowercase occurrences of uppercase patterns"
    );
}

#[test]
fn ci_equals_host_fold_reference_on_gpu() {
    use vyre_driver_wgpu::WgpuBackend;
    let backend = match WgpuBackend::shared() {
        Ok(backend) => backend,
        Err(e) => {
            eprintln!("no wgpu backend ({e}); skipping GPU case-insensitive gate");
            return;
        }
    };
    let cases: usize = std::env::var("VYRE_CI_GPU_CASES")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(200);
    let mut rng = Lcg(0x6770_755f_6369u64);
    for case in 0..cases {
        let patterns = random_patterns(&mut rng);
        let haystack = random_haystack(&mut rng);
        assert_ci_equals_host_fold(
            backend.as_ref(),
            &patterns,
            &haystack,
            &format!("gpu case {case}"),
        );
    }
}
