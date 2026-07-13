//! W8-3: the fast path from `docs/scanning-a-corpus-the-right-way.md`, made real.
//!
//! This is the "scan a corpus of files with regions and resident tables" demo the
//! guide describes: NOT a single-string literal demo. It coalesces a set of
//! "files" into one contiguous haystack plus the `region_starts` array the
//! by-region scans consume, compiles a literal-set matcher ONCE, prepares a
//! RESIDENT FUSED session (the immutable tables upload once), and runs one launch
//! that produces BOTH the per-region literal-presence bitmap AND the positioned
//! `(pattern_id, start, end)` matches (with timed attribution left on).
//!
//! Run on a GPU box:
//!   cargo run -p vyre-libs --example scan_corpus_fast_path
//! Optionally scan a real directory (each regular file becomes one region):
//!   cargo run -p vyre-libs --example scan_corpus_fast_path -- /path/to/tree
//!
//! With no GPU backend the resident fast path is unavailable (resident buffers are
//! device-only); the example says so LOUDLY and falls back to the portable
//! `scan_all` on the CPU reference backend so it still prints the match set, a
//! stated fallback, never a silent degrade (Law 10).

use std::path::Path;

use vyre::VyreBackend;
use vyre_driver_reference::CpuRefBackend;
use vyre_driver_wgpu::WgpuBackend;
use vyre_foundation::match_result::Match;
use vyre_libs::scan::GpuLiteralSet;

/// The demo pattern set, a handful of secret-shaped literals a consumer might
/// hunt for. In a real consumer these come from a rule catalog (Tier-B data).
const PATTERNS: &[&[u8]] = &[
    b"AKIA",      // aws access key id prefix
    b"password",  // generic
    b"BEGIN RSA", // private key header fragment
    b"token",     // generic
    b"secret",    // generic
];

fn main() {
    let args: Vec<String> = std::env::args().collect();

    // 1. Assemble the corpus. Either read a real directory tree (each file is one
    //    region) or fall back to a built-in multi-file corpus so the example runs
    //    standalone.
    let files: Vec<Vec<u8>> = match args.get(1) {
        Some(dir) => match read_regular_files(Path::new(dir)) {
            Ok(files) if !files.is_empty() => {
                println!("scanning {} file(s) under {dir}", files.len());
                files
            }
            Ok(_) => {
                eprintln!("no regular files under {dir}; using the built-in corpus");
                builtin_corpus()
            }
            Err(error) => {
                eprintln!("could not read {dir} ({error}); using the built-in corpus");
                builtin_corpus()
            }
        },
        None => {
            println!(
                "no directory argument; scanning the built-in {}-file corpus",
                builtin_corpus().len()
            );
            builtin_corpus()
        }
    };

    let (haystack, region_starts) = coalesce_regions(&files);
    println!(
        "corpus: {} bytes across {} region(s)",
        haystack.len(),
        region_starts.len()
    );

    // 2. Compile the matcher ONCE (Aho–Corasick DFA + prefilter masks).
    let matcher = GpuLiteralSet::compile(PATTERNS);
    let pattern_count = PATTERNS.len() as u32;
    let words_per_region = pattern_count.div_ceil(32).max(1) as usize;
    let max_matches = 4_096u32;

    // 3. Prefer the GPU resident fast path; fall back LOUDLY to the CPU reference
    //    backend's portable one-shot scan when no GPU is present.
    match WgpuBackend::shared() {
        Ok(backend) => {
            println!("\n== GPU resident fused fast path (RTX-class device) ==");
            run_resident_fused(
                backend.as_ref(),
                &matcher,
                &haystack,
                &region_starts,
                words_per_region,
                pattern_count,
                max_matches,
            );
        }
        Err(error) => {
            eprintln!(
                "\n!! no GPU backend available ({error}).\n!! The resident-table fast path is device-only; running the portable\n!! `scan_all` on the CPU reference backend instead (this path re-uploads\n!! tables and has no per-region presence (it is the fallback, not the fast path))."
            );
            run_portable_scan_all(&CpuRefBackend, &matcher, &haystack);
        }
    }
}

/// Coalesce a sequence of byte buffers ("files") into one contiguous haystack plus
/// the `region_starts` array (region `i` spans `region_starts[i]..region_starts[i+1]`,
/// the last to `haystack.len()`; the first start is always `0`). This is the
/// consumer's in-memory "assemble a corpus" step, done inline (not a public API 
/// the by-region scans just want the two arrays).
fn coalesce_regions(files: &[Vec<u8>]) -> (Vec<u8>, Vec<u32>) {
    let mut haystack = Vec::new();
    let mut region_starts = Vec::with_capacity(files.len());
    for file in files {
        region_starts.push(haystack.len() as u32);
        haystack.extend_from_slice(file);
    }
    if region_starts.is_empty() {
        // A single empty region keeps the by-region contract (first start == 0).
        region_starts.push(0);
    }
    (haystack, region_starts)
}

/// The GPU fast path: one resident fused session, one dispatch producing BOTH
/// outputs, timing left on.
fn run_resident_fused(
    backend: &dyn VyreBackend,
    matcher: &GpuLiteralSet,
    haystack: &[u8],
    region_starts: &[u32],
    words_per_region: usize,
    pattern_count: u32,
    max_matches: u32,
) {
    let session = match matcher.prepare_resident_fused_scan(
        backend,
        haystack.len() + 64,
        region_starts.len() as u32,
        max_matches,
    ) {
        Ok(session) => session,
        Err(error) => {
            eprintln!("prepare_resident_fused_scan failed: {error}");
            return;
        }
    };

    let mut presence = Vec::new();
    let mut matches = Vec::new();
    let mut scratch = Vec::new();
    let timed = match session.scan_into_timed(
        backend,
        haystack,
        region_starts,
        0,
        &mut presence,
        &mut matches,
        &mut scratch,
    ) {
        Ok(timed) => timed,
        Err(error) => {
            eprintln!("resident fused scan failed: {error}");
            let _ = session.free(backend);
            return;
        }
    };

    print_presence(&presence, words_per_region, pattern_count);
    print_matches(&matches);

    // Timed attribution: the kernel-vs-staging split, left on in production.
    let staging = timed
        .device_ns
        .map(|device_ns| timed.wall_ns.saturating_sub(device_ns));
    match (timed.device_ns, staging) {
        (Some(device_ns), Some(staging_ns)) => println!(
            "\ntiming: wall={} ns  device={} ns  staging={} ns",
            timed.wall_ns, device_ns, staging_ns
        ),
        _ => println!(
            "\ntiming: wall={} ns  device=<no device timer on this backend>",
            timed.wall_ns
        ),
    }

    if let Err(error) = session.free(backend) {
        eprintln!("free session failed: {error}");
    }
}

/// The portable fallback: `scan_all` returns every match with no cap tuning and no
/// host paging. It works on any backend (no resident buffers), but re-uploads the
/// tables per call and has no per-region presence output.
fn run_portable_scan_all(backend: &dyn VyreBackend, matcher: &GpuLiteralSet, haystack: &[u8]) {
    match matcher.scan_all(backend, haystack) {
        Ok(matches) => print_matches(&matches),
        Err(error) => eprintln!("scan_all failed: {error}"),
    }
}

fn print_presence(presence: &[u32], words_per_region: usize, pattern_count: u32) {
    if words_per_region == 0 {
        return;
    }
    println!("\nper-region presence (which patterns occur in each region):");
    for (region, words) in presence.chunks(words_per_region).enumerate() {
        let mut present = Vec::new();
        for pattern_id in 0..pattern_count {
            let word = (pattern_id / 32) as usize;
            let bit = pattern_id % 32;
            if words.get(word).is_some_and(|w| (w >> bit) & 1 == 1) {
                present.push(pattern_label(pattern_id));
            }
        }
        if present.is_empty() {
            println!("  region {region}: (none)");
        } else {
            println!("  region {region}: {}", present.join(", "));
        }
    }
}

fn print_matches(matches: &[Match]) {
    println!("\npositioned matches ({} total):", matches.len());
    for hit in matches {
        println!(
            "  {} @ [{}..{})",
            pattern_label(hit.pattern_id),
            hit.start,
            hit.end
        );
    }
}

fn pattern_label(pattern_id: u32) -> String {
    PATTERNS
        .get(pattern_id as usize)
        .map(|pattern| String::from_utf8_lossy(pattern).into_owned())
        .unwrap_or_else(|| format!("pattern#{pattern_id}"))
}

/// A built-in multi-"file" corpus so the example runs with no arguments and no
/// filesystem. Each entry is one region.
fn builtin_corpus() -> Vec<Vec<u8>> {
    vec![
        b"user config: no secrets here, just a token= placeholder".to_vec(),
        b"AKIAEXAMPLE0000 and a password=hunter2 in the same file".to_vec(),
        b"-----BEGIN RSA PRIVATE KEY----- ... a secret key blob".to_vec(),
        b"plain prose with nothing sensitive at all".to_vec(),
    ]
}

/// Read every regular file directly under `dir` (one level; each becomes a region).
fn read_regular_files(dir: &Path) -> std::io::Result<Vec<Vec<u8>>> {
    let mut files = Vec::new();
    for entry in std::fs::read_dir(dir)? {
        let entry = entry?;
        if entry.file_type()?.is_file() {
            files.push(std::fs::read(entry.path())?);
        }
    }
    files.sort();
    Ok(files)
}
