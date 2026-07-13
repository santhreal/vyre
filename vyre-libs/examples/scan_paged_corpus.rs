//! W8-3: the DISK-INGRESS paging path from `docs/scanning-a-corpus-the-right-way.md`,
//! made real: "a corpus larger than one window".
//!
//! `scan_corpus_fast_path` shows the single-shot resident fast path: it coalesces
//! the WHOLE corpus into one contiguous host haystack and one launch. That is
//! correct only while the corpus fits the u32 haystack ABI (<4 GiB) AND fits host
//! RAM. This example shows the other half: a corpus that does NOT fit one window.
//!
//! `scan_paths_paged_prefetched` takes file PATHS, plans windows at file
//! boundaries under a byte budget, and streams them past the GPU, a background
//! reader prefetches window k+1 off disk while the device scans window k, so host
//! RSS is bounded to ~2 windows no matter how large the corpus is. The driver
//! stitches every window's results back into ONE global numbering: u64 byte
//! positions (unbounded by the per-window u32 ABI) and the global region id (==
//! the original file index the match starts in). A match that straddles a window
//! boundary is found exactly once (overlap + start-dedup), so the paged result is
//! byte-identical to a single-shot scan (Law 10).
//!
//! Run on a GPU box (builds a temp corpus, forces multi-window paging):
//!   cargo run -p vyre-libs --example scan_paged_corpus
//! Or page a real directory tree (each regular file becomes one region):
//!   cargo run -p vyre-libs --example scan_paged_corpus -- /path/to/tree 65536
//!                                                          ^dir       ^window budget bytes
//!
//! With no GPU backend the resident paged path is unavailable (resident buffers
//! are device-only). The example says so LOUDLY and falls back to the portable
//! `scan_paged_fused_async` on the CPU reference backend, which reads every file
//! into memory (surrendering the bounded-RSS disk-ingress property) but still
//! produces the same global match set. A stated fallback, never a silent degrade
//! (Law 10).

use std::path::{Path, PathBuf};

use vyre_driver_reference::CpuRefBackend;
use vyre_driver_wgpu::WgpuBackend;
use vyre_libs::scan::{
    scan_paged_fused_async, scan_paths_paged_prefetched, GlobalMatch, GpuLiteralSet,
    PagedScanResult,
};

/// The demo pattern set, a handful of secret-shaped literals. In a real consumer
/// these come from a rule catalog (Tier-B data).
const PATTERNS: &[&[u8]] = &[
    b"AKIA",      // aws access key id prefix
    b"password",  // generic
    b"BEGIN RSA", // private key header fragment
    b"token",     // generic
    b"secret",    // generic
];

/// Default per-window byte budget when building the temp corpus. Deliberately tiny
/// so the built-in corpus spans several windows and the paging is exercised, not
/// bypassed. Real consumers size this to a fraction of device memory.
const DEFAULT_WINDOW_BUDGET_BYTES: usize = 96;

fn main() {
    let args: Vec<String> = std::env::args().collect();

    // Resolve (paths, window budget, temp-dir-guard). The guard keeps the built-in
    // corpus's temp directory alive until the scan completes.
    let (paths, window_budget, _temp_guard): (Vec<PathBuf>, usize, Option<TempCorpus>) =
        match args.get(1) {
            Some(dir) => {
                let budget = args
                    .get(2)
                    .and_then(|b| b.parse::<usize>().ok())
                    .unwrap_or(64 * 1024);
                match read_regular_file_paths(Path::new(dir)) {
                    Ok(paths) if !paths.is_empty() => {
                        println!(
                            "paging {} file(s) under {dir} (window budget {budget} bytes)",
                            paths.len()
                        );
                        (paths, budget, None)
                    }
                    Ok(_) => {
                        eprintln!("no regular files under {dir}; using the built-in temp corpus");
                        build_builtin_corpus()
                    }
                    Err(error) => {
                        eprintln!("could not read {dir} ({error}); using the built-in temp corpus");
                        build_builtin_corpus()
                    }
                }
            }
            None => {
                println!("no directory argument; paging the built-in temp corpus");
                build_builtin_corpus()
            }
        };

    let total_bytes: u64 = paths
        .iter()
        .map(|p| std::fs::metadata(p).map(|m| m.len()).unwrap_or(0))
        .sum();
    println!(
        "corpus: {} file(s), {total_bytes} bytes total, window budget {window_budget} bytes -> multi-window paging\n",
        paths.len()
    );

    // Compile the matcher ONCE (Aho–Corasick DFA + prefilter masks).
    let matcher = GpuLiteralSet::compile(PATTERNS);
    let words_per_region = (PATTERNS.len() as u32).div_ceil(32).max(1) as usize;
    let max_matches = 4_096u32;

    let path_refs: Vec<&Path> = paths.iter().map(PathBuf::as_path).collect();

    match WgpuBackend::shared() {
        Ok(backend) => {
            println!(
                "== GPU resident paged fast path (prefetched disk ingress, RTX-class device) =="
            );
            match scan_paths_paged_prefetched(
                &matcher,
                backend.as_ref(),
                &path_refs,
                window_budget,
                max_matches,
            ) {
                Ok(result) => report(&result, words_per_region),
                Err(error) => eprintln!("scan_paths_paged_prefetched failed: {error}"),
            }
        }
        Err(error) => {
            eprintln!(
                "!! no GPU backend available ({error}).\n!! The resident paged path is device-only; reading every file into memory\n!! and running the portable `scan_paged_fused_async` on the CPU reference\n!! backend instead (this surrenders the bounded-RSS disk-ingress property\n!! but yields the same global match set (the fallback, not the fast path))."
            );
            match read_all_into_memory(&paths) {
                Ok(files) => {
                    let borrowed: Vec<&[u8]> = files.iter().map(Vec::as_slice).collect();
                    match scan_paged_fused_async(
                        &matcher,
                        &CpuRefBackend,
                        &borrowed,
                        window_budget,
                        max_matches,
                    ) {
                        Ok(result) => report(&result, words_per_region),
                        Err(error) => eprintln!("scan_paged_fused_async failed: {error}"),
                    }
                }
                Err(error) => eprintln!("could not read corpus into memory: {error}"),
            }
        }
    }
}

/// Print the unified global result: per-region presence + every positioned match
/// in global (file-index, u64-byte) coordinates.
fn report(result: &PagedScanResult, words_per_region: usize) {
    println!(
        "\n{} region(s), {} presence word(s)/region, {} global match(es):",
        result.region_count,
        result.presence_words,
        result.matches.len()
    );
    print_presence(result, words_per_region);
    print_matches(&result.matches);
}

fn print_presence(result: &PagedScanResult, words_per_region: usize) {
    if words_per_region == 0 {
        return;
    }
    println!("\nper-region presence (which patterns occur in each file):");
    for (region, words) in result.presence.chunks(words_per_region).enumerate() {
        let mut present = Vec::new();
        for pattern_id in 0..PATTERNS.len() as u32 {
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

fn print_matches(matches: &[GlobalMatch]) {
    println!("\npositioned global matches:");
    for hit in matches {
        println!(
            "  {} in region {} @ [{}..{})",
            pattern_label(hit.pattern_id),
            hit.region_id,
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

/// Read every regular file path directly under `dir` (one level; each becomes a
/// region), sorted for determinism.
fn read_regular_file_paths(dir: &Path) -> std::io::Result<Vec<PathBuf>> {
    let mut paths = Vec::new();
    for entry in std::fs::read_dir(dir)? {
        let entry = entry?;
        if entry.file_type()?.is_file() {
            paths.push(entry.path());
        }
    }
    paths.sort();
    Ok(paths)
}

fn read_all_into_memory(paths: &[PathBuf]) -> std::io::Result<Vec<Vec<u8>>> {
    paths.iter().map(std::fs::read).collect()
}

/// A temp directory holding the built-in corpus; deleted when dropped. Kept alive
/// by the caller until the scan finishes.
struct TempCorpus {
    dir: PathBuf,
}

impl Drop for TempCorpus {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.dir);
    }
}

/// Materialize a multi-file corpus on disk so the example pages real files off the
/// filesystem with no arguments. Each entry is one file/region; several patterns
/// deliberately straddle window boundaries under the tiny default budget so the
/// overlap/dedup path is exercised. Returns the sorted paths, the default window
/// budget, and the temp-dir guard.
fn build_builtin_corpus() -> (Vec<PathBuf>, usize, Option<TempCorpus>) {
    let base = std::env::temp_dir().join(format!("vyre_paged_corpus_{}", std::process::id()));
    // Best-effort clean slate; ignore if absent.
    let _ = std::fs::remove_dir_all(&base);
    if let Err(error) = std::fs::create_dir_all(&base) {
        eprintln!(
            "could not create temp corpus dir {}: {error}",
            base.display()
        );
        return (Vec::new(), DEFAULT_WINDOW_BUDGET_BYTES, None);
    }

    let files: &[(&str, &[u8])] = &[
        (
            "00_config.txt",
            b"user config: just a token= placeholder and nothing else here",
        ),
        (
            "01_creds.txt",
            b"AKIAEXAMPLE0000 sits next to a password=hunter2 in this file",
        ),
        (
            "02_key.pem",
            b"-----BEGIN RSA PRIVATE KEY----- and later the word secret appears",
        ),
        (
            "03_prose.txt",
            b"plain prose with nothing sensitive at all, filler filler filler",
        ),
        (
            "04_mixed.log",
            b"line one\nline two has a secret token here\nline three plain",
        ),
    ];

    let mut paths = Vec::with_capacity(files.len());
    for (name, body) in files {
        let path = base.join(name);
        if let Err(error) = std::fs::write(&path, body) {
            eprintln!("could not write temp file {}: {error}", path.display());
            return (
                Vec::new(),
                DEFAULT_WINDOW_BUDGET_BYTES,
                Some(TempCorpus { dir: base }),
            );
        }
        paths.push(path);
    }
    paths.sort();
    (
        paths,
        DEFAULT_WINDOW_BUDGET_BYTES,
        Some(TempCorpus { dir: base }),
    )
}
