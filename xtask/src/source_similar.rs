//! Repo-wide Rust source duplication scanner.
//!
//! `whats-similar` catches duplicate registered IR programs. This command
//! catches the other class: forked Rust source that has not reached inventory
//! registration yet. It uses normalized token shingles so renamed variables do
//! not hide duplicated implementation skeletons.

use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::{Path, PathBuf};
use std::process;

const DEFAULT_TOP_N: usize = 20;
const DEFAULT_MIN_SCORE: f64 = 0.86;
const DEFAULT_MAX_FILE_BYTES: u64 = 512 * 1024;
const SHINGLE_WIDTH: usize = 8;
const MAX_CANDIDATE_SHINGLE_FANOUT: usize = 64;
const MIN_SHARED_RARE_SHINGLES: u16 = 16;

#[derive(Debug, Clone)]
struct Config {
    roots: Vec<PathBuf>,
    top_n: usize,
    min_score: f64,
    max_file_bytes: u64,
    fail_on_findings: bool,
}

#[derive(Debug, Clone)]
struct SourceFingerprint {
    path: PathBuf,
    bytes: u64,
    tokens: usize,
    shingles: HashMap<u64, u32>,
    magnitude: f64,
}

#[derive(Debug, Clone)]
struct SimilarPair {
    score: f64,
    left: usize,
    right: usize,
}

pub(crate) fn run(args: &[String]) {
    let config = match parse_args(args) {
        Ok(config) => config,
        Err(error) => {
            eprintln!("Fix: {error}");
            print_usage();
            process::exit(1);
        }
    };

    let files = match collect_rust_files(&config.roots, config.max_file_bytes) {
        Ok(files) => files,
        Err(error) => {
            eprintln!("Fix: source-similar scan failed: {error}");
            process::exit(1);
        }
    };
    let fingerprints = fingerprint_files(&files);
    let pairs = score_pairs(&fingerprints, config.top_n, config.min_score);

    println!(
        "source-similar: scanned {} Rust files under {} root(s) (min={:.2}, top={}, shingle_width={})",
        fingerprints.len(),
        config.roots.len(),
        config.min_score,
        config.top_n,
        SHINGLE_WIDTH
    );
    if pairs.is_empty() {
        println!("  no Rust source file pairs crossed the duplication floor.");
        return;
    }
    for (index, pair) in pairs.iter().enumerate() {
        let left = &fingerprints[pair.left];
        let right = &fingerprints[pair.right];
        println!(
            "  {:>2}. {:>5.1}%  {}",
            index + 1,
            pair.score * 100.0,
            pair_verdict(pair.score)
        );
        println!(
            "      A: {} tokens={} bytes={}",
            display_path(&left.path),
            left.tokens,
            left.bytes
        );
        println!(
            "      B: {} tokens={} bytes={}",
            display_path(&right.path),
            right.tokens,
            right.bytes
        );
    }
    if config.fail_on_findings {
        eprintln!(
            "Fix: source-similar found {} duplicate/similar Rust source pair(s) at score >= {:.2}. Extract a shared module or lower --min only for exploratory scans.",
            pairs.len(),
            config.min_score
        );
        process::exit(1);
    }
}

fn parse_args(args: &[String]) -> Result<Config, String> {
    let mut roots = Vec::new();
    let mut top_n = DEFAULT_TOP_N;
    let mut min_score = DEFAULT_MIN_SCORE;
    let mut max_file_bytes = DEFAULT_MAX_FILE_BYTES;
    let mut fail_on_findings = false;
    let mut index = 2usize;
    while index < args.len() {
        match args[index].as_str() {
            "--root" => {
                index += 1;
                let Some(root) = args.get(index) else {
                    return Err("--root requires a path".to_string());
                };
                roots.push(PathBuf::from(root));
            }
            "--top" => {
                index += 1;
                let Some(raw) = args.get(index) else {
                    return Err("--top requires a positive integer".to_string());
                };
                top_n = raw
                    .parse::<usize>()
                    .map_err(|_| format!("--top must be an integer, got `{raw}`"))?;
                if top_n == 0 {
                    return Err("--top must be greater than zero".to_string());
                }
            }
            "--min" => {
                index += 1;
                let Some(raw) = args.get(index) else {
                    return Err("--min requires a score in 0.0..=1.0".to_string());
                };
                min_score = raw
                    .parse::<f64>()
                    .map_err(|_| format!("--min must be a float, got `{raw}`"))?;
                if !(0.0..=1.0).contains(&min_score) {
                    return Err("--min must be in 0.0..=1.0".to_string());
                }
            }
            "--max-file-bytes" => {
                index += 1;
                let Some(raw) = args.get(index) else {
                    return Err("--max-file-bytes requires a positive integer".to_string());
                };
                max_file_bytes = raw
                    .parse::<u64>()
                    .map_err(|_| format!("--max-file-bytes must be an integer, got `{raw}`"))?;
                if max_file_bytes == 0 {
                    return Err("--max-file-bytes must be greater than zero".to_string());
                }
            }
            "--fail-on-findings" | "--check" => {
                fail_on_findings = true;
            }
            "--help" | "-h" => {
                print_usage();
                process::exit(0);
            }
            other => return Err(format!("unknown source-similar option `{other}`")),
        }
        index += 1;
    }
    if roots.is_empty() {
        roots = default_roots();
    }
    Ok(Config {
        roots,
        top_n,
        min_score,
        max_file_bytes,
        fail_on_findings,
    })
}

fn print_usage() {
    eprintln!(
        "USAGE:\n  cargo_full run --bin xtask -- source-similar [--root PATH] [--top N] [--min SCORE] [--max-file-bytes BYTES] [--fail-on-findings]\n\n\
         Defaults scan Rust files under the Vyre workspace source roots and report high-confidence renamed/forked source skeletons."
    );
}

fn default_roots() -> Vec<PathBuf> {
    [
        "vyre-core",
        "vyre-foundation",
        "vyre-driver",
        "vyre-driver-cuda",
        "vyre-driver-wgpu",
        "vyre-driver-spirv",
        "vyre-reference",
        "vyre-spec",
        "vyre-primitives",
        "vyre-self-substrate",
        "vyre-runtime",
        "vyre-libs",
        "vyre-intrinsics",
        "vyre-aot",
        "vyre-frontend-c",
        "vyre-bench",
        "vyre-lower",
        "vyre-emit-ptx",
        "vyre-emit-spirv",
        "vyre-emit-naga",
        "xtask",
        "conform",
    ]
    .into_iter()
    .map(PathBuf::from)
    .filter(|path| path.exists())
    .collect()
}

fn collect_rust_files(roots: &[PathBuf], max_file_bytes: u64) -> Result<Vec<PathBuf>, String> {
    let mut files = Vec::new();
    let mut seen = HashSet::new();
    for root in roots {
        collect_rust_files_recursive(root, max_file_bytes, &mut files, &mut seen)?;
    }
    files.sort();
    Ok(files)
}

fn collect_rust_files_recursive(
    path: &Path,
    max_file_bytes: u64,
    files: &mut Vec<PathBuf>,
    seen: &mut HashSet<PathBuf>,
) -> Result<(), String> {
    if should_skip_path(path) {
        return Ok(());
    }
    let meta = fs::metadata(path)
        .map_err(|error| format!("could not stat `{}`: {error}", path.display()))?;
    if meta.is_dir() {
        for entry in
            fs::read_dir(path).map_err(|error| format!("could not read `{}`: {error}", path.display()))?
        {
            let entry =
                entry.map_err(|error| format!("could not read entry in `{}`: {error}", path.display()))?;
            collect_rust_files_recursive(&entry.path(), max_file_bytes, files, seen)?;
        }
        return Ok(());
    }
    if meta.len() > max_file_bytes || path.extension().and_then(|ext| ext.to_str()) != Some("rs") {
        return Ok(());
    }
    let canonical = fs::canonicalize(path)
        .map_err(|error| format!("could not canonicalize `{}`: {error}", path.display()))?;
    if seen.insert(canonical) {
        files.push(path.to_path_buf());
    }
    Ok(())
}

fn should_skip_path(path: &Path) -> bool {
    path.components().any(|component| {
        let name = component.as_os_str().to_string_lossy();
        matches!(
            name.as_ref(),
            ".git"
                | "target"
                | ".pytest_cache"
                | "__pycache__"
                | ".cursor"
                | ".internals"
                | "jules_tickets"
                | "__law7_split"
        )
    })
}

fn fingerprint_files(files: &[PathBuf]) -> Vec<SourceFingerprint> {
    let mut out = Vec::new();
    for path in files {
        let Ok(source) = fs::read_to_string(path) else {
            continue;
        };
        let tokens = normalize_tokens(&source);
        if tokens.len() < SHINGLE_WIDTH * 2 {
            continue;
        }
        let shingles = shingle_counts(&tokens, SHINGLE_WIDTH);
        if shingles.is_empty() {
            continue;
        }
        let magnitude = magnitude(&shingles);
        out.push(SourceFingerprint {
            path: path.clone(),
            bytes: source.len() as u64,
            tokens: tokens.len(),
            shingles,
            magnitude,
        });
    }
    out
}

fn score_pairs(
    fingerprints: &[SourceFingerprint],
    top_n: usize,
    min_score: f64,
) -> Vec<SimilarPair> {
    let candidates = candidate_pairs(fingerprints);
    let mut pairs = Vec::new();
    for (left, right) in candidates {
        if same_generated_family(&fingerprints[left].path, &fingerprints[right].path) {
            continue;
        }
        let score = cosine(&fingerprints[left], &fingerprints[right]);
        if score >= min_score {
            pairs.push(SimilarPair { score, left, right });
        }
    }
    pairs.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    pairs.truncate(top_n);
    pairs
}

fn candidate_pairs(fingerprints: &[SourceFingerprint]) -> HashSet<(usize, usize)> {
    let mut by_shingle: HashMap<u64, Vec<usize>> = HashMap::new();
    for (file_index, fingerprint) in fingerprints.iter().enumerate() {
        for shingle in fingerprint.shingles.keys() {
            by_shingle.entry(*shingle).or_default().push(file_index);
        }
    }
    let mut shared_rare_counts: HashMap<(usize, usize), u16> = HashMap::new();
    for files in by_shingle.values() {
        if files.len() < 2 || files.len() > MAX_CANDIDATE_SHINGLE_FANOUT {
            continue;
        }
        for left_pos in 0..files.len() {
            for &right in &files[left_pos + 1..] {
                let left = files[left_pos];
                let key = if left < right {
                    (left, right)
                } else {
                    (right, left)
                };
                let count = shared_rare_counts.entry(key).or_insert(0);
                *count = count.saturating_add(1);
            }
        }
    }
    shared_rare_counts
        .into_iter()
        .filter_map(|(pair, count)| (count >= MIN_SHARED_RARE_SHINGLES).then_some(pair))
        .collect()
}

fn same_generated_family(left: &Path, right: &Path) -> bool {
    let left = display_path(left);
    let right = display_path(right);
    (left.contains("/tests/__split/") && right.contains("/tests/__split/"))
        || (left.contains("/parse/vast/classify/nodes_")
            && right.contains("/parse/vast/classify/nodes_"))
}

fn normalize_tokens(source: &str) -> Vec<String> {
    let mut tokens = Vec::new();
    let bytes = source.as_bytes();
    let mut index = 0usize;
    while index < bytes.len() {
        let b = bytes[index];
        if b.is_ascii_whitespace() {
            index += 1;
            continue;
        }
        if b == b'/' && bytes.get(index + 1) == Some(&b'/') {
            index += 2;
            while index < bytes.len() && bytes[index] != b'\n' {
                index += 1;
            }
            continue;
        }
        if b == b'/' && bytes.get(index + 1) == Some(&b'*') {
            index += 2;
            while index + 1 < bytes.len() && !(bytes[index] == b'*' && bytes[index + 1] == b'/') {
                index += 1;
            }
            index = (index + 2).min(bytes.len());
            continue;
        }
        if b == b'"' {
            tokens.push("str".to_string());
            index += 1;
            while index < bytes.len() {
                if bytes[index] == b'\\' {
                    index = (index + 2).min(bytes.len());
                    continue;
                }
                if bytes[index] == b'"' {
                    index += 1;
                    break;
                }
                index += 1;
            }
            continue;
        }
        if b == b'\'' {
            tokens.push("chr".to_string());
            index += 1;
            while index < bytes.len() {
                if bytes[index] == b'\\' {
                    index = (index + 2).min(bytes.len());
                    continue;
                }
                if bytes[index] == b'\'' {
                    index += 1;
                    break;
                }
                index += 1;
            }
            continue;
        }
        if b.is_ascii_alphabetic() || b == b'_' {
            let start = index;
            index += 1;
            while index < bytes.len() && (bytes[index].is_ascii_alphanumeric() || bytes[index] == b'_') {
                index += 1;
            }
            let ident = &source[start..index];
            if is_rust_keyword(ident) {
                tokens.push(ident.to_string());
            } else {
                tokens.push("id".to_string());
            }
            continue;
        }
        if b.is_ascii_digit() {
            tokens.push("num".to_string());
            index += 1;
            while index < bytes.len()
                && (bytes[index].is_ascii_alphanumeric()
                    || matches!(bytes[index], b'_' | b'.'))
            {
                index += 1;
            }
            continue;
        }
        tokens.push((b as char).to_string());
        index += 1;
    }
    tokens
}

fn is_rust_keyword(token: &str) -> bool {
    matches!(
        token,
        "as"
            | "async"
            | "await"
            | "break"
            | "const"
            | "continue"
            | "crate"
            | "else"
            | "enum"
            | "extern"
            | "false"
            | "fn"
            | "for"
            | "if"
            | "impl"
            | "in"
            | "let"
            | "loop"
            | "match"
            | "mod"
            | "move"
            | "mut"
            | "pub"
            | "ref"
            | "return"
            | "self"
            | "Self"
            | "static"
            | "struct"
            | "super"
            | "trait"
            | "true"
            | "type"
            | "unsafe"
            | "use"
            | "where"
            | "while"
    )
}

fn shingle_counts(tokens: &[String], width: usize) -> HashMap<u64, u32> {
    let mut counts = HashMap::new();
    if tokens.len() < width {
        return counts;
    }
    for window in tokens.windows(width) {
        *counts.entry(hash_window(window)).or_insert(0) += 1;
    }
    counts
}

fn hash_window(window: &[String]) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    for token in window {
        for &byte in token.as_bytes() {
            hash ^= u64::from(byte);
            hash = hash.wrapping_mul(0x100_0000_01b3);
        }
        hash ^= 0xff;
        hash = hash.wrapping_mul(0x100_0000_01b3);
    }
    hash
}

fn magnitude(counts: &HashMap<u64, u32>) -> f64 {
    (counts
        .values()
        .map(|count| {
            let c = f64::from(*count);
            c * c
        })
        .sum::<f64>())
    .sqrt()
}

fn cosine(left: &SourceFingerprint, right: &SourceFingerprint) -> f64 {
    if left.magnitude == 0.0 || right.magnitude == 0.0 {
        return 0.0;
    }
    let (small, large) = if left.shingles.len() <= right.shingles.len() {
        (&left.shingles, &right.shingles)
    } else {
        (&right.shingles, &left.shingles)
    };
    let dot = small
        .iter()
        .filter_map(|(key, left_count)| large.get(key).map(|right_count| (*left_count, *right_count)))
        .map(|(left_count, right_count)| f64::from(left_count) * f64::from(right_count))
        .sum::<f64>();
    dot / (left.magnitude * right.magnitude)
}

fn pair_verdict(score: f64) -> &'static str {
    if score >= 0.97 {
        "DUPLICATE"
    } else if score >= 0.90 {
        "VERY SIMILAR"
    } else {
        "SIMILAR"
    }
}

fn display_path(path: &Path) -> String {
    path.to_string_lossy().replace('\\', "/")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalization_catches_renamed_function_skeletons() {
        let left = normalize_tokens(
            "pub fn alpha(input: u32) -> u32 { let value = input + 1; value * 2 }",
        );
        let right = normalize_tokens(
            "pub fn beta(other: u32) -> u32 { let tmp = other + 9; tmp * 7 }",
        );
        let left_counts = shingle_counts(&left, 4);
        let right_counts = shingle_counts(&right, 4);
        let left_fp = SourceFingerprint {
            path: PathBuf::from("left.rs"),
            bytes: 1,
            tokens: left.len(),
            magnitude: magnitude(&left_counts),
            shingles: left_counts,
        };
        let right_fp = SourceFingerprint {
            path: PathBuf::from("right.rs"),
            bytes: 1,
            tokens: right.len(),
            magnitude: magnitude(&right_counts),
            shingles: right_counts,
        };
        assert!(
            cosine(&left_fp, &right_fp) > 0.70,
            "renamed/literal-changed function skeletons should stay similar"
        );
    }

    #[test]
    fn comments_and_strings_do_not_dominate_similarity() {
        let tokens = normalize_tokens(
            "//! doc words should vanish\nfn x() { let s = \"different payload\"; /* block */ 7 }",
        );
        assert!(!tokens.iter().any(|token| token == "doc"));
        assert!(tokens.iter().any(|token| token == "str"));
        assert!(tokens.iter().any(|token| token == "num"));
    }

    #[test]
    fn parse_args_defaults_to_existing_roots() {
        let args = vec!["xtask".to_string(), "source-similar".to_string()];
        let config = parse_args(&args).expect("default args");
        assert!(config.top_n > 0);
        assert!((0.0..=1.0).contains(&config.min_score));
    }

    #[test]
    fn parse_args_rejects_zero_top() {
        let args = vec![
            "xtask".to_string(),
            "source-similar".to_string(),
            "--top".to_string(),
            "0".to_string(),
        ];
        let error = parse_args(&args).unwrap_err();
        assert!(error.contains("--top"));
    }

    #[test]
    fn parse_args_accepts_check_mode_for_ci_duplicate_gates() {
        let args = vec![
            "xtask".to_string(),
            "source-similar".to_string(),
            "--check".to_string(),
            "--min".to_string(),
            "0.95".to_string(),
        ];
        let config = parse_args(&args).expect("check args");
        assert!(config.fail_on_findings);
        assert_eq!(config.min_score, 0.95);
    }

    #[test]
    fn skips_generated_split_scratch_and_internal_planning_trees() {
        assert!(should_skip_path(Path::new(
            "vyre-macros/src/__law7_split/lib_part1.rs"
        )));
        assert!(should_skip_path(Path::new(
            ".internals/audits/notes/generated.rs"
        )));
        assert!(should_skip_path(Path::new("jules_tickets/ticket.rs")));
        assert!(!should_skip_path(Path::new("vyre-primitives/src/graph/toposort.rs")));
    }

    #[test]
    fn generated_family_filter_suppresses_split_test_pairs_only() {
        assert!(same_generated_family(
            Path::new("vyre-driver-cuda/tests/__split/a.rs"),
            Path::new("vyre-driver-cuda/tests/__split/b.rs")
        ));
        assert!(!same_generated_family(
            Path::new("vyre-driver-cuda/tests/a.rs"),
            Path::new("vyre-driver-cuda/tests/__split/b.rs")
        ));
    }

    #[test]
    fn candidate_pairs_use_shared_rare_shingles_without_full_quadratic_scan() {
        let source_a = normalize_tokens(
            "fn alpha() { let special = 1; special + 2; let again = special + 3; again * 4; let tail = again + special; tail }",
        );
        let source_b = normalize_tokens(
            "fn beta() { let renamed = 9; renamed + 7; let more = renamed + 8; more * 6; let end = more + renamed; end }",
        );
        let source_c = normalize_tokens("struct CompletelyDifferent { field: u32 }");
        let fingerprints = [source_a, source_b, source_c]
            .iter()
            .enumerate()
            .map(|(idx, tokens)| {
                let shingles = shingle_counts(tokens, 4);
                SourceFingerprint {
                    path: PathBuf::from(format!("{idx}.rs")),
                    bytes: 1,
                    tokens: tokens.len(),
                    magnitude: magnitude(&shingles),
                    shingles,
                }
            })
            .collect::<Vec<_>>();
        let candidates = candidate_pairs(&fingerprints);
        assert!(
            candidates.contains(&(0, 1)),
            "renamed duplicate skeletons must become scoring candidates"
        );
    }
}
