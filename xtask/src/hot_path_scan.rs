//! `cargo_full run --bin xtask -- hot-path-scan`  -  ROADMAP S11 enforcement.
//!
//! Reads `docs/optimization/HOT_PATHS.toml` and scans every listed file
//! for allocation, clone, lock, and string-construction patterns that
//! are usually evidence of hot-path waste:
//!
//! - `.clone()`  -  almost always hidden allocation; scratch reuse or
//!   `Cow` / `Arc` is cheaper.
//! - `.to_owned()` / `.to_string()`  -  allocates on every call.
//! - `Vec::new()` / `Vec::with_capacity(N)` (in non-init code)  -
//!   per-call vector; consider scratch reuse.
//! - `HashMap::new()` / `BTreeMap::new()`  -  per-call map.
//! - `String::new()` / `String::from(...)`  -  per-call string.
//! - `Mutex::new(...)` / `RwLock::new(...)`  -  per-call lock primitive
//!   in code that runs many times per dispatch.
//!
//! Each finding prints `file:line | pattern | line content`. Exit 0
//! when the scan is informational (passed `--report` or default), exit
//! 1 when `--strict` is set and any finding fires.
//!
//! The scanner is line-oriented + regex-free to keep it deterministic
//! across rust-fmt rewrites; no AST parsing. It does NOT short-circuit
//! on test modules  -  hot-path files often have inline `#[cfg(test)]`
//! blocks that legitimately allocate; the audit ignores `#[cfg(test)]`
//! lines but does NOT skip the rest of the file.

use std::io::{self, Read};
use std::path::{Path, PathBuf};
use std::process::{self};

use serde::Deserialize;

const MAX_HOT_PATH_SCAN_FILE_BYTES: u64 = 2_097_152;

#[derive(Debug, Deserialize)]
struct HotPathsConfig {
    #[serde(default)]
    schema: u32,
    #[serde(default)]
    hot_path: Vec<HotPathEntry>,
}

#[derive(Debug, Deserialize)]
struct HotPathEntry {
    file: String,
    #[serde(default)]
    reason: String,
}

#[derive(Debug)]
struct Finding {
    file: String,
    line: u32,
    pattern: &'static str,
    content: String,
}

const PATTERNS: &[(&str, &str)] = &[
    ("clone", ".clone()"),
    ("to_owned", ".to_owned()"),
    ("to_string", ".to_string()"),
    ("Vec::new", "Vec::new()"),
    ("Vec::with_capacity", "Vec::with_capacity"),
    ("HashMap::new", "HashMap::new()"),
    ("BTreeMap::new", "BTreeMap::new()"),
    ("FxHashMap::new", "FxHashMap::new()"),
    ("String::new", "String::new()"),
    ("String::from", "String::from("),
    ("Mutex::new", "Mutex::new("),
    ("RwLock::new", "RwLock::new("),
    ("format!", "format!("),
];

pub(crate) fn run(args: &[String]) {
    let strict = args.iter().any(|a| a == "--strict");
    let root = match workspace_root() {
        Some(r) => r,
        None => {
            eprintln!("Fix: hot-path-scan must run from the vyre workspace.");
            process::exit(1);
        }
    };
    let config_path = root
        .join("docs")
        .join("optimization")
        .join("HOT_PATHS.toml");
    let entries = match load_config(&config_path) {
        Ok(e) => e,
        Err(err) => {
            eprintln!("Fix: failed to load {}: {err}", config_path.display());
            process::exit(1);
        }
    };
    let mut findings: Vec<Finding> = Vec::new();
    let mut scanned = 0usize;
    let mut missing: Vec<String> = Vec::new();
    for entry in &entries {
        let path = root.join(&entry.file);
        if !path.exists() {
            missing.push(entry.file.clone());
            continue;
        }
        scanned += 1;
        match read_text_bounded(&path) {
            Ok(text) => collect_findings(&entry.file, &text, &mut findings),
            Err(err) => eprintln!("warn: could not read {}: {err}", path.display()),
        }
    }
    findings.sort_by(|a, b| {
        (a.file.as_str(), a.line, a.pattern).cmp(&(b.file.as_str(), b.line, b.pattern))
    });
    let mut by_file: std::collections::BTreeMap<&str, usize> = std::collections::BTreeMap::new();
    for f in &findings {
        *by_file.entry(f.file.as_str()).or_insert(0) += 1;
    }

    println!("=== vyre hot-path scan ===");
    println!(
        "Listed: {} | scanned: {} | missing: {} | findings: {}",
        entries.len(),
        scanned,
        missing.len(),
        findings.len()
    );
    if !missing.is_empty() {
        println!();
        println!("Missing files (listed in HOT_PATHS.toml but not on disk):");
        for path in &missing {
            println!("  ✗ {path}");
        }
    }
    if !findings.is_empty() {
        println!();
        println!("Per-file finding counts:");
        // Attach the operator-supplied `reason` (from HOT_PATHS.toml) so
        // the report explains WHY each file is on the hot-path watchlist.
        // Without this the `reason` field is read but never surfaced  -
        // dead documentation.
        let reason_by_file: std::collections::BTreeMap<&str, &str> = entries
            .iter()
            .map(|e| (e.file.as_str(), e.reason.as_str()))
            .collect();
        for (file, count) in &by_file {
            let reason = reason_by_file.get(file).copied().unwrap_or("");
            if reason.is_empty() {
                println!("  {file}: {count}");
            } else {
                println!("  {file}: {count}   -  {reason}");
            }
        }
        println!();
        println!("Findings:");
        for f in &findings {
            println!(
                "  {}:{} | {} | {}",
                f.file,
                f.line,
                f.pattern,
                f.content.trim()
            );
        }
    } else {
        println!();
        println!("✓ no hot-path patterns found");
    }
    if strict && (!findings.is_empty() || !missing.is_empty()) {
        println!();
        println!(
            "hot-path-scan: STRICT mode failed  -  {} finding(s), {} missing file(s).",
            findings.len(),
            missing.len()
        );
        process::exit(1);
    }
}

fn workspace_root() -> Option<PathBuf> {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .map(Path::to_path_buf)
}

fn load_config(path: &Path) -> Result<Vec<HotPathEntry>, String> {
    let text = read_text_bounded(path).map_err(|e| e.to_string())?;
    let cfg: HotPathsConfig = toml::from_str(&text).map_err(|e| e.to_string())?;
    if cfg.schema != 1 {
        return Err(format!(
            "expected schema = 1, got {}  -  update the loader before changing the schema",
            cfg.schema
        ));
    }
    Ok(cfg.hot_path)
}

fn read_text_bounded(path: &Path) -> io::Result<String> {
    let mut reader =
        std::fs::File::open(path)?.take(MAX_HOT_PATH_SCAN_FILE_BYTES.saturating_add(1));
    let mut text = String::new();
    reader.read_to_string(&mut text)?;
    if text.len() as u64 > MAX_HOT_PATH_SCAN_FILE_BYTES {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "{} exceeds {MAX_HOT_PATH_SCAN_FILE_BYTES} byte hot-path scan read cap",
                path.display()
            ),
        ));
    }
    Ok(text)
}

fn collect_findings(file: &str, text: &str, out: &mut Vec<Finding>) {
    for (line_no, line) in text.lines().enumerate() {
        let trimmed = line.trim_start();
        // Skip comments and cfg(test) attributes  -  those are intentional
        // dev-only or annotation lines, not runtime cost.
        if trimmed.starts_with("//") || trimmed.starts_with("#[cfg(test)]") {
            continue;
        }
        for (name, pat) in PATTERNS {
            if line.contains(pat) {
                out.push(Finding {
                    file: file.to_string(),
                    line: (line_no + 1) as u32,
                    pattern: name,
                    content: line.to_string(),
                });
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn collect_findings_picks_up_clone() {
        let mut out = Vec::new();
        collect_findings("x.rs", "let y = x.clone();\n", &mut out);
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].pattern, "clone");
        assert_eq!(out[0].line, 1);
    }

    #[test]
    fn collect_findings_skips_comments() {
        let mut out = Vec::new();
        collect_findings("x.rs", "// uses x.clone() in docs\n", &mut out);
        assert!(out.is_empty());
    }

    #[test]
    fn collect_findings_picks_up_multiple_patterns() {
        let mut out = Vec::new();
        collect_findings(
            "x.rs",
            "let v: Vec<u32> = Vec::new();\nlet s = String::from(\"a\");\nlet l = Mutex::new(0);\n",
            &mut out,
        );
        let pats: Vec<&str> = out.iter().map(|f| f.pattern).collect();
        assert!(pats.contains(&"Vec::new"));
        assert!(pats.contains(&"String::from"));
        assert!(pats.contains(&"Mutex::new"));
    }

    #[test]
    fn collect_findings_picks_up_format_macro() {
        let mut out = Vec::new();
        collect_findings("x.rs", "let s = format!(\"{}\", 5);\n", &mut out);
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].pattern, "format!");
    }

    #[test]
    fn load_config_rejects_wrong_schema() {
        let dir = tempfile::TempDir::new().unwrap();
        let path = dir.path().join("hp.toml");
        std::fs::write(&path, "schema = 99\nhot_path = []\n").unwrap();
        let err = load_config(&path).unwrap_err();
        assert!(err.contains("schema = 1"));
    }

    #[test]
    fn load_config_parses_entries() {
        let dir = tempfile::TempDir::new().unwrap();
        let path = dir.path().join("hp.toml");
        std::fs::write(
            &path,
            "schema = 1\n[[hot_path]]\nfile = \"a.rs\"\nreason = \"x\"\n",
        )
        .unwrap();
        let entries = load_config(&path).unwrap();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].file, "a.rs");
    }
}
