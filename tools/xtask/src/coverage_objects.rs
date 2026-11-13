//! Instrumented executables that wrote no profile in a coverage run.
//!
//! `cargo llvm-cov report` reads every executable under the instrumented
//! profile directory. On a persistent runner that directory also holds builds
//! of other feature sets, of other checkouts and of removed targets. Their
//! regions have no profile in the run, so each one is reported as missed lines.
//!
//! An executable took part in the run exactly when a raw profile of the run
//! records its GNU build ID, so the selection is every executable whose build
//! ID no raw profile records. Each target of a `--tests` build is a test
//! executable the run starts, so the source of a target stays in the report
//! through that executable even when a binary built beside it never ran.
//!
//! Shared libraries are not selected. A proc-macro library runs only while a
//! dependent compiles, so a warm run records none, and deleting one rebuilds
//! every dependent in the next run.

use std::collections::BTreeSet;
use std::fs::{self, File};
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::process::Command;

/// A GNU build ID, the bytes of an `NT_GNU_BUILD_ID` note.
pub type BuildId = Vec<u8>;

/// Executables to delete before the report, and the count of those kept.
#[derive(Debug, Eq, PartialEq)]
pub struct Selection {
    /// Executables whose build ID no raw profile of the run records.
    pub stale: Vec<PathBuf>,
    /// Executables whose build ID a raw profile records.
    pub kept: usize,
}

/// Directories under the profile directory that hold no measured executable:
/// incremental state, fingerprints, and build scripts, which the report skips.
const SKIPPED_DIRECTORIES: [&str; 3] = ["incremental", ".fingerprint", "build"];

/// Extensions of files that are not measured executables, or that the
/// selection never deletes. `so` is a shared library; see the module notes.
const SKIPPED_EXTENSIONS: [&str; 4] = ["d", "rlib", "rmeta", "so"];

/// The objects `cargo llvm-cov report` reads: executables under `directory`
/// whose file name is a workspace package or target name, optionally prefixed
/// with `lib` and suffixed with a `-<hash>`.
#[derive(Debug)]
pub struct CoverageScope {
    /// `llvm-cov-target` under the target directory, where the instrumented
    /// build and its raw profiles are written.
    pub directory: PathBuf,
    /// Workspace package and target names, with `-` written as `_`.
    pub names: BTreeSet<String>,
}

impl CoverageScope {
    /// The scope of the workspace at `root`, read from `cargo metadata`.
    ///
    /// # Errors
    ///
    /// Returns the corrective action when `cargo metadata` fails or reports no
    /// target directory.
    pub fn of_workspace(root: &Path) -> Result<Self, String> {
        let output = crate::cargo_runner::command(root)
            .args(["metadata", "--no-deps", "--format-version", "1"])
            .output()
            .map_err(|error| format!("Fix: cannot start `cargo metadata`: {error}"))?;
        if !output.status.success() {
            return Err(format!(
                "Fix: `cargo metadata` failed: {}",
                String::from_utf8_lossy(&output.stderr).trim()
            ));
        }
        let metadata: serde_json::Value = serde_json::from_slice(&output.stdout)
            .map_err(|error| format!("Fix: `cargo metadata` printed no JSON document: {error}"))?;
        let directory = metadata
            .get("target_directory")
            .and_then(serde_json::Value::as_str)
            .map(|target| Path::new(target).join("llvm-cov-target"))
            .ok_or_else(|| "Fix: `cargo metadata` reported no target directory; run the prune inside the workspace.".to_string())?;
        let mut names = BTreeSet::new();
        for package in metadata
            .get("packages")
            .and_then(serde_json::Value::as_array)
            .into_iter()
            .flatten()
        {
            let targets = package
                .get("targets")
                .and_then(serde_json::Value::as_array)
                .into_iter()
                .flatten();
            for name in std::iter::once(package)
                .chain(targets)
                .filter_map(|item| item.get("name").and_then(serde_json::Value::as_str))
            {
                names.insert(name.replace('-', "_"));
            }
        }
        Ok(Self { directory, names })
    }

    /// Whether the report reads an object named `file_name`.
    #[must_use]
    pub fn reads(&self, file_name: &str) -> bool {
        let stem = file_name
            .split('.')
            .next()
            .unwrap_or(file_name)
            .replace('-', "_");
        let unhashed = stem.rsplit_once('_').and_then(|(base, hash)| {
            (!hash.is_empty() && hash.bytes().all(|byte| byte.is_ascii_hexdigit())).then_some(base)
        });
        std::iter::once(stem.as_str()).chain(unhashed).any(|base| {
            self.names.contains(base)
                || base
                    .strip_prefix("lib")
                    .is_some_and(|rest| self.names.contains(rest))
        })
    }
}

/// `llvm-profdata` of the active toolchain, the one `cargo llvm-cov` reads raw
/// profiles with.
///
/// # Errors
///
/// Returns the corrective action when `rustc` cannot report its sysroot and
/// host, or when the toolchain has no `llvm-tools` component.
pub fn llvm_profdata() -> Result<PathBuf, String> {
    let rustc = std::env::var_os("RUSTC").unwrap_or_else(|| "rustc".into());
    let ask = |arguments: &[&str]| -> Result<String, String> {
        let output = Command::new(&rustc)
            .args(arguments)
            .output()
            .map_err(|error| format!("Fix: cannot start `rustc`: {error}"))?;
        if !output.status.success() {
            return Err(format!(
                "Fix: `rustc {}` failed: {}",
                arguments.join(" "),
                String::from_utf8_lossy(&output.stderr).trim()
            ));
        }
        Ok(String::from_utf8_lossy(&output.stdout).into_owned())
    };
    let sysroot = ask(&["--print", "sysroot"])?;
    let version = ask(&["-vV"])?;
    let host = version
        .lines()
        .find_map(|line| line.strip_prefix("host: "))
        .ok_or_else(|| "Fix: `rustc -vV` reported no host triple.".to_string())?;
    let tool = Path::new(sysroot.trim())
        .join("lib/rustlib")
        .join(host.trim())
        .join("bin")
        .join(format!("llvm-profdata{}", std::env::consts::EXE_SUFFIX));
    if tool.is_file() {
        Ok(tool)
    } else {
        Err(format!(
            "Fix: `{}` does not exist. Install it with `rustup component add llvm-tools-preview`.",
            tool.display()
        ))
    }
}

/// Select every executable the report reads under `scope.directory/debug`
/// whose build ID `profiled` does not contain.
///
/// # Errors
///
/// Returns the corrective action when `profiled` is empty, because a run that
/// recorded nothing would select every executable, or when a file cannot be
/// read as an ELF object.
pub fn select(scope: &CoverageScope, profiled: &BTreeSet<BuildId>) -> Result<Selection, String> {
    if profiled.is_empty() {
        return Err("Fix: no raw profile records a build ID, so nothing identifies the executables of this run. Run the coverage build before the prune.".to_string());
    }
    let mut stale = Vec::new();
    let mut kept = 0;
    for path in executables(&scope.directory.join("debug"))? {
        if !path
            .file_name()
            .and_then(|name| name.to_str())
            .is_some_and(|name| scope.reads(name))
        {
            continue;
        }
        match gnu_build_id(&path)? {
            Some(id) if profiled.contains(&id) => kept += 1,
            _ => stale.push(path),
        }
    }
    stale.sort();
    Ok(Selection { stale, kept })
}

/// Delete every executable of `selection`.
///
/// # Errors
///
/// Returns the path that could not be deleted.
pub fn delete(selection: &Selection) -> Result<(), String> {
    for path in &selection.stale {
        fs::remove_file(path).map_err(|error| {
            format!(
                "Fix: cannot delete stale instrumented object `{}`: {error}",
                path.display()
            )
        })?;
    }
    Ok(())
}

/// The build IDs recorded by every raw profile directly inside `coverage_dir`,
/// read with `llvm_profdata show --binary-ids`.
///
/// # Errors
///
/// Returns the corrective action when the directory holds no raw profile, when
/// `llvm-profdata` fails, or when a raw profile records no build ID: such a
/// profile belongs to an executable the selection cannot recognize, and
/// deleting that executable would drop measured lines from the report.
pub fn profiled_build_ids(
    coverage_dir: &Path,
    llvm_profdata: &Path,
) -> Result<BTreeSet<BuildId>, String> {
    let entries = fs::read_dir(coverage_dir).map_err(|error| {
        format!(
            "Fix: cannot read the coverage directory `{}`: {error}",
            coverage_dir.display()
        )
    })?;
    let mut profiles: Vec<PathBuf> = entries
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .filter(|path| {
            path.extension()
                .is_some_and(|extension| extension == "profraw")
        })
        .collect();
    profiles.sort();
    if profiles.is_empty() {
        return Err(format!(
            "Fix: `{}` holds no raw profile. Run `cargo llvm-cov --no-report` before the prune.",
            coverage_dir.display()
        ));
    }
    let mut ids = BTreeSet::new();
    for profile in &profiles {
        let output = Command::new(llvm_profdata)
            .args(["show", "--binary-ids"])
            .arg(profile)
            .output()
            .map_err(|error| {
                format!(
                    "Fix: cannot start `{}`: {error}. Install it with `rustup component add llvm-tools-preview`.",
                    llvm_profdata.display()
                )
            })?;
        if !output.status.success() {
            return Err(format!(
                "Fix: `{} show --binary-ids {}` failed: {}",
                llvm_profdata.display(),
                profile.display(),
                String::from_utf8_lossy(&output.stderr).trim()
            ));
        }
        let recorded = recorded_build_ids(&String::from_utf8_lossy(&output.stdout))?;
        if recorded.is_empty() {
            return Err(format!(
                "Fix: the raw profile `{}` records no build ID. Link instrumented executables with a GNU build-ID note.",
                profile.display()
            ));
        }
        ids.extend(recorded);
    }
    Ok(ids)
}

/// The build IDs listed by `llvm-profdata show --binary-ids`: the hexadecimal
/// lines after the `Binary IDs:` line.
///
/// # Errors
///
/// Returns the corrective action when a line in that block is not hexadecimal.
pub fn recorded_build_ids(show_output: &str) -> Result<Vec<BuildId>, String> {
    let mut lines = show_output
        .lines()
        .skip_while(|line| !line.starts_with("Binary IDs:"));
    if lines.next().is_none() {
        return Ok(Vec::new());
    }
    lines
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .map(|line| {
            decode_hex(line).ok_or_else(|| {
                format!("Fix: `llvm-profdata` listed `{line}` as a binary ID, which is not hexadecimal. Use the llvm-profdata of the toolchain that built the coverage run.")
            })
        })
        .collect()
}

/// The `NT_GNU_BUILD_ID` note of an ELF64 little-endian object, or `None` when
/// the object carries no such note.
///
/// Only the file header, the section header table and the note sections are
/// read, so an executable of hundreds of megabytes costs a few reads.
///
/// # Errors
///
/// Returns the corrective action when the file cannot be read or is not an
/// ELF64 little-endian object.
pub fn gnu_build_id(path: &Path) -> Result<Option<BuildId>, String> {
    let fail = |what: &str| {
        format!(
            "Fix: `{}` {what}. The coverage lane measures ELF64 little-endian executables only; delete the file or run the lane on Linux.",
            path.display()
        )
    };
    let mut file = File::open(path).map_err(|error| fail(&format!("cannot be opened: {error}")))?;
    let mut header = [0u8; 64];
    file.read_exact(&mut header)
        .map_err(|_| fail("is not an ELF64 little-endian object"))?;
    if header[..4] != *b"\x7fELF" || header[4] != 2 || header[5] != 1 {
        return Err(fail("is not an ELF64 little-endian object"));
    }
    let section_table = le_u64(&header[0x28..]);
    let entry_size = usize::from(le_u16(&header[0x3a..]));
    let entries = usize::from(le_u16(&header[0x3c..]));
    if entries == 0 {
        return Ok(None);
    }
    if entry_size < 64 {
        return Err(fail("declares section headers shorter than 64 bytes"));
    }
    let mut table = vec![0u8; entry_size * entries];
    read_at(&mut file, section_table, &mut table)
        .map_err(|error| fail(&format!("has an unreadable section header table: {error}")))?;
    for section in table.chunks_exact(entry_size) {
        const SHT_NOTE: u32 = 7;
        if le_u32(&section[4..]) != SHT_NOTE {
            continue;
        }
        let size = usize::try_from(le_u64(&section[0x20..]))
            .ok()
            .filter(|size| *size <= MAX_NOTE_SECTION)
            .ok_or_else(|| fail("declares a note section larger than 64 KiB"))?;
        let mut notes = vec![0u8; size];
        read_at(&mut file, le_u64(&section[0x18..]), &mut notes)
            .map_err(|error| fail(&format!("has an unreadable note section: {error}")))?;
        if let Some(id) = build_id_note(&notes) {
            return Ok(Some(id));
        }
    }
    Ok(None)
}

/// Upper bound on one note section. A build-ID section is 36 bytes.
const MAX_NOTE_SECTION: usize = 64 * 1024;

/// The descriptor of the first `GNU` note of type `NT_GNU_BUILD_ID` in a note
/// section, or `None` when the section has none or ends inside a note.
fn build_id_note(mut notes: &[u8]) -> Option<BuildId> {
    const NT_GNU_BUILD_ID: u32 = 3;
    while notes.len() >= 12 {
        let name_size = usize::try_from(le_u32(notes)).ok()?;
        let descriptor_size = usize::try_from(le_u32(&notes[4..])).ok()?;
        let kind = le_u32(&notes[8..]);
        let name_end = 12usize.checked_add(name_size)?;
        let descriptor_start = 12usize.checked_add(name_size.checked_next_multiple_of(4)?)?;
        let descriptor_end = descriptor_start.checked_add(descriptor_size)?;
        if descriptor_end > notes.len() {
            return None;
        }
        if kind == NT_GNU_BUILD_ID && notes[12..name_end] == *b"GNU\0" {
            return Some(notes[descriptor_start..descriptor_end].to_vec());
        }
        let next = descriptor_start.checked_add(descriptor_size.checked_next_multiple_of(4)?)?;
        notes = notes.get(next..)?;
    }
    None
}

/// Regular files with an execute bit under `directory`, outside
/// [`SKIPPED_DIRECTORIES`] and without a [`SKIPPED_EXTENSIONS`] extension.
/// A missing directory holds none.
fn executables(directory: &Path) -> Result<Vec<PathBuf>, String> {
    let mut found = Vec::new();
    let mut pending = vec![directory.to_path_buf()];
    while let Some(dir) = pending.pop() {
        let Ok(entries) = fs::read_dir(&dir) else {
            continue;
        };
        for entry in entries.filter_map(Result::ok) {
            let path = entry.path();
            let kind = entry.file_type().map_err(|error| {
                format!("Fix: cannot read the type of `{}`: {error}", path.display())
            })?;
            if kind.is_dir() {
                if !entry
                    .file_name()
                    .to_str()
                    .is_some_and(|name| SKIPPED_DIRECTORIES.contains(&name))
                {
                    pending.push(path);
                }
            } else if kind.is_file()
                && !path
                    .extension()
                    .and_then(|extension| extension.to_str())
                    .is_some_and(|extension| SKIPPED_EXTENSIONS.contains(&extension))
                && is_executable(&path)
            {
                found.push(path);
            }
        }
    }
    Ok(found)
}

#[cfg(unix)]
fn is_executable(path: &Path) -> bool {
    use std::os::unix::fs::PermissionsExt;
    fs::metadata(path).is_ok_and(|meta| meta.permissions().mode() & 0o111 != 0)
}

#[cfg(not(unix))]
fn is_executable(path: &Path) -> bool {
    path.extension().is_some_and(|extension| extension == "exe")
}

fn read_at(file: &mut File, offset: u64, buffer: &mut [u8]) -> std::io::Result<()> {
    file.seek(SeekFrom::Start(offset))?;
    file.read_exact(buffer)
}

fn decode_hex(text: &str) -> Option<BuildId> {
    if text.len() % 2 != 0 || !text.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return None;
    }
    (0..text.len())
        .step_by(2)
        .map(|index| u8::from_str_radix(text.get(index..index + 2)?, 16).ok())
        .collect()
}

fn le_u16(bytes: &[u8]) -> u16 {
    u16::from_le_bytes([bytes[0], bytes[1]])
}

fn le_u32(bytes: &[u8]) -> u32 {
    u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]])
}

fn le_u64(bytes: &[u8]) -> u64 {
    let mut word = [0u8; 8];
    word.copy_from_slice(&bytes[..8]);
    u64::from_le_bytes(word)
}
