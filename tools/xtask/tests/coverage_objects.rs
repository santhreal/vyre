//! The coverage report reads only executables that wrote a profile in the run.
//!
//! Closes the class where an instrumented executable left in the shared
//! profile directory by another feature set, another checkout, or a removed
//! target is read by `cargo llvm-cov report`. Its regions have no profile in
//! the run, so every one of its lines counts as missed and the measured
//! coverage falls with no source change.
//!
//! The selection keys on the GNU build ID a raw profile records, so these
//! cases build ELF objects with and without that note, place them where cargo
//! does, and check which ones the selection deletes. Not covered here: the
//! `llvm-profdata` invocation itself, which the coverage workflow runs against
//! real raw profiles.
#![cfg(unix)]

use std::collections::BTreeSet;
use std::fs;
use std::os::unix::fs::PermissionsExt;
use std::path::PathBuf;

use xtask::coverage_objects::{
    delete, gnu_build_id, recorded_build_ids, select, BuildId, CoverageScope,
};

const NT_GNU_ABI_TAG: u32 = 1;
const NT_GNU_BUILD_ID: u32 = 3;

/// One ELF note: name, type, descriptor.
fn note(name: &[u8], kind: u32, descriptor: &[u8]) -> Vec<u8> {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&u32::try_from(name.len()).unwrap().to_le_bytes());
    bytes.extend_from_slice(&u32::try_from(descriptor.len()).unwrap().to_le_bytes());
    bytes.extend_from_slice(&kind.to_le_bytes());
    bytes.extend_from_slice(name);
    bytes.resize(bytes.len().next_multiple_of(4), 0);
    bytes.extend_from_slice(descriptor);
    bytes.resize(bytes.len().next_multiple_of(4), 0);
    bytes
}

/// An ELF64 little-endian object whose single note section holds `notes`,
/// after a null section header as a linker writes one.
fn elf(notes: &[u8]) -> Vec<u8> {
    let note_offset = 64u64;
    let table_offset = note_offset + notes.len() as u64;
    let mut bytes = vec![0u8; 64];
    bytes[..4].copy_from_slice(b"\x7fELF");
    bytes[4] = 2;
    bytes[5] = 1;
    bytes[0x28..0x30].copy_from_slice(&table_offset.to_le_bytes());
    bytes[0x3a..0x3c].copy_from_slice(&64u16.to_le_bytes());
    bytes[0x3c..0x3e].copy_from_slice(&2u16.to_le_bytes());
    bytes.extend_from_slice(notes);
    bytes.extend_from_slice(&[0u8; 64]);
    let mut section = [0u8; 64];
    section[4..8].copy_from_slice(&7u32.to_le_bytes());
    section[0x18..0x20].copy_from_slice(&note_offset.to_le_bytes());
    section[0x20..0x28].copy_from_slice(&(notes.len() as u64).to_le_bytes());
    bytes.extend_from_slice(&section);
    bytes
}

/// An object whose build ID is `id`, after an ABI-tag note that shares the
/// `GNU` name, so the reader must match the note type and not the first name.
fn object(id: &[u8]) -> Vec<u8> {
    let mut notes = note(
        b"GNU\0",
        NT_GNU_ABI_TAG,
        &[0, 0, 0, 0, 3, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0],
    );
    notes.extend(note(b"GNU\0", NT_GNU_BUILD_ID, id));
    elf(&notes)
}

struct Layout {
    _dir: tempfile::TempDir,
    profile: PathBuf,
    scope: CoverageScope,
}

impl Layout {
    fn new() -> Self {
        let dir = tempfile::tempdir().expect("Fix: fixture directory must exist");
        let directory = dir.path().join("llvm-cov-target");
        let profile = directory.join("debug");
        for sub in [
            "deps",
            "examples",
            "build/crate-1/out",
            "incremental/x",
            ".fingerprint/x",
        ] {
            fs::create_dir_all(profile.join(sub)).expect("Fix: fixture directory must exist");
        }
        // Workspace names as `cargo metadata` lists them; `foreign` is not one.
        let names = ["all_tests", "tool", "demo", "crate", "macro", "script", "x"]
            .into_iter()
            .map(str::to_string)
            .collect();
        Self {
            _dir: dir,
            profile,
            scope: CoverageScope { directory, names },
        }
    }

    fn file(&self, relative: &str, contents: &[u8], executable: bool) -> PathBuf {
        let path = self.profile.join(relative);
        fs::write(&path, contents).expect("Fix: fixture file must be writable");
        let mode = if executable { 0o755 } else { 0o644 };
        fs::set_permissions(&path, fs::Permissions::from_mode(mode))
            .expect("Fix: fixture permissions must be settable");
        path
    }
}

const RAN_TEST: &[u8] = &[0xa1; 20];
const RAN_BIN: &[u8] = &[0xb2; 20];
const STALE: &[u8] = &[0xc3; 20];

fn profiled() -> BTreeSet<BuildId> {
    [RAN_TEST.to_vec(), RAN_BIN.to_vec()].into_iter().collect()
}

/// A run's profile directory: a test executable and a binary that ran (the
/// binary with its `deps` hard link), and every kind of file the selection
/// must delete or leave alone. Returns the layout and the sorted stale paths.
fn populated() -> (Layout, Vec<PathBuf>) {
    let layout = Layout::new();
    layout.file("deps/all_tests-0a1", &object(RAN_TEST), true);
    let bin = layout.file("deps/tool-0b2", &object(RAN_BIN), true);
    fs::hard_link(&bin, layout.profile.join("tool")).expect("Fix: hard link must be creatable");
    // Never candidates: dependency files, libraries, build scripts,
    // incremental and fingerprint state, files without an execute bit, and
    // executables of a name outside the workspace, which the report skips.
    layout.file("deps/all_tests-0a1.d", b"deps", true);
    layout.file("deps/libcrate-0c3.rlib", b"rlib", true);
    layout.file("deps/libcrate-0c3.rmeta", b"rmeta", true);
    layout.file("deps/libmacro-0d4.so", &object(STALE), true);
    layout.file("build/crate-1/tool-2a1", &object(STALE), true);
    layout.file("build/crate-1/out/all_tests-2b2", &object(STALE), true);
    layout.file("incremental/x/tool-2c3", &object(STALE), true);
    layout.file(".fingerprint/x/demo-2d4", &object(STALE), true);
    layout.file("deps/tool-0e5", &object(STALE), false);
    layout.file("deps/foreign-0f6", &object(STALE), true);
    let mut stale = vec![
        layout.file("deps/all_tests-1a1", &object(STALE), true),
        layout.file("deps/libtool-1b2", &object(STALE), true),
        layout.file("examples/demo-1c3", &object(STALE), true),
        layout.file("demo", &object(STALE), true),
        layout.file("deps/tool-1d4", &elf(&[]), true),
    ];
    stale.sort();
    (layout, stale)
}

#[test]
fn only_executables_no_profile_records_are_selected() {
    let (layout, stale) = populated();
    let selection = select(&layout.scope, &profiled()).expect("Fix: a recorded run must select");
    assert_eq!(selection.stale, stale);
    // The test executable, the binary, and the binary's second name.
    assert_eq!(selection.kept, 3);
}

#[test]
fn delete_removes_the_selection_and_nothing_else() {
    let (layout, stale) = populated();
    let selection = select(&layout.scope, &profiled()).expect("Fix: a recorded run must select");
    delete(&selection).expect("Fix: deletion must succeed on a writable layout");
    for path in &stale {
        assert!(!path.exists(), "{} must be deleted", path.display());
    }
    for kept in [
        "deps/all_tests-0a1",
        "deps/tool-0b2",
        "tool",
        "deps/all_tests-0a1.d",
        "deps/libcrate-0c3.rlib",
        "deps/libmacro-0d4.so",
        "build/crate-1/tool-2a1",
        "incremental/x/tool-2c3",
        ".fingerprint/x/demo-2d4",
        "deps/tool-0e5",
        "deps/foreign-0f6",
    ] {
        assert!(layout.profile.join(kept).exists(), "{kept} must be kept");
    }
    let again = select(&layout.scope, &profiled()).expect("Fix: a pruned layout still selects");
    assert_eq!(
        again.stale,
        Vec::<PathBuf>::new(),
        "a second prune must find nothing"
    );
    assert_eq!(again.kept, 3);
}

#[test]
fn a_run_that_recorded_no_build_id_is_rejected() {
    let (layout, _stale) = populated();
    let error = select(&layout.scope, &BTreeSet::new()).unwrap_err();
    assert!(
        error.contains("no raw profile records a build ID"),
        "{error}"
    );
}

#[test]
fn an_executable_that_is_not_elf64_is_rejected() {
    let layout = Layout::new();
    layout.file("deps/all_tests-0a1", &object(RAN_TEST), true);
    let script = layout.file("deps/script", b"#!/bin/sh\nexit 0\n", true);
    let error = select(&layout.scope, &profiled()).unwrap_err();
    assert!(
        error.contains("is not an ELF64 little-endian object"),
        "{error}"
    );
    assert!(error.contains(&script.display().to_string()), "{error}");
    let mut big_endian = object(RAN_TEST);
    big_endian[5] = 2;
    layout.file("deps/script", &big_endian, true);
    assert!(select(&layout.scope, &profiled()).is_err());
}

#[test]
fn the_build_id_is_the_gnu_note_of_type_three() {
    let layout = Layout::new();
    let path = layout.file("deps/x", &object(STALE), true);
    assert_eq!(gnu_build_id(&path).unwrap(), Some(STALE.to_vec()));
    let other_owner = layout.file(
        "deps/y",
        &elf(&note(b"Go\0\0", NT_GNU_BUILD_ID, STALE)),
        true,
    );
    assert_eq!(gnu_build_id(&other_owner).unwrap(), None);
    // A note whose declared descriptor runs past the section ends the scan.
    let mut truncated = note(b"GNU\0", NT_GNU_BUILD_ID, STALE);
    truncated[4..8].copy_from_slice(&1000u32.to_le_bytes());
    let cut = layout.file("deps/z", &elf(&truncated), true);
    assert_eq!(gnu_build_id(&cut).unwrap(), None);
}

#[test]
fn recorded_build_ids_are_the_hex_lines_after_the_header() {
    let output = "Instrumentation level: Front-end\nTotal functions: 3\nBinary IDs: \nfe94952be371230695fa3eb348a2b53ea1f6ae52\n\nA1B2\n";
    assert_eq!(
        recorded_build_ids(output).unwrap(),
        vec![
            vec![
                0xfe, 0x94, 0x95, 0x2b, 0xe3, 0x71, 0x23, 0x06, 0x95, 0xfa, 0x3e, 0xb3, 0x48, 0xa2,
                0xb5, 0x3e, 0xa1, 0xf6, 0xae, 0x52
            ],
            vec![0xa1, 0xb2],
        ]
    );
    assert_eq!(
        recorded_build_ids("Total functions: 3\n").unwrap(),
        Vec::<BuildId>::new()
    );
    for malformed in [
        "Binary IDs:\nabc\n",
        "Binary IDs:\n+a\n",
        "Binary IDs:\nzz\n",
    ] {
        let error = recorded_build_ids(malformed).unwrap_err();
        assert!(error.contains("is not hexadecimal"), "{error}");
    }
}

#[test]
fn the_report_reads_workspace_names_with_an_optional_prefix_and_hash() {
    let layout = Layout::new();
    for name in [
        "all_tests",
        "all_tests-0a1b",
        "all-tests-0a1b",
        "libtool-9f",
        "libtool.so",
        "tool-0a1.exe",
        "demo",
    ] {
        assert!(layout.scope.reads(name), "{name} must be read");
    }
    for name in [
        "foreign",
        "foreign-0a1",
        "tool-xyz",
        "libforeign-0a1",
        "tools-0a1",
        "all",
        "tool-",
    ] {
        assert!(!layout.scope.reads(name), "{name} must not be read");
    }
}
