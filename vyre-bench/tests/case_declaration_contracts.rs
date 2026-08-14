//! Permanent guard for the bench-case declaration invariants.
//!
//! WHY: the honest cases each carried a verbatim copy of the same three
//! declarations (metadata record, suite list, GPU requirement shape) and the
//! copies had drifted. `search.binary.u32.1m` omitted `SuiteKind::Smoke` from
//! its private suite list, so that case never ran in the smoke suite and a
//! regression in it was invisible to every smoke run. Collapsing the copies onto
//! `crate::cases::honest_case` removes the place drift can live, but only for the
//! cases that exist today. This file states the invariants as properties over
//! the registry, so a case added tomorrow is covered without editing a list.
//!
//! Coverage is derived at run time twice over: the case set is walked from the
//! inventory registry, and the timing-narrowing gate walks the crate's own source
//! tree. A new case, or a new source file, is inside both gates the moment it
//! exists.
//!
//! What it does NOT catch: anything only observable by dispatching a program on a
//! device. Whether a case's measured span is nonzero, and whether its device
//! outputs match its reference, need a GPU. The record assembly that carries both
//! of those is unit-tested in `cases::reference_sample` and
//! `cases::release_workloads::resident_batch`.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
use vyre_bench::api::case::{BenchCase, BenchLayer};
use vyre_bench::api::suite::SuiteKind;

fn registered() -> Vec<&'static dyn BenchCase> {
    let cases: Vec<&'static dyn BenchCase> = vyre_bench::registry::collect_all().iter().collect();
    assert!(
        !cases.is_empty(),
        "Fix: the bench registry published no cases, so every property below is vacuous"
    );
    cases
}

/// A case's metadata record must name the case it came from.
///
/// The collapsed metadata helper takes the id as its first argument, so a
/// copy-paste that passes a neighbour's id compiles and produces a case whose
/// reports are filed under the wrong name.
#[test]
fn metadata_names_its_own_case() {
    let mismatched: Vec<String> = registered()
        .iter()
        .filter(|case| case.metadata().id != case.id())
        .map(|case| {
            format!(
                "{} reports metadata id {}",
                case.id().0,
                case.metadata().id.0
            )
        })
        .collect();

    assert!(
        mismatched.is_empty(),
        "Fix: a case's metadata must carry its own id: {mismatched:?}"
    );
}

/// A case names each suite it runs in exactly once.
///
/// An empty list is not a defect: `BenchCase::active_in_suite` reads it as
/// membership in every suite, which is the documented default. A repeated suite
/// always is, because the case is then selected twice by one suite run and its
/// samples are pooled with themselves.
#[test]
fn suite_membership_is_free_of_duplicates() {
    let offenders: Vec<String> = registered()
        .iter()
        .filter(|case| {
            let suites = case.suites();
            let unique: BTreeSet<String> = suites.iter().map(|s| format!("{s:?}")).collect();
            unique.len() != suites.len()
        })
        .map(|case| {
            format!(
                "{} declares a suite twice: {:?}",
                case.id().0,
                case.suites()
            )
        })
        .collect();

    assert!(offenders.is_empty(), "Fix: {offenders:?}");
}

/// Every honest-layer case runs in the smoke suite.
///
/// This is the invariant `search.binary.u32.1m` broke while six sibling copies of
/// the same list held it. Membership is read from the layer the case declares, so
/// a new honest case is inside this gate without being named here.
#[test]
fn every_honest_case_runs_in_the_smoke_suite() {
    let excluded: Vec<String> = registered()
        .iter()
        .filter(|case| matches!(case.metadata().layer, BenchLayer::Honest))
        .filter(|case| !case.suites().contains(&SuiteKind::Smoke))
        .map(|case| case.id().0)
        .collect();

    assert!(
        excluded.is_empty(),
        "Fix: an honest case outside the smoke suite is never smoke-tested: {excluded:?}"
    );
}

/// Every case carries a non-empty owner crate and at least one non-blank tag,
/// each tag named once. Tags are how the release matrix and the dashboard select
/// cases, so a blank or repeated tag silently changes which set a case lands in.
#[test]
fn metadata_prose_is_populated_and_tags_are_distinct() {
    let offenders: Vec<String> = registered()
        .iter()
        .filter_map(|case| {
            let metadata = case.metadata();
            let id = case.id().0;
            if metadata.owner_crate.trim().is_empty() {
                return Some(format!("{id} declares no owner crate"));
            }
            if metadata.name.trim().is_empty() {
                return Some(format!("{id} declares no name"));
            }
            if metadata.description.trim().is_empty() {
                return Some(format!("{id} declares no description"));
            }
            if metadata.tags.is_empty() {
                return Some(format!("{id} declares no tags"));
            }
            if metadata.tags.iter().any(|tag| tag.trim().is_empty()) {
                return Some(format!("{id} declares a blank tag: {:?}", metadata.tags));
            }
            let unique: BTreeSet<&String> = metadata.tags.iter().collect();
            if unique.len() != metadata.tags.len() {
                return Some(format!("{id} declares a tag twice: {:?}", metadata.tags));
            }
            None
        })
        .collect();

    assert!(offenders.is_empty(), "Fix: {offenders:?}");
}

/// A declared memory floor is nonzero.
///
/// Declaring no floor is the documented micro-benchmark default and the runner
/// treats it as "any device will do". Declaring `Some(0)` is not that: the case
/// reads as gated on memory it does not actually require, so the runner reports a
/// check it never performed. The collapsed requirement helper takes the floor as
/// its only argument, which is exactly the argument an arithmetic slip zeroes.
#[test]
fn a_declared_memory_floor_is_nonzero() {
    let offenders: Vec<String> = registered()
        .iter()
        .map(|case| (case.id().0, case.requirements()))
        .filter(|(_, requirements)| {
            requirements.min_vram_bytes == Some(0) || requirements.min_input_bytes == Some(0)
        })
        .map(|(id, requirements)| {
            format!(
                "{id} declares vram={:?} input={:?}",
                requirements.min_vram_bytes, requirements.min_input_bytes
            )
        })
        .collect();

    assert!(
        offenders.is_empty(),
        "Fix: a zero memory floor gates nothing while reading as a declared bound: {offenders:?}"
    );
}

fn crate_source_files() -> Vec<PathBuf> {
    fn walk(dir: &Path, found: &mut Vec<PathBuf>) {
        let entries = std::fs::read_dir(dir)
            .unwrap_or_else(|error| panic!("Fix: cannot read {}: {error}", dir.display()));
        for entry in entries {
            let path = entry.expect("Fix: cannot read a directory entry").path();
            if path.is_dir() {
                walk(&path, found);
            } else if path.extension().is_some_and(|ext| ext == "rs") {
                found.push(path);
            }
        }
    }

    let mut found = Vec::new();
    walk(
        &Path::new(env!("CARGO_MANIFEST_DIR")).join("src"),
        &mut found,
    );
    found.sort();
    assert!(
        found.len() > 50,
        "Fix: the source walk found only {} files, so the gate below is vacuous",
        found.len()
    );
    found
}

/// `api::metric::elapsed_ns` is the only place a measured span narrows to `u64`.
///
/// `Duration::as_nanos` is a `u128`. Three spellings of the narrowing coexisted
/// across the crate: a bare `as u64` cast, a `min(u64::MAX)` clamp, and a
/// `try_from().unwrap_or()`. The bare cast wraps, so a span past about 18.4
/// seconds reported as a short one and the slowest samples read as the fastest.
/// Whitespace is stripped before matching, so the multi-line spellings are caught
/// too, and the walk covers the whole tree rather than the files that had the
/// defect.
///
/// A `SystemTime` timestamp formatted into a filename is not a measured span and
/// does not match: it reads the clock through `duration_since`, never `elapsed`.
#[test]
fn only_the_metric_owner_narrows_a_measured_span() {
    let owner = Path::new(env!("CARGO_MANIFEST_DIR")).join("src/api/metric.rs");
    let offenders: Vec<String> = crate_source_files()
        .into_iter()
        .filter(|path| *path != owner)
        .filter(|path| stripped_code(path).contains("elapsed().as_nanos()"))
        .map(|path| path.display().to_string())
        .collect();

    assert!(
        offenders.is_empty(),
        "Fix: narrow a measured span with api::metric::elapsed_ns, which saturates. \
         A bare `as u64` cast wraps past 18.4 seconds: {offenders:?}"
    );
}

/// The truncating spelling appears nowhere, including inside the owner.
///
/// The gate above exempts `api::metric`, because that file is where the one
/// legitimate read of a measured span lives. Nothing exempts it from the cast
/// rule: writing the bare cast directly in `elapsed_ns` would reintroduce the
/// wrap for every caller at once and leave `narrow_nanos` dead. This gate walks
/// the whole tree with no exemption.
#[test]
fn no_nanosecond_count_is_narrowed_by_a_truncating_cast() {
    let offenders: Vec<String> = crate_source_files()
        .into_iter()
        .filter(|path| stripped_code(path).contains("as_nanos()asu64"))
        .map(|path| path.display().to_string())
        .collect();

    assert!(
        offenders.is_empty(),
        "Fix: a `Duration::as_nanos() as u64` cast wraps instead of saturating. \
         Narrow through api::metric, which uses `u64::try_from`: {offenders:?}"
    );
}

/// A source file with comments removed and all whitespace collapsed away, so a
/// spelling split across lines matches the same as a single-line one.
fn stripped_code(path: &Path) -> String {
    let source = std::fs::read_to_string(path)
        .unwrap_or_else(|error| panic!("Fix: cannot read {}: {error}", path.display()));
    source
        .lines()
        .map(|line| line.split("//").next().unwrap_or(line))
        .collect::<Vec<&str>>()
        .join("")
        .split_whitespace()
        .collect()
}
