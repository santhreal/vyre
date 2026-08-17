//! No production crate outside the reference oracle ships a host dispatcher.
//!
//! WHY. Vyre executes on a device. A `ProgramDispatcher` that runs a Program on
//! the host is a second execution route, and a second route that is reachable
//! from a default build is a fallback whether or not anything currently takes
//! it: the type is public, so a consumer constructs it and runs off the device
//! with no error, no telemetry and no probe failure. This is not hypothetical.
//! `vyre-libs` shipped a graph-primitive oracle dispatcher as an ungated
//! `pub mod` for the whole of 0.7, and every one of its callers was a test.
//!
//! This closes the class rather than that instance. The implementor set is read
//! out of the tree at run time, so a host dispatcher added to a crate nobody
//! thought of is caught on the commit that adds it, and the assertion is an
//! exact equality against the empty set so the failure names the file and the
//! type instead of a count.
//!
//! The only allowed production host dispatchers belong to `vyre-reference` and
//! `vyre-driver-reference`, which form the single seam every parity suite uses.
//! A dispatcher under `#[cfg(test)]` is also excluded because it cannot reach a
//! shipped build. Feature-gated host dispatchers are forbidden: dormant CPU
//! execution is still a second production route.
//!
//! What it does not catch: a host dispatcher that implements the trait through a
//! blanket impl or a macro, and one that executes a Program without implementing
//! the trait at all. The first is a shape this workspace does not use; the
//! second is what the hygiene matrix's hidden-fallback scan covers.

#![forbid(unsafe_code)]

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

/// Crate directories permitted to execute a Program on the host.
const ORACLE_CRATES: &[&str] = &["vyre-reference", "vyre-driver-reference"];

/// Below this many implementors the walk has stopped reading the tree, and an
/// empty offender set would prove nothing. Every backend driver ships one, so
/// this floor cannot be met by an accident of the scanner.
const MINIMUM_IMPLEMENTORS: usize = 3;

#[test]
fn no_production_crate_outside_the_oracle_ships_a_host_dispatcher() {
    let root = checkout_root();
    let offenders: BTreeMap<String, Vec<String>> = implementors(&root)
        .into_iter()
        .filter(|(path, declared)| {
            is_host_dispatcher_offender(path)
                && (declared.reaches_interpreter || declared.types.iter().any(|n| is_host_type(n)))
        })
        .map(|(path, declared)| (path, declared.types))
        .collect();

    let offending: BTreeSet<&String> = offenders.keys().collect();
    assert_eq!(
        offending,
        BTreeSet::new(),
        "Fix: a host `ProgramDispatcher` is reachable from a shipped build. Vyre \
         executes on a device: move the implementation into one of \
         {ORACLE_CRATES:?}, or make it a test-only module. Feature gates do not \
         make dormant host execution an acceptable production route. Offenders: \
         {offenders:?}"
    );
}

#[test]
fn the_walk_actually_reaches_the_dispatchers_it_judges() {
    let root = checkout_root();
    let found = implementors(&root);
    let total: usize = found.values().map(|declared| declared.types.len()).sum();
    assert!(
        total >= MINIMUM_IMPLEMENTORS,
        "Fix: the source walk found {total} `ProgramDispatcher` implementors, \
         below the floor of {MINIMUM_IMPLEMENTORS}. The walk is not reading the \
         tree, so an empty offender set proves nothing. Files seen: {:?}",
        found.keys().collect::<Vec<_>>()
    );
}

/// WHY: the judgement is the whole test, and it has four cases that must not
/// collapse into each other. Exercised on synthetic paths so it holds for the
/// rule and not for today's tree.
#[test]
fn the_judgement_admits_reference_and_test_dispatchers_only() {
    for oracle in ORACLE_CRATES {
        assert!(
            !is_host_dispatcher_offender(&format!("{oracle}/src/dispatcher.rs")),
            "the parity oracle crate {oracle} exists to execute on the host"
        );
    }
    assert!(
        !is_host_dispatcher_offender("vyre-libs/src/graph/dispatch/tests/mod.rs"),
        "a test module never reaches a default build"
    );
    assert!(
        is_host_dispatcher_offender("vyre-libs/src/graph/dispatch/oracle.rs"),
        "an ungated production module is the defect this closes"
    );
    assert!(
        !is_host_dispatcher_offender("vyre-driver-cuda/tests/parity.rs"),
        "an integration test is not production surface"
    );
    assert!(
        is_test_cfg("#[cfg(test)]"),
        "a test cfg must exclude the dispatcher from shipped builds"
    );
    assert!(
        !is_test_cfg("#[cfg(feature = \"cpu-parity\")]"),
        "a feature gate must not exempt dormant host execution"
    );
}

/// Whether a file declaring an implementor is a production host dispatcher.
///
/// Paths decide whether code is reachable from a shipped build: oracle crates
/// and `tests` or `benches` directories are excluded here. A test `cfg` on the
/// declaring module is read separately in [`implementors`].
fn is_host_dispatcher_offender(path: &str) -> bool {
    let path = Path::new(path);
    if ORACLE_CRATES
        .iter()
        .any(|crate_dir| path.starts_with(crate_dir))
    {
        return false;
    }
    !path
        .components()
        .any(|component| matches!(component.as_os_str().to_str(), Some("tests" | "benches")))
}

/// Every tracked source file declaring a `ProgramDispatcher` implementor, mapped
/// to the type names it declares.
///
/// Every implementor, device ones included, because that is what makes the reach
/// floor mean something: the host subset is a handful and would sit under any
/// honest floor, so a walk that had stopped reading the tree would still pass a
/// floor set against it. The host judgement is applied by the caller.
///
/// A file whose declaring module is gated on a parity feature or on `test` is
/// omitted, because it is not reachable from a default build and reachability is
/// the property under judgement.
fn implementors(root: &Path) -> BTreeMap<String, Declared> {
    let mut found = BTreeMap::new();
    for path in tracked_sources(root) {
        let Ok(text) = std::fs::read_to_string(root.join(&path)) else {
            continue;
        };
        let types: Vec<String> = text
            .lines()
            .filter_map(|line| implemented_type(line.trim()))
            .map(str::to_string)
            .collect();
        if types.is_empty() || is_test_gated(&text) || declaration_is_gated(root, &path) {
            continue;
        }
        let reaches_interpreter = reaches_the_interpreter(&text);
        found.insert(
            path,
            Declared {
                types,
                reaches_interpreter,
            },
        );
    }
    found
}

/// What one source file declares about dispatch.
struct Declared {
    /// Every type it implements `ProgramDispatcher` for.
    types: Vec<String>,
    /// Whether it also calls the host interpreter.
    reaches_interpreter: bool,
}

/// The type name in an `impl ProgramDispatcher for T` line, if that is the line.
fn implemented_type(line: &str) -> Option<&str> {
    let rest = line.strip_prefix("impl ")?;
    let rest = rest.strip_prefix("ProgramDispatcher for ")?;
    let name = rest
        .split(|character: char| !character.is_alphanumeric() && character != '_')
        .next()?;
    (!name.is_empty()).then_some(name)
}

/// Whether a dispatcher name says it executes on the host.
///
/// This workspace spells a host implementation with `Cpu`, `Host` or `Oracle` in
/// the type name, and the naming rule is itself enforced by the neutral-vocabulary
/// gate. A name is not enough on its own: a dispatcher named for what it wraps
/// rather than for where it runs carries none of those markers and still executes
/// on the host, which is why [`reaches_the_interpreter`] is checked beside this.
fn is_host_type(name: &str) -> bool {
    ["Cpu", "Host", "Oracle"]
        .iter()
        .any(|marker| name.contains(marker))
}

/// Whether the file that declares an implementor also calls the host interpreter.
///
/// The definitive signal, and independent of naming: `vyre_reference::reference_eval`
/// is the one entry point that executes a Program on the CPU, so a file holding
/// both an `impl ProgramDispatcher` and a call to it dispatches on the host
/// whatever its type is called.
fn reaches_the_interpreter(text: &str) -> bool {
    text.contains("reference_eval")
}

/// Whether the file's own inner attributes restrict it to test builds.
fn is_test_gated(text: &str) -> bool {
    text.lines().any(|line| is_test_cfg(line.trim()))
}

/// Whether a `cfg` attribute line names `test`.
fn is_test_cfg(line: &str) -> bool {
    if !line.starts_with("#![cfg(") && !line.starts_with("#[cfg(") {
        return false;
    }
    line.contains("test")
}

/// Whether the `mod` statement that brings this file into the crate is gated.
///
/// This, not the file's own attributes, is where the gate belongs and where it
/// usually sits: a test module carries no `cfg` of its own and is excluded from
/// shipped builds by the attribute on its `mod` line. A check that read only the
/// file would call a correctly gated tree broken and send the next reader to add
/// a redundant inner attribute.
fn declaration_is_gated(root: &Path, path: &str) -> bool {
    let path = Path::new(path);
    let Some(stem) = path.file_stem().and_then(|stem| stem.to_str()) else {
        return false;
    };
    let Some(directory) = path.parent() else {
        return false;
    };
    // A `mod.rs` is declared by its directory's name, one level further up.
    let (name, directory) = if stem == "mod" {
        match (
            directory.file_name().and_then(|name| name.to_str()),
            directory.parent(),
        ) {
            (Some(name), Some(parent)) => (name, parent),
            _ => return false,
        }
    } else {
        (stem, directory)
    };
    for parent in [directory.join("mod.rs"), directory.join("lib.rs")] {
        let Ok(text) = std::fs::read_to_string(root.join(&parent)) else {
            continue;
        };
        let mut attributes = Vec::new();
        for line in text.lines() {
            let line = line.trim();
            if line.starts_with("#[") {
                attributes.push(line);
                continue;
            }
            let declares = line == format!("mod {name};")
                || line == format!("pub mod {name};")
                || line == format!("pub(crate) mod {name};")
                || line == format!("pub(super) mod {name};");
            if declares {
                return attributes.iter().copied().any(is_test_cfg);
            }
            if !line.starts_with("///") && !line.starts_with("//") && !line.is_empty() {
                attributes.clear();
            }
        }
    }
    false
}

/// Every `.rs` path git tracks, relative to the checkout root.
///
/// Tracked rather than walked: a scratch file in a working tree is not part of
/// the product, and counting one would report a defect no clone of the
/// repository has.
fn tracked_sources(root: &Path) -> Vec<String> {
    let output = std::process::Command::new("git")
        .arg("-C")
        .arg(root)
        .args(["ls-files", "*.rs"])
        .output()
        .expect("Fix: git must be available to read the tracked source set");
    assert!(
        output.status.success(),
        "Fix: `git ls-files` failed in {}",
        root.display()
    );
    String::from_utf8_lossy(&output.stdout)
        .lines()
        .map(str::to_string)
        .collect()
}

/// Absolute root of the checkout this test runs in.
///
/// Resolved from the working directory rather than `CARGO_MANIFEST_DIR`: a target
/// directory shared by several checkouts computes the same unit hash for a member
/// in each of them, so a compiled-in path can name a different tree than the one
/// under test.
fn checkout_root() -> PathBuf {
    let start = std::env::current_dir().expect("Fix: the working directory must be readable");
    for candidate in start.ancestors() {
        let Ok(text) = std::fs::read_to_string(candidate.join("Cargo.toml")) else {
            continue;
        };
        if text.lines().any(|line| line.trim_start() == "[workspace]") {
            return candidate.to_path_buf();
        }
    }
    panic!("Fix: no workspace manifest above the working directory");
}
