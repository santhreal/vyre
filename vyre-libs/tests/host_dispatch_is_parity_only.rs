//! No production crate outside the reference oracle executes a Program on the host.
//!
//! WHY. Vyre executes on a device. An execution-seam implementation that runs a
//! Program on the host is a second execution route, and a second route that is
//! reachable from a default build is a fallback whether or not anything
//! currently takes it: the type is public, so a consumer constructs it and runs
//! off the device with no error, no telemetry and no probe failure. This is not
//! hypothetical. `vyre-libs` shipped a graph-primitive oracle dispatcher as an
//! ungated `pub mod` for the whole of 0.7, and every one of its callers was a
//! test.
//!
//! This closes the class rather than that instance. The implementor set is read
//! out of the tree at run time, so a host implementation added to a crate nobody
//! thought of is caught on the commit that adds it, and the assertion is an
//! exact equality against the empty set so the failure names the file and the
//! type instead of a count.
//!
//! Both execution seams are policed, and each name in [`SEAM_TRAITS`] must
//! resolve to a trait declaration in tracked source. A seam that is renamed or
//! deleted therefore turns this suite red instead of leaving the walk to search
//! for a name nothing declares, which is how a scan starts certifying nothing.
//!
//! The only allowed production host implementations belong to `vyre-reference`
//! and `vyre-driver-reference`, which form the single seam every parity suite
//! uses. An implementation excluded from shipped builds by a `cfg(test)` on the
//! item, on an enclosing module, or on the `mod` statement that includes its
//! file is also excluded here, because it cannot reach a shipped build.
//! Feature-gated host execution is forbidden: dormant CPU execution is still a
//! second production route.
//!
//! What it does not catch: an implementation reached through a blanket impl or a
//! macro, and one that executes a Program without implementing a seam trait at
//! all. The first is a shape this workspace does not use; the second is what the
//! hygiene matrix's hidden-fallback scan covers.

#![forbid(unsafe_code)]

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

/// Crate directories permitted to execute a Program on the host.
const ORACLE_CRATES: &[&str] = &["vyre-reference", "vyre-driver-reference"];

/// The execution seam a Program crosses.
const SEAM_TRAITS: &[&str] = &["SemanticExecutor"];

/// Below this many implementors the walk has stopped reading the tree, and an
/// empty offender set would prove nothing. Counted across every implementor,
/// test doubles included, because the production subset is a handful and would
/// sit under any honest floor.
const MINIMUM_IMPLEMENTORS: usize = 20;

#[test]
fn no_production_crate_outside_the_oracle_executes_on_the_host() {
    let root = checkout_root();
    let offenders: BTreeMap<String, Vec<String>> = implementors(&root)
        .into_iter()
        .filter_map(|(path, declared)| {
            let shipped: Vec<String> = declared
                .types
                .into_iter()
                .filter(|declared| !declared.test_only)
                .map(|declared| declared.name)
                .collect();
            let host = declared.reaches_interpreter || shipped.iter().any(|n| is_host_type(n));
            (is_host_execution_offender(&path) && host && !shipped.is_empty())
                .then_some((path, shipped))
        })
        .collect();

    let offending: BTreeSet<&String> = offenders.keys().collect();
    assert_eq!(
        offending,
        BTreeSet::new(),
        "Fix: a host implementation of {SEAM_TRAITS:?} is reachable from a \
         shipped build. Vyre executes on a device: move the implementation into \
         one of {ORACLE_CRATES:?}, or gate it on `test`. Feature gates do not \
         make dormant host execution an acceptable production route. Offenders: \
         {offenders:?}"
    );
}

#[test]
fn the_walk_actually_reaches_the_implementors_it_judges() {
    let root = checkout_root();
    let found = implementors(&root);
    let total: usize = found.values().map(|declared| declared.types.len()).sum();
    assert!(
        total >= MINIMUM_IMPLEMENTORS,
        "Fix: the source walk found {total} implementors of {SEAM_TRAITS:?}, \
         below the floor of {MINIMUM_IMPLEMENTORS}. The walk is not reading the \
         tree, so an empty offender set proves nothing. Files seen: {:?}",
        found.keys().collect::<Vec<_>>()
    );
}

/// WHY: a seam name that nothing declares makes the walk above search for a
/// string no source contains, and an empty offender set then means the scan
/// found nothing rather than that nothing is wrong. Renaming or retiring a seam
/// must land in this file in the same change.
#[test]
fn every_policed_seam_is_a_trait_this_tree_declares() {
    let root = checkout_root();
    let declarations: BTreeSet<String> = tracked_sources(&root)
        .iter()
        .filter_map(|path| std::fs::read_to_string(root.join(path)).ok())
        .flat_map(|text| {
            text.lines()
                .filter_map(declared_trait)
                .map(str::to_string)
                .collect::<Vec<_>>()
        })
        .collect();
    let missing: Vec<&&str> = SEAM_TRAITS
        .iter()
        .filter(|seam| !declarations.contains(**seam))
        .collect();
    assert!(
        missing.is_empty(),
        "Fix: {missing:?} names no trait this tree declares. Update SEAM_TRAITS \
         to the seams that exist, and state what stopped being policed."
    );
}

/// WHY: the judgement is the whole test, and it has four cases that must not
/// collapse into each other. Exercised on synthetic paths so it holds for the
/// rule and not for today's tree.
#[test]
fn the_judgement_admits_reference_and_test_implementations_only() {
    for oracle in ORACLE_CRATES {
        assert!(
            !is_host_execution_offender(&format!("{oracle}/src/dispatcher.rs")),
            "the parity oracle crate {oracle} exists to execute on the host"
        );
    }
    assert!(
        !is_host_execution_offender("vyre-libs/src/graph/dispatch/tests/mod.rs"),
        "a test module never reaches a default build"
    );
    assert!(
        is_host_execution_offender("vyre-libs/src/graph/dispatch/oracle.rs"),
        "an ungated production module is the defect this closes"
    );
    assert!(
        !is_host_execution_offender("vyre-driver-cuda/tests/parity.rs"),
        "an integration test is not production surface"
    );
    assert!(
        is_test_cfg("#[cfg(test)]"),
        "a test cfg must exclude the implementation from shipped builds"
    );
    assert!(
        !is_test_cfg("#[cfg(feature = \"cpu-parity\")]"),
        "a feature gate must not exempt dormant host execution"
    );
}

/// WHY: every gating shape this tree uses must be recognized, or a correctly
/// gated double is reported as a shipped host route and the next reader silences
/// the gate. The shapes are an attribute on the item, an attribute on an inline
/// module, and an attribute on the `mod` statement that includes a separate file.
#[test]
fn every_test_gating_shape_this_tree_uses_is_recognized() {
    let source = "\
#[cfg(test)]
impl SemanticExecutor for GatedItem {}

#[cfg(test)]
mod inner {
    impl SemanticExecutor for GatedByModule {}
}

impl SemanticExecutor for Shipped {}
";
    let declared = declared_types(source);
    let shipped: Vec<&str> = declared
        .iter()
        .filter(|declared| !declared.test_only)
        .map(|declared| declared.name.as_str())
        .collect();
    assert_eq!(declared.len(), 3, "every implementor must be counted");
    assert_eq!(shipped, ["Shipped"]);
}

/// Whether a file declaring an implementor is production host execution surface.
///
/// Paths decide whether code is reachable from a shipped build: oracle crates
/// and `tests` or `benches` directories are excluded here. Test `cfg`s are read
/// separately, on the item, its enclosing module, and its `mod` statement.
fn is_host_execution_offender(path: &str) -> bool {
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

/// Every tracked source file declaring a seam implementor, mapped to what it
/// declares.
///
/// Every implementor, device ones included, because that is what makes the reach
/// floor mean something: the host subset is a handful and would sit under any
/// honest floor, so a walk that had stopped reading the tree would still pass a
/// floor set against it. The host judgement is applied by the caller.
fn implementors(root: &Path) -> BTreeMap<String, Declared> {
    let mut found = BTreeMap::new();
    for path in tracked_sources(root) {
        let Ok(text) = std::fs::read_to_string(root.join(&path)) else {
            continue;
        };
        let types = declared_types(&text);
        if types.is_empty() {
            continue;
        }
        let file_is_test_only = is_test_gated(&text)
            || declaration_is_gated(root, &path)
            || include_is_gated(root, &path);
        let types = types
            .into_iter()
            .map(|declared| DeclaredType {
                test_only: declared.test_only || file_is_test_only,
                ..declared
            })
            .collect();
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

/// What one source file declares about execution.
struct Declared {
    /// Every seam implementor it declares.
    types: Vec<DeclaredType>,
    /// Whether it also calls the host interpreter.
    reaches_interpreter: bool,
}

/// One seam implementor and whether shipped builds can reach it.
struct DeclaredType {
    /// Implementing type name.
    name: String,
    /// Whether a `cfg(test)` on the item or an enclosing module excludes it.
    test_only: bool,
}

/// Every seam implementor in one file, with its item-level gating resolved.
///
/// Attributes accumulate until a non-attribute line consumes them, so a
/// `#[cfg(test)]` sitting above the `impl` gates that impl, and one sitting
/// above an inline `mod` gates everything the module's braces enclose.
fn declared_types(text: &str) -> Vec<DeclaredType> {
    let mut declared = Vec::new();
    let mut attributes: Vec<&str> = Vec::new();
    let mut test_modules: Vec<i32> = Vec::new();
    let mut depth: i32 = 0;
    for line in text.lines() {
        let line = line.trim();
        if line.starts_with("#[") {
            attributes.push(line);
            continue;
        }
        let gated = !test_modules.is_empty() || attributes.iter().copied().any(is_test_cfg);
        if opens_module(line) && attributes.iter().copied().any(is_test_cfg) {
            test_modules.push(depth);
        }
        if let Some(name) = implemented_type(line) {
            declared.push(DeclaredType {
                name: name.to_string(),
                test_only: gated,
            });
        }
        depth += i32::try_from(line.matches('{').count()).unwrap_or(0)
            - i32::try_from(line.matches('}').count()).unwrap_or(0);
        while test_modules.last().is_some_and(|opened| depth <= *opened) {
            test_modules.pop();
        }
        if !line.is_empty() {
            attributes.clear();
        }
    }
    declared
}

/// Whether the line opens an inline module body.
fn opens_module(line: &str) -> bool {
    let rest = ["mod ", "pub mod ", "pub(crate) mod ", "pub(super) mod "]
        .iter()
        .find_map(|prefix| line.strip_prefix(prefix));
    rest.is_some_and(|rest| rest.contains('{'))
}

/// The type name in an `impl <Seam> for T` line, if that is the line.
///
/// A qualified seam path is accepted: an impl written against
/// `vyre_megakernel::SemanticExecutor` is the same route as one written against
/// the imported name.
fn implemented_type(line: &str) -> Option<&str> {
    let rest = line.strip_prefix("impl ")?;
    let rest = SEAM_TRAITS.iter().find_map(|seam| {
        let position = rest.find(*seam)?;
        let qualifier = &rest[..position];
        (qualifier.is_empty() || qualifier.ends_with("::"))
            .then(|| rest[position + seam.len()..].trim_start())
    })?;
    let rest = rest.strip_prefix("for ")?;
    let name = rest
        .split(|character: char| !character.is_alphanumeric() && character != '_')
        .next()?;
    (!name.is_empty()).then_some(name)
}

/// The trait name a `pub trait <Name>` line declares, if that is the line.
fn declared_trait(line: &str) -> Option<&str> {
    let rest = line.trim().strip_prefix("pub trait ")?;
    let name = rest
        .split(|character: char| !character.is_alphanumeric() && character != '_')
        .next()?;
    (!name.is_empty()).then_some(name)
}

/// Whether an implementor name says it executes on the host.
///
/// This workspace spells a host implementation with `Cpu`, `Host` or `Oracle` in
/// the type name, and the naming rule is itself enforced by the neutral-vocabulary
/// gate. A name is not enough on its own: an implementation named for what it
/// wraps rather than for where it runs carries none of those markers and still
/// executes on the host, which is why [`reaches_the_interpreter`] is checked
/// beside this.
fn is_host_type(name: &str) -> bool {
    ["Cpu", "Host", "Oracle"]
        .iter()
        .any(|marker| name.contains(marker))
}

/// Whether the file that declares an implementor also calls the host interpreter.
///
/// The definitive signal, and independent of naming: `vyre_reference::reference_eval`
/// is the one entry point that executes a Program on the CPU, so a file holding
/// both a seam impl and a call to it executes on the host whatever its type is
/// called.
fn reaches_the_interpreter(text: &str) -> bool {
    text.contains("reference_eval")
}

/// Whether the file's own inner attribute restricts the whole file to test
/// builds.
///
/// Only an inner `#![cfg(test)]` applies to the file. An outer `#[cfg(test)]`
/// applies to one item and is resolved in [`declared_types`], so reading it
/// here would exempt every production file that carries a unit-test module.
fn is_test_gated(text: &str) -> bool {
    text.lines()
        .map(str::trim)
        .any(|line| line.starts_with("#![cfg(") && line.contains("test"))
}

/// Whether a `cfg` attribute line names `test`.
fn is_test_cfg(line: &str) -> bool {
    if !line.starts_with("#![cfg(") && !line.starts_with("#[cfg(") {
        return false;
    }
    line.contains("test")
}

/// Whether a `#[path]` include of this file carries a test `cfg`.
///
/// A `*_tests.rs` file beside its subject is brought in by
/// `#[cfg(test)] #[path = "..."] mod tests;`, where the module name matches
/// neither the file stem nor its directory, so [`declaration_is_gated`] cannot
/// see it.
fn include_is_gated(root: &Path, path: &str) -> bool {
    let path = Path::new(path);
    let Some(file_name) = path.file_name().and_then(|name| name.to_str()) else {
        return false;
    };
    let Some(directory) = path.parent() else {
        return false;
    };
    let needle = format!("#[path = \"{file_name}\"]");
    let Ok(siblings) = root.join(directory).read_dir() else {
        return false;
    };
    for entry in siblings.flatten() {
        let name = entry.file_name();
        let Some(name) = name.to_str() else {
            continue;
        };
        if !name.ends_with(".rs") {
            continue;
        }
        let Ok(text) = std::fs::read_to_string(entry.path()) else {
            continue;
        };
        let mut attributes: Vec<&str> = Vec::new();
        for line in text.lines() {
            let line = line.trim();
            if line.starts_with("#[") {
                attributes.push(line);
                if line == needle && attributes.iter().copied().any(is_test_cfg) {
                    return true;
                }
                continue;
            }
            if line.contains(&needle) && attributes.iter().copied().any(is_test_cfg) {
                return true;
            }
            if !line.starts_with("//") && !line.is_empty() {
                attributes.clear();
            }
        }
    }
    false
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
