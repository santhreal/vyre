//! Manifest parsing, crate identification, and workspace path resolution.
//!
//! Every gate that resolves a name against the tree or judges source layout
//! relies on root-relative directory resolution and member manifests. This
//! module reads workspace manifests, locates member and crate source roots,
//! and scans discarding imports and inventory registration submitters.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::{Arc, LazyLock, Mutex, PoisonError};

use toml::Value;
use walkdir::WalkDir;

use crate::module_layout::CrateRoot;
use crate::source_scan::code_offsets;

/// Largest source or manifest file this gate will read.
///
/// The gate walks whatever tree it is pointed at, so an unbounded
/// `read_to_string` lets one pathological file decide the process's memory.
/// Every read in this crate goes through here.
pub const MAX_SOURCE_BYTES: u64 = 16_777_216;

/// Read a source or manifest file, refusing anything over [`MAX_SOURCE_BYTES`].
pub(crate) fn read_source_bounded(path: &Path) -> std::io::Result<String> {
    use std::io::Read as _;

    let file = fs::File::open(path)?;
    let length = file.metadata()?.len();
    if length > MAX_SOURCE_BYTES {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!(
                "`{}` is {length} bytes; refusing to read more than {MAX_SOURCE_BYTES}",
                path.display()
            ),
        ));
    }
    let mut text = String::with_capacity(length as usize);
    file.take(MAX_SOURCE_BYTES + 1).read_to_string(&mut text)?;
    Ok(text)
}

/// The checkout-relative directory of a workspace member, by package name.
///
/// A member's directory is not always its package name: `vyre-conform` lives at
/// `conform/vyre-conform`. The roster is read from the root manifest at run
/// time, so a gate that needs its own crate directory gets it without a
/// compiled-in manifest path, which would name whichever checkout built the
/// binary.
///
/// # Panics
/// Panics when no member directory's manifest declares `package`.
#[must_use]
pub fn member_directory(root: &Path, package: &str) -> PathBuf {
    for member in workspace_members(root) {
        let member_dir = root.join(&member);
        let manifest_path = member_dir.join("Cargo.toml");
        let Ok(text) = read_source_bounded(&manifest_path) else {
            continue;
        };
        let declared = toml::from_str::<toml::Table>(&text).ok().and_then(|table| {
            Value::Table(table)
                .get("package")
                .and_then(|pkg| pkg.get("name"))
                .and_then(Value::as_str)
                .map(str::to_string)
        });
        if declared.as_deref() == Some(package) {
            return member_dir;
        }
    }
    panic!(
        "Fix: no workspace member under {} declares package `{package}`; the roster in the root \
         Cargo.toml is what this resolves against.",
        root.display()
    );
}

/// The workspace member roster, as the root manifest declares it.
///
/// Every gate that resolves a name against the tree needs this list, so it has
/// one owner: a second copy drifts the moment a member is added under a path
/// one copy filters and the other does not.
///
/// # Panics
///
/// Panics when the root manifest cannot be read or parsed.
#[must_use]
pub fn workspace_members(root: &Path) -> Vec<String> {
    workspace_paths(root, "members")
}

/// The paths the root manifest excludes from the workspace.
///
/// `exclude` is the other half of the roster: a directory that is neither a
/// member nor excluded is a directory cargo will pull in the day it grows a
/// manifest. Reading it beside [`workspace_members`] keeps both answers coming
/// from one parse of one file.
///
/// # Panics
///
/// Panics when the root manifest cannot be read or parsed.
#[must_use]
pub fn workspace_excludes(root: &Path) -> Vec<String> {
    workspace_paths(root, "exclude")
}

/// One `[workspace]` array of paths, empty when the key is absent.
///
/// # Panics
///
/// Panics when the root manifest cannot be read or parsed. Every gate in this
/// crate answers for the roster that manifest declares, so a gate that carried
/// on with an empty roster would report a clean tree it never read.
pub(crate) fn workspace_paths(root: &Path, key: &str) -> Vec<String> {
    let manifest_path = root.join("Cargo.toml");
    let text = read_source_bounded(&manifest_path)
        .unwrap_or_else(|error| panic!("Fix: cannot read {}: {error}", manifest_path.display()));
    let table: toml::Table = toml::from_str(&text)
        .unwrap_or_else(|error| panic!("Fix: parse {}: {error}", manifest_path.display()));
    string_list(
        Value::Table(table)
            .get("workspace")
            .and_then(|workspace| workspace.get(key)),
    )
}

/// Every string a TOML array holds, and nothing when it is not one.
///
/// A manifest states a roster, a feature list and a dependency's features all
/// the same way, as an array of strings, and every reader of one wrote the same
/// four-combinator chain to unwrap it. An absent key and a value of another type
/// both answer with an empty list: a caller that needs to tell those apart reads
/// the value itself.
#[must_use]
pub fn string_list(value: Option<&Value>) -> Vec<String> {
    value
        .and_then(Value::as_array)
        .map(|entries| {
            entries
                .iter()
                .filter_map(Value::as_str)
                .map(str::to_string)
                .collect()
        })
        .unwrap_or_default()
}

/// Crate identifier for a crate name, e.g. `vyre_libs` for `vyre-libs`.
#[must_use]
pub fn crate_ident(crate_name: &str) -> String {
    crate_name.replace('-', "_")
}

/// This crate's own sources. Its tests carry example registrations that name
/// other crates on purpose, so scanning itself would report its own fixtures.
pub(crate) const SELF_CRATE: &str = "structure-gate";

/// One member source file and its text, read once per pass over the workspace.
///
/// Six rosters are derived from member source text, and each one used to walk
/// every member tree and read every file again: a registration scan, a
/// substrate-path scan, a frontend scan, a materializer scan, a submitter scan
/// that reads the corpus twice by itself, and a discarding-import scan. That is
/// ten passes over the same 4495 files to answer questions one pass answers,
/// and on a network-mounted checkout each pass is minutes rather than
/// milliseconds: one call held a validation host for half an hour.
pub(crate) struct MemberSource {
    /// Last path component of the member, which is its crate name.
    pub(crate) crate_name: String,
    /// Checkout-relative path of the file, with forward slashes.
    pub(crate) file: String,
    /// The text, or why it could not be read. An unreadable file still counts
    /// as a path: a roster over file names judges it, and a roster over text
    /// skips it, which is what each scan did when it read the file itself.
    pub(crate) text: std::io::Result<String>,
}

/// Every member source file with its text.
///
/// The roster is a prefix range of [`tree_files`] per member rather than a walk
/// per member. The reads are then dealt out over equal slices of the flattened
/// roster, because the members are wildly uneven: one member holds 718 of the
/// files and thirty hold fewer than twenty each, so a lane per member would
/// wait on the one lane that drew the large crate.
///
/// The lane count deliberately exceeds the host's core count. A read of a
/// network-mounted file is a round trip rather than work, so a lane spends its
/// time blocked and the useful width is the number of requests in flight, not
/// the number of cores. Order is restored by sorting the roster before the read
/// and joining the lanes in order, so every roster derived from the corpus
/// reads the same on every run.
///
/// Read once per process per roster. One consolidated test binary runs every
/// contract of a crate in one process, and each contract that judges the tree
/// asks for the whole corpus: reading it per contract is 4495 network round
/// trips per contract, and the corpus of a checked-in tree does not change
/// under a test run. The caller that finds it missing reads it while the others
/// wait, so contracts starting together share one sweep instead of racing to
/// repeat it. The roster is part of the key because a caller may judge a subset
/// of the members.
///
/// # Panics
///
/// Resumes the panic of a read lane that failed. A lane records a refusal per
/// file rather than unwinding, so the only way one panics is a defect in this
/// module; a partial roster would let a membership contract pass on the members
/// that happened to load.
pub(crate) fn member_sources(root: &Path, members: &[String]) -> Arc<[MemberSource]> {
    static READ: LazyLock<Mutex<BTreeMap<(PathBuf, Vec<String>), Arc<[MemberSource]>>>> =
        LazyLock::new(|| Mutex::new(BTreeMap::new()));

    let key = (root.to_path_buf(), members.to_vec());
    let mut cache = READ.lock().unwrap_or_else(PoisonError::into_inner);
    if let Some(found) = cache.get(&key) {
        return Arc::clone(found);
    }
    let read: Arc<[MemberSource]> = read_member_sources(root, members).into();
    cache.insert(key, Arc::clone(&read));
    read
}

/// The corpus read, without the cache in front of it.
///
/// # Panics
///
/// Resumes the panic of a read lane. A partial roster would let a membership
/// contract pass on the members that happened to load.
fn read_member_sources(root: &Path, members: &[String]) -> Vec<MemberSource> {
    let tree = tree_files(root);
    let mut roster: Vec<(String, PathBuf)> = Vec::new();
    for member in members {
        // This crate's own tests carry example registrations that name other
        // crates on purpose, so scanning itself reports its own fixtures.
        if member.as_str() == SELF_CRATE {
            continue;
        }
        let crate_name = member.rsplit('/').next().unwrap_or(member).to_string();
        for path in tree.rust_sources_under(&root.join(member)) {
            roster.push((crate_name.clone(), path.clone()));
        }
    }
    roster.sort_by(|left, right| left.1.cmp(&right.1));
    let per_lane = roster.len().div_ceil(read_lanes(roster.len())).max(1);
    std::thread::scope(|scope| {
        let lanes: Vec<_> = roster
            .chunks(per_lane)
            .map(|chunk| {
                scope.spawn(move || {
                    chunk
                        .iter()
                        .map(|(crate_name, path)| MemberSource {
                            crate_name: crate_name.clone(),
                            file: relative(root, path),
                            text: read_source_bounded(path),
                        })
                        .collect::<Vec<_>>()
                })
            })
            .collect();
        lanes
            .into_iter()
            .flat_map(|lane| {
                lane.join().expect("a member source lane must not panic. Fix: repair the defect stated by the lane panic printed before this one")
            })
            .collect()
    })
}

/// The most files one read may hold open at once.
///
/// A lane is a thread with a file open, so the lane count is a descriptor
/// count. A process gets 256 descriptors by default on macOS and 1024 on Linux,
/// shared by every reader in the binary, the test harness, and cargo: an
/// unbounded multiple of the core count exhausted them, and a corpus read then
/// failed with `Too many open files` rather than reading slowly. The bound
/// leaves room for several readers to be live at once, because the member
/// corpus, the source corpus and a consumer's own roster are separate reads.
/// Thirty-two requests in flight already covers the round-trip latency of a
/// network mount, which is the only thing extra lanes buy: 4495 files at 2 ms
/// each is under a third of a second.
const MAX_READ_LANES: usize = 32;

/// How many lanes a latency-bound read of `files` files should use.
///
/// Eight times the host's parallelism, capped at `MAX_READ_LANES` and never
/// more than one lane per file. A blocked read occupies no core, so the width
/// that matters is requests in flight; on a local disk the extra lanes cost a
/// thread each and finish in the same wall time, and on a network mount they
/// are the difference between seconds and minutes.
#[must_use]
pub fn read_lanes(files: usize) -> usize {
    let cores = std::thread::available_parallelism().map_or(1, std::num::NonZeroUsize::get);
    cores
        .saturating_mul(8)
        .min(MAX_READ_LANES)
        .min(files.max(1))
}

/// Every crate in the checkout that keeps its sources in a `src/` directory.
///
/// Read from the tree rather than the workspace roster, because the layout
/// rules judge tree shape and a crate outside the workspace grows the same
/// pairs and the same nameless modules: the external extension examples are
/// separate packages on purpose. A directory earns a place here by declaring
/// `[package]` and holding Rust source under `src/`, so a crate added anywhere
/// in the checkout is judged without an edit here. A `src/` emptied by a
/// deletion is not a crate root: the directory survives the pull that removed
/// every file in it. This crate is included; the member corpus exempts it only
/// because its registration fixtures name other crates, and a rule over file
/// names has no such fixtures.
///
/// Build outputs and hidden directories are skipped: a manifest under `target/`
/// is a vendored dependency cargo unpacked, not a crate of this checkout, and
/// walking `.git` costs one round trip per loose object to find no manifest at
/// all.
///
/// Whether a candidate holds source is a prefix range of [`tree_files`], so the
/// answer costs a binary search rather than a directory walk per manifest. The
/// roster is sorted afterwards so it does not depend on which lane finished
/// first.
///
/// # Panics
///
/// Resumes the panic of a walk lane that failed. A lane resolves manifests
/// against an already-walked tree and skips what it cannot parse, so a panic is
/// a defect here; a partial root set would silently exempt whole crates.
pub(crate) fn crate_source_roots(root: &Path) -> Vec<CrateRoot> {
    let walked = tree_files(root);
    let tree: &TreeFiles = &walked;
    let mut roots: Vec<CrateRoot> = std::thread::scope(|scope| {
        let lanes: Vec<_> = tree
            .manifests
            .iter()
            .map(|manifest| {
                scope.spawn(move || {
                    let directory = manifest.parent()?;
                    let ident = manifest_crate_ident(manifest)?;
                    tree.carries_rust_source(&directory.join("src"))
                        .then(|| CrateRoot {
                            directory: relative(root, directory),
                            ident,
                        })
                })
            })
            .collect();
        lanes
            .into_iter()
            .filter_map(|lane| {
                lane.join().expect("a crate root lane must not panic. Fix: repair the defect stated by the lane panic printed before this one")
            })
            .collect()
    });
    roots.sort_by(|left, right| left.directory.cmp(&right.directory));
    roots
}

/// Every manifest and every Rust source in the checkout, from one walk.
pub(crate) struct TreeFiles {
    /// Every `Cargo.toml`, sorted.
    pub(crate) manifests: Vec<PathBuf>,
    /// Every `.rs` file, sorted.
    rust_sources: Vec<PathBuf>,
}

impl TreeFiles {
    /// Every Rust source under `directory`, at any depth.
    ///
    /// A path sorts by components, so every descendant of `directory` is
    /// contiguous in the sorted roster and the range is two binary searches.
    pub(crate) fn rust_sources_under(&self, directory: &Path) -> &[PathBuf] {
        let from = self
            .rust_sources
            .partition_point(|path| path.as_path() < directory);
        let rest = &self.rust_sources[from..];
        let count = rest.partition_point(|path| path.starts_with(directory));
        &rest[..count]
    }

    /// Every Rust source in the checkout, sorted.
    pub(crate) fn rust_sources(&self) -> &[PathBuf] {
        &self.rust_sources
    }

    /// Whether any Rust source lives under `directory`, at any depth.
    pub(crate) fn carries_rust_source(&self, directory: &Path) -> bool {
        !self.rust_sources_under(directory).is_empty()
    }
}

/// The checkout's manifests and Rust sources, walked once per process.
///
/// Crate-root discovery, the source-file roster, and the per-crate source trees
/// each used to walk the tree for themselves, and every contract in a
/// consolidated test binary asked all three: over a network-mounted checkout
/// each walk is one round trip per directory, and the walks rather than the
/// reads are what stalled a run. One walk answers every one of them, and the
/// checked-in tree does not change while a test binary runs.
///
/// Build outputs and hidden directories are skipped: a manifest under `target/`
/// is a vendored dependency cargo unpacked, not a crate of this checkout, and
/// walking `.git` costs one round trip per loose object to find no manifest at
/// all.
pub(crate) fn tree_files(root: &Path) -> Arc<TreeFiles> {
    static WALKED: LazyLock<Mutex<BTreeMap<PathBuf, Arc<TreeFiles>>>> =
        LazyLock::new(|| Mutex::new(BTreeMap::new()));

    let mut cache = WALKED.lock().unwrap_or_else(PoisonError::into_inner);
    if let Some(found) = cache.get(root) {
        return Arc::clone(found);
    }
    let walked = Arc::new(walk_tree_files(root));
    #[cfg(test)]
    {
        *TREE_WALKS
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .entry(root.to_path_buf())
            .or_insert(0) += 1;
    }
    cache.insert(root.to_path_buf(), Arc::clone(&walked));
    walked
}

/// Times the tree walk ran, per root.
///
/// Sharing the walk is otherwise unobservable: a cache that stopped caching
/// returns the same roster, so every contract over the roster keeps passing
/// while the binary goes back to one full walk of the checkout per contract.
/// Counting per root keeps the count independent of whatever else in the binary
/// is walking another tree at the same time.
#[cfg(test)]
static TREE_WALKS: LazyLock<Mutex<BTreeMap<PathBuf, usize>>> =
    LazyLock::new(|| Mutex::new(BTreeMap::new()));

/// The walk, without the cache in front of it.
fn walk_tree_files(root: &Path) -> TreeFiles {
    let mut manifests = Vec::new();
    let mut rust_sources = Vec::new();
    for entry in WalkDir::new(root)
        .into_iter()
        .filter_entry(|entry| !crate::source_scan::is_pruned(entry))
        .filter_map(Result::ok)
        .filter(|entry| entry.file_type().is_file())
    {
        if entry.file_name() == "Cargo.toml" {
            manifests.push(entry.into_path());
        } else if entry.path().extension().is_some_and(|ext| ext == "rs") {
            rust_sources.push(entry.into_path());
        }
    }
    manifests.sort();
    rust_sources.sort();
    TreeFiles {
        manifests,
        rust_sources,
    }
}

/// The identifier a manifest's library carries, or `None` for no package.
///
/// `[lib] name` wins where it is written, because that is the name a consumer
/// and `cargo public-api` both use; the package name is the default Cargo
/// applies when it is not.
pub(crate) fn manifest_crate_ident(manifest: &Path) -> Option<String> {
    let text = read_source_bounded(manifest).ok()?;
    let table: toml::Table = toml::from_str(&text).ok()?;
    let value = Value::Table(table);
    let package = value.get("package")?.get("name")?.as_str()?;
    let name = value
        .get("lib")
        .and_then(|lib| lib.get("name"))
        .and_then(Value::as_str)
        .unwrap_or(package);
    Some(crate_ident(name))
}

pub(crate) fn relative(root: &Path, path: &Path) -> String {
    path.strip_prefix(root)
        .unwrap_or(path)
        .to_string_lossy()
        .replace('\\', "/")
}

/// Workspace root, resolved from the directory the gate was invoked in.
///
/// Never compiled in. A target directory shared by several checkouts computes the
/// same unit hash for all of them, so cargo hands one checkout a binary another
/// one built; a path baked into that binary then names the wrong tree, and
/// `VYRE_CHECKOUT_ROOT` did not prevent it, because cargo does not export a
/// `relative = true` config variable to the process it runs. The tree the
/// operator invoked cargo in is the tree the gate must answer for.
///
/// Memoized per working directory. A consolidated test binary runs every
/// contract against one checkout, so the ancestor scan is one walk rather than
/// one per contract. Over a network mount that scan is a round trip per
/// ancestor, and running it from a hundred threads at once exhausted the
/// descriptor limit of the invoking shell, which surfaced as every contract
/// failing on an unreadable working directory. Caching a single answer for the
/// process instead would be wrong: the root follows the directory the process
/// stands in, and a test that moves there to check that would be served the
/// checkout that happened to ask first.
///
/// # Panics
///
/// Panics when the working directory is unreadable, when no ancestor of it
/// declares a `[workspace]`, and when the memo lock is poisoned, which means an
/// earlier caller panicked partway through resolving a root.
#[must_use]
pub fn workspace_root() -> PathBuf {
    static ROOTS: LazyLock<Mutex<HashMap<PathBuf, PathBuf>>> =
        LazyLock::new(|| Mutex::new(HashMap::new()));
    const POISONED: &str =
        "Fix: a thread panicked while resolving the workspace root; report that panic.";

    let start = std::env::current_dir()
        .expect("Fix: the working directory must be readable to locate the vyre checkout");
    if let Some(root) = ROOTS.lock().expect(POISONED).get(&start) {
        return root.clone();
    }
    // Walked outside the lock: a cold miss must not serialize every other
    // caller behind one network round trip per ancestor.
    let root = workspace_root_from(&start).unwrap_or_else(|| {
        panic!(
            "Fix: run this from inside the vyre checkout; no ancestor of `{}` has a \
             Cargo.toml declaring [workspace].",
            start.display()
        )
    });
    ROOTS.lock().expect(POISONED).insert(start, root.clone());
    root
}

/// The nearest ancestor of `start`, inclusive, whose manifest declares a workspace.
#[must_use]
pub fn workspace_root_from(start: &Path) -> Option<PathBuf> {
    start
        .ancestors()
        .find(|directory| {
            read_source_bounded(&directory.join("Cargo.toml")).is_ok_and(|text| {
                text.lines()
                    .any(|line| line.trim_start().starts_with("[workspace]"))
            })
        })
        .map(Path::to_path_buf)
}

/// True when this source text submits an inventory registration.
///
/// Comments are skipped, so a doc comment explaining the linkage rule is not
/// mistaken for a registration. An invocation of a macro in `submitting_macros`
/// counts: the expansion holds the `inventory::submit!` the caller never wrote,
/// and a crate that only invokes one still needs its registrations linked.
/// [`crate::registration_macro::submitting_macros`] derives that set from the
/// tree.
#[must_use]
pub fn submits_registrations(text: &str, submitting_macros: &BTreeSet<String>) -> bool {
    if crate::registration_macro::writes_inventory_submit(text) {
        return true;
    }
    submitting_macros
        .iter()
        .any(|name| !crate::registration_macro::find_macro_invocations(text, name).is_empty())
}

/// Crate identifiers named by a discarding import, as written.
///
/// Only a bare crate identifier counts. `use std::io::Read as _;` imports a
/// trait into scope, which is the legitimate use of the form and references a
/// symbol at every call site.
#[must_use]
pub fn discarding_imports(text: &str) -> Vec<String> {
    let mut imports = Vec::new();
    for offset in code_offsets(text) {
        let rest = &text[offset..];
        if let Some(rest) = rest.strip_prefix("use ") {
            if let Some(crate_name) = discarded_crate(rest) {
                imports.push(crate_name);
            }
        }
    }
    imports
}

/// Crate identifier of a `use <crate> as _;` statement starting at `rest`.
fn discarded_crate(rest: &str) -> Option<String> {
    let mut tokens = rest.split_whitespace();
    let imported = tokens.next()?;
    let as_token = tokens.next()?;
    let underscore = tokens.next()?;
    if as_token != "as" {
        return None;
    }
    if !underscore.starts_with('_') {
        return None;
    }
    let without_semi = imported.strip_suffix(';').unwrap_or(imported);
    if without_semi.contains("::") {
        return None;
    }
    Some(without_semi.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The smallest default descriptor budget a supported host gives a process.
    ///
    /// macOS. Linux gives 1024. The cap is judged against the smaller one
    /// because a read that exhausts descriptors fails outright rather than
    /// running slowly.
    const SMALLEST_DESCRIPTOR_BUDGET: usize = 256;

    /// A read never asks for more descriptors than a process can spare.
    ///
    /// A lane is a thread holding a file open, and the harness, cargo, and every
    /// other reader in the binary draw on the same budget. An unbounded multiple
    /// of the core count turned a corpus read into `Too many open files` on a
    /// ten-core host and failed 59 contracts at once, so the bound is asserted
    /// rather than left to whatever host the suite lands on.
    #[test]
    fn a_read_asks_for_fewer_lanes_than_the_descriptor_budget() {
        // Four concurrent readers: the member corpus, the source corpus, a
        // consumer's own roster, and one more, none of which coordinate.
        assert!(
            MAX_READ_LANES * 4 <= SMALLEST_DESCRIPTOR_BUDGET,
            "Fix: {MAX_READ_LANES} lanes leaves no room for four concurrent readers inside the \
             {SMALLEST_DESCRIPTOR_BUDGET} descriptors a process starts with."
        );
        assert!(
            read_lanes(usize::MAX) <= MAX_READ_LANES,
            "Fix: a corpus of any size must still read under the cap."
        );
        assert_eq!(
            read_lanes(3),
            3,
            "Fix: a lane with no file to read is a thread spawned for nothing."
        );
        assert_eq!(
            read_lanes(0),
            1,
            "Fix: an empty corpus must still yield a usable lane count rather than zero, which \
             every chunking caller divides by."
        );
    }

    /// The checkout is walked once per process, however many contracts ask.
    #[test]
    fn a_second_request_for_the_tree_does_not_walk_it_again() {
        let root = workspace_root().join("structure-gate");

        let first = tree_files(&root);
        let second = tree_files(&root);

        assert!(
            !first.manifests.is_empty() && first.carries_rust_source(&root.join("src")),
            "Fix: the walk of {} found no manifest or no source, so sharing it proves nothing.",
            root.display()
        );
        assert!(
            Arc::ptr_eq(&first, &second),
            "Fix: the second request re-walked the tree."
        );
        assert_eq!(
            TREE_WALKS
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .get(&root)
                .copied(),
            Some(1),
            "Fix: the tree was walked more than once for one root. Crate-root discovery, the \
             source roster and the per-crate source trees all ask for it, so a walk per request is \
             a full pass over the checkout per request."
        );
    }

    /// The prefix range names the subtree and stops at its boundary.
    ///
    /// A range that ran past the subtree would hand a crate root the files of
    /// the crate that sorts after it, and the layout rules would report a module
    /// under the wrong owner. A range that stopped short would hand it nothing,
    /// which every per-file assertion passes.
    #[test]
    fn a_subtree_range_holds_exactly_the_sources_under_it() {
        let root = workspace_root();
        let tree = tree_files(&root);
        let directory = root.join("structure-gate/src");

        let under = tree.rust_sources_under(&directory);

        assert!(
            under
                .iter()
                .any(|path| path.ends_with("workspace_manifest.rs")),
            "Fix: the range under {} is missing a file that is there.",
            directory.display()
        );
        assert!(
            under.iter().all(|path| path.starts_with(&directory)),
            "Fix: the range under {} reached outside it, so a crate root would be handed another \
             crate's sources.",
            directory.display()
        );
        assert!(
            !tree.carries_rust_source(&root.join("structure-gate/src/nothing_declares_this")),
            "Fix: a directory holding no source answered that it does, so a `src/` emptied by a \
             deletion would still read as a crate root."
        );
    }

    /// The corpus is read once per process, not once per contract.
    ///
    /// Pointer identity is the only observable form of the contract: a second
    /// read returns equal text either way, so a cache that silently stopped
    /// caching would leave every assertion over the corpus passing while a
    /// consolidated test binary went back to one full sweep of the tree per
    /// contract. Over a network-mounted checkout that was the difference
    /// between a run and a stall.
    #[test]
    fn a_second_request_for_the_same_roster_shares_the_first_read() {
        let root = workspace_root();
        let members = workspace_members(&root);
        let first = member_sources(&root, &members);
        let second = member_sources(&root, &members);

        assert!(
            !first.is_empty(),
            "Fix: the corpus under {} is empty, so nothing was read and sharing it proves nothing.",
            root.display()
        );
        assert!(
            Arc::ptr_eq(&first, &second),
            "Fix: the second request re-read the corpus. Every contract in this binary sweeps it, \
             so a per-request read is one full pass over the tree per contract."
        );
    }

    /// A different roster is a different corpus.
    #[test]
    fn a_narrower_roster_is_not_served_the_wider_read() {
        let root = workspace_root();
        let members = workspace_members(&root);
        let narrowed = &members[..1];

        let wide = member_sources(&root, &members);
        let narrow = member_sources(&root, narrowed);

        assert!(
            narrow.len() < wide.len(),
            "Fix: a request for {narrowed:?} was served the whole workspace corpus, so the cache \
             key ignores the roster and a caller judging one member judges all of them."
        );
    }

    #[test]
    fn a_trait_import_is_not_a_discarding_crate_import() {
        let named = discarding_imports("fn read() {\n    use std::io::Read as _;\n}\n");

        assert!(named.is_empty(), "{named:?}");
    }

    #[test]
    fn a_crate_named_only_inside_a_comment_is_not_an_import() {
        let named = discarding_imports(
            "/// Naming the crate with `use vyre_libs as _;` references nothing.\npub fn anchor() {}\n",
        );

        assert!(named.is_empty(), "{named:?}");
    }

    #[test]
    fn a_discarding_crate_import_is_read_from_source() {
        let named = discarding_imports(
            "#[cfg(feature = \"gpu\")]\nuse vyre_driver_metal as _;\nuse vyre_libs as _;\n",
        );

        assert_eq!(named, vec!["vyre_driver_metal", "vyre_libs"]);
    }

    #[test]
    fn a_submission_inside_a_comment_does_not_make_a_crate_a_submitter() {
        let submitting: BTreeSet<String> = [
            "submit_hardware_intrinsic",
            "define_unary_u32_hardware_intrinsic",
        ]
        .into_iter()
        .map(str::to_string)
        .collect();

        assert!(!submits_registrations(
            "// This crate reads the registry; inventory::submit! lives in the driver.\n",
            &submitting
        ));
        assert!(submits_registrations(
            "inventory::submit! {\n    ExampleRegistration { id: \"example\" }\n}\n",
            &submitting
        ));
        assert!(submits_registrations(
            "submit_hardware_intrinsic! {\n    id: \"example\",\n}\n",
            &submitting
        ));
        assert!(submits_registrations(
            "define_unary_u32_hardware_intrinsic!(foo, \"example\", expr);\n",
            &submitting
        ));
        assert!(
            !submits_registrations(
                "define_unary_u32_hardware_intrinsic!(foo, \"example\", expr);\n",
                &BTreeSet::new()
            ),
            "Fix: a macro outside the derived set must not count, or every macro invocation in \
             the workspace reads as a registration submission."
        );
    }

    #[test]
    fn manifest_with_leading_comments_parses_workspace_members() {
        let temp_dir = tempfile::tempdir().expect("tempdir");
        let manifest_path = temp_dir.path().join("Cargo.toml");
        let manifest_content = r#"# Workspace members are listed explicitly.
# Comments at top of file.
[workspace]
resolver = "2"
members = [
    # Core compiler
    "vyre",
    "vyre-libs",
]
"#;
        std::fs::write(&manifest_path, manifest_content).expect("write manifest");
        let members = workspace_members(temp_dir.path());
        assert_eq!(members, vec!["vyre", "vyre-libs"]);
    }

    #[test]
    fn member_directory_resolves_package_from_manifest_with_comments() {
        let temp_dir = tempfile::tempdir().expect("tempdir");
        let root = temp_dir.path();
        std::fs::write(
            root.join("Cargo.toml"),
            "# Workspace root\n[workspace]\nmembers = [\"nested/pkg-a\"]\n",
        )
        .expect("write root Cargo.toml");
        let pkg_dir = root.join("nested").join("pkg-a");
        std::fs::create_dir_all(&pkg_dir).expect("create pkg dir");
        std::fs::write(
            pkg_dir.join("Cargo.toml"),
            "# Package manifest\n[package]\nname = \"pkg-a\"\nversion = \"0.1.0\"\n",
        )
        .expect("write member Cargo.toml");

        let found = member_directory(root, "pkg-a");
        assert_eq!(found, pkg_dir);
    }
}
