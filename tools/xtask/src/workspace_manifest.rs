//! Manifest parsing, crate identification, and workspace path resolution.
//!
//! Every gate that resolves a name against the tree or judges source layout
//! relies on root-relative directory resolution and member manifests. This
//! module reads workspace manifests, locates member and crate source roots,
//! and scans discarding imports and inventory registration submitters.

use std::collections::{BTreeMap, HashMap};
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::{Arc, LazyLock, Mutex, PoisonError};

use serde::Deserialize;
use walkdir::WalkDir;

/// The most bytes one tracked source or manifest may hold before a reader
/// refuses it.
///
/// The largest tracked Rust source is under 100 KiB, so a file over the cap is
/// generated output or a binary that reached the walk, not a large source.
/// Reading it in full would let one file decide how much memory a tree scan
/// takes, and truncating it would let a rule judge part of a file and report
/// the rest clean.
pub const MAX_SOURCE_BYTES: u64 = 4 * 1024 * 1024;

/// Read one source or manifest whole, or fail naming why.
///
/// Every gate that reads a tracked Rust source or `Cargo.toml` reads it here,
/// so one cap holds for all of them. A file that cannot be opened, is not
/// UTF-8, or holds more than [`MAX_SOURCE_BYTES`] is an error with the path in
/// it, never a shorter text and never a skip.
///
/// # Errors
///
/// The open or read failure, or `InvalidData` when the file exceeds the cap.
pub fn read_source_bounded(path: &Path) -> std::io::Result<String> {
    use std::io::Read as _;

    let file = fs::File::open(path).map_err(|error| {
        std::io::Error::new(
            error.kind(),
            format!("cannot open `{}`: {error}", path.display()),
        )
    })?;
    let length = file.metadata().map_or(0, |metadata| metadata.len());
    let mut text = String::with_capacity(length.min(MAX_SOURCE_BYTES) as usize);
    file.take(MAX_SOURCE_BYTES + 1)
        .read_to_string(&mut text)
        .map_err(|error| {
            std::io::Error::new(
                error.kind(),
                format!("cannot read `{}` as UTF-8: {error}", path.display()),
            )
        })?;
    if text.len() as u64 > MAX_SOURCE_BYTES {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!(
                "`{}` holds more than {MAX_SOURCE_BYTES} bytes, which is generated output or a binary rather than a source",
                path.display()
            ),
        ));
    }
    Ok(text)
}

/// The directory of a workspace member, by package name.
///
/// A member's directory is not its package name: `vyre-conform` is at
/// `conform/vyre-conform`. The roster is read from the root manifest at run
/// time, so a gate that needs its own crate directory gets it without a
/// compiled-in manifest path, which would name whichever checkout built the
/// binary.
///
/// # Panics
/// Panics when no member directory's manifest declares `package`.
#[must_use]
pub fn member_directory(root: &Path, package: &str) -> PathBuf {
    find_member_directory(root, package).unwrap_or_else(|| {
        panic!(
            "Fix: no workspace member under {} declares package `{package}`; the roster in the \
             root Cargo.toml is what this resolves against.",
            root.display()
        )
    })
}

/// [`member_directory`], or `None` when no member declares `package`.
#[must_use]
pub fn find_member_directory(root: &Path, package: &str) -> Option<PathBuf> {
    member_directories(root).remove(package)
}

/// Every workspace member's directory, keyed by the package name its manifest
/// declares. A caller resolving several names reads the roster once through
/// this instead of once per name.
///
/// A member whose manifest is unreadable or declares no package is absent.
///
/// # Panics
///
/// Panics when the root manifest cannot be read or parsed.
#[must_use]
pub fn member_directories(root: &Path) -> BTreeMap<String, PathBuf> {
    #[derive(Deserialize)]
    struct PackageManifest {
        package: Option<PackageName>,
    }
    #[derive(Deserialize)]
    struct PackageName {
        name: String,
    }
    workspace_members(root)
        .into_iter()
        .filter_map(|member| {
            let member_dir = root.join(&member);
            let text = read_source_bounded(&member_dir.join("Cargo.toml")).ok()?;
            let name = toml::from_str::<PackageManifest>(&text).ok()?.package?.name;
            Some((name, member_dir))
        })
        .collect()
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
#[derive(Deserialize)]
struct RootWorkspaceManifest {
    workspace: Option<WorkspaceSection>,
}

#[derive(Deserialize)]
struct WorkspaceSection {
    members: Option<Vec<String>>,
    exclude: Option<Vec<String>>,
}

/// The `workspace.<key>` path list from the root manifest.
///
/// # Panics
///
/// Panics when the root manifest cannot be read or parsed. An empty list would
/// report a workspace with no members as a workspace with nothing to lint, so
/// every structural gate would pass on an unreadable tree.
pub(crate) fn workspace_paths(root: &Path, key: &str) -> Vec<String> {
    let manifest_path = root.join("Cargo.toml");
    let text = read_source_bounded(&manifest_path)
        .unwrap_or_else(|error| panic!("Fix: cannot read {}: {error}", manifest_path.display()));
    let manifest: RootWorkspaceManifest = toml::from_str(&text)
        .unwrap_or_else(|error| panic!("Fix: parse {}: {error}", manifest_path.display()));
    let ws = manifest.workspace.unwrap_or(WorkspaceSection {
        members: None,
        exclude: None,
    });
    match key {
        "members" => ws.members.unwrap_or_default(),
        "exclude" => ws.exclude.unwrap_or_default(),
        _ => Vec::new(),
    }
}

/// Every checkout directory that holds a `Cargo.toml`, root-relative.
///
/// The root manifest declares which of these cargo loads and which it leaves
/// out. A directory on neither list is a crate the workspace never decided
/// about: cargo pulls it into the graph or refuses to build it depending on
/// where it sits, and nothing records which was meant.
///
/// # Panics
///
/// Panics when a manifest path is not under `root`.
#[must_use]
pub fn manifest_directories(root: &Path) -> Vec<String> {
    tree_files(root)
        .manifests
        .iter()
        .filter_map(|manifest| manifest.parent())
        .filter(|directory| *directory != root)
        .map(|directory| {
            let relative = directory.strip_prefix(root).unwrap_or_else(|_| {
                panic!(
                    "Fix: manifest directory {} is not under {}",
                    directory.display(),
                    root.display()
                )
            });
            relative.to_string_lossy().replace('\\', "/")
        })
        .collect()
}

/// Crate identifier for a crate name, e.g. `vyre_libs` for `vyre-libs`.
#[must_use]
pub fn crate_ident(crate_name: &str) -> String {
    crate_name.replace('-', "_")
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

/// Sorted checkout-relative `.rs` paths across all workspace members.
#[must_use]
pub fn source_file_roster(root: &Path) -> Vec<String> {
    let walked = tree_files(root);
    let members = workspace_members(root);
    let mut out = Vec::new();
    for member in &members {
        for path in walked.rust_sources_under(&root.join(member)) {
            out.push(relative(root, path));
        }
    }
    out.sort();
    out.dedup();
    out
}
