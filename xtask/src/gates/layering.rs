//! Whether the resolved crate graph stays inside the layering the ownership
//! registry declares.
//!
//! Three rules over one graph. A member may reach another member only if its
//! `docs/CRATE_OWNERSHIP.toml` entry allows it, directly or through a declared
//! edge. A member in a substrate-neutral layer may not reach a backend API crate
//! at all, whatever the intermediate was, and may not name a concrete backend,
//! vendor or dialect in its own production sources: a neutral crate that
//! describes its work in one vendor's words is where a rule meant for every
//! backend ends up written for one, which is the drift the third rule reports
//! before the code follows the prose.
//!
//! The graph comes from the manifests and the lockfile, never from cargo. The
//! shell form ran `cargo tree` once per member and once more per violation, so a
//! workspace that did not resolve produced no verdict, and the `--edges=normal`
//! trees were the only statement of which edges counted. Manifest edges are read
//! with the member's own default features activated, which is the same edge set
//! `cargo tree` prints, and third-party edges come from `Cargo.lock`, so a
//! neutral crate that reaches a backend API through another third-party crate is
//! still caught.

use std::collections::{BTreeMap, BTreeSet};

use structure_gate::backend_vocabulary::segments_of;

use crate::gate::{Finding, Gate, GateCtx, GateError, Report};
use crate::gates::manifest_contract::{dep_lines, dependency_hosts, entries, target_package};
use crate::gates::scan::{self, Tree};

/// Production dependency tables. Development edges are excluded because a test
/// may depend upward deliberately, which is the same allowance `check-tier-deps`
/// makes.
const PRODUCTION_TABLES: &[&str] = &["dependencies", "build-dependencies"];

/// Crates that must name no concrete backend or driver product in a production
/// dependency table.
///
/// This is the direct-edge half of the layering contract, kept beside the
/// transitive half. The two answer different questions: the closure rule asks
/// whether an edge is declared, this one asks whether a named crate has an edge
/// at all, and a crate can satisfy the first while carrying the second.
///
/// `vyre-runtime` is here rather than in [`FORBIDDEN_DEPENDENCIES`], where it
/// stated a boundary the architecture does not have: its own manifest names no
/// backend, so the facade depending on it drags in no substrate, and admitting an
/// artifact and submitting resident work is the product path a consumer reaches
/// through the facade. What is worth holding is that the runtime stays neutral,
/// which is the rule it is now under.
const NEUTRAL_CRATES: &[&str] = &[
    "vyre",
    "vyre-driver",
    "vyre-foundation",
    "vyre-primitives",
    "vyre-reference",
    "vyre-runtime",
    "vyre-spec",
];

/// Packages a neutral crate must not depend on outside `[dev-dependencies]`.
const FORBIDDEN_DEPENDENCIES: &[&str] = &[
    "naga",
    "vyre-aot",
    "vyre-driver-cuda",
    "vyre-driver-spirv",
    "vyre-driver-wgpu",
    "wgpu",
];

/// Crate names that no manifest may declare, because the crates they name are
/// gone and a manifest citing one describes an architecture the tree does not
/// have.
const RETIRED_CRATES: &[&str] = &["vyre-ir", "vyre-wgpu"];

/// Whether each layer is substrate-neutral.
///
/// Every layer a member declares needs a decision here. A member whose layer is
/// missing is an unreviewed crate, and a decision no member uses is an allowance
/// nothing needs; both are fatal rather than reported, because either one makes
/// the neutrality half of the gate answer for a roster nobody checked.
const NEUTRAL_LAYERS: &[(&str, bool)] = &[
    ("backend-neutral", true),
    ("compiler-boundary", true),
    ("concrete-backend", false),
    ("conformance", false),
    ("emitter", false),
    ("facade", true),
    ("foundation", true),
    ("libraries", true),
    ("lowering", true),
    ("packaging", true),
    // Optimizer passes expressed as Vyre programs, dispatched through the
    // `ProgramDispatcher` seam, so the crate names no backend API.
    ("pass-engine", true),
    ("primitives", true),
    // Substrate-bound by function, not by accident: the link crate must name
    // every source whose registrations a build links, and the concrete drivers
    // are sources, so it reaches each backend API through them.
    ("registry-link", false),
    ("runtime", true),
    ("semantics", true),
    // `structure-gate` depends on no vyre crate, so it keeps running while the
    // workspace does not compile. Nothing it reads is substrate-bound.
    ("standalone-tooling", true),
    ("test-tooling", false),
    ("tooling", false),
];

/// Third-party crates that are the substrate boundary. A neutral crate reaching
/// one of these has crossed it whatever the intermediate was.
const BACKEND_APIS: &[&str] = &["ash", "cudarc", "metal", "naga", "wgpu"];

/// Concrete backend, vendor and dialect names. A crate in a substrate-neutral
/// layer names the neutral concept instead: primary text, primary binary,
/// secondary text, native module, backend, target, device, artifact.
///
/// Matched case-insensitively and only where the hit is a whole word, so
/// `cudarc` is not `CUDA` and `hash` is not `ash`. Every spelling of a workspace
/// member name is masked out first, because a crate that must name
/// `vyre-driver-wgpu` is naming a package rather than describing its own work in
/// one substrate's words.
const BACKEND_WORDS: &[&str] = &[
    "cubin", "CUDA", "cudarc", "GLSL", "HLSL", "Metal", "MSL", "naga", "NVIDIA", "NVRTC", "NVVM",
    "OpenCL", "PTX", "ptxas", "SPIR-V", "SPIRV", "Vulkan", "WGPU", "WGSL",
];

/// Substrate-neutral layers whose crates may still name a concrete backend, and
/// the reason each may.
///
/// A crate whose job is to police the backends names every one of them: a roster
/// it may not write is a roster it cannot check. Every other neutral layer
/// describes work that must read the same for every target, so this list stays
/// short and each row carries why. A row naming a layer no member declares is
/// fatal, because an exemption nothing uses records a rule that stopped covering
/// anything.
const VOCABULARY_EXEMPT_LAYERS: &[(&str, &str)] = &[(
    "standalone-tooling",
    "a tooling crate names the backends its own rules police",
)];

/// Directory prefix, word, and reason for a backend word that identifies an
/// external interface instead of describing the crate's own work.
///
/// A name the kernel exports cannot be restated in neutral words: the probe
/// opens that exact path, and a rename would make it read the wrong file or
/// nothing. Every other site states the neutral concept, so each row here
/// carries the reason it is not one of them.
const INTERFACE_NAMES: &[(&str, &str, &str)] = &[(
    "vyre-runtime/src/uring/",
    "nvidia-fs",
    "the Linux kernel module, and the /proc path it exports, that the GPUDirect probe reads",
)];

/// Every internal edge stays inside its declared closure, and no substrate-neutral
/// crate reaches a backend API.
pub struct Layering;

impl Gate for Layering {
    fn name(&self) -> &'static str {
        "layering"
    }

    fn help(&self) -> &'static str {
        "hold every member inside the dependency closure its ownership entry declares, and keep substrate-neutral layers away from backend API crates and from concrete backend vocabulary"
    }

    fn run(&self, ctx: &GateCtx) -> Result<Report, GateError> {
        let tree = Tree::open(&ctx.root)?;
        let graph = Graph::read(&tree)?;
        let registry = Registry::read(&tree, &graph.members)?;
        let mut report = Report::clean();
        let mut edges = 0usize;
        let mut scanned = 0usize;

        for member in &graph.members {
            let allowed = registry.closure(member);
            let internal = graph.reachable_members(member);
            edges += internal.len();
            for reached in &internal {
                if allowed.contains(reached) {
                    continue;
                }
                report.find(Finding::in_file(
                    format!("{}/Cargo.toml", graph.directory(member)),
                    format!(
                        "`{member}` reaches `{reached}`, which its ownership entry does not \
                         allow directly or through a declared edge: {}",
                        graph.path_to(member, reached)
                    ),
                    "remove the edge, or declare it in the docs/CRATE_OWNERSHIP.toml entry \
                     for the crate that owns it and regenerate the ownership docs",
                ));
            }
            if !registry.neutral(member) {
                continue;
            }
            for api in graph.reachable_backend_apis(member) {
                report.find(Finding::in_file(
                    format!("{}/Cargo.toml", graph.directory(member)),
                    format!(
                        "`{member}` is in the substrate-neutral layer `{}` and reaches the \
                         backend API crate `{api}`: {}",
                        registry.layer(member),
                        graph.path_to(member, &api)
                    ),
                    "move the code that needs the backend API into the crate that owns that \
                     backend, or move this crate out of the neutral layer",
                ));
            }
            if exempt_from_vocabulary(registry.layer(member)) {
                continue;
            }
            for (file, line, words) in backend_vocabulary(&tree, graph.directory(member))? {
                scanned += 1;
                report.find(Finding::at(
                    file,
                    line,
                    format!(
                        "`{member}` is in the substrate-neutral layer `{}` and names {words} in \
                         production source",
                        registry.layer(member)
                    ),
                    "state the neutral concept the rule names (primary text, primary binary, \
                     secondary text, native module, backend, target, device, artifact), or move \
                     the code into the crate that owns that backend when the concrete detail is \
                     load-bearing",
                ));
            }
        }

        report.note(format!(
            "{edges} resolved internal edge(s) across {} workspace member(s)",
            graph.members.len()
        ));
        report.note(format!(
            "{} member(s) in a substrate-neutral layer, checked against {} and {} backend word(s)",
            graph
                .members
                .iter()
                .filter(|member| registry.neutral(member))
                .count(),
            BACKEND_APIS.join(", "),
            BACKEND_WORDS.len()
        ));
        if scanned != 0 {
            report.note(format!("{scanned} line(s) name a backend word"));
        }
        for (layer, reason) in VOCABULARY_EXEMPT_LAYERS {
            report.note(format!(
                "the `{layer}` layer is excused from the vocabulary rule: {reason}"
            ));
        }
        for (prefix, name, reason) in INTERFACE_NAMES {
            report.note(format!(
                "`{name}` under {prefix} is read as an interface name rather than vocabulary: \
                 {reason}"
            ));
        }
        Ok(report)
    }
}

/// Whether `layer` is excused from the vocabulary rule by
/// [`VOCABULARY_EXEMPT_LAYERS`].
fn exempt_from_vocabulary(layer: &str) -> bool {
    VOCABULARY_EXEMPT_LAYERS
        .iter()
        .any(|(exempt, _)| *exempt == layer)
}

/// Backend words in the production source under `directory`, one entry per line.
///
/// Test code is excluded twice over: a file the tree reaches only as test support
/// is skipped by path, and a line inside a `#[cfg(test)]` item is skipped by the
/// same reader the hot-path scan uses. A backend word in a test is the test
/// naming the backend it drives, which is what a backend test is for.
fn backend_vocabulary(
    tree: &Tree,
    directory: &str,
) -> Result<Vec<(String, u32, String)>, GateError> {
    let prefix = format!("{directory}/src/");
    let mut found = Vec::new();
    for path in tree.paths() {
        let Some(relative) = path.to_str() else {
            continue;
        };
        if !relative.starts_with(&prefix) || !relative.ends_with(".rs") || is_test_source(relative)
        {
            continue;
        }
        let text = tree.read(relative)?;
        let lines: Vec<&str> = text.lines().collect();
        let test_only = scan::cfg_test_lines(&lines);
        for (number, line) in scan::numbered(&text) {
            let index = usize::try_from(number)
                .unwrap_or(usize::MAX)
                .saturating_sub(1);
            if test_only.get(index).copied().unwrap_or(false) {
                continue;
            }
            let words = words_in(&mask_interface_names(line, relative));
            if !words.is_empty() {
                found.push((relative.to_string(), number, words.join(", ")));
            }
        }
    }
    Ok(found)
}

/// Whether the tree reaches `relative` only as test support.
///
/// A `tests` directory or a `tests.rs` module is test code whatever declared it,
/// and the `#[cfg(test)]` attribute that gates it sits in the parent file rather
/// than in the file being read, so the line reader cannot see it from here.
fn is_test_source(relative: &str) -> bool {
    relative
        .split('/')
        .any(|part| part == "tests" || part == "tests.rs")
}

/// `line` with every interface name allowed for its directory blanked to spaces
/// of the same width.
///
/// Blanked rather than removed so a reported column still maps to the source, and
/// so a blanked name cannot join its neighbours into a word that was never there.
fn mask_interface_names(line: &str, relative: &str) -> String {
    let mut masked = line.to_string();
    for (_, name, _) in INTERFACE_NAMES
        .iter()
        .filter(|(prefix, _, _)| relative.starts_with(prefix))
    {
        while let Some(at) = masked.find(name) {
            masked.replace_range(at..at + name.len(), &" ".repeat(name.len()));
        }
    }
    masked
}

/// The backend words `line` names, in [`BACKEND_WORDS`] order, without repeats.
///
/// Compared segment by segment rather than by substring. A name is a run of
/// identifier segments, split on every non-alphanumeric byte and at camel-case
/// boundaries, so `CudaDevice` names `CUDA`, `barracuda` does not, and a word
/// spelled with a separator matches the run its own spelling splits into.
/// Substring matching would need an allowance for every unrelated identifier that
/// happens to carry a vendor's letters, and camel case is where a backend type
/// name hides from a whole-word rule.
fn words_in(line: &str) -> Vec<String> {
    let segments = segments_of(line);
    let mut found = Vec::new();
    for word in BACKEND_WORDS {
        let wanted = segments_of(word);
        if !wanted.is_empty()
            && segments
                .windows(wanted.len())
                .any(|run| run == wanted.as_slice())
        {
            found.push(format!("`{word}`"));
        }
    }
    found
}

/// No named neutral crate carries a production edge to a backend, driver product
/// or runtime, and no manifest names a retired crate.
pub struct NeutralCrates;

impl Gate for NeutralCrates {
    fn name(&self) -> &'static str {
        "neutral-crates"
    }

    fn help(&self) -> &'static str {
        "keep the named substrate-neutral crates free of production edges to backends, driver products and the runtime, and keep retired crate names out of every manifest"
    }

    fn run(&self, ctx: &GateCtx) -> Result<Report, GateError> {
        let tree = Tree::open(&ctx.root)?;
        let root = tree.read_toml("Cargo.toml")?;
        let workspace_deps = root
            .get("workspace")
            .and_then(toml::Value::as_table)
            .and_then(|workspace| workspace.get("dependencies"))
            .and_then(toml::Value::as_table)
            .cloned()
            .unwrap_or_default();
        let mut report = Report::clean();
        let mut optional = Vec::new();

        for crate_name in NEUTRAL_CRATES {
            let manifest = format!("{crate_name}/Cargo.toml");
            if !tree.has(&manifest) {
                return Err(GateError::new(
                    format!("neutral crate `{crate_name}` has no manifest at {manifest}"),
                    "point the rule at the directory the crate moved to, or drop the name; a \
                     neutrality rule over a manifest that does not exist reports success \
                     forever",
                ));
            }
            let text = tree.read(&manifest)?;
            let table = toml::from_str::<toml::Table>(&text).map_err(|error| {
                GateError::new(
                    format!("{manifest} is not readable as TOML: {error}"),
                    "repair the manifest",
                )
            })?;
            let lines = dep_lines(&text);
            for (prefix, host) in dependency_hosts(&table) {
                for section in PRODUCTION_TABLES {
                    for (key, spec) in entries(&host, section) {
                        let package = target_package(&key, &spec, &workspace_deps);
                        if !FORBIDDEN_DEPENDENCIES.contains(&package.as_str()) {
                            continue;
                        }
                        let table_name = format!("{prefix}{section}");
                        if spec
                            .get("optional")
                            .and_then(toml::Value::as_bool)
                            .unwrap_or(false)
                        {
                            optional.push(format!("{crate_name} -> {package} ({table_name})"));
                            continue;
                        }
                        let message = format!(
                            "`{crate_name}` depends on `{package}` in [{table_name}], which a \
                             substrate-neutral crate may not name outside [dev-dependencies]"
                        );
                        let fix = "move the dependency under [dev-dependencies], move the \
                                   production code that needs it into the crate that owns the \
                                   backend, or take the crate out of NEUTRAL_CRATES in \
                                   xtask/src/gates/layering.rs once an owner decides it is no \
                                   longer neutral";
                        report.find(
                            match lines.get(&(table_name.clone(), key.clone())).copied() {
                                Some(line) => Finding::at(&manifest, line, message, fix),
                                None => Finding::in_file(&manifest, message, fix),
                            },
                        );
                    }
                }
            }
        }

        for manifest in tree
            .paths()
            .iter()
            .filter(|path| path.file_name().and_then(|name| name.to_str()) == Some("Cargo.toml"))
        {
            let text = tree.read(manifest)?;
            for (number, line) in crate::gates::scan::numbered(&text) {
                let key = line
                    .trim()
                    .trim_start_matches('"')
                    .split(['.', ' ', '=', '"'])
                    .next()
                    .unwrap_or_default();
                if !line.contains('=') || !RETIRED_CRATES.contains(&key) {
                    continue;
                }
                report.find(Finding::at(
                    manifest.clone(),
                    number,
                    format!("manifest names the retired crate `{key}`"),
                    "delete the entry, or rename it to the crate that took the name's place; \
                     a manifest citing a crate the workspace does not have describes an \
                     architecture nobody can build",
                ));
            }
        }

        report.note(format!(
            "{} neutral crate(s) checked against {}",
            NEUTRAL_CRATES.len(),
            FORBIDDEN_DEPENDENCIES.join(", ")
        ));
        if !optional.is_empty() {
            report.note(format!(
                "optional edge(s) permitted by the rule, activated only by a feature: {}",
                optional.join(", ")
            ));
        }
        Ok(report)
    }
}

/// The resolved dependency graph: member edges from the manifests, third-party
/// edges from the lockfile.
struct Graph {
    /// Member package names, in manifest order.
    members: BTreeSet<String>,
    /// Repository-relative directory per member package.
    directories: BTreeMap<String, String>,
    /// Direct production edges of each member, with default features activated.
    edges: BTreeMap<String, BTreeSet<String>>,
    /// Direct edges of each locked package, which is every crate the build can
    /// reach including the ones no manifest here names.
    locked: BTreeMap<String, BTreeSet<String>>,
}

impl Graph {
    /// Read the graph from the root manifest, the member manifests and the lockfile.
    fn read(tree: &Tree) -> Result<Self, GateError> {
        let root = tree.read_toml("Cargo.toml")?;
        let workspace = root
            .get("workspace")
            .and_then(toml::Value::as_table)
            .ok_or_else(|| {
                GateError::new(
                    "the root Cargo.toml declares no [workspace] table",
                    "restore the workspace table; a layering scan over no members reports \
                     success forever",
                )
            })?;
        let workspace_deps = workspace
            .get("dependencies")
            .and_then(toml::Value::as_table)
            .cloned()
            .unwrap_or_default();
        for api in BACKEND_APIS {
            if !workspace_deps.contains_key(*api) {
                return Err(GateError::new(
                    format!("backend API crate `{api}` is not in [workspace.dependencies]"),
                    "name the crate in [workspace.dependencies] or drop it from BACKEND_APIS \
                     in xtask/src/gates/layering.rs; a boundary named after a crate the \
                     workspace never resolves cannot be crossed",
                ));
            }
        }

        let mut members = BTreeSet::new();
        let mut directories = BTreeMap::new();
        let mut edges = BTreeMap::new();
        let listed = tree.members()?;
        if listed.is_empty() {
            return Err(GateError::new(
                "the root manifest declares no workspace members",
                "declare workspace.members; a layering scan over an empty roster reports \
                 success forever",
            ));
        }
        for directory in listed {
            let manifest = tree.read_toml(format!("{directory}/Cargo.toml"))?;
            let name = manifest
                .get("package")
                .and_then(toml::Value::as_table)
                .and_then(|package| package.get("name"))
                .and_then(toml::Value::as_str)
                .ok_or_else(|| {
                    GateError::new(
                        format!("{directory}/Cargo.toml declares no [package] name"),
                        "give the member a package name, or remove it from workspace.members",
                    )
                })?
                .to_string();
            edges.insert(name.clone(), direct_edges(&manifest, &workspace_deps));
            directories.insert(name.clone(), directory);
            members.insert(name);
        }

        Ok(Self {
            members,
            directories,
            edges,
            locked: locked_edges(&tree.read_toml("Cargo.lock")?),
        })
    }

    /// The directory a member's manifest sits in.
    fn directory(&self, member: &str) -> &str {
        self.directories
            .get(member)
            .map_or("Cargo.toml", String::as_str)
    }

    /// Every member the given member reaches through production edges, itself
    /// excluded.
    fn reachable_members(&self, member: &str) -> BTreeSet<String> {
        let mut reached = self.reach(member, false);
        reached.remove(member);
        reached.retain(|name| self.members.contains(name));
        reached
    }

    /// Every backend API crate the given member reaches, through members or
    /// through third-party crates.
    fn reachable_backend_apis(&self, member: &str) -> BTreeSet<String> {
        let reached = self.reach(member, true);
        BACKEND_APIS
            .iter()
            .filter(|api| reached.contains(**api))
            .map(|api| (*api).to_string())
            .collect()
    }

    /// Transitive closure over member edges, optionally following third-party
    /// edges out of the lockfile as well.
    fn reach(&self, member: &str, third_party: bool) -> BTreeSet<String> {
        let mut reached = BTreeSet::new();
        let mut pending: Vec<String> = self.next(member, third_party);
        while let Some(current) = pending.pop() {
            if !reached.insert(current.clone()) {
                continue;
            }
            pending.extend(self.next(&current, third_party));
        }
        reached
    }

    /// One node's outgoing edges.
    fn next(&self, package: &str, third_party: bool) -> Vec<String> {
        if let Some(edges) = self.edges.get(package) {
            return edges.iter().cloned().collect();
        }
        if !third_party {
            return Vec::new();
        }
        self.locked
            .get(package)
            .map(|edges| edges.iter().cloned().collect())
            .unwrap_or_default()
    }

    /// The shortest edge chain from one package to another, rendered for a
    /// reader who has to remove one of its edges.
    ///
    /// A finding that names only the endpoints leaves the reader to rediscover
    /// the intermediate, which is the whole cost of a transitive violation.
    fn path_to(&self, from: &str, to: &str) -> String {
        let mut previous: BTreeMap<String, String> = BTreeMap::new();
        let mut queue: Vec<String> = vec![from.to_string()];
        let mut seen: BTreeSet<String> = [from.to_string()].into_iter().collect();
        let mut at = 0usize;
        while at < queue.len() {
            let current = queue[at].clone();
            at += 1;
            for next in self.next(&current, true) {
                if !seen.insert(next.clone()) {
                    continue;
                }
                previous.insert(next.clone(), current.clone());
                if next == to {
                    let mut chain = vec![to.to_string()];
                    let mut step = to.to_string();
                    while let Some(before) = previous.get(&step) {
                        chain.push(before.clone());
                        step = before.clone();
                    }
                    chain.reverse();
                    return chain.join(" -> ");
                }
                queue.push(next);
            }
        }
        format!("{from} -> ... -> {to}")
    }
}

/// The layering the ownership registry declares.
struct Registry {
    /// Directly declared edges per package.
    declared: BTreeMap<String, BTreeSet<String>>,
    /// Declared layer per package.
    layers: BTreeMap<String, String>,
    /// Neutrality decision per layer name.
    neutrality: BTreeMap<String, bool>,
}

impl Registry {
    /// Read the registry and hold it against the member roster.
    fn read(tree: &Tree, members: &BTreeSet<String>) -> Result<Self, GateError> {
        let table = tree.read_toml("docs/CRATE_OWNERSHIP.toml")?;
        let crates = table
            .get("crate")
            .and_then(toml::Value::as_array)
            .filter(|entries| !entries.is_empty())
            .ok_or_else(|| {
                GateError::new(
                    "docs/CRATE_OWNERSHIP.toml declares no [[crate]] entries",
                    "record each crate's layer and allowed internal edges; a closure read \
                     from an empty registry allows nothing and reports nothing",
                )
            })?;
        let mut declared = BTreeMap::new();
        let mut layers = BTreeMap::new();
        for entry in crates {
            let Some(package) = entry.get("package").and_then(toml::Value::as_str) else {
                return Err(GateError::new(
                    "a docs/CRATE_OWNERSHIP.toml [[crate]] entry declares no package",
                    "name the package the entry describes",
                ));
            };
            let edges: BTreeSet<String> = entry
                .get("dependency")
                .and_then(toml::Value::as_array)
                .map(|list| {
                    list.iter()
                        .filter_map(|dependency| dependency.get("package"))
                        .filter_map(toml::Value::as_str)
                        .map(str::to_string)
                        .collect()
                })
                .unwrap_or_default();
            declared.insert(package.to_string(), edges);
            layers.insert(
                package.to_string(),
                entry
                    .get("layer")
                    .and_then(toml::Value::as_str)
                    .unwrap_or_default()
                    .to_string(),
            );
        }

        let unregistered: Vec<&String> = members
            .iter()
            .filter(|member| !declared.contains_key(*member))
            .collect();
        if !unregistered.is_empty() {
            let names: Vec<&str> = unregistered.iter().map(|name| name.as_str()).collect();
            return Err(GateError::new(
                format!(
                    "workspace member(s) with no docs/CRATE_OWNERSHIP.toml entry: {}",
                    names.join(", ")
                ),
                "record the crate's layer and allowed internal edges in the registry, then \
                 run `python3 scripts/crate_ownership.py --write`; an unregistered crate has \
                 an empty closure, so every edge it has would report at once",
            ));
        }

        let neutrality: BTreeMap<String, bool> = NEUTRAL_LAYERS
            .iter()
            .map(|(layer, neutral)| ((*layer).to_string(), *neutral))
            .collect();
        let used: BTreeSet<&str> = members
            .iter()
            .map(|member| layers.get(member).map_or("", String::as_str))
            .collect();
        let undecided: Vec<&str> = used
            .iter()
            .filter(|layer| !neutrality.contains_key(**layer))
            .copied()
            .collect();
        if !undecided.is_empty() {
            return Err(GateError::new(
                format!(
                    "layer(s) a member declares with no neutrality decision: {}",
                    undecided.join(", ")
                ),
                "record whether the layer is substrate-neutral in NEUTRAL_LAYERS in \
                 xtask/src/gates/layering.rs; a layer with no decision would be skipped",
            ));
        }
        let stale: Vec<&str> = neutrality
            .keys()
            .map(String::as_str)
            .filter(|layer| !used.contains(*layer))
            .collect();
        if !stale.is_empty() {
            return Err(GateError::new(
                format!(
                    "neutrality decision(s) no member uses: {}",
                    stale.join(", ")
                ),
                "delete the entry from NEUTRAL_LAYERS in xtask/src/gates/layering.rs; a \
                 decision for a layer nobody declares records a rule that stopped covering \
                 anything",
            ));
        }
        let vacant: Vec<&str> = VOCABULARY_EXEMPT_LAYERS
            .iter()
            .map(|(layer, _)| *layer)
            .filter(|layer| {
                !used.contains(*layer) || !neutrality.get(*layer).copied().unwrap_or(false)
            })
            .collect();
        if !vacant.is_empty() {
            return Err(GateError::new(
                format!(
                    "vocabulary exemption(s) for a layer no member declares as substrate-neutral: {}",
                    vacant.join(", ")
                ),
                "delete the entry from VOCABULARY_EXEMPT_LAYERS in xtask/src/gates/layering.rs; \
                 the vocabulary rule only reaches neutral layers, so an exemption outside them \
                 excuses nothing",
            ));
        }

        Ok(Self {
            declared,
            layers,
            neutrality,
        })
    }

    /// Every package the given one may reach, through its own declared edges and
    /// through theirs.
    fn closure(&self, package: &str) -> BTreeSet<String> {
        let mut reached = BTreeSet::new();
        let mut pending: Vec<String> = self
            .declared
            .get(package)
            .map(|edges| edges.iter().cloned().collect())
            .unwrap_or_default();
        while let Some(current) = pending.pop() {
            if !reached.insert(current.clone()) {
                continue;
            }
            if let Some(edges) = self.declared.get(&current) {
                pending.extend(edges.iter().cloned());
            }
        }
        reached
    }

    /// The layer the registry gives a package.
    fn layer(&self, package: &str) -> &str {
        self.layers.get(package).map_or("", String::as_str)
    }

    /// Whether a package's layer is substrate-neutral.
    fn neutral(&self, package: &str) -> bool {
        self.neutrality
            .get(self.layer(package))
            .copied()
            .unwrap_or(false)
    }
}

/// One member's direct production edges, with its own default features activated.
///
/// An optional dependency is an edge only when a feature in the default closure
/// activates it, which is the edge set a plain build has. A weak activation
/// (`dep?/feature`) is not an activation.
fn direct_edges(manifest: &toml::Table, workspace_deps: &toml::Table) -> BTreeSet<String> {
    let activated = activated_dependencies(manifest);
    let mut edges = BTreeSet::new();
    for (_, host) in dependency_hosts(manifest) {
        for table in PRODUCTION_TABLES {
            for (key, spec) in entries(&host, table) {
                let optional = spec
                    .get("optional")
                    .and_then(toml::Value::as_bool)
                    .unwrap_or(false);
                if optional && !activated.contains(&key) {
                    continue;
                }
                edges.insert(target_package(&key, &spec, workspace_deps));
            }
        }
    }
    edges
}

/// Dependency keys the manifest's default features activate.
fn activated_dependencies(manifest: &toml::Table) -> BTreeSet<String> {
    let features = manifest
        .get("features")
        .and_then(toml::Value::as_table)
        .cloned()
        .unwrap_or_default();
    let mut enabled = BTreeSet::new();
    let mut pending = vec!["default".to_string()];
    let mut activated = BTreeSet::new();
    while let Some(feature) = pending.pop() {
        if !enabled.insert(feature.clone()) {
            continue;
        }
        // A feature named after an optional dependency activates it implicitly.
        activated.insert(feature.clone());
        let Some(list) = features.get(&feature).and_then(toml::Value::as_array) else {
            continue;
        };
        for entry in list.iter().filter_map(toml::Value::as_str) {
            if let Some(dependency) = entry.strip_prefix("dep:") {
                activated.insert(dependency.to_string());
            } else if let Some((dependency, _)) = entry.split_once('/') {
                match dependency.strip_suffix('?') {
                    // `dep?/feature` enables the feature only if something else
                    // already pulled the dependency in, so it is not an edge.
                    Some(_) => {}
                    None => {
                        activated.insert(dependency.to_string());
                        pending.push(dependency.to_string());
                    }
                }
            } else {
                pending.push(entry.to_string());
            }
        }
    }
    activated
}

/// Each locked package's direct edges, which is how a third-party crate's own
/// dependencies become visible without resolving features.
fn locked_edges(lock: &toml::Table) -> BTreeMap<String, BTreeSet<String>> {
    let mut edges: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
    let Some(packages) = lock.get("package").and_then(toml::Value::as_array) else {
        return edges;
    };
    for package in packages {
        let Some(name) = package.get("name").and_then(toml::Value::as_str) else {
            continue;
        };
        let entry = edges.entry(name.to_string()).or_default();
        let Some(dependencies) = package.get("dependencies").and_then(toml::Value::as_array) else {
            continue;
        };
        for dependency in dependencies.iter().filter_map(toml::Value::as_str) {
            // A lockfile edge is `name`, `name version`, or `name version source`.
            let dependency = dependency.split_whitespace().next().unwrap_or(dependency);
            entry.insert(dependency.to_string());
        }
    }
    edges
}

#[cfg(test)]
mod tests {
    use super::*;

    fn table(text: &str) -> toml::Table {
        toml::from_str(text).expect("fixture parses")
    }

    #[test]
    fn an_optional_dependency_is_an_edge_only_when_a_default_feature_activates_it() {
        let manifest = table(
            r#"
            [features]
            default = ["fast"]
            fast = ["dep:present"]
            slow = ["dep:absent"]

            [dependencies]
            present = { version = "1", optional = true }
            absent = { version = "1", optional = true }
            always = "1"
            "#,
        );
        let edges = direct_edges(&manifest, &toml::Table::new());
        assert!(edges.contains("present"), "{edges:?}");
        assert!(edges.contains("always"), "{edges:?}");
        assert!(!edges.contains("absent"), "{edges:?}");
    }

    #[test]
    fn a_weak_activation_is_not_an_edge_and_a_bare_feature_name_is() {
        let manifest = table(
            r#"
            [features]
            default = ["weak", "bare"]
            weak = ["maybe?/extra"]
            bare = []

            [dependencies]
            maybe = { version = "1", optional = true }
            bare = { version = "1", optional = true }
            "#,
        );
        let edges = direct_edges(&manifest, &toml::Table::new());
        assert!(!edges.contains("maybe"), "{edges:?}");
        assert!(edges.contains("bare"), "{edges:?}");
    }

    #[test]
    fn a_renamed_workspace_dependency_resolves_to_the_package_it_names() {
        let workspace = table(
            r#"
            renamed = { version = "1", package = "real-name" }
            "#,
        );
        let manifest = table(
            r#"
            [dependencies]
            renamed = { workspace = true }
            "#,
        );
        let edges = direct_edges(&manifest, &workspace);
        assert!(edges.contains("real-name"), "{edges:?}");
    }

    #[test]
    fn development_edges_are_not_production_edges() {
        let manifest = table(
            r#"
            [dev-dependencies]
            harness = "1"

            [build-dependencies]
            generator = "1"
            "#,
        );
        let edges = direct_edges(&manifest, &toml::Table::new());
        assert!(!edges.contains("harness"), "{edges:?}");
        assert!(edges.contains("generator"), "{edges:?}");
    }

    #[test]
    fn a_lockfile_edge_keeps_only_the_package_name() {
        let lock = table(
            r#"
            [[package]]
            name = "wgpu"
            dependencies = ["naga 0.19.0", "raw-window-handle"]

            [[package]]
            name = "naga"
            "#,
        );
        let edges = locked_edges(&lock);
        assert_eq!(
            edges.get("wgpu").expect("wgpu is locked"),
            &["naga".to_string(), "raw-window-handle".to_string()]
                .into_iter()
                .collect::<BTreeSet<String>>()
        );
        assert!(edges.contains_key("naga"));
    }

    #[test]
    fn a_neutral_crate_that_reaches_a_backend_api_only_through_a_third_party_crate_is_found() {
        let graph = Graph {
            members: ["neutral".to_string()].into_iter().collect(),
            directories: [("neutral".to_string(), "neutral".to_string())]
                .into_iter()
                .collect(),
            edges: [("neutral".to_string(), ["middle".to_string()].into())]
                .into_iter()
                .collect(),
            locked: [
                ("middle".to_string(), ["wgpu".to_string()].into()),
                ("wgpu".to_string(), BTreeSet::new()),
            ]
            .into_iter()
            .collect(),
        };
        assert_eq!(
            graph.reachable_backend_apis("neutral"),
            ["wgpu".to_string()].into_iter().collect::<BTreeSet<_>>()
        );
        assert_eq!(
            graph.path_to("neutral", "wgpu"),
            "neutral -> middle -> wgpu"
        );
        assert!(graph.reachable_members("neutral").is_empty());
    }

    #[test]
    fn a_declared_closure_follows_the_edges_of_the_crates_it_names() {
        let registry = Registry {
            declared: [
                ("top".to_string(), ["middle".to_string()].into()),
                ("middle".to_string(), ["bottom".to_string()].into()),
                ("bottom".to_string(), BTreeSet::new()),
            ]
            .into_iter()
            .collect(),
            layers: [("top".to_string(), "runtime".to_string())]
                .into_iter()
                .collect(),
            neutrality: [("runtime".to_string(), true)].into_iter().collect(),
        };
        assert_eq!(
            registry.closure("top"),
            ["middle".to_string(), "bottom".to_string()]
                .into_iter()
                .collect::<BTreeSet<_>>()
        );
        assert!(registry.neutral("top"));
        assert!(!registry.neutral("unknown"));
    }

    #[test]
    fn every_backend_word_is_reported_from_the_list_rather_than_a_sample() {
        for word in BACKEND_WORDS {
            let line = format!("/// the {word} path");
            assert_eq!(
                words_in(&line),
                vec![format!("`{word}`")],
                "Fix: every entry of BACKEND_WORDS must be reportable; `{word}` was not."
            );
        }
    }

    #[test]
    fn a_backend_word_that_is_only_a_substring_of_an_unrelated_name_is_not_a_hit() {
        for line in [
            "let hash = compute();",
            "let barracuda = 1;",
            "let metallic = 1;",
            "let vulkanish = 0;",
            "fn aims_lower() {}",
        ] {
            assert!(
                words_in(line).is_empty(),
                "Fix: a vendor's letters inside an unrelated name are not vocabulary: {line}"
            );
        }
    }

    #[test]
    fn a_backend_word_in_camel_case_or_under_scores_is_a_hit() {
        for (line, expected) in [
            ("let device = CudaDevice::new();", "`CUDA`"),
            ("const MAX_NVIDIA_FS_BYTES: u64 = 1;", "`NVIDIA`"),
            ("fn ptxas_like() {}", "`ptxas`"),
            ("struct WGSLModule;", "`WGSL`"),
            ("let words = SpirvWords;", "`SPIRV`"),
            ("let table = spir_v_table;", "`SPIR-V`"),
        ] {
            assert!(
                words_in(line).contains(&expected.to_string()),
                "Fix: {expected} must be found in {line}, got {:?}",
                words_in(line)
            );
        }
    }

    #[test]
    fn a_member_crate_name_is_still_vocabulary_when_a_neutral_crate_writes_it() {
        assert_eq!(
            words_in("/// the vyre-driver-cuda fork answered a per-launch topology"),
            vec!["`CUDA`".to_string()],
            "Fix: naming the crate that owns a backend states the backend; only a layer \
             exempted by VOCABULARY_EXEMPT_LAYERS may write the roster."
        );
    }

    #[test]
    fn every_vocabulary_exemption_names_a_neutral_layer() {
        for (layer, _) in VOCABULARY_EXEMPT_LAYERS {
            assert!(exempt_from_vocabulary(layer));
            assert!(
                NEUTRAL_LAYERS
                    .iter()
                    .any(|(name, neutral)| name == layer && *neutral),
                "Fix: `{layer}` is excused from the vocabulary rule but is not a neutral layer, \
                 so the exemption excuses nothing."
            );
        }
        assert!(
            !exempt_from_vocabulary("lowering"),
            "Fix: a product layer is never excused from the vocabulary rule."
        );
    }

    #[test]
    fn an_interface_name_is_allowed_only_under_the_directory_that_reads_it() {
        let line = "let mut file = fs::File::open(\"/proc/driver/nvidia-fs/stats\")?;";
        let inside = mask_interface_names(line, "vyre-runtime/src/uring/gpudirect.rs");
        assert!(
            words_in(&inside).is_empty(),
            "Fix: the path the GPUDirect probe opens is an interface name under its own module."
        );
        assert_eq!(
            inside.len(),
            line.len(),
            "Fix: masking must preserve width so a reported column still maps to the source."
        );
        let outside = mask_interface_names(line, "vyre-foundation/src/lib.rs");
        assert_eq!(
            words_in(&outside),
            vec!["`NVIDIA`".to_string()],
            "Fix: the allowance must not reach a crate that does not read the interface."
        );
    }

    #[test]
    fn a_test_only_directory_or_module_is_not_production_source() {
        assert!(is_test_source(
            "vyre-libs/src/solvers/tests/dot_contracts.rs"
        ));
        assert!(is_test_source("vyre-libs/src/solvers/tests.rs"));
        assert!(!is_test_source("vyre-libs/src/solvers/contracts.rs"));
        assert!(!is_test_source("vyre-libs/src/tested/mod.rs"));
    }
}
