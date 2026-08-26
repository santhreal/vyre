//! Closure gates for the three duplication classes the composition domains have
//! repeatedly reintroduced, each derived from the tree at run time.
//!
//! A dedup pass fixes the copies that exist. It cannot fix the copy somebody adds
//! next month, and every class below has already come back at least once after
//! being cleaned up: launch grids were published per operation, cleaned up, and
//! published again under new names; the Bellman binding record documented itself
//! as "the only place the six names are spelled" while a second spelling of the
//! same six lived in another crate; the persistent-fixpoint routing assertions
//! were copied per op and the copies drifted in what they accepted.
//!
//! So each gate derives its MEMBER SET from source at run time rather than from a
//! list maintained here. A new launch-grid declaration, a new binding record or a
//! new routed convergence op turns the suite RED until it is routed onto the
//! owner. A hardcoded roster would go stale in silence, which is the same failure
//! as having no gate.
//!
//! What these gates do NOT catch: a copy written in a spelling none of the three
//! signatures below match, and a copy in a crate other than the one named by
//! `SUBJECT_CRATE`. They are structural signatures over source text, not a
//! semantic equivalence check.

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

use vyre_test_support::monorepo::{vyre_crate_directory, vyre_workspace_root};

/// The crate whose sources every gate below reads.
///
/// The three classes are properties of the composition domains, which live here.
const SUBJECT_CRATE: &str = "vyre-libs";

/// The crate that owns the one ceiling division from a lane count to a grid.
///
/// Nothing here reaches it: launch geometry is derived below admission by
/// `vyre_driver::infer_dispatch_grid`, which is the only remaining caller class.
/// The name stays because the meta gate resolves both crate directories.
const OWNER_CRATE: &str = "vyre-primitives";

/// Every `.rs` file under this crate's `src/`, as (crate-relative path, text).
///
/// Resolved at run time rather than from a compile-time manifest constant: every
/// checkout here shares one target directory, so a binary baked with one tree's
/// path reads another tree's files.
fn source_files() -> Vec<(String, String)> {
    let root = vyre_crate_directory(SUBJECT_CRATE);
    let src = root.join("src");
    let mut out = Vec::new();
    collect_rs(&src, &root, &mut out);
    out.sort_by(|a, b| a.0.cmp(&b.0));
    assert_walk_is_closed_under_the_module_tree(&out, &src);
    out
}

/// Panics unless every file-backed module the walk read is itself in the walk.
///
/// The guard the gates below need is that the walk found the crate, not that the
/// crate is a particular size: a walk rooted at the wrong directory or one that
/// never descends makes every signature below match nothing and pass. A file
/// count states that as a number, which then has to be lowered by hand every
/// time the crate legitimately shrinks, and a lowered floor guards nothing.
///
/// Closure over the declarations is the same guard without the number. Each
/// `mod name;` in a collected file names a sibling `name.rs` or `name/mod.rs`,
/// and the compiler already proved those files exist, so a missing one means the
/// walk skipped it. `src/lib.rs` is required outright because it is the root the
/// closure starts from and an empty walk is closed over nothing.
///
/// Inline `mod name { ... }` blocks are not file-backed and are skipped. A
/// `#[path]` attribute would point a declaration somewhere else; this crate has
/// none, and one would surface here as a missing file rather than pass silently.
fn assert_walk_is_closed_under_the_module_tree(files: &[(String, String)], src: &Path) {
    let present: BTreeSet<&str> = files.iter().map(|(path, _)| path.as_str()).collect();
    assert!(
        present.contains("src/lib.rs"),
        "Fix: the walk under {} did not read src/lib.rs, so it is not reading this crate; every gate below would pass by finding no members.",
        src.display()
    );
    let mut missing = Vec::new();
    for (path, text) in files {
        let directory = match path.rsplit_once('/') {
            Some((head, tail)) if tail == "mod.rs" || tail == "lib.rs" => head.to_string(),
            Some((head, tail)) => format!("{head}/{}", tail.trim_end_matches(".rs")),
            None => String::new(),
        };
        for name in file_backed_modules(text) {
            let flat = format!("{directory}/{name}.rs");
            let nested = format!("{directory}/{name}/mod.rs");
            if !present.contains(flat.as_str()) && !present.contains(nested.as_str()) {
                missing.push(format!(
                    "{path} declares `mod {name};` but neither {flat} nor {nested} was read"
                ));
            }
        }
    }
    assert!(
        missing.is_empty(),
        "Fix: the walk under {} skipped {} file-backed module(s), so the gates below cover less than the crate: {}",
        src.display(),
        missing.len(),
        missing.join("; ")
    );
}

/// The names of the `mod name;` declarations in `text`, ignoring inline modules and `#[path]` overrides.
fn file_backed_modules(text: &str) -> Vec<&str> {
    let mut out = Vec::new();
    let mut has_path_override = false;
    for line in text.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with("#[path") {
            has_path_override = true;
            continue;
        }
        let code = line.split("//").next().unwrap_or(line).trim();
        let Some(rest) = code.strip_suffix(';') else {
            has_path_override = false;
            continue;
        };
        let rest = rest.trim_start_matches("pub ");
        let rest = match rest.find(") ") {
            Some(end) if rest.starts_with("pub(") => &rest[end + 2..],
            _ => rest,
        };
        if let Some(name) = rest.strip_prefix("mod ") {
            let name = name.trim();
            if !has_path_override
                && !name.is_empty()
                && name.chars().all(|c| c.is_alphanumeric() || c == '_')
            {
                out.push(name);
            }
        }
        has_path_override = false;
    }
    out
}

/// Every `.rs` file under every workspace member's `src/`, as (workspace-relative
/// path, text).
///
/// A binding record is public API, so a second spelling of its names can live in
/// any consumer, and the copy that motivated the owner did: the record documented
/// itself as the only place the six names were spelled while a second spelling of
/// the same six sat in another crate. A gate scoped to the declaring crate cannot
/// see that, which makes it a gate that passes on the exact defect it exists for.
/// The member list is read from the workspace manifest, so a new crate is enrolled
/// without touching this file.
fn workspace_source_files() -> Vec<(String, String)> {
    let root = vyre_workspace_root();
    let manifest = std::fs::read_to_string(root.join("Cargo.toml"))
        .expect("Fix: cannot read the workspace Cargo.toml.");
    let mut members = Vec::new();
    let mut in_members = false;
    for line in manifest.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with("members") && trimmed.contains('[') {
            in_members = true;
            continue;
        }
        if in_members {
            if trimmed.starts_with(']') {
                break;
            }
            let name = trimmed.trim_end_matches(',').trim_matches('"');
            if !name.is_empty() && !name.starts_with('#') {
                members.push(name.to_string());
            }
        }
    }
    assert!(
        members.len() >= 5,
        "Fix: the workspace manifest parse found {} members; the `[workspace] members` layout changed and the scan would cover almost nothing.",
        members.len()
    );

    let mut out = Vec::new();
    for member in &members {
        let src = root.join(member).join("src");
        if src.is_dir() {
            collect_rs(&src, &root, &mut out);
        }
    }
    assert!(
        out.len() > 500,
        "Fix: only {} source files were found across {} workspace members; the walk is wrong, so the gate would pass by finding no members.",
        out.len(),
        members.len()
    );
    out.sort_by(|a, b| a.0.cmp(&b.0));
    out
}

fn collect_rs(dir: &Path, root: &Path, out: &mut Vec<(String, String)>) {
    let entries = std::fs::read_dir(dir)
        .unwrap_or_else(|e| panic!("Fix: cannot read {}: {e}", dir.display()));
    for entry in entries {
        let path = entry
            .unwrap_or_else(|e| panic!("Fix: cannot read an entry of {}: {e}", dir.display()))
            .path();
        if path.is_dir() {
            collect_rs(&path, root, out);
        } else if path.extension().is_some_and(|e| e == "rs") {
            let text = std::fs::read_to_string(&path)
                .unwrap_or_else(|e| panic!("Fix: cannot read {}: {e}", path.display()));
            let relative = path
                .strip_prefix(root)
                .unwrap_or(&path)
                .to_string_lossy()
                .replace('\\', "/");
            out.push((relative, text));
        }
    }
}

// ---------------------------------------------------------------------------
// Class 1: no operation publishes launch geometry.
// ---------------------------------------------------------------------------

/// Launch geometry an operation publishes, as (path, kind, ident).
///
/// Three shapes carry a grid out of an operation and into a caller: a function
/// that returns one, a struct field that stores one, and a constant that names
/// one. Each is recognized by its declared `[u32; 3]` type rather than by its
/// identifier, so a rename cannot hide one and a name that only reads like
/// geometry is not convicted: `persistent_fixpoint_grid` returns a `Program`
/// built around a grid-sync barrier and publishes no launch. All three are
/// derived from source text at run time, so a new one enrolls itself and turns
/// the gate below red.
fn published_launch_geometry(files: &[(String, String)]) -> Vec<(String, String, String)> {
    let mut found = Vec::new();
    for (path, text) in files {
        let lines: Vec<&str> = text.lines().collect();
        let mut cfg_test = false;
        for (index, line) in lines.iter().enumerate() {
            let trimmed = line.trim_start();
            // A `#[cfg(test)]` item is unreachable from any caller, so it is not
            // published surface. The marker applies to the next item only.
            if trimmed.starts_with("#[cfg(test)]") {
                cfg_test = true;
                continue;
            }
            if trimmed.starts_with("//") || trimmed.is_empty() {
                continue;
            }
            if let Some((kind, ident)) = launch_geometry_declaration(&lines, index) {
                if !cfg_test {
                    found.push((path.clone(), kind.to_string(), ident));
                }
            }
            cfg_test = false;
        }
    }
    found
}

/// The declaration starting at `index`, joined up to where its type is settled.
///
/// A signature that wraps across lines puts its return type several lines below
/// the identifier, so reading one line would miss every wrapped declaration.
/// The join stops at the body brace or the terminating semicolon.
fn declaration_text(lines: &[&str], index: usize) -> String {
    let mut text = String::new();
    for line in &lines[index..] {
        let trimmed = line.trim();
        if trimmed.starts_with("//") {
            continue;
        }
        text.push(' ');
        text.push_str(trimmed);
        if trimmed.ends_with('{') || trimmed.ends_with(';') || trimmed.ends_with(',') {
            break;
        }
    }
    text
}

/// The kind and identifier of the launch-geometry declaration at `index`.
///
/// A launch grid and a workgroup shape are both `[u32; 3]`, so the type alone
/// cannot separate them. A workgroup shape is the extent one invocation group
/// covers and is a backend lowering fact `structure-gate` owns; a launch grid is
/// the number of groups, which is the compiler's to choose. The identifier is
/// the only place the declaration states which of the two it is.
fn launch_geometry_declaration(lines: &[&str], index: usize) -> Option<(&'static str, String)> {
    let (kind, name) = launch_shaped_declaration(lines, index)?;
    (!name.to_ascii_uppercase().contains("WORKGROUP")).then_some((kind, name))
}

/// The kind and identifier of any `[u32; 3]` declaration at `index`.
fn launch_shaped_declaration(lines: &[&str], index: usize) -> Option<(&'static str, String)> {
    let trimmed = lines[index].trim_start();
    if let Some(rest) = trimmed
        .strip_prefix("pub const fn ")
        .or_else(|| trimmed.strip_prefix("pub fn "))
        .or_else(|| trimmed.strip_prefix("pub(crate) const fn "))
        .or_else(|| trimmed.strip_prefix("pub(crate) fn "))
    {
        let (name, _) = rest.split_once('(')?;
        return declaration_text(lines, index)
            .contains("-> [u32; 3]")
            .then(|| ("function", name.to_string()));
    }
    if let Some(rest) = trimmed
        .strip_prefix("pub const ")
        .or_else(|| trimmed.strip_prefix("pub(crate) const "))
    {
        let (name, tail) = rest.split_once(':')?;
        return tail
            .contains("[u32; 3]")
            .then(|| ("constant", name.trim().to_string()));
    }
    let rest = trimmed
        .strip_prefix("pub ")
        .or_else(|| trimmed.strip_prefix("pub(crate) "))?;
    let (name, tail) = rest.split_once(':')?;
    tail.contains("[u32; 3]")
        .then(|| ("field", name.trim().to_string()))
}

/// No operation in this crate may publish launch geometry.
///
/// A grid an operation hands back is a launch a caller can pass to a dispatch,
/// and a caller-chosen launch is not the one the compiler admits. The defect this
/// closes is a scatter: a paged cache append guards on one decoded chunk and
/// declares a cache-sized destination, so a published grid sized from the widest
/// buffer fired one lane per cache element on every decode step. The span is a
/// property of the program's own guard, so `vyre_foundation::guarded_logical_span`
/// reads it, and target lowering and `vyre_driver::infer_dispatch_grid` narrow to
/// it. Nothing below admission needs an operation to publish a number.
///
/// Does NOT catch geometry returned under a name this signature does not match,
/// such as a tuple field or a method returning `LaunchGeometry`, nor a launch
/// grid declared under a name containing `workgroup`, which is read as the
/// workgroup shape it claims to be.
#[test]
fn no_operation_publishes_launch_geometry() {
    let offenders: Vec<String> = published_launch_geometry(&source_files())
        .into_iter()
        .map(|(path, kind, ident)| format!("{path}: {kind} `{ident}`"))
        .collect();

    assert!(
        offenders.is_empty(),
        "Fix: these declarations publish a launch grid out of an operation. A caller that passes one back overrides the domain the program's guard admits. Delete the declaration and let `vyre_driver::infer_dispatch_grid` derive the launch, or assert `vyre_foundation::guarded_logical_span` where the shape is the contract:\n  {}",
        offenders.join("\n  ")
    );
}

/// The signature must still match the shapes it claims to reject.
///
/// An empty result is the passing state of the gate above, so the scan has to be
/// proven able to see a member at all. Otherwise a broken prefix match would read
/// as a clean tree forever.
#[test]
fn the_launch_geometry_signature_matches_every_shape_it_rejects() {
    let sample = vec![(
        "src/probe.rs".to_string(),
        concat!(
            "pub const fn probe_dispatch_grid(n: u32) -> [u32; 3] { [n, 1, 1] }\n",
            "pub const PROBE_DISPATCH_GRID: [u32; 3] = [1, 1, 1];\n",
            "pub grid: [u32; 3],\n",
            "pub high_grid: [u32; 3],\n",
            "pub const fn probe_words(n: u32) -> u32 { n }\n",
            "pub const PROBE_WORKGROUP_SIZE: [u32; 3] = [256, 1, 1];\n",
            "#[cfg(test)]\n",
            "pub const fn test_only_dispatch_grid(n: u32) -> [u32; 3] { [n, 1, 1] }\n",
        )
        .to_string(),
    )];
    let kinds: Vec<(String, String)> = published_launch_geometry(&sample)
        .into_iter()
        .map(|(_, kind, ident)| (kind, ident))
        .collect();
    assert_eq!(
        kinds,
        vec![
            ("function".to_string(), "probe_dispatch_grid".to_string()),
            ("constant".to_string(), "PROBE_DISPATCH_GRID".to_string()),
            ("field".to_string(), "grid".to_string()),
            ("field".to_string(), "high_grid".to_string()),
        ],
        "Fix: the launch-geometry signature no longer sees a function, constant or field it must reject, or it now flags a workgroup shape or a word count that is not a launch."
    );

    // The workgroup constant must be seen as a `[u32; 3]` declaration and then
    // excluded by role. A prefix match that stopped seeing it altogether would
    // satisfy the assertion above while going blind to every launch grid too.
    let lines: Vec<&str> = sample[0].1.lines().collect();
    let workgroup_index = lines
        .iter()
        .position(|line| line.contains("PROBE_WORKGROUP_SIZE"))
        .expect("the sample declares a workgroup shape");
    assert_eq!(
        launch_shaped_declaration(&lines, workgroup_index),
        Some(("constant", "PROBE_WORKGROUP_SIZE".to_string())),
        "Fix: the `[u32; 3]` shape scan no longer sees a workgroup constant, so it sees no launch constant either."
    );
    assert_eq!(
        launch_geometry_declaration(&lines, workgroup_index),
        None,
        "Fix: a workgroup shape is not a launch grid; `structure-gate` owns it."
    );
}

// ---------------------------------------------------------------------------
// Class 2: buffer binding names are spelled once.
// ---------------------------------------------------------------------------

/// Binding-record types that publish canonical names, as `(type, [fields])`.
///
/// A record is a member because it declares `const CANONICAL`, so adding one
/// enrolls it automatically and adding one WITHOUT canonical names is caught by
/// the companion test below.
fn binding_records(files: &[(String, String)]) -> BTreeMap<String, Vec<String>> {
    let mut fields: BTreeMap<String, Vec<String>> = BTreeMap::new();
    let mut canonical = BTreeSet::new();
    for (_, text) in files {
        let lines: Vec<&str> = text.lines().collect();
        for (index, line) in lines.iter().enumerate() {
            if let Some(rest) = line.trim().strip_prefix("pub struct ") {
                let Some(name) = rest.split(['<', ' ', '{']).next() else {
                    continue;
                };
                if !name.ends_with("Buffers") {
                    continue;
                }
                let mut names = Vec::new();
                for body in &lines[index + 1..] {
                    let trimmed = body.trim();
                    if trimmed == "}" {
                        break;
                    }
                    if let Some(field) = trimmed
                        .strip_prefix("pub ")
                        .and_then(|f| f.split_once(':'))
                        .filter(|(_, ty)| ty.contains("str"))
                    {
                        names.push(field.0.to_string());
                    }
                }
                if !names.is_empty() {
                    fields.insert(name.to_string(), names);
                }
            }
            if let Some(rest) = line.trim().strip_prefix("impl ") {
                if let Some(name) = rest.split(['<', ' ']).next() {
                    if lines[index..]
                        .iter()
                        .take_while(|l| !l.starts_with("}"))
                        .any(|l| l.contains("const CANONICAL"))
                    {
                        canonical.insert(name.to_string());
                    }
                }
            }
        }
    }
    fields.retain(|name, _| canonical.contains(name));
    fields
}

/// A binding record's names are spelled in exactly one place.
///
/// A record whose fields are all `&str` is a positional list wearing field names:
/// spelling the full set a second time is where a transposed `src`/`dst` or
/// `k`/`k_t` hides, unprovable from either copy because both compile and both look
/// deliberate. The signature is a struct literal that assigns every field of the
/// record to a string literal. A partial literal with `..CANONICAL` is not a
/// member: it states only what it changes, which is the point.
///
/// Does NOT catch a re-spelling that routes each name through a `const` first.
#[test]
fn no_binding_record_spells_its_full_name_set_twice() {
    let records = binding_records(&source_files());
    let files = workspace_source_files();
    assert!(
        !records.is_empty(),
        "Fix: no binding record with canonical names was derived; the signature no longer matches, so the gate would pass by finding nothing."
    );

    let mut offenders = Vec::new();
    for (record, fields) in &records {
        let literal = format!("{record} {{");
        for (path, text) in &files {
            let lines: Vec<&str> = text.lines().collect();
            for (index, line) in lines.iter().enumerate() {
                if !line.contains(&literal) {
                    continue;
                }
                // The owner's own `const CANONICAL`/`const TERSE` use `Self {`,
                // never the type name, so they are not candidates at all.
                let mut spelled = BTreeSet::new();
                for body in &lines[index + 1..] {
                    let trimmed = body.trim();
                    if trimmed.starts_with('}') {
                        break;
                    }
                    if let Some((field, value)) = trimmed.split_once(": ") {
                        if value.trim_start().starts_with('"') {
                            spelled.insert(field.trim().to_string());
                        }
                    }
                }
                if fields.iter().all(|f| spelled.contains(f)) {
                    offenders.push(format!(
                        "{path}:{} respells all {} names of {record}",
                        index + 1,
                        fields.len()
                    ));
                }
            }
        }
    }

    assert!(
        offenders.is_empty(),
        "Fix: these sites spell a binding record's full name set a second time instead of naming the record's own canonical constant:\n  {}",
        offenders.join("\n  ")
    );
}

/// A new binding record must publish canonical names rather than leave every
/// caller to invent them.
#[test]
fn every_buffer_binding_record_publishes_canonical_names() {
    let files = source_files();
    let with_canonical: BTreeSet<String> = binding_records(&files).into_keys().collect();

    let mut declared = BTreeMap::new();
    for (path, text) in &files {
        for line in text.lines() {
            if let Some(rest) = line.trim().strip_prefix("pub struct ") {
                if let Some(name) = rest.split(['<', ' ', '{']).next() {
                    if name.ends_with("Buffers") {
                        declared.insert(name.to_string(), path.clone());
                    }
                }
            }
        }
    }
    assert!(
        declared.len() >= 2,
        "Fix: only {} buffer binding records were derived; the signature no longer matches this crate's declarations.",
        declared.len()
    );

    let missing: Vec<String> = declared
        .iter()
        .filter(|(name, _)| !with_canonical.contains(name.as_str()))
        .map(|(name, path)| format!("{name} in {path}"))
        .collect();
    assert!(
        missing.is_empty(),
        "Fix: these binding records publish no `const CANONICAL`, so every caller spells the names itself and two callers cannot be compared:\n  {}",
        missing.join("\n  ")
    );
}

// ---------------------------------------------------------------------------
// Class 3: routed convergence ops assert the routing contract once.
// ---------------------------------------------------------------------------

/// The routing-contract owner and the harness it asserts over, both excluded.
const ROUTING_OWNERS: [&str; 2] = [
    "src/fixpoint/routing_contract.rs",
    "src/fixpoint/persistent_fixpoint.rs",
];

/// Every op that drives a convergence loop through the routed harness must
/// register with the one routing contract.
///
/// The four obligations are obligations of the ROUTING, not of the op, so an op
/// that asserts them itself is asserting a rule it does not own, and the copies
/// had already drifted in what they accepted. Membership is derived from calling
/// `routed_persistent_fixpoint`, so a new routed op is RED on the first run.
///
/// Does NOT catch an op that registers with the contract but supplies a fixture
/// that never crosses one workgroup; the contract itself asserts that instead.
#[test]
fn every_routed_convergence_op_registers_with_the_routing_contract() {
    let files = source_files();
    let mut members = Vec::new();
    for (path, text) in &files {
        if ROUTING_OWNERS.contains(&path.as_str()) {
            continue;
        }
        if text.contains("routed_persistent_fixpoint(") {
            members.push(path.clone());
        }
    }
    assert!(
        members.len() >= 2,
        "Fix: only {} routed convergence ops were derived ({members:?}); the signature no longer matches, so the gate would pass by finding nothing.",
        members.len()
    );

    let by_path: BTreeMap<&str, &str> = files
        .iter()
        .map(|(path, text)| (path.as_str(), text.as_str()))
        .collect();
    let mut unregistered = Vec::new();
    for path in &members {
        // An op's tests live in the op's own file, in a `tests/mod.rs` under a
        // directory named for it, or - once the op is split across files - in a
        // `tests/mod.rs` beside the file in the op's own directory. All three are
        // the same module. A registration in a neighbouring file counts only when
        // it names the op, so a dialect directory shared by several ops can never
        // lend one op's registration to another.
        let named_directory = path.trim_end_matches(".rs").to_string() + "/tests/mod.rs";
        let (directory, file) = path.rsplit_once('/').unwrap_or((".", path.as_str()));
        let stem = file.trim_end_matches(".rs");
        let op = if stem == "program" || stem == "mod" {
            directory.rsplit('/').next().unwrap_or(directory)
        } else {
            stem
        };
        let dir_without_src = directory.strip_prefix("src/").unwrap_or(directory);
        let beside = format!("{directory}/tests/mod.rs");
        let internal_dir = format!("tests/internal/{dir_without_src}/mod.rs");
        let internal_op = format!("tests/internal/{op}/mod.rs");
        let candidates = [
            path.as_str(),
            named_directory.as_str(),
            beside.as_str(),
            internal_dir.as_str(),
            internal_op.as_str(),
        ];
        let registered = candidates.iter().any(|candidate| {
            if let Some(text) = by_path.get(*candidate) {
                text.contains("assert_routes_on_dispatch_span")
                    && (*candidate == path.as_str() || text.contains(op))
            } else if let Ok(text) =
                std::fs::read_to_string(vyre_crate_directory(SUBJECT_CRATE).join(candidate))
            {
                text.contains("assert_routes_on_dispatch_span")
                    && (*candidate == path.as_str() || text.contains(op))
            } else {
                false
            }
        });
        if !registered {
            unregistered.push(path.clone());
        }
    }

    assert!(
        unregistered.is_empty(),
        "Fix: these ops route a convergence loop through `routed_persistent_fixpoint` but never register with `fixpoint::routing_contract::assert_routes_on_dispatch_span`, so nothing proves they switch off the single shared cleared convergence word once the dispatch spans more than one workgroup:\n  {}",
        unregistered.join("\n  ")
    );
}

/// No op may re-assert the routing obligations privately.
///
/// The drift that made one contract necessary was per-op copies of these
/// assertions, and a copy is re-added by pasting the assertion back next to the op
/// rather than by deleting the shared one. Membership is every non-owner file, so
/// the check does not depend on knowing which ops exist.
#[test]
fn no_op_re_asserts_the_routing_obligations_privately() {
    let files = source_files();
    // `count_grid_sync` and `declared_words` are the two observations the
    // obligations are expressed in. Reaching for both outside the owner is an op
    // re-deriving the routing rule instead of registering with it.
    let mut offenders = Vec::new();
    for (path, text) in &files {
        if ROUTING_OWNERS.contains(&path.as_str()) {
            continue;
        }
        if text.contains("count_grid_sync(") && text.contains("declared_words(") {
            offenders.push(path.clone());
        }
    }
    assert!(
        offenders.is_empty(),
        "Fix: these files observe both `count_grid_sync` and `declared_words`, which together are the persistent-fixpoint routing obligations. Register the op with `fixpoint::routing_contract::assert_routes_on_dispatch_span` instead; a rule asserted once per op is a rule that can be weakened for one op:\n  {}",
        offenders.join("\n  ")
    );
}

/// No production source file may call `persistent_fixpoint` or
/// `persistent_fixpoint_grid` directly.
///
/// A direct call bypasses [`crate::fixpoint::persistent_fixpoint::routed_persistent_fixpoint`],
/// which owns the single-workgroup vs grid-sync routing decision and the
/// convergence-flag width that goes with it. Above one workgroup width a direct
/// call to `persistent_fixpoint` exposes the launch to the lost-set race and
/// false convergence, while a direct call to `persistent_fixpoint_grid` forces a
/// cooperative launch at one workgroup for nothing.
///
/// Production source text is stripped of inline test modules (`mod tests`) and
/// `#[cfg(test)]` sections before matching, so tests that deliberately drive the
/// pre-routing single-word harness for divergence verification are preserved,
/// while any new production caller fails until routed.
#[test]
fn no_production_op_calls_unrouted_persistent_fixpoint() {
    let files = source_files();
    let mut offenders = Vec::new();
    for (path, text) in &files {
        if ROUTING_OWNERS.contains(&path.as_str()) {
            continue;
        }
        let prod = non_test_source_text(text);
        if calls_unrouted_persistent_fixpoint(&prod) {
            offenders.push(path.clone());
        }
    }
    assert!(
        offenders.is_empty(),
        "Fix: these production source files call `persistent_fixpoint` or `persistent_fixpoint_grid` directly without routing through `routed_persistent_fixpoint`. Direct calls bypass the multi-workgroup GridSync switch and race when the dispatch span exceeds 256 lanes. Route through `routed_persistent_fixpoint` and register with `fixpoint::routing_contract::assert_routes_on_dispatch_span`:\n  {}",
        offenders.join("\n  ")
    );
}

/// Strip test modules and `#[cfg(test)]` attributes from `text` so only production
/// code is inspected.
fn non_test_source_text(text: &str) -> String {
    let mut out = String::new();
    let mut in_test_module = false;
    let mut depth = 0usize;
    for line in text.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with("#[cfg(test)]") || trimmed.starts_with("mod tests") {
            in_test_module = true;
        }
        if in_test_module {
            for c in trimmed.chars() {
                if c == '{' {
                    depth += 1;
                } else if c == '}' {
                    depth = depth.saturating_sub(1);
                    if depth == 0 {
                        in_test_module = false;
                    }
                }
            }
            if trimmed.ends_with(';') && depth == 0 {
                in_test_module = false;
            }
            continue;
        }
        out.push_str(line);
        out.push('\n');
    }
    out
}

/// Detect calls to `persistent_fixpoint(...)` or `persistent_fixpoint_grid(...)` that are
/// not calls to `routed_persistent_fixpoint(...)`.
fn calls_unrouted_persistent_fixpoint(text: &str) -> bool {
    let is_ident_char = |c: char| c.is_ascii_alphanumeric() || c == '_';
    for target in [
        "persistent_fixpoint(",
        "persistent_fixpoint_grid(",
        "persistent_fixpoint (",
        "persistent_fixpoint_grid (",
    ] {
        let mut rem = text;
        while let Some(idx) = rem.find(target) {
            let prefix = &rem[..idx];
            if let Some(last_char) = prefix.chars().last() {
                if !is_ident_char(last_char) {
                    return true;
                }
            } else {
                return true;
            }
            rem = &rem[idx + target.len()..];
        }
    }
    false
}

/// Keeps the crate directory resolution honest for both axes: a stale path
/// resolves to another checkout, and every gate above would then read the wrong
/// tree.
///
/// The walk itself is guarded on every call by
/// [`assert_walk_is_closed_under_the_module_tree`], which needs no list of
/// required files to maintain.
#[test]
fn both_crate_directories_are_the_ones_this_workspace_holds() {
    for crate_name in [SUBJECT_CRATE, OWNER_CRATE] {
        let root = vyre_crate_directory(crate_name);
        let manifest = std::fs::read_to_string(root.join("Cargo.toml"))
            .expect("Fix: the resolved crate directory holds no Cargo.toml.");
        assert!(
            manifest.contains(&format!("name = \"{crate_name}\"")),
            "Fix: {} is not the {crate_name} crate directory.",
            root.display()
        );
    }
    let subject = vyre_crate_directory(SUBJECT_CRATE);
    assert!(
        subject.join("tests/shared_owner_closure.rs").is_file(),
        "Fix: {} does not hold this test file, so the walk is reading a different checkout.",
        subject.display()
    );
}
