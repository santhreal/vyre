//! Closure gates for the three duplication classes this crate has repeatedly
//! reintroduced, each derived from the tree at run time.
//!
//! A dedup pass fixes the copies that exist. It cannot fix the copy somebody adds
//! next month, and every class below has already come back at least once after
//! being cleaned up: `lane_grid` replaced four hand-rolled ceiling helpers and two
//! more were added afterwards; the Bellman binding record documented itself as
//! "the only place the six names are spelled" while a second spelling of the same
//! six lived in another crate; the persistent-fixpoint routing assertions were
//! copied per op and the copies drifted in what they accepted.
//!
//! So each gate derives its MEMBER SET from source at run time rather than from a
//! list maintained here. A new dispatch-grid function, a new binding record or a
//! new routed convergence op turns the suite RED until it is routed onto the
//! owner. A hardcoded roster would go stale in silence, which is the same failure
//! as having no gate.
//!
//! What these gates do NOT catch: a copy written in a spelling none of the three
//! signatures below match, and a copy in a crate other than this one. They are
//! structural signatures over source text, not a semantic equivalence check.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

use vyre_test_support::monorepo::{vyre_crate_directory, vyre_workspace_root};

/// Every `.rs` file under this crate's `src/`, as (crate-relative path, text).
///
/// Resolved at run time rather than from a compile-time manifest constant: every
/// checkout here shares one target directory, so a binary baked with one tree's
/// path reads another tree's files.
fn source_files() -> Vec<(String, String)> {
    let root = vyre_crate_directory("vyre-primitives");
    let src = root.join("src");
    let mut out = Vec::new();
    collect_rs(&src, &root, &mut out);
    assert!(
        out.len() > 100,
        "Fix: only {} source files were found under {}; the walk is wrong, so every gate below would pass by finding no members.",
        out.len(),
        src.display()
    );
    out.sort_by(|a, b| a.0.cmp(&b.0));
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

/// The lines of `text` making up the body of the free function declared on
/// `start`, up to and including its closing brace at the declaration's indent.
///
/// Indent-terminated rather than brace-counted so a brace inside a string or a
/// comment cannot desynchronize the scan.
fn function_body<'a>(lines: &[&'a str], start: usize) -> Vec<&'a str> {
    let indent = lines[start].len() - lines[start].trim_start().len();
    let closer = " ".repeat(indent) + "}";
    let mut body = Vec::new();
    for line in &lines[start..] {
        body.push(*line);
        if *line == closer.as_str() {
            break;
        }
    }
    body
}

// ---------------------------------------------------------------------------
// Class 1: one ceiling division from a lane count to a dispatch grid.
// ---------------------------------------------------------------------------

/// The owner module path, excluded from its own gate.
const DISPATCH_GRID_OWNER: &str = "src/dispatch_grid.rs";

/// Free functions that compute a dispatch grid, as (path, function name, body).
///
/// Methods are excluded by requiring no `&self` receiver: a `fn dispatch_grid(&self)`
/// that returns a stored field is an accessor, not a computation, and holding it to
/// the ceiling-division rule would be asserting the wrong thing.
fn dispatch_grid_functions(files: &[(String, String)]) -> Vec<(String, String, String)> {
    let mut found = Vec::new();
    for (path, text) in files {
        if path == DISPATCH_GRID_OWNER {
            continue;
        }
        let lines: Vec<&str> = text.lines().collect();
        for (index, line) in lines.iter().enumerate() {
            let trimmed = line.trim_start();
            let Some(rest) = trimmed
                .strip_prefix("pub const fn ")
                .or_else(|| trimmed.strip_prefix("pub fn "))
                .or_else(|| trimmed.strip_prefix("const fn "))
                .or_else(|| trimmed.strip_prefix("fn "))
                .or_else(|| trimmed.strip_prefix("pub(crate) const fn "))
                .or_else(|| trimmed.strip_prefix("pub(crate) fn "))
            else {
                continue;
            };
            let Some((name, args)) = rest.split_once('(') else {
                continue;
            };
            if !name.ends_with("dispatch_grid") && !name.ends_with("grid_x") {
                continue;
            }
            if args.starts_with("&self") || args.starts_with("self") {
                continue;
            }
            found.push((
                path.clone(),
                name.to_string(),
                function_body(&lines, index).join("\n"),
            ));
        }
    }
    found
}

/// No dispatch-grid function may hand-roll the ceiling division `lane_grid` owns.
///
/// The signature is a `/` or `%` operator inside the body. Ceiling division is the
/// whole of what these functions do, so a division in one is either the owner's
/// arithmetic restated or a second, differently-rounded answer to the same
/// question. Both are the defect: the copies that existed disagreed at zero, where
/// three underflowed `value - 1` and one produced a grid of zero groups that the
/// CUDA launcher rejects outright.
///
/// Does NOT catch a copy that reaches the same result without a division operator,
/// such as a lookup table or a loop.
#[test]
fn no_dispatch_grid_function_hand_rolls_the_ceiling_division() {
    let files = source_files();
    let members = dispatch_grid_functions(&files);
    assert!(
        members.len() >= 10,
        "Fix: only {} dispatch-grid functions were derived; the signature no longer matches this crate's declarations, so the gate would pass by finding nothing.",
        members.len()
    );

    let mut offenders = Vec::new();
    for (path, name, body) in &members {
        let arithmetic: Vec<&str> = body
            .lines()
            .filter(|line| {
                let code = line.split("//").next().unwrap_or(line);
                code.contains(" / ") || code.contains(" % ")
            })
            .map(str::trim)
            .collect();
        if !arithmetic.is_empty() {
            offenders.push(format!("{path}::{name}: {}", arithmetic.join(" | ")));
        }
    }

    assert!(
        offenders.is_empty(),
        "Fix: these dispatch-grid functions compute their own ceiling division instead of calling the one owner, vyre_primitives::dispatch_grid::lane_grid, whose zero case is the launchable one:\n  {}",
        offenders.join("\n  ")
    );
}

/// Every dispatch-grid function must be able to reach the owner.
///
/// An owner inside a domain module is invisible to a domain that does not enable
/// it, and the primitive that could not reach it hand-rolled the arithmetic
/// instead: that is exactly how the `decode` copy survived a cleanup that removed
/// four others. Reachability is a property of the feature graph, so it is computed
/// from `Cargo.toml` and `lib.rs` at run time rather than asserted by hand.
#[test]
fn every_domain_with_a_dispatch_grid_can_reach_the_owner() {
    let root = vyre_crate_directory("vyre-primitives");
    let features = feature_closures(&root);
    let module_features = module_features(&root);
    let files = source_files();

    let owner_module = DISPATCH_GRID_OWNER
        .trim_start_matches("src/")
        .trim_end_matches(".rs");
    assert!(
        !module_features.contains_key(owner_module),
        "Fix: `{owner_module}` is now gated on feature `{}`. The dispatch-grid owner must stay ungated, or a domain that does not enable that feature cannot reach it and will hand-roll the arithmetic again.",
        module_features[owner_module]
    );

    let mut unreachable = Vec::new();
    for (path, name, _) in dispatch_grid_functions(&files) {
        let Some(module) = path
            .trim_start_matches("src/")
            .split('/')
            .next()
            .map(|m| m.trim_end_matches(".rs"))
        else {
            continue;
        };
        let Some(feature) = module_features.get(module) else {
            continue;
        };
        let closure = features.get(feature).unwrap_or_else(|| {
            panic!("Fix: `lib.rs` gates module `{module}` on feature `{feature}`, which `Cargo.toml` does not declare.")
        });
        if !closure.contains(owner_module) && !closure.contains(feature.as_str()) {
            unreachable.push(format!("{path}::{name} (feature `{feature}`)"));
        }
    }
    assert!(
        unreachable.is_empty(),
        "Fix: {unreachable:?} name a feature `Cargo.toml` does not declare."
    );
}

/// `feature -> transitive feature closure`, from `Cargo.toml`'s `[features]`.
fn feature_closures(root: &Path) -> BTreeMap<String, BTreeSet<String>> {
    let manifest = std::fs::read_to_string(root.join("Cargo.toml"))
        .expect("Fix: cannot read vyre-primitives/Cargo.toml.");
    let mut direct: BTreeMap<String, Vec<String>> = BTreeMap::new();
    let mut in_features = false;
    for line in manifest.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with('[') {
            in_features = trimmed == "[features]";
            continue;
        }
        if !in_features || trimmed.starts_with('#') {
            continue;
        }
        let Some((name, list)) = trimmed.split_once(" = [") else {
            continue;
        };
        let deps = list
            .trim_end_matches(']')
            .split(',')
            .filter_map(|part| {
                let dep = part.trim().trim_matches('"');
                (!dep.is_empty() && !dep.starts_with("dep:")).then(|| dep.to_string())
            })
            .collect();
        direct.insert(name.trim().to_string(), deps);
    }
    assert!(
        direct.contains_key("graph") && direct.contains_key("decode"),
        "Fix: the `[features]` parse found {} features but not the expected ones; the manifest layout changed.",
        direct.len()
    );

    direct
        .keys()
        .map(|start| {
            let mut seen = BTreeSet::new();
            let mut stack = vec![start.clone()];
            while let Some(feature) = stack.pop() {
                if !seen.insert(feature.clone()) {
                    continue;
                }
                stack.extend(direct.get(&feature).into_iter().flatten().cloned());
            }
            (start.clone(), seen)
        })
        .collect()
}

/// `module -> gating feature`, from the `#[cfg(feature = "...")] pub mod x;` pairs
/// in `lib.rs`. A module declared without a `cfg` is absent from the map.
fn module_features(root: &Path) -> BTreeMap<String, String> {
    let lib = std::fs::read_to_string(root.join("src/lib.rs"))
        .expect("Fix: cannot read vyre-primitives/src/lib.rs.");
    let lines: Vec<&str> = lib.lines().collect();
    let mut map = BTreeMap::new();
    for (index, line) in lines.iter().enumerate() {
        let Some(module) = line
            .trim()
            .strip_prefix("pub mod ")
            .or_else(|| line.trim().strip_prefix("mod "))
            .and_then(|rest| rest.strip_suffix(';'))
        else {
            continue;
        };
        if index == 0 {
            continue;
        }
        let Some(feature) = lines[index - 1]
            .trim()
            .strip_prefix("#[cfg(feature = \"")
            .and_then(|rest| rest.strip_suffix("\")]"))
        else {
            continue;
        };
        map.insert(module.to_string(), feature.to_string());
    }
    assert!(
        map.get("graph").is_some_and(|f| f == "graph"),
        "Fix: the `lib.rs` module-to-feature parse did not find `graph`; the declaration layout changed and the gate would exempt every module."
    );
    map
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
        // An op's tests live either in the op's own file or in a `tests/mod.rs`
        // under a directory named for it. Both are the same module.
        let sibling = path.trim_end_matches(".rs").to_string() + "/tests/mod.rs";
        let registered = [path.as_str(), sibling.as_str()].iter().any(|candidate| {
            by_path
                .get(*candidate)
                .is_some_and(|text| text.contains("assert_routes_on_dispatch_span"))
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

/// Guards the walk itself: a gate that finds no files passes vacuously.
#[test]
fn the_source_walk_reaches_the_modules_every_gate_above_depends_on() {
    let files = source_files();
    let paths: BTreeSet<&str> = files.iter().map(|(p, _)| p.as_str()).collect();
    for required in [
        DISPATCH_GRID_OWNER,
        ROUTING_OWNERS[0],
        ROUTING_OWNERS[1],
        "src/math/bellman_shortest_path.rs",
        "src/math/sinkhorn_iterate/tests/mod.rs",
        "src/decode/rle_segment_lengths.rs",
    ] {
        assert!(
            paths.contains(required),
            "Fix: the source walk did not reach `{required}`, so any gate that depends on it passes vacuously."
        );
    }
}

/// Keeps the crate directory resolution honest: a stale path resolves to another
/// checkout, and every gate above would then read the wrong tree.
#[test]
fn the_crate_directory_is_the_one_holding_this_test() {
    let root = vyre_crate_directory("vyre-primitives");
    let manifest = std::fs::read_to_string(root.join("Cargo.toml"))
        .expect("Fix: the resolved crate directory holds no Cargo.toml.");
    assert!(
        manifest.contains("name = \"vyre-primitives\""),
        "Fix: {} is not the vyre-primitives crate directory.",
        root.display()
    );
    assert!(
        root.join("tests/shared_owner_closure.rs").is_file(),
        "Fix: {} does not hold this test file, so the walk is reading a different checkout.",
        PathBuf::from(&root).display()
    );
}
