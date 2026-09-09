//! Publication class validation and release ordering derivation.
//!
//! Asserts that:
//! 1. Every workspace member declares an explicit, valid publication class in both its
//!    Cargo.toml manifest under `[package.metadata.vyre.publication_class]` and in
//!    `docs/CRATE_OWNERSHIP.toml`.
//! 2. No publishable package depends on an internal `publish = false` package via normal or
//!    build dependencies.
//! 3. Newly publishable packages are caught and validated against the declared class roster.
//! 4. Release ordering of publishable packages is derived topologically from the dependency DAG.

use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::path::{Path, PathBuf};

/// Valid publication classes for workspace members.
pub(crate) const VALID_PUBLICATION_CLASSES: &[&str] = &[
    "stable-consumer-sdk",
    "extension-sdk",
    "concrete-backend",
    "internal-engine",
    "conformance-tooling",
    "private-test-support",
];

/// Known publishable package roster.
pub(crate) const EXPECTED_PUBLISHABLE_PACKAGES: &[&str] = &[
    "vyre",
    "vyre-foundation",
    "vyre-megakernel",
    "vyre-driver",
    "vyre-driver-metal",
    "vyre-driver-wgpu",
    "vyre-driver-spirv",
    "vyre-driver-cuda",
    "vyre-driver-reference",
    "vyre-reference",
    "vyre-spec",
    "vyre-macros",
    "vyre-primitives",
    "vyre-pass-engine",
    "vyre-runtime",
    "vyre-safetensors",
    "vyre-libs",
    "vyre-aot",
    "vyre-lints",
    "vyre-lower",
    "vyre-emit-naga",
    "vyre-emit-ptx",
    "vyre-emit-spirv",
    "vyre-emit-metal",
    "vyre-debug",
];

fn workspace_root() -> PathBuf {
    let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    manifest_dir.parent().expect("workspace root").to_path_buf()
}

#[derive(Debug, Clone)]
#[allow(dead_code)]
pub(crate) struct MemberInfo {
    pub(crate) name: String,
    pub(crate) path: String,
    pub(crate) publication_class: Option<String>,
    pub(crate) publish: bool,
    pub(crate) normal_deps: Vec<String>,
    pub(crate) build_deps: Vec<String>,
    pub(crate) dev_deps: Vec<String>,
}

fn load_workspace_members(root: &Path) -> BTreeMap<String, MemberInfo> {
    let root_cargo_path = root.join("Cargo.toml");
    let root_cargo_str = std::fs::read_to_string(&root_cargo_path)
        .expect("root Cargo.toml must exist and be readable");
    let root_toml: toml::Value = toml::from_str(&root_cargo_str).expect("parse root Cargo.toml");

    let members = root_toml
        .get("workspace")
        .and_then(|w| w.get("members"))
        .and_then(|m| m.as_array())
        .expect("workspace.members must be an array");

    let mut map = BTreeMap::new();

    for m in members {
        let member_path_str = m.as_str().expect("member must be a string");
        let member_cargo_path = root.join(member_path_str).join("Cargo.toml");
        let content = std::fs::read_to_string(&member_cargo_path)
            .unwrap_or_else(|e| panic!("failed to read {}: {e}", member_cargo_path.display()));
        let member_toml: toml::Value = toml::from_str(&content).expect("parse member Cargo.toml");

        let pkg = member_toml.get("package").expect("package table");
        let name = pkg
            .get("name")
            .and_then(|n| n.as_str())
            .unwrap_or(member_path_str)
            .to_string();

        let publish = match pkg.get("publish") {
            Some(toml::Value::Boolean(b)) => *b,
            Some(toml::Value::Array(arr)) => !arr.is_empty(),
            _ => true,
        };

        let pub_class = pkg
            .get("metadata")
            .and_then(|m| m.get("vyre"))
            .and_then(|v| v.get("publication_class"))
            .and_then(|c| c.as_str())
            .map(|s| s.to_string());

        let mut normal_deps = Vec::new();
        let mut build_deps = Vec::new();
        let mut dev_deps = Vec::new();

        if let Some(deps) = member_toml.get("dependencies").and_then(|d| d.as_table()) {
            normal_deps.extend(deps.keys().cloned());
        }
        if let Some(deps) = member_toml
            .get("build-dependencies")
            .and_then(|d| d.as_table())
        {
            build_deps.extend(deps.keys().cloned());
        }
        if let Some(deps) = member_toml
            .get("dev-dependencies")
            .and_then(|d| d.as_table())
        {
            dev_deps.extend(deps.keys().cloned());
        }

        if let Some(target) = member_toml.get("target").and_then(|t| t.as_table()) {
            for (_, target_val) in target {
                if let Some(deps) = target_val.get("dependencies").and_then(|d| d.as_table()) {
                    normal_deps.extend(deps.keys().cloned());
                }
                if let Some(deps) = target_val
                    .get("build-dependencies")
                    .and_then(|d| d.as_table())
                {
                    build_deps.extend(deps.keys().cloned());
                }
                if let Some(deps) = target_val
                    .get("dev-dependencies")
                    .and_then(|d| d.as_table())
                {
                    dev_deps.extend(deps.keys().cloned());
                }
            }
        }

        map.insert(
            name.clone(),
            MemberInfo {
                name,
                path: member_path_str.to_string(),
                publication_class: pub_class,
                publish,
                normal_deps,
                build_deps,
                dev_deps,
            },
        );
    }

    map
}

fn load_ownership_classes(root: &Path) -> BTreeMap<String, String> {
    let ownership_path = root.join("docs/CRATE_OWNERSHIP.toml");
    let content =
        std::fs::read_to_string(&ownership_path).expect("CRATE_OWNERSHIP.toml must exist");
    let toml_val: toml::Value = toml::from_str(&content).expect("parse CRATE_OWNERSHIP.toml");

    let mut map = BTreeMap::new();
    if let Some(crates) = toml_val.get("crate").and_then(|c| c.as_array()) {
        for c in crates {
            let pkg = c.get("package").and_then(|p| p.as_str()).expect("package");
            let pub_class = c
                .get("publication_class")
                .and_then(|p| p.as_str())
                .unwrap_or_default()
                .to_string();
            map.insert(pkg.to_string(), pub_class);
        }
    }
    map
}

/// Validates that all members declare explicit and valid publication classes.
pub(crate) fn validate_publication_classes(
    members: &BTreeMap<String, MemberInfo>,
    ownership_classes: &BTreeMap<String, String>,
) -> Result<(), Vec<String>> {
    let mut errors = Vec::new();

    for (name, info) in members {
        let Some(manifest_class) = &info.publication_class else {
            errors.push(format!(
                "member `{name}` ({}) does not declare `[package.metadata.vyre.publication_class]`",
                info.path
            ));
            continue;
        };

        if !VALID_PUBLICATION_CLASSES.contains(&manifest_class.as_str()) {
            errors.push(format!(
                "member `{name}` declares invalid publication_class `{manifest_class}`; must be one of: {}",
                VALID_PUBLICATION_CLASSES.join(", ")
            ));
        }

        if let Some(ownership_class) = ownership_classes.get(name) {
            if ownership_class != manifest_class {
                errors.push(format!(
                    "member `{name}` publication_class mismatch: Cargo.toml declares `{manifest_class}` but CRATE_OWNERSHIP.toml declares `{ownership_class}`"
                ));
            }
        } else {
            errors.push(format!(
                "member `{name}` has no entry in docs/CRATE_OWNERSHIP.toml"
            ));
        }
    }

    if errors.is_empty() {
        Ok(())
    } else {
        Err(errors)
    }
}

/// Validates that no publishable package depends on a non-publishable package via normal or build deps.
pub(crate) fn validate_publishable_dependency_closure(
    members: &BTreeMap<String, MemberInfo>,
) -> Result<(), Vec<String>> {
    let mut errors = Vec::new();

    for (name, info) in members {
        if !info.publish {
            continue;
        }

        for dep in &info.normal_deps {
            if let Some(dep_info) = members.get(dep) {
                if !dep_info.publish {
                    errors.push(format!(
                        "publishable crate `{name}` has normal dependency on non-publishable crate `{dep}` (publish = false)"
                    ));
                }
            }
        }

        for dep in &info.build_deps {
            if let Some(dep_info) = members.get(dep) {
                if !dep_info.publish {
                    errors.push(format!(
                        "publishable crate `{name}` has build-dependency on non-publishable crate `{dep}` (publish = false)"
                    ));
                }
            }
        }
    }

    if errors.is_empty() {
        Ok(())
    } else {
        Err(errors)
    }
}

/// Validates that every publishable crate is known in the expected roster.
pub(crate) fn validate_publishable_roster(
    members: &BTreeMap<String, MemberInfo>,
) -> Result<(), Vec<String>> {
    let mut errors = Vec::new();
    let expected_set: BTreeSet<&str> = EXPECTED_PUBLISHABLE_PACKAGES.iter().copied().collect();

    for (name, info) in members {
        if info.publish && !expected_set.contains(name.as_str()) {
            errors.push(format!(
                "unexpected newly publishable package `{name}` appeared without updated roster and publication class policy"
            ));
        }
    }

    if errors.is_empty() {
        Ok(())
    } else {
        Err(errors)
    }
}

/// Computes the topological release order for all publishable packages.
pub(crate) fn derive_release_ordering(
    members: &BTreeMap<String, MemberInfo>,
) -> Result<Vec<String>, String> {
    let publishable_names: BTreeSet<String> = members
        .iter()
        .filter(|(_, info)| info.publish)
        .map(|(name, _)| name.clone())
        .collect();

    // in-degree and adjacency for the publishable subgraph
    let mut in_degree: BTreeMap<String, usize> = BTreeMap::new();
    let mut adj: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();

    for name in &publishable_names {
        in_degree.insert(name.clone(), 0);
        adj.insert(name.clone(), BTreeSet::new());
    }

    for (name, info) in members {
        if !info.publish {
            continue;
        }
        let mut deps = BTreeSet::new();
        for dep in &info.normal_deps {
            if publishable_names.contains(dep) && dep != name {
                deps.insert(dep.clone());
            }
        }
        for dep in &info.build_deps {
            if publishable_names.contains(dep) && dep != name {
                deps.insert(dep.clone());
            }
        }

        for dep in deps {
            adj.get_mut(&dep).unwrap().insert(name.clone());
            *in_degree.get_mut(name).unwrap() += 1;
        }
    }

    // Kahn's algorithm
    let mut queue: VecDeque<String> = in_degree
        .iter()
        .filter(|(_, &deg)| deg == 0)
        .map(|(name, _)| name.clone())
        .collect();

    let mut order = Vec::new();

    while let Some(node) = queue.pop_front() {
        order.push(node.clone());
        if let Some(neighbors) = adj.get(&node) {
            for neighbor in neighbors {
                let deg = in_degree.get_mut(neighbor).unwrap();
                *deg -= 1;
                if *deg == 0 {
                    queue.push_back(neighbor.clone());
                }
            }
        }
    }

    if order.len() != publishable_names.len() {
        return Err(format!(
            "cycle detected in publishable dependency graph; resolved only {} of {} packages",
            order.len(),
            publishable_names.len()
        ));
    }

    Ok(order)
}

#[test]
fn every_workspace_member_declares_explicit_valid_publication_class() {
    let root = workspace_root();
    let members = load_workspace_members(&root);
    let ownership_classes = load_ownership_classes(&root);

    assert!(
        members.len() >= 30,
        "expected at least 30 workspace members, found {}",
        members.len()
    );

    let result = validate_publication_classes(&members, &ownership_classes);
    if let Err(errors) = result {
        panic!(
            "publication class validation failed with {} error(s):\n  {}",
            errors.len(),
            errors.join("\n  ")
        );
    }
}

#[test]
fn publishable_packages_depend_only_on_publishable_packages() {
    let root = workspace_root();
    let members = load_workspace_members(&root);

    let result = validate_publishable_dependency_closure(&members);
    if let Err(errors) = result {
        panic!(
            "publishable dependency closure check failed with {} error(s):\n  {}",
            errors.len(),
            errors.join("\n  ")
        );
    }
}

#[test]
fn publishable_roster_matches_expected_packages() {
    let root = workspace_root();
    let members = load_workspace_members(&root);

    let result = validate_publishable_roster(&members);
    if let Err(errors) = result {
        panic!(
            "publishable roster check failed with {} error(s):\n  {}",
            errors.len(),
            errors.join("\n  ")
        );
    }
}

#[test]
fn release_ordering_is_derivable_from_dependency_graph() {
    let root = workspace_root();
    let members = load_workspace_members(&root);

    let order = derive_release_ordering(&members)
        .expect("release ordering must derive cleanly without cycles");

    assert!(
        order.len() == EXPECTED_PUBLISHABLE_PACKAGES.len(),
        "expected {} publishable packages in release order, got {}",
        EXPECTED_PUBLISHABLE_PACKAGES.len(),
        order.len()
    );

    // Verify topological property: for any package in the order, all its publishable dependencies appear earlier
    let positions: BTreeMap<String, usize> = order
        .iter()
        .enumerate()
        .map(|(idx, name)| (name.clone(), idx))
        .collect();

    for (name, info) in &members {
        if !info.publish {
            continue;
        }
        let pkg_pos = positions[name];
        for dep in &info.normal_deps {
            if let Some(&dep_pos) = positions.get(dep) {
                assert!(
                    dep_pos < pkg_pos,
                    "release order violation: dependency `{dep}` (at {dep_pos}) must precede `{name}` (at {pkg_pos})"
                );
            }
        }
    }
}

#[test]
fn mutation_missing_publication_class_is_caught() {
    let root = workspace_root();
    let mut members = load_workspace_members(&root);
    let ownership_classes = load_ownership_classes(&root);

    // Mutate: strip publication class from vyre
    if let Some(vyre_info) = members.get_mut("vyre") {
        vyre_info.publication_class = None;
    }

    let result = validate_publication_classes(&members, &ownership_classes);
    assert!(
        result.is_err(),
        "stripping publication class must fail validation"
    );
    let errors = result.unwrap_err();
    assert!(errors
        .iter()
        .any(|e| e.contains("does not declare `[package.metadata.vyre.publication_class]`")));
}

#[test]
fn mutation_invalid_publication_class_is_caught() {
    let root = workspace_root();
    let mut members = load_workspace_members(&root);
    let ownership_classes = load_ownership_classes(&root);

    // Mutate: set invalid publication class
    if let Some(vyre_info) = members.get_mut("vyre") {
        vyre_info.publication_class = Some("invalid-class-xyz".to_string());
    }

    let result = validate_publication_classes(&members, &ownership_classes);
    assert!(
        result.is_err(),
        "invalid publication class must fail validation"
    );
    let errors = result.unwrap_err();
    assert!(errors
        .iter()
        .any(|e| e.contains("declares invalid publication_class")));
}

#[test]
fn mutation_unpublished_dependency_in_publishable_crate_is_caught() {
    let root = workspace_root();
    let mut members = load_workspace_members(&root);

    // Mutate: make vyre depend on vyre-registry-link (publish = false)
    if let Some(vyre_info) = members.get_mut("vyre") {
        vyre_info.normal_deps.push("vyre-registry-link".to_string());
    }

    let result = validate_publishable_dependency_closure(&members);
    assert!(
        result.is_err(),
        "depending on publish = false package must fail validation"
    );
    let errors = result.unwrap_err();
    assert!(
        errors
            .iter()
            .any(|e| e
                .contains("has normal dependency on non-publishable crate `vyre-registry-link`"))
    );
}

#[test]
fn mutation_newly_publishable_unclassified_package_is_caught() {
    let root = workspace_root();
    let mut members = load_workspace_members(&root);

    // Mutate: make a new package publishable
    members.insert(
        "vyre-unclassified-new-pkg".to_string(),
        MemberInfo {
            name: "vyre-unclassified-new-pkg".to_string(),
            path: "vyre-unclassified-new-pkg".to_string(),
            publication_class: Some("stable-consumer-sdk".to_string()),
            publish: true,
            normal_deps: vec![],
            build_deps: vec![],
            dev_deps: vec![],
        },
    );

    let result = validate_publishable_roster(&members);
    assert!(
        result.is_err(),
        "unexpected publishable package must fail validation"
    );
    let errors = result.unwrap_err();
    assert!(errors
        .iter()
        .any(|e| e.contains("unexpected newly publishable package `vyre-unclassified-new-pkg`")));
}
