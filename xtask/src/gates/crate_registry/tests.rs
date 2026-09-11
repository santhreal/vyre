//! Contract tests for the architecture manifest gate.

use super::workspace::*;
use super::*;
use toml::Value;

/// WHY: a `workspace = true` entry that also names features enables both
/// sets, so reading only the local list under-reports the edge.
#[test]
fn inherited_and_local_features_are_unioned() {
    let workspace: toml::Table = toml::from_str("[serde]\nversion = \"1\"\nfeatures = [\"std\"]\n")
        .expect("the workspace table parses");
    let specification: Value =
        toml::from_str::<toml::Table>("workspace = true\nfeatures = [\"derive\"]\n")
            .expect("the specification parses")
            .into();
    let merged = merged_specification("serde", &specification, &workspace);
    assert_eq!(feature_list(&merged), vec!["derive", "std"]);
}

/// WHY: a renamed dependency key is the alias, not the destination, and
/// every rule is keyed on the destination package.
#[test]
fn a_renamed_package_key_is_the_destination() {
    let workspace = toml::Table::new();
    let specification: Value =
        toml::from_str::<toml::Table>("package = \"vyre-libs\"\nversion = \"0.7\"\n")
            .expect("the specification parses")
            .into();
    let merged = merged_specification("libs", &specification, &workspace);
    assert_eq!(
        merged.get("package").and_then(Value::as_str),
        Some("vyre-libs")
    );
}

/// WHY: the two documents are the reviewable form of the manifest, so an
/// empty list has to render as a word rather than as nothing: a blank cell
/// reads as an unfilled table rather than as an edge with no features.
#[test]
fn an_empty_list_renders_as_none() {
    assert_eq!(format_list(&[]), "None");
    assert_eq!(
        format_list(&["gpu".to_string(), "std".to_string()]),
        "`gpu`, `std`"
    );
}

fn manifest(text: &str) -> toml::Table {
    toml::from_str(text).expect("the manifest parses")
}

/// WHY: this is the hidden-edge class. An optional dependency a feature
/// turns on is absent from the default resolution, and the destination
/// features that feature enables are absent from the dependency table's
/// own `features` key. A gate that read only the table would record the
/// edge as carrying no features and no activation, which is the graph
/// cargo resolves with `--no-default-features`, not the one it resolves
/// with `--all-features`.
#[test]
fn a_feature_activated_edge_carries_its_activation_and_its_features() {
    let table = manifest(
            "[dependencies]\nvyre-libs = { path = \"../vyre-libs\", optional = true }\n\
             [features]\ndefault = []\nlibs-compositions = [\"dep:vyre-libs\", \"vyre-libs/encoding\"]\n\
             everything = [\"libs-compositions\"]\n",
        );
    let effect = feature_effect(&table, &BTreeSet::from(["vyre-libs".to_string()]));
    assert_eq!(
        effect.activated_by.get("vyre-libs"),
        Some(&BTreeSet::from([
            "everything".to_string(),
            "libs-compositions".to_string()
        ])),
        "a feature that names another feature activates what that one activates"
    );
    assert_eq!(
        effect.enabled.get("vyre-libs"),
        Some(&BTreeSet::from(["encoding".to_string()]))
    );
    assert!(effect.explicit.contains("vyre-libs"));
}

/// WHY: a weak `dep?/feature` enables a destination feature without
/// activating the edge, so recording it as an activation would report an
/// edge the build never resolves.
#[test]
fn a_weak_feature_reference_is_not_an_activation() {
    let table = manifest(
        "[dependencies]\nserde = { version = \"1\", optional = true }\n\
             [features]\njson = [\"serde?/derive\"]\n",
    );
    let effect = feature_effect(&table, &BTreeSet::from(["serde".to_string()]));
    assert!(effect.activated_by.get("serde").is_none());
    assert_eq!(
        effect.enabled.get("serde"),
        Some(&BTreeSet::from(["derive".to_string()]))
    );
}

/// WHY: cargo derives a feature named after an optional dependency unless
/// some feature spells it `dep:`. The derived feature resolves the edge, so
/// it belongs in `activating_features`, and no line of the manifest
/// declares it, so it is not a named activation. Reading the two as one
/// left the rule that rejects an unnamed optional edge unable to fire.
#[test]
fn an_implicit_feature_is_a_resolution_and_not_a_named_activation() {
    let table = manifest("[dependencies]\nureq = { version = \"2\", optional = true }\n");
    let effect = feature_effect(&table, &BTreeSet::from(["ureq".to_string()]));
    assert!(effect.explicit.is_empty());
    assert!(effect.activated_by.is_empty());

    let named = manifest(
        "[dependencies]\nureq = { version = \"2\", optional = true }\n\
             [features]\nhttp = [\"dep:ureq\"]\n",
    );
    let effect = feature_effect(&named, &BTreeSet::from(["ureq".to_string()]));
    assert_eq!(
        effect.activated_by.get("ureq"),
        Some(&BTreeSet::from(["http".to_string()]))
    );
}

/// One consumer, one dependency, and the rows to judge them by.
struct Case {
    records: Vec<CrateRecord>,
    layers: Vec<LayerRecord>,
    state: WorkspaceState,
}

impl Case {
    fn new(source_layer: &str, source_rank: i64, target_layer: &str, target_rank: i64) -> Self {
        let records = vec![
            CrateRecord {
                package: "consumer".to_string(),
                path: "consumer".to_string(),
                layer: source_layer.to_string(),
                publication_class: "internal-engine".to_string(),
                seam: "consumer-seam".to_string(),
                interface: "consume".to_string(),
                responsibility: "consume".to_string(),
                facade_exported: false,
            },
            CrateRecord {
                package: "dependency".to_string(),
                path: "dependency".to_string(),
                layer: target_layer.to_string(),
                publication_class: "internal-engine".to_string(),
                seam: "dependency-seam".to_string(),
                interface: "be consumed".to_string(),
                responsibility: "be consumed".to_string(),
                facade_exported: false,
            },
        ];
        let mut layers = vec![LayerRecord {
            name: source_layer.to_string(),
            rank: source_rank,
            purpose: "consume".to_string(),
            ..LayerRecord::default()
        }];
        if target_layer != source_layer {
            layers.push(LayerRecord {
                name: target_layer.to_string(),
                rank: target_rank,
                purpose: "be consumed".to_string(),
                ..LayerRecord::default()
            });
        }
        let state = WorkspaceState {
            members: vec!["consumer".to_string(), "dependency".to_string()],
            paths: BTreeMap::from([
                ("consumer".to_string(), "consumer".to_string()),
                ("dependency".to_string(), "dependency".to_string()),
            ]),
            dependencies: BTreeMap::from([
                ("consumer".to_string(), BTreeMap::new()),
                ("dependency".to_string(), BTreeMap::new()),
            ]),
            development: BTreeMap::from([
                ("consumer".to_string(), BTreeMap::new()),
                ("dependency".to_string(), BTreeMap::new()),
            ]),
        };
        Self {
            records,
            layers,
            state,
        }
    }

    /// Declare the one edge, and record the admission a production edge
    /// needs so each case below isolates the rule it changes.
    fn edge(mut self, use_: DependencyUse) -> Self {
        if use_.is_production() {
            let source = self.records[0].layer.clone();
            let target = self.records[1].layer.clone();
            if let Some(layer) = self.layers.iter_mut().find(|layer| layer.name == target) {
                layer.consumed_by = vec![source];
            }
        }
        self.state.dependencies.insert(
            "consumer".to_string(),
            BTreeMap::from([("dependency".to_string(), use_)]),
        );
        self
    }

    fn direction(&self) -> Vec<Finding> {
        direction_findings(&self.state, &self.records, &self.layers)
    }

    fn contract(&self) -> Vec<Finding> {
        contract_findings(&self.state, &self.records)
    }
}

fn production(kinds: &[&str]) -> DependencyUse {
    DependencyUse {
        kinds: kinds.iter().map(|kind| (*kind).to_string()).collect(),
        conditions: vec!["always".to_string()],
        default_features: true,
        ..DependencyUse::default()
    }
}

/// WHY: the rank comparison is the whole direction contract, so the case it
/// exists for has to fail. A consumer in a layer the dependency's layer
/// outranks is a reversal, and an equal rank across two layers is one too:
/// two layers share a rank only when neither depends on the other.
#[test]
fn a_layer_reversal_is_a_finding() {
    let reversed = Case::new("low", 1, "high", 4)
        .edge(production(&["normal"]))
        .direction();
    assert_eq!(reversed.len(), 1, "{reversed:?}");
    assert!(
            reversed[0]
                .message
                .contains("`consumer` in layer `low` (rank 1) depends always on `dependency` in layer `high` (rank 4)"),
            "{reversed:?}"
        );
    let equal = Case::new("left", 3, "right", 3)
        .edge(production(&["normal"]))
        .direction();
    assert_eq!(equal.len(), 1, "{equal:?}");
    assert!(Case::new("high", 4, "low", 1)
        .edge(production(&["normal"]))
        .direction()
        .is_empty());
}

/// WHY: an edge that exists only when a feature is on is still an edge the
/// build resolves, and it is the one a gate reading the default resolution
/// never sees. It is held to the same rank rule, and the finding names the
/// feature so a reader knows which build has it.
#[test]
fn a_feature_activated_reversal_is_a_finding_naming_the_feature() {
    let findings = Case::new("low", 1, "high", 4)
        .edge(DependencyUse {
            optional: true,
            activating_features: vec!["extra".to_string()],
            ..production(&["normal"])
        })
        .direction();
    assert_eq!(findings.len(), 1, "{findings:?}");
    assert!(
        findings[0]
            .message
            .contains("depends under feature `extra` on `dependency`"),
        "{findings:?}"
    );
}

/// WHY: a dev-dependency on a higher layer is how a crate tests against the
/// facade that consumes it. Cargo resolves it in a separate graph that
/// cannot form a production cycle, so judging it would reject the intended
/// shape. A build-dependency is not exempt: cargo links it.
#[test]
fn a_development_edge_carries_no_direction() {
    assert!(Case::new("low", 1, "high", 4)
        .edge(production(&["dev"]))
        .direction()
        .is_empty());
    assert_eq!(
        Case::new("low", 1, "high", 4)
            .edge(production(&["dev", "normal"]))
            .direction()
            .len(),
        1
    );
    assert_eq!(
        Case::new("low", 1, "high", 4)
            .edge(production(&["build"]))
            .direction()
            .len(),
        1
    );
}

/// WHY: a layer a crate row names and no `[[layer]]` row declares has no
/// rank, so every edge touching it would be skipped rather than judged. The
/// unranked layer itself is the finding, and so is a declared layer no
/// member occupies: it is a rank nothing is held to.
#[test]
fn an_unmatched_layer_is_a_finding() {
    let mut case = Case::new("undeclared", 0, "undeclared", 0);
    case.layers = vec![LayerRecord {
        name: "empty".to_string(),
        rank: 0,
        purpose: "nothing".to_string(),
        ..LayerRecord::default()
    }];
    let findings = case.direction();
    assert_eq!(findings.len(), 3, "{findings:?}");
}

/// WHY: `consumed_by` is the rule rank cannot state. Rank permits a runtime
/// crate to reach a composition library; the admitted set decides whether
/// it may, and every layer states one. An edge from a layer the destination
/// does not admit is the undeclared half, and an admitted layer no edge
/// crosses is the stale half.
#[test]
fn an_unadmitted_consumer_layer_is_a_finding_and_an_unused_admission_is_too() {
    let mut case = Case::new("high", 4, "low", 1).edge(production(&["normal"]));
    case.layers[1].consumed_by = vec!["other".to_string()];
    case.layers.push(LayerRecord {
        name: "other".to_string(),
        rank: 3,
        purpose: "elsewhere".to_string(),
        ..LayerRecord::default()
    });
    case.records.push(CrateRecord {
        package: "other-member".to_string(),
        path: "other-member".to_string(),
        layer: "other".to_string(),
        publication_class: "internal-engine".to_string(),
        seam: "other-seam".to_string(),
        interface: "elsewhere".to_string(),
        responsibility: "elsewhere".to_string(),
        facade_exported: false,
    });
    let findings = case.direction();
    assert!(
        findings.iter().any(|finding| finding
            .message
            .contains("layer `low` admits no consumer in `high`")),
        "{findings:?}"
    );
    assert!(
        findings.iter().any(|finding| finding.message.contains(
            "layer `low` admits consumer layer `other` and no production edge crosses it"
        )),
        "{findings:?}"
    );
}

/// WHY: rank does not judge an edge inside one layer, so the admitted set
/// is the only thing that does. A layer whose members reach each other
/// records its own name, and one that does not rejects the edge. Without
/// this the five layers that carry an intra-layer edge admitted every
/// further edge among their own members with nothing recording it.
#[test]
fn an_intra_layer_edge_needs_the_layer_to_admit_itself() {
    let admitted = Case::new("libraries", 3, "libraries", 3)
        .edge(production(&["normal"]))
        .direction();
    assert!(admitted.is_empty(), "{admitted:?}");

    let mut case = Case::new("libraries", 3, "libraries", 3).edge(production(&["normal"]));
    case.layers[0].consumed_by.clear();
    let findings = case.direction();
    assert_eq!(findings.len(), 1, "{findings:?}");
    assert!(
        findings[0]
            .message
            .contains("layer `libraries` admits no consumer in `libraries`"),
        "{findings:?}"
    );
}

/// WHY: this is the facade rule, and rank cannot express it. The exporting
/// layer outranks everything, so rank permits it to reach any crate in the
/// workspace; the export flag on the destination is what keeps the curated
/// surface from re-exporting an emitter. A flag no exporting edge crosses
/// is a stale declaration.
#[test]
fn an_unexported_seam_reached_from_an_exporting_layer_is_a_finding() {
    let mut case = Case::new("facade", 6, "emitter", 2).edge(production(&["normal"]));
    case.layers[0].exports_declared_seams = true;
    let findings = case.direction();
    assert_eq!(findings.len(), 1, "{findings:?}");
    assert!(
            findings[0].message.contains(
                "`consumer` exports declared seams and depends always on `dependency`, whose row does not set `facade_exported`"
            ),
            "{findings:?}"
        );

    case.records[1].facade_exported = true;
    assert!(case.direction().is_empty());

    let mut stale = Case::new("facade", 6, "emitter", 2);
    stale.layers[0].exports_declared_seams = true;
    stale.records[1].facade_exported = true;
    let findings = stale.direction();
    assert!(
        findings.iter().any(|finding| finding.message.contains(
            "`dependency` declares `facade_exported` and no exporting layer depends on it"
        )),
        "{findings:?}"
    );
}

/// WHY: a member cargo resolves and no row describes has no layer and no
/// seam, so every rule silently skips it. Adding a workspace member has to
/// turn this red.
#[test]
fn an_undeclared_member_is_a_finding() {
    let mut case = Case::new("high", 4, "low", 1);
    case.state.members.push("newcomer".to_string());
    case.state
        .paths
        .insert("newcomer".to_string(), "newcomer".to_string());
    let findings = case.contract();
    assert!(
        findings
            .iter()
            .any(|finding| finding.message == "workspace member `newcomer` has no manifest row"),
        "{findings:?}"
    );
    assert!(
        findings
            .iter()
            .any(|finding| finding.message == "workspace package `newcomer` has no manifest row"),
        "{findings:?}"
    );
}

/// WHY: a row describing no member is the surviving stale-declaration
/// class. The per-edge roster it replaced could go stale one edge at a
/// time; a member row can only go stale as a whole, and it does so loudly.
#[test]
fn a_row_that_is_not_a_member_is_a_finding() {
    let mut case = Case::new("high", 4, "low", 1);
    case.records.push(CrateRecord {
        package: "vyre-ghost".to_string(),
        path: "vyre-ghost".to_string(),
        layer: "low".to_string(),
        publication_class: "internal-engine".to_string(),
        seam: "ghost".to_string(),
        interface: "nothing".to_string(),
        responsibility: "nothing".to_string(),
        facade_exported: false,
    });
    let findings = case.contract();
    assert!(
            findings
                .iter()
                .any(|finding| finding.message
                    == "manifest row `vyre-ghost` is not a workspace member"),
            "{findings:?}"
        );
}

/// WHY: two rows claiming one seam name leave every edge into that seam
/// crossing an interface with two owners, which is what the 22-package
/// split produced: 23 packages owned `product-libraries`.
#[test]
fn two_rows_owning_one_seam_is_a_finding() {
    let mut case = Case::new("high", 4, "low", 1);
    case.records[1].seam = "consumer-seam".to_string();
    let findings = case.contract();
    assert!(
        findings.iter().any(|finding| finding
            .message
            .contains("`consumer` and `dependency` both own the `consumer-seam` seam")),
        "{findings:?}"
    );
}

/// WHY: the publication class decides whether a crate is published and
/// whether a consumer may depend on it. It has one home, beside the
/// `publish` key it qualifies, and this reader is the only thing that finds
/// it: a class written into the architecture manifest instead is a retired
/// key, not a second source, so nothing here may fall back to one.
#[test]
fn the_publication_class_is_read_only_from_the_member_manifest() {
    let declared: toml::Table = toml::from_str(
        "[package]\nname = \"a\"\n[package.metadata.vyre]\npublication_class = \"extension-sdk\"\n",
    )
    .expect("the fixture manifest parses");
    assert_eq!(
        manifest_publication_class(&declared).as_deref(),
        Some("extension-sdk")
    );

    for elsewhere in [
        "[package]\nname = \"a\"\n",
        "[package]\nname = \"a\"\npublication_class = \"extension-sdk\"\n",
        "[package]\nname = \"a\"\n[package.metadata]\npublication_class = \"extension-sdk\"\n",
        "[package]\nname = \"a\"\n[metadata.vyre]\npublication_class = \"extension-sdk\"\n",
    ] {
        let manifest: toml::Table = toml::from_str(elsewhere).expect("the fixture manifest parses");
        assert_eq!(
            manifest_publication_class(&manifest),
            None,
            "`{MANIFEST_PUBLICATION_KEY}` is the only home; `{elsewhere}` declares no class"
        );
    }
}

/// WHY: internal production dependency cycles, including intra-layer ones
/// that the rank rule cannot see, must fail closed with a finding naming
/// the exact cycle path.
#[test]
fn dependency_cycle_is_a_finding() {
    let state = WorkspaceState {
        members: vec!["a".to_string(), "b".to_string()],
        paths: BTreeMap::from([
            ("a".to_string(), "a".to_string()),
            ("b".to_string(), "b".to_string()),
        ]),
        development: BTreeMap::new(),
        dependencies: BTreeMap::from([
            (
                "a".to_string(),
                BTreeMap::from([("b".to_string(), production(&["normal"]))]),
            ),
            (
                "b".to_string(),
                BTreeMap::from([("a".to_string(), production(&["normal"]))]),
            ),
        ]),
    };
    let findings = cycle_findings(&state);
    assert_eq!(findings.len(), 1, "{findings:?}");
    assert!(findings[0]
        .message
        .contains("dependency cycle detected: a -> b -> a"));
}

/// WHY: an optional dependency no feature names is an edge no build
/// resolves. It reads as a live dependency to anyone counting rebuild
/// fan-out, and it is the shape a half-finished feature rename leaves.
#[test]
fn an_unactivatable_optional_edge_is_a_finding() {
    let state = WorkspaceState {
        members: vec!["a".to_string()],
        paths: BTreeMap::from([("a".to_string(), "a".to_string())]),
        development: BTreeMap::new(),
        dependencies: BTreeMap::from([(
            "a".to_string(),
            BTreeMap::from([(
                "b".to_string(),
                DependencyUse {
                    optional: true,
                    ..production(&["normal"])
                },
            )]),
        )]),
    };
    let findings = activation_findings(&state);
    assert_eq!(findings.len(), 1, "{findings:?}");
    assert!(findings[0]
        .message
        .contains("`a` declares `b` optional and no feature activates it"));
}
