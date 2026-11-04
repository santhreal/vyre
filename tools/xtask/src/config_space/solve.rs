//! Whether the configuration space has a valid assignment, and a witness for it.
//!
//! The previous verdict was `no other finding was reported`, which restates the
//! rest of the gate and can never fail on its own claim: it consulted no
//! constraint, no target and no default selection, so a genuinely conflicting
//! space would have been called satisfiable the moment the naming and ownership
//! scans came back clean. This evaluates the declared constraints over the
//! cells the model schedules, and reports the assignment it proved.

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

use super::feature_graph::{BodyEntry, FeatureGraph, RESERVED_DEFAULT};
use super::roster::{Constraints, FacadeRoster};

/// One conflict that leaves a cell with no valid assignment.
#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
pub struct ConstraintConflict {
    /// The cell that has no assignment.
    pub cell: String,
    /// What the cell violates.
    pub violated: String,
}

/// The assignment the model proved for the widest cell it schedules.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct SatisfyingAssignment {
    /// The cell the assignment belongs to.
    pub cell: String,
    /// Every `package/feature` the cell enables, in sorted order.
    pub enabled: Vec<String>,
    /// Every package the cell activates, in sorted order.
    pub activated: Vec<String>,
}

/// The verdict on the configuration space.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum Satisfiability {
    /// Every scheduled cell has an assignment that violates no constraint, and
    /// the witness is one such assignment.
    Satisfiable(SatisfyingAssignment),
    /// At least one scheduled cell violates a constraint, so the space has no
    /// valid configuration there.
    Unsatisfiable(Vec<ConstraintConflict>),
}

impl Satisfiability {
    /// Whether the space has a valid assignment everywhere it is scheduled.
    #[must_use]
    pub fn is_satisfiable(&self) -> bool {
        match self {
            Self::Satisfiable(_) => true,
            Self::Unsatisfiable(_) => false,
        }
    }

    /// The conflicts a caller reports, empty when the space is satisfiable.
    #[must_use]
    pub fn conflicts(&self) -> &[ConstraintConflict] {
        match self {
            Self::Satisfiable(_) => &[],
            Self::Unsatisfiable(conflicts) => conflicts,
        }
    }
}

/// Everything one feature selection turns on.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct Activation {
    /// Every `(package, feature)` the selection enables.
    pub enabled: BTreeSet<(String, String)>,
    /// Every package the selection activates.
    pub activated: BTreeSet<String>,
}

/// Propagate `selection` of `package` across the whole feature graph.
///
/// A weak forward `X?/G` fires only once something else activates `X`, so the
/// pass repeats until no deferred edge becomes live. That is the same fixed
/// point cargo resolves, and skipping it would report a closure a build never
/// produces.
#[must_use]
pub fn activate(graph: &FeatureGraph, package: &str, selection: &[String]) -> Activation {
    let mut state = Activation::default();
    state.activated.insert(package.to_string());
    let mut pending: Vec<(String, String)> = selection
        .iter()
        .map(|feature| (package.to_string(), feature.clone()))
        .collect();
    let mut deferred: Vec<(String, String)> = Vec::new();

    loop {
        while let Some((owner, feature)) = pending.pop() {
            let Some(table) = graph.packages.get(&owner) else {
                continue;
            };
            if !table.features.contains_key(&feature) {
                // An implicit optional-dependency feature activates the
                // dependency and nothing else.
                if table.optional_dependencies.contains(&feature) {
                    activate_package(graph, &feature, &mut state, &mut pending);
                }
                continue;
            }
            if !state.enabled.insert((owner.clone(), feature.clone())) {
                continue;
            }
            for entry in &table.features[&feature] {
                match entry {
                    BodyEntry::Local { feature } => pending.push((owner.clone(), feature.clone())),
                    BodyEntry::Activates { dependency } => {
                        activate_package(graph, dependency, &mut state, &mut pending);
                    }
                    BodyEntry::Forwards {
                        dependency,
                        feature,
                        weak,
                    } => {
                        if *weak {
                            deferred.push((dependency.clone(), feature.clone()));
                        } else {
                            activate_package(graph, dependency, &mut state, &mut pending);
                            pending.push((dependency.clone(), feature.clone()));
                        }
                    }
                }
            }
        }
        let live: Vec<(String, String)> = deferred
            .iter()
            .filter(|(dependency, feature)| {
                state.activated.contains(dependency)
                    && !state
                        .enabled
                        .contains(&(dependency.clone(), feature.clone()))
            })
            .cloned()
            .collect();
        if live.is_empty() {
            break;
        }
        pending.extend(live);
    }

    // A package a cell activates drags in its own non-optional dependencies.
    let mut frontier: Vec<String> = state.activated.iter().cloned().collect();
    while let Some(owner) = frontier.pop() {
        let Some(table) = graph.packages.get(&owner) else {
            continue;
        };
        for dependency in &table.dependencies {
            if table.optional_dependencies.contains(dependency) {
                continue;
            }
            if graph.packages.contains_key(dependency) && state.activated.insert(dependency.clone())
            {
                frontier.push(dependency.clone());
            }
        }
    }

    state
}

/// Activate `dependency` with its own default selection.
fn activate_package(
    graph: &FeatureGraph,
    dependency: &str,
    state: &mut Activation,
    pending: &mut Vec<(String, String)>,
) {
    if !state.activated.insert(dependency.to_string()) {
        return;
    }
    if graph.packages.contains_key(dependency) {
        pending.push((dependency.to_string(), RESERVED_DEFAULT.to_string()));
    }
}

/// Every cell the model proves an assignment for.
///
/// The cells are the selections a build can actually ask for: nothing, the
/// default, every declared feature one at a time, and everything at once. A
/// constraint that no cell can violate constrains nothing, and one that the
/// widest cell violates makes the space unsatisfiable.
#[must_use]
pub fn cells(graph: &FeatureGraph) -> Vec<(String, String, Vec<String>)> {
    let mut cells = Vec::new();
    for (package, table) in &graph.packages {
        let names: Vec<String> = table
            .enable_able()
            .into_iter()
            .filter(|name| name != RESERVED_DEFAULT)
            .collect();
        cells.push((
            package.clone(),
            format!("{package} --no-default-features"),
            Vec::new(),
        ));
        if table.features.contains_key(RESERVED_DEFAULT) {
            cells.push((
                package.clone(),
                format!("{package} (default features)"),
                vec![RESERVED_DEFAULT.to_string()],
            ));
        }
        for name in &names {
            cells.push((
                package.clone(),
                format!("{package} --no-default-features --features {name}"),
                vec![name.clone()],
            ));
        }
        if names.len() > 1 {
            cells.push((package.clone(), format!("{package} --all-features"), names));
        }
    }
    cells
}

/// Prove the declared constraints over every scheduled cell.
///
/// `roster_findings` collects the constraints that name nothing the workspace
/// declares. A constraint over an absent feature can never bind, so leaving it
/// in the record is the same as having no constraint while reading as one.
#[must_use]
pub fn solve(
    graph: &FeatureGraph,
    roster: &FacadeRoster,
    constraints: &Constraints,
    roster_findings: &mut Vec<String>,
) -> Satisfiability {
    let mut exclusive: Vec<&super::roster::ExclusiveFeatures> = Vec::new();
    for pair in &constraints.exclusive {
        let declared = graph.packages.get(&pair.package);
        let missing: Vec<&str> = pair
            .features
            .iter()
            .filter(|feature| {
                declared.is_none_or(|table| !table.features.contains_key(feature.as_str()))
            })
            .map(String::as_str)
            .collect();
        if missing.is_empty() {
            exclusive.push(pair);
        } else {
            roster_findings.push(format!(
                "Constraint on `{}` names {} which the package does not declare, so it can never bind",
                pair.package,
                missing
                    .iter()
                    .map(|feature| format!("`{feature}`"))
                    .collect::<Vec<_>>()
                    .join(" and "),
            ));
        }
    }

    // A compiler-self-use domain cannot enter the default consumer
    // closure. The domain packages declare their own publication class, so the
    // rule reads it rather than repeating a list of them.
    let internal: BTreeMap<&str, &str> = roster
        .feature
        .iter()
        .filter(|entry| {
            graph
                .packages
                .get(&entry.domain)
                .is_some_and(|table| table.publication_class == "internal-engine")
        })
        .map(|entry| (entry.name.as_str(), entry.domain.as_str()))
        .collect();

    let mut conflicts = Vec::new();
    let mut widest: Option<SatisfyingAssignment> = None;

    for (package, cell, selection) in cells(graph) {
        let activation = activate(graph, &package, &selection);
        for pair in &exclusive {
            let [first, second] = &pair.features;
            if activation
                .enabled
                .contains(&(pair.package.clone(), first.clone()))
                && activation
                    .enabled
                    .contains(&(pair.package.clone(), second.clone()))
            {
                conflicts.push(ConstraintConflict {
                    cell: cell.clone(),
                    violated: format!(
                        "`{}` enables both `{first}` and `{second}`, which cannot coexist: {}",
                        pair.package, pair.reason
                    ),
                });
            }
        }
        if package == roster.package
            && selection.first().map(String::as_str) == Some(RESERVED_DEFAULT)
        {
            for (feature, domain) in &internal {
                if activation
                    .enabled
                    .contains(&(roster.package.clone(), (*feature).to_string()))
                {
                    conflicts.push(ConstraintConflict {
                        cell: cell.clone(),
                        violated: format!(
                            "the default consumer closure reaches `{feature}`, whose owning package `{domain}` is internal-engine"
                        ),
                    });
                }
            }
        }
        let candidate = SatisfyingAssignment {
            cell,
            enabled: activation
                .enabled
                .iter()
                .map(|(owner, feature)| format!("{owner}/{feature}"))
                .collect(),
            activated: activation.activated.into_iter().collect(),
        };
        if widest
            .as_ref()
            .is_none_or(|held| held.enabled.len() < candidate.enabled.len())
        {
            widest = Some(candidate);
        }
    }

    if conflicts.is_empty() {
        // `cells` always yields the no-features cell for every member, and the
        // graph is never empty, so a witness always exists here.
        widest.map_or_else(
            || {
                Satisfiability::Unsatisfiable(vec![ConstraintConflict {
                    cell: "workspace".to_string(),
                    violated: "the workspace schedules no configuration cell at all".to_string(),
                }])
            },
            Satisfiability::Satisfiable,
        )
    } else {
        conflicts.sort();
        conflicts.dedup();
        Satisfiability::Unsatisfiable(conflicts)
    }
}
