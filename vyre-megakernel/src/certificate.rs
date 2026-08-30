//! The reproducible record of one bounded candidate search.
//!
//! A certificate states which grammar derived the candidate set, how many
//! candidates each production family contributed, why every eliminated family
//! was eliminated, and whether the budget ran out before the search did. It is
//! recorded in the artifact, so a selection can be reproduced from the graph,
//! the facts, the budget, and this record alone.

use serde::{Deserialize, Serialize};

use crate::grammar::ScheduleProduction;

/// Stable reason one derived candidate family cannot be admitted.
///
/// Every variant is reachable: a reason no candidate can ever earn would report
/// a check that never runs.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum PruneReason {
    /// A derived transform would change the numerical result of the program.
    Numerical,
    /// The derived plan breaks the dependence order of the graph.
    Dependence,
    /// Two arms of the derived plan alias storage or conflict on an effect.
    AliasOrEffect,
    /// A write is not visible where a barrier phase or proxy read consumes it.
    BarrierVisibility,
    /// The derived pipeline ring cannot be held in the available storage.
    PipelineCapacity,
    /// The derived plan exceeds the occupancy the device can hold resident.
    Occupancy,
    /// The derived plan exceeds the workgroup-shared scratch the device grants.
    Scratch,
    /// The derived workspace is not representable in the addressable range.
    Workspace,
    /// The derived plan has no forward-progress guarantee on this device.
    Progress,
    /// The candidate's proved bound is no better than the best proved candidate
    /// under the requested objective.
    ObjectiveDominated,
    /// The authenticated target facts do not grant the derived capability.
    TargetFacts,
    /// The artifact cannot represent the derived plan.
    Representation,
    /// A schedule transform precondition failed.
    ScheduleLegality,
    /// Target compilation rejected the plan, so it never reached measurement.
    Emission,
    /// The candidate's aggregated figures exceed a hard bound the objective
    /// states, so it is refused rather than ranked last.
    ObjectiveBound,
    /// The candidate does not exercise the schedule family the caller required,
    /// so it is eliminated before ranking rather than ranked and passed over.
    ScheduleRequirement,
}

impl PruneReason {
    /// Every reason, for closure over the eliminated classes.
    pub const ALL: &'static [Self] = &[
        Self::Numerical,
        Self::Dependence,
        Self::AliasOrEffect,
        Self::BarrierVisibility,
        Self::PipelineCapacity,
        Self::Occupancy,
        Self::Scratch,
        Self::Workspace,
        Self::Progress,
        Self::ObjectiveDominated,
        Self::TargetFacts,
        Self::Representation,
        Self::ScheduleLegality,
        Self::Emission,
        Self::ObjectiveBound,
        Self::ScheduleRequirement,
    ];

    /// Stable machine-readable diagnostic code.
    #[must_use]
    pub const fn code(self) -> &'static str {
        match self {
            Self::Numerical => "MKC001_NUMERICAL",
            Self::Dependence => "MKC002_DEPENDENCE",
            Self::AliasOrEffect => "MKC003_ALIAS_OR_EFFECT",
            Self::BarrierVisibility => "MKC004_BARRIER_VISIBILITY",
            Self::PipelineCapacity => "MKC005_PIPELINE_CAPACITY",
            Self::Occupancy => "MKC006_OCCUPANCY",
            Self::Scratch => "MKC007_SCRATCH",
            Self::Workspace => "MKC008_WORKSPACE",
            Self::Progress => "MKC009_PROGRESS",
            Self::ObjectiveDominated => "MKC010_OBJECTIVE_DOMINATED",
            Self::TargetFacts => "MKC011_TARGET_FACTS",
            Self::Representation => "MKC012_REPRESENTATION",
            Self::ScheduleLegality => "MKC013_SCHEDULE_LEGALITY",
            Self::Emission => "MKC014_EMISSION",
            Self::ObjectiveBound => "MKC015_OBJECTIVE_BOUND",
            Self::ScheduleRequirement => "MKC016_SCHEDULE_REQUIREMENT",
        }
    }
}

/// One production family eliminated for one reason, with how often.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct PrunedFamily {
    /// Production whose candidates were eliminated.
    pub production: ScheduleProduction,
    /// Stable reason the candidates were eliminated.
    pub reason: PruneReason,
    /// Candidates eliminated for this production and reason.
    pub count: u32,
}

/// One production family and what it contributed to the search.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct DerivedFamily {
    /// Production that derived the candidates.
    pub production: ScheduleProduction,
    /// Candidates the production derived, admitted or not.
    pub derived: u32,
    /// Candidates the production derived that constraint propagation admitted.
    pub admitted: u32,
}

/// One node's program derived through a chain of declared laws.
///
/// The chain is what makes a derived candidate a derivation rather than a
/// recipe: a ranked alternative that cannot name the laws it came from is
/// indistinguishable from a shape somebody wrote down.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct LawCitation {
    /// Graph node whose program the laws rewrote.
    pub node: u32,
    /// Law and rewrite names in application order.
    pub laws: Vec<String>,
}

/// One law-derived candidate eliminated, and why.
///
/// Held apart from [`PrunedFamily`] because a law states an equality between
/// programs and is not a schedule production: charging its refusal to a
/// production would report a family the search never proposed.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct PrunedLaw {
    /// Law chain whose candidate was eliminated.
    pub citation: LawCitation,
    /// Stable reason the candidate was eliminated.
    pub reason: PruneReason,
}

/// Reproducible record of one bounded candidate search.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct SearchCertificate {
    /// Grammar version whose productions derived the candidate set.
    pub grammar_version: u32,
    /// Expansion depth the search reached.
    pub depth: u32,
    /// Contributing families in canonical order.
    pub derived: Vec<DerivedFamily>,
    /// Eliminated families in canonical order.
    pub pruned: Vec<PrunedFamily>,
    /// Whether a bound stopped the search before the grammar was exhausted.
    pub budget_exhausted: bool,
    /// Law chains the admitted candidate set was derived through, in canonical
    /// order.
    ///
    /// Defaulted on read so a certificate recorded before law derivation
    /// existed deserializes as one that cited no law, which is what it was.
    #[serde(default)]
    pub law_derived: Vec<LawCitation>,
    /// Law-derived candidates the constraints eliminated, in canonical order.
    #[serde(default)]
    pub law_pruned: Vec<PrunedLaw>,
    /// Whether a bound stopped a law derivation before its laws were
    /// exhausted.
    #[serde(default)]
    pub law_budget_reached: bool,
}

impl SearchCertificate {
    /// Start a certificate for one grammar version.
    #[must_use]
    pub(crate) fn new(grammar_version: u32) -> Self {
        Self {
            grammar_version,
            ..Self::default()
        }
    }

    /// Record one candidate derived by `production`.
    pub(crate) fn derived(&mut self, production: ScheduleProduction) {
        let family = self.family(production);
        family.derived = family.derived.saturating_add(1);
    }

    /// Record one candidate of `production` that constraint propagation admitted.
    pub(crate) fn admitted(&mut self, production: ScheduleProduction) {
        let family = self.family(production);
        family.admitted = family.admitted.saturating_add(1);
    }

    /// The recorded family of one production, inserted when it is new.
    fn family(&mut self, production: ScheduleProduction) -> &mut DerivedFamily {
        if let Some(index) = self
            .derived
            .iter()
            .position(|family| family.production == production)
        {
            return &mut self.derived[index];
        }
        self.derived.push(DerivedFamily {
            production,
            derived: 0,
            admitted: 0,
        });
        let inserted = self.derived.len() - 1;
        &mut self.derived[inserted]
    }

    /// Record the depth the search reached.
    pub(crate) fn reached_depth(&mut self, depth: u32) {
        self.depth = self.depth.max(depth);
    }

    /// Record one eliminated candidate against its family.
    pub(crate) fn pruned(&mut self, production: ScheduleProduction, reason: PruneReason) {
        match self
            .pruned
            .iter_mut()
            .find(|family| family.production == production && family.reason == reason)
        {
            Some(family) => family.count = family.count.saturating_add(1),
            None => self.pruned.push(PrunedFamily {
                production,
                reason,
                count: 1,
            }),
        }
    }

    /// Record that a bound stopped the search.
    pub(crate) fn exhausted(&mut self) {
        self.budget_exhausted = true;
    }

    /// Record one law chain the admitted candidate set was derived through.
    pub(crate) fn cite_law(&mut self, citation: LawCitation) {
        if !self.law_derived.contains(&citation) {
            self.law_derived.push(citation);
        }
    }

    /// Record that a bound stopped a law derivation.
    pub(crate) fn reached_law_budget(&mut self) {
        self.law_budget_reached = true;
    }

    /// Record one law-derived candidate the constraints eliminated.
    pub(crate) fn prune_law(&mut self, citation: LawCitation, reason: PruneReason) {
        let pruned = PrunedLaw { citation, reason };
        if !self.law_pruned.contains(&pruned) {
            self.law_pruned.push(pruned);
        }
    }

    /// Every law name the admitted candidate set cites, once each, sorted.
    #[must_use]
    pub fn cited_laws(&self) -> Vec<&str> {
        let mut cited: Vec<&str> = self
            .law_derived
            .iter()
            .flat_map(|citation| citation.laws.iter().map(String::as_str))
            .collect();
        cited.sort_unstable();
        cited.dedup();
        cited
    }

    /// Order every recorded family so one search records one certificate.
    pub(crate) fn canonicalize(&mut self) {
        self.derived.sort_unstable();
        self.pruned.sort_unstable();
        self.law_derived.sort_unstable();
        self.law_pruned.sort_unstable();
    }

    /// Candidates one production derived.
    #[must_use]
    pub fn derived_by(&self, production: ScheduleProduction) -> u32 {
        self.derived
            .iter()
            .find(|family| family.production == production)
            .map_or(0, |family| family.derived)
    }

    /// Candidates one production contributed to ranking.
    #[must_use]
    pub fn admitted_by(&self, production: ScheduleProduction) -> u32 {
        self.derived
            .iter()
            .find(|family| family.production == production)
            .map_or(0, |family| family.admitted)
    }

    /// Candidates every production derived.
    #[must_use]
    pub fn derived_total(&self) -> u32 {
        self.derived
            .iter()
            .fold(0, |total, family| total.saturating_add(family.derived))
    }

    /// Candidates constraint propagation admitted for ranking.
    #[must_use]
    pub fn admitted_total(&self) -> u32 {
        self.derived
            .iter()
            .fold(0, |total, family| total.saturating_add(family.admitted))
    }

    /// Candidates eliminated for one reason across every production.
    #[must_use]
    pub fn pruned_for(&self, reason: PruneReason) -> u32 {
        self.pruned
            .iter()
            .filter(|family| family.reason == reason)
            .fold(0, |total, family| total.saturating_add(family.count))
    }
}
