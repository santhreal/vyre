//! Tier-B TOML rule database for the egraph saturation engine.
//!
//! The Rust-coded `Family::rules` pattern in
//! [`crate::optimizer::eqsat`] keeps every rewrite in source code,
//! which means new equivalences need a recompile. The Tier-B contract
//! says community-contributable rule families should live in TOML so
//! a domain expert can add `(matmul_strassen_2x2 == matmul_2x2)`
//! without touching Rust.
//!
//! This module ships the MVP: a TOML schema for **op-id equivalence
//! rules** plus a `Rule` implementation that unions every pair of
//! e-classes whose enodes name two equivalent op ids. The richer
//! pattern DSL (LHS sub-tree match + RHS substitution) is a
//! follow-up; the equivalence-pair shape covers the most common
//! "these two ops compute the same thing" rewrites that drive
//! algebraic-canonicalisation Families today.
//!
//! ## TOML format
//!
//! ```toml
//! schema = 2
//!
//! [[equivalence]]
//! left = "vyre-libs::math::matmul"
//! right = "vyre-libs::math::matmul_strassen_one_level"
//! law = "algebraic"
//!
//! [[equivalence]]
//! left = "vyre-libs::math::elementwise_add"
//! right = "vyre-libs::math::add"
//! law = "algebraic"
//! ```
//!
//! Each `[[equivalence]]` row tells the rule that whenever both
//! `left` and `right` op ids appear as enodes anywhere in the
//! egraph, their e-classes are equivalent.
//!
//! `law` names the [`vyre_spec::RegionLawFamily`] that authorizes the pair,
//! and is what makes a data-file rewrite admissible in candidate search: an
//! unrecognized name is rejected at load rather than saturating on the claim
//! of the file that wrote it. Schema 1 files carry no law and are rejected.
//!
//! ## Trait expectations
//!
//! The rule walks `egraph.iter_nodes()` and groups e-classes by the
//! op-id string returned by `OpIdNode::op_id`. Languages that want
//! to consume TOML equivalence rules implement `OpIdNode` in
//! addition to [`crate::optimizer::eqsat::ENodeLang`]. Languages
//! that don't (e.g. pure-arithmetic toy languages from the eqsat
//! tests) don't pay the trait cost.

use std::path::Path;

use rustc_hash::FxHashMap;
use serde::Deserialize;

use vyre_spec::RegionLawFamily;

use crate::optimizer::eqsat::{EClassId, EGraph, ENodeLang, Rule};
use crate::optimizer::rewrite_contract::RewriteWitness;

/// Languages that participate in TOML equivalence rules expose the
/// op-id string of each enode. The string is the registry id
/// (`vyre-libs::math::matmul`, etc.).
pub trait OpIdNode {
    /// Stable op-id string. `None` for terminal/leaf nodes that don't
    /// carry an op id (literals, builtins)  -  they're skipped by the
    /// equivalence rule.
    fn op_id(&self) -> Option<&str>;
}

/// One TOML-loaded equivalence pair.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EquivalenceRule {
    /// Op-id of the left side.
    pub left: String,
    /// Op-id of the right side.
    pub right: String,
    /// Law family that authorizes the pair.
    pub law: RegionLawFamily,
}

/// Wire form of one row, before the cited law is resolved.
#[derive(Debug, Clone, Deserialize)]
struct WireEquivalenceRule {
    left: String,
    right: String,
    law: String,
}

/// TOML schema container.
#[derive(Debug, Clone, Deserialize)]
struct RuleFile {
    #[serde(default)]
    schema: u32,
    #[serde(default)]
    equivalence: Vec<WireEquivalenceRule>,
}

/// Schema version this loader accepts.
pub const EQSAT_TOML_SCHEMA_VERSION: u32 = 2;

/// A loaded TOML equivalence rule set.
///
/// Implements [`Rule`] for any language `L: ENodeLang + OpIdNode`. On
/// each `matches` call, walks the egraph once, groups e-classes by
/// op-id, and emits `(a, b)` pairs for every (left, right) op-id
/// pair where both sides have at least one e-class.
#[derive(Debug, Clone)]
pub struct TomlEquivalenceRules {
    name: &'static str,
    rules: Vec<EquivalenceRule>,
}

impl TomlEquivalenceRules {
    /// Construct an empty rule set with the given debug name.
    #[must_use]
    pub fn new(name: &'static str) -> Self {
        Self {
            name,
            rules: Vec::new(),
        }
    }

    /// Load rule pairs from a TOML file.
    ///
    /// # Errors
    ///
    /// Returns the underlying `std::io::Error` on read failure or a
    /// `toml::de::Error` projected through `std::io::Error::other` on
    /// parse failure. Schema-version mismatch returns
    /// `std::io::ErrorKind::InvalidData`.
    pub fn load(name: &'static str, path: &Path) -> std::io::Result<Self> {
        let text = read_eqsat_toml_bounded(path)?;
        Self::from_toml_str(name, &text)
    }

    /// Parse rule pairs from an in-memory TOML string.
    ///
    /// # Errors
    ///
    /// Returns `std::io::ErrorKind::InvalidData` when the TOML text cannot be
    /// decoded, declares an unsupported schema version, or cites a law family
    /// outside [`RegionLawFamily`].
    pub fn from_toml_str(name: &'static str, text: &str) -> std::io::Result<Self> {
        let parsed: RuleFile = toml::from_str(text)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        if parsed.schema != EQSAT_TOML_SCHEMA_VERSION {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!(
                    "Fix: TOML rule file declares schema = {}, expected schema = {EQSAT_TOML_SCHEMA_VERSION}. Schema {EQSAT_TOML_SCHEMA_VERSION} requires a `law` key naming the law family that authorizes each equivalence.",
                    parsed.schema
                ),
            ));
        }
        let mut rules = Vec::with_capacity(parsed.equivalence.len());
        for row in parsed.equivalence {
            let law = RegionLawFamily::from_name(&row.law).ok_or_else(|| {
                let known = RegionLawFamily::all()
                    .iter()
                    .map(|family| family.name())
                    .collect::<Vec<_>>()
                    .join(", ");
                std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!(
                        "TOML rule file cites law `{}` for `{}` == `{}`, which is not a registered law family. Fix: cite one of {known}.",
                        row.law, row.left, row.right
                    ),
                )
            })?;
            rules.push(EquivalenceRule {
                left: row.left,
                right: row.right,
                law,
            });
        }
        Ok(Self { name, rules })
    }

    /// Number of equivalence rules loaded.
    #[must_use]
    pub fn len(&self) -> usize {
        self.rules.len()
    }

    /// True when no rules are loaded.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.rules.is_empty()
    }

    /// Iterate the loaded rules in declaration order.
    pub fn iter(&self) -> impl Iterator<Item = &EquivalenceRule> {
        self.rules.iter()
    }
}

const MAX_EQSAT_TOML_RULE_BYTES: u64 = 4 * 1024 * 1024;

fn read_eqsat_toml_bounded(path: &Path) -> std::io::Result<String> {
    let mut reader = std::fs::File::open(path)?;
    let mut bytes = Vec::new();
    let mut total = 0u64;
    let mut chunk = [0u8; 8192];
    loop {
        let read = std::io::Read::read(&mut reader, &mut chunk)?;
        if read == 0 {
            return String::from_utf8(bytes)
                .map_err(|error| std::io::Error::new(std::io::ErrorKind::InvalidData, error));
        }
        let read = read as u64;
        total = total.saturating_add(read);
        if total > MAX_EQSAT_TOML_RULE_BYTES {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!(
                    "eqsat TOML `{}` exceeds {MAX_EQSAT_TOML_RULE_BYTES} byte rule cap",
                    path.display()
                ),
            ));
        }
        bytes.extend_from_slice(&chunk[..read as usize]);
    }
}

impl<L> Rule<L> for TomlEquivalenceRules
where
    L: ENodeLang + OpIdNode,
{
    fn name(&self) -> &'static str {
        self.name
    }

    /// Every loaded pair cites a registered law family, checked at load, so
    /// the rule set is authorized by the laws it names rather than by the file
    /// that declared it.
    fn witness(&self) -> RewriteWitness {
        RewriteWitness::Structural(
            "each equivalence pair cites a registered region law, resolved when the file is loaded",
        )
    }

    fn matches(&self, egraph: &EGraph<L>) -> Vec<(EClassId, EClassId)> {
        if self.rules.is_empty() {
            return Vec::new();
        }
        // Group e-classes by op-id in one pass.
        let mut by_op: FxHashMap<&str, Vec<EClassId>> = FxHashMap::default();
        for (cid, node) in egraph.iter_nodes() {
            if let Some(op_id) = node.op_id() {
                by_op.entry(op_id).or_default().push(cid);
            }
        }
        let mut equivs = Vec::new();
        for rule in &self.rules {
            let lefts = by_op.get(rule.left.as_str());
            let rights = by_op.get(rule.right.as_str());
            if let (Some(lefts), Some(rights)) = (lefts, rights) {
                // Emit the cross product. The egraph union-find
                // collapses redundant unions so duplicate pairs are
                // cheap.
                for &a in lefts {
                    for &b in rights {
                        if a != b {
                            equivs.push((a, b));
                        }
                    }
                }
            }
        }
        equivs
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::optimizer::eqsat::{EChildren, EGraph};
    use std::hash::{Hash, Hasher};

    /// Toy language for the TOML rule tests: a `Named(op_id, children)`
    /// node and a leaf `Lit`. `Named.op_id` is what the equivalence
    /// rule keys on.
    #[derive(Debug, Clone, PartialEq, Eq)]
    enum Toy {
        Named(&'static str, Vec<EClassId>),
        Lit(u32),
    }

    impl Hash for Toy {
        fn hash<H: Hasher>(&self, state: &mut H) {
            match self {
                Toy::Named(name, children) => {
                    state.write_u8(0);
                    name.hash(state);
                    for c in children {
                        c.hash(state);
                    }
                }
                Toy::Lit(v) => {
                    state.write_u8(1);
                    v.hash(state);
                }
            }
        }
    }

    impl ENodeLang for Toy {
        fn children(&self) -> EChildren {
            match self {
                Toy::Named(_, kids) => kids.iter().copied().collect(),
                Toy::Lit(_) => EChildren::new(),
            }
        }
        fn with_children(&self, children: &[EClassId]) -> Self {
            match self {
                Toy::Named(name, _) => Toy::Named(name, children.to_vec()),
                Toy::Lit(v) => Toy::Lit(*v),
            }
        }
    }

    impl OpIdNode for Toy {
        fn op_id(&self) -> Option<&str> {
            match self {
                Toy::Named(name, _) => Some(name),
                Toy::Lit(_) => None,
            }
        }
    }

    #[test]
    fn from_toml_str_parses_equivalence_pairs() {
        let toml = r#"
schema = 2
[[equivalence]]
left = "a"
right = "b"
law = "algebraic"
[[equivalence]]
left = "c"
right = "d"
law = "layout"
"#;
        let rules = TomlEquivalenceRules::from_toml_str("test", toml).unwrap();
        assert_eq!(rules.len(), 2);
        let first = rules.iter().next().unwrap();
        assert_eq!(first.left, "a");
        assert_eq!(first.law, RegionLawFamily::Algebraic);
        assert_eq!(
            rules.iter().nth(1).unwrap().law,
            RegionLawFamily::Layout,
            "each row resolves its own cited law"
        );
    }

    #[test]
    fn from_toml_str_rejects_wrong_schema() {
        let toml = "schema = 99\nequivalence = []\n";
        let err = TomlEquivalenceRules::from_toml_str("test", toml).unwrap_err();
        assert!(format!("{err}").contains("expected schema = 2"));
    }

    #[test]
    fn from_toml_str_accepts_empty_equivalence() {
        let toml = "schema = 2\n";
        let rules = TomlEquivalenceRules::from_toml_str("test", toml).unwrap();
        assert!(rules.is_empty());
    }

    #[test]
    fn matches_returns_empty_when_no_rules() {
        let mut egraph: EGraph<Toy> = EGraph::new();
        let _ = egraph.add(Toy::Named("a", vec![]));
        let _ = egraph.add(Toy::Named("b", vec![]));
        let rules = TomlEquivalenceRules::new("empty");
        assert!(rules.matches(&egraph).is_empty());
    }

    #[test]
    fn matches_emits_pair_when_both_op_ids_present() {
        let mut egraph: EGraph<Toy> = EGraph::new();
        let a = egraph.add(Toy::Named("a", vec![]));
        let b = egraph.add(Toy::Named("b", vec![]));
        let toml =
            "schema = 2\n[[equivalence]]\nleft = \"a\"\nright = \"b\"\nlaw = \"algebraic\"\n";
        let rules = TomlEquivalenceRules::from_toml_str("test", toml).unwrap();
        let pairs = rules.matches(&egraph);
        assert_eq!(pairs.len(), 1);
        assert!(
            (pairs[0].0 == a && pairs[0].1 == b) || (pairs[0].0 == b && pairs[0].1 == a),
            "expected (a, b) pair; got {pairs:?}"
        );
    }

    #[test]
    fn matches_empty_when_one_side_absent() {
        let mut egraph: EGraph<Toy> = EGraph::new();
        let _ = egraph.add(Toy::Named("a", vec![]));
        // "b" is absent.
        let toml =
            "schema = 2\n[[equivalence]]\nleft = \"a\"\nright = \"b\"\nlaw = \"algebraic\"\n";
        let rules = TomlEquivalenceRules::from_toml_str("test", toml).unwrap();
        assert!(rules.matches(&egraph).is_empty());
    }

    #[test]
    fn matches_skips_leaf_nodes_without_op_id() {
        let mut egraph: EGraph<Toy> = EGraph::new();
        let _ = egraph.add(Toy::Lit(7));
        let _ = egraph.add(Toy::Lit(8));
        // Lit has no op_id, so a rule keying on "anything" finds
        // nothing.
        let toml =
            "schema = 2\n[[equivalence]]\nleft = \"7\"\nright = \"8\"\nlaw = \"algebraic\"\n";
        let rules = TomlEquivalenceRules::from_toml_str("test", toml).unwrap();
        assert!(rules.matches(&egraph).is_empty());
    }

    #[test]
    fn rule_name_forwards_constructor_name() {
        let rules: TomlEquivalenceRules = TomlEquivalenceRules::new("algebra_v1");
        let r: &dyn Rule<Toy> = &rules;
        assert_eq!(r.name(), "algebra_v1");
    }

    #[test]
    fn from_toml_str_rejects_a_row_without_a_law() {
        let toml = "schema = 2\n[[equivalence]]\nleft = \"a\"\nright = \"b\"\n";
        let err = TomlEquivalenceRules::from_toml_str("test", toml).unwrap_err();
        assert_eq!(err.kind(), std::io::ErrorKind::InvalidData);
        assert!(
            format!("{err}").contains("law"),
            "the decode error must name the missing key: {err}"
        );
    }

    #[test]
    fn from_toml_str_rejects_an_unregistered_law() {
        let toml = "schema = 2\n[[equivalence]]\nleft = \"a\"\nright = \"b\"\nlaw = \"vibes\"\n";
        let err = TomlEquivalenceRules::from_toml_str("test", toml).unwrap_err();
        assert_eq!(err.kind(), std::io::ErrorKind::InvalidData);
        let rendered = format!("{err}");
        assert!(rendered.contains("vibes"), "{rendered}");
        for family in RegionLawFamily::all() {
            assert!(
                rendered.contains(family.name()),
                "the rejection must list every citable law: {rendered}"
            );
        }
    }

    #[test]
    fn a_loaded_rule_set_is_admissible_in_candidate_search() {
        let rules = TomlEquivalenceRules::new("algebra_v1");
        let rule: &dyn Rule<Toy> = &rules;
        assert!(
            rule.witness().admits_candidate_search(),
            "a law-citing rule set must be admissible"
        );
    }
}
