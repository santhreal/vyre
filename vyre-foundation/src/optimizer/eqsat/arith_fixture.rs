//! Toy arithmetic `ENode` language, cost function, rules, and family used by
//! the engine tests in this module.

use rustc_hash::FxHashMap;
use smallvec::smallvec;

use super::{EChildren, EClassId, EGraph, ENodeLang, Family, Rule};

/// A minimal arithmetic `ENode` language for engine tests.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(super) enum Arith {
    Const(u32),
    Add(EClassId, EClassId),
    Mul(EClassId, EClassId),
}

impl ENodeLang for Arith {
    fn children(&self) -> EChildren {
        match self {
            Self::Const(_) => EChildren::new(),
            Self::Add(a, b) | Self::Mul(a, b) => smallvec![*a, *b],
        }
    }

    fn with_children(&self, children: &[EClassId]) -> Self {
        match self {
            Self::Const(n) => Self::Const(*n),
            Self::Add(_, _) => Self::Add(children[0], children[1]),
            Self::Mul(_, _) => Self::Mul(children[0], children[1]),
        }
    }
}

/// Simple cost: 1 per Const, 2 per Add, 3 per Mul.
pub(super) fn arith_cost(node: &Arith) -> u64 {
    match node {
        Arith::Const(_) => 1,
        Arith::Add(_, _) => 2,
        Arith::Mul(_, _) => 3,
    }
}

/// Simpler test rule: union every two `Const(a)` with `Const(a)` (idempotent).
/// Used to verify `saturate` calls `matches` and `rebuild` correctly.
pub(super) struct UnionEqualConstsRule;

impl Rule<Arith> for UnionEqualConstsRule {
    fn name(&self) -> &'static str {
        "union_equal_consts"
    }

    fn matches(&self, egraph: &EGraph<Arith>) -> Vec<(EClassId, EClassId)> {
        let mut by_value: FxHashMap<u32, Vec<EClassId>> = FxHashMap::default();
        for (cid, node) in egraph.iter_nodes() {
            if let Arith::Const(v) = node {
                by_value.entry(*v).or_default().push(cid);
            }
        }
        let mut out = Vec::new();
        for ids in by_value.values() {
            for window in ids.windows(2) {
                out.push((window[0], window[1]));
            }
        }
        out
    }
}

/// Rule that pairs every Const id with itself  -  guaranteed to
/// produce at least one match whenever the egraph holds any Const.
/// Used purely as a forwarding-test fixture.
pub(super) struct PairConstSelfRule;

impl Rule<Arith> for PairConstSelfRule {
    fn name(&self) -> &'static str {
        "pair_const_self"
    }

    fn matches(&self, egraph: &EGraph<Arith>) -> Vec<(EClassId, EClassId)> {
        let mut out = Vec::new();
        for (cid, node) in egraph.iter_nodes() {
            if let Arith::Const(_) = node {
                out.push((cid, cid));
            }
        }
        out
    }
}

pub(super) struct ForeignClassRule;

impl Rule<Arith> for ForeignClassRule {
    fn name(&self) -> &'static str {
        "foreign_class"
    }

    fn matches(&self, _egraph: &EGraph<Arith>) -> Vec<(EClassId, EClassId)> {
        vec![(EClassId(999), EClassId(999))]
    }
}

/// A toy family with one rule, used for the per-family budget tests.
pub(super) struct ConstUnionFamily {
    pub(super) name: &'static str,
}

impl Family<Arith> for ConstUnionFamily {
    fn name(&self) -> &'static str {
        self.name
    }
    fn rules(&self) -> Vec<Box<dyn Rule<Arith>>> {
        vec![Box::new(UnionEqualConstsRule)]
    }
}
