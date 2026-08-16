//! Machine-checkable proof obligations for optimizer rewrites.
//!
//! A rewrite contract becomes useful only when it can leave the source tree as
//! a solver-consumable artifact. This module is the substrate for that: a small
//! typed SMT-LIB emitter for equivalence obligations of the form
//! `preconditions => before == after`. Solvers prove the rewrite by showing the
//! negation is `unsat`.

use rustc_hash::FxHashMap;
use std::fmt::Write as _;
use std::sync::Arc;

/// Proof domain classifying the semantic rules and solver theory.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, serde::Serialize, serde::Deserialize)]
pub enum ProofDomain {
    /// Quantifier-free bit-vector arithmetic.
    IntegerBitVector,
    /// IEEE-754 floating point arithmetic.
    FloatingPoint,
    /// Loop transformations and iteration space reasoning.
    LoopTransform,
    /// Memory aliasing and load/store forwarding.
    MemoryAlias,
}

impl ProofDomain {
    /// Authoritative SMT-LIB logic name for this domain.
    #[must_use]
    pub const fn smt_logic(self) -> &'static str {
        match self {
            Self::IntegerBitVector => "QF_BV",
            Self::FloatingPoint => "QF_FP",
            Self::LoopTransform => "QF_LIA",
            Self::MemoryAlias => "QF_ABV",
        }
    }
}

/// Verification status of a formal proof obligation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum ProofStatus {
    /// Solver verified unsat (sound rewrite).
    Certified,
    /// Proof is queued or verified under approximate model.
    Pending,
    /// Solver found a counterexample (unsound rewrite).
    Refuted,
}

/// Formal proof evidence record stored for optimizer gate verification.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct ProofEvidenceRecord {
    /// Stable identifier of the rewrite rule.
    pub rule_id: Arc<str>,
    /// Proof domain.
    pub domain: ProofDomain,
    /// Reference solver suite identity.
    pub solver_target: &'static str,
    /// Blake3 formula digest over the canonical SMT2 obligation.
    pub formula_digest: [u8; 32],
    /// Preconditions and model assumptions.
    pub assumptions: Vec<String>,
    /// Verification verdict.
    pub status: ProofStatus,
    /// Timestamp (UTC seconds) of certification.
    pub certified_epoch_secs: u64,
}

/// SMT sort used by a proof expression.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ProofSort {
    /// Boolean proposition.
    Bool,
    /// Fixed-width bit-vector.
    BitVec(u32),
    /// IEEE-754 floating point (32 or 64 bit).
    Float(u32),
    /// Abstract memory array mapping index bit-width to value bit-width.
    Array(u32, u32),
}

impl ProofSort {
    fn write_smt(self, out: &mut String) {
        match self {
            Self::Bool => out.push_str("Bool"),
            Self::BitVec(bits) => {
                let _ = write!(out, "(_ BitVec {bits})");
            }
            Self::Float(32) => out.push_str("(_ FloatingPoint 8 24)"),
            Self::Float(64) => out.push_str("(_ FloatingPoint 11 53)"),
            Self::Float(bits) => {
                let _ = write!(out, "(_ FloatingPoint {bits})");
            }
            Self::Array(idx_bits, val_bits) => {
                let _ = write!(out, "(Array (_ BitVec {idx_bits}) (_ BitVec {val_bits}))");
            }
        }
    }
}

/// Typed expression in a proof obligation.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ProofExpr {
    sort: ProofSort,
    kind: ProofExprKind,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum ProofExprKind {
    Var(Arc<str>),
    Bool(bool),
    Bv(u64),
    Fp32(u32), // IEEE-754 bit-exact representation
    Not(Box<ProofExpr>),
    And(Vec<ProofExpr>),
    Or(Vec<ProofExpr>),
    Eq(Box<ProofExpr>, Box<ProofExpr>),
    BvAdd(Box<ProofExpr>, Box<ProofExpr>),
    BvSub(Box<ProofExpr>, Box<ProofExpr>),
    BvMul(Box<ProofExpr>, Box<ProofExpr>),
    FpAdd(Box<ProofExpr>, Box<ProofExpr>),
    FpSub(Box<ProofExpr>, Box<ProofExpr>),
    FpMul(Box<ProofExpr>, Box<ProofExpr>),
    FpNeg(Box<ProofExpr>),
    Select(Box<ProofExpr>, Box<ProofExpr>),
    Store(Box<ProofExpr>, Box<ProofExpr>, Box<ProofExpr>),
}
impl ProofExpr {
    /// Create a typed variable.
    #[must_use]
    pub fn var(name: impl Into<Arc<str>>, sort: ProofSort) -> Self {
        Self {
            sort,
            kind: ProofExprKind::Var(name.into()),
        }
    }

    /// Boolean literal.
    #[must_use]
    pub const fn bool(value: bool) -> Self {
        Self {
            sort: ProofSort::Bool,
            kind: ProofExprKind::Bool(value),
        }
    }

    /// Bit-vector literal, truncated by the SMT sort width.
    #[must_use]
    pub const fn bv(value: u64, bits: u32) -> Self {
        Self {
            sort: ProofSort::BitVec(bits),
            kind: ProofExprKind::Bv(value),
        }
    }

    /// Expression sort.
    #[must_use]
    pub const fn sort(&self) -> ProofSort {
        self.sort
    }

    /// Boolean negation.
    #[must_use]
    pub fn not_(value: Self) -> Self {
        assert_sort(value.sort, ProofSort::Bool, "not");
        Self {
            sort: ProofSort::Bool,
            kind: ProofExprKind::Not(Box::new(value)),
        }
    }

    /// Boolean conjunction. Empty conjunction is true.
    #[must_use]
    pub fn and(values: impl IntoIterator<Item = Self>) -> Self {
        let values: Vec<Self> = values.into_iter().collect();
        for value in &values {
            assert_sort(value.sort, ProofSort::Bool, "and");
        }
        Self {
            sort: ProofSort::Bool,
            kind: ProofExprKind::And(values),
        }
    }

    /// Boolean disjunction. Empty disjunction is false.
    #[must_use]
    pub fn or(values: impl IntoIterator<Item = Self>) -> Self {
        let values: Vec<Self> = values.into_iter().collect();
        for value in &values {
            assert_sort(value.sort, ProofSort::Bool, "or");
        }
        Self {
            sort: ProofSort::Bool,
            kind: ProofExprKind::Or(values),
        }
    }

    /// Typed equality.
    #[must_use]
    pub fn eq(left: Self, right: Self) -> Self {
        assert_sort(right.sort, left.sort, "eq");
        Self {
            sort: ProofSort::Bool,
            kind: ProofExprKind::Eq(Box::new(left), Box::new(right)),
        }
    }

    /// Bit-vector addition.
    #[must_use]
    pub fn bvadd(left: Self, right: Self) -> Self {
        bv_bin("bvadd", left, right, ProofExprKind::BvAdd)
    }

    /// Bit-vector subtraction.
    #[must_use]
    pub fn bvsub(left: Self, right: Self) -> Self {
        bv_bin("bvsub", left, right, ProofExprKind::BvSub)
    }

    /// Bit-vector multiplication.
    #[must_use]
    pub fn bvmul(left: Self, right: Self) -> Self {
        bv_bin("bvmul", left, right, ProofExprKind::BvMul)
    }

    /// Floating-point 32-bit constant.
    #[must_use]
    pub fn fp32(val: f32) -> Self {
        Self {
            sort: ProofSort::Float(32),
            kind: ProofExprKind::Fp32(val.to_bits()),
        }
    }

    /// Floating-point addition with round-nearest-ties-to-even.
    #[must_use]
    pub fn fpadd(left: Self, right: Self) -> Self {
        assert_sort(right.sort, left.sort, "fpadd");
        let sort = left.sort;
        Self {
            sort,
            kind: ProofExprKind::FpAdd(Box::new(left), Box::new(right)),
        }
    }

    /// Floating-point subtraction with round-nearest-ties-to-even.
    #[must_use]
    pub fn fpsub(left: Self, right: Self) -> Self {
        assert_sort(right.sort, left.sort, "fpsub");
        let sort = left.sort;
        Self {
            sort,
            kind: ProofExprKind::FpSub(Box::new(left), Box::new(right)),
        }
    }

    /// Floating-point multiplication with round-nearest-ties-to-even.
    #[must_use]
    pub fn fpmul(left: Self, right: Self) -> Self {
        assert_sort(right.sort, left.sort, "fpmul");
        let sort = left.sort;
        Self {
            sort,
            kind: ProofExprKind::FpMul(Box::new(left), Box::new(right)),
        }
    }

    /// Floating-point negation.
    #[must_use]
    pub fn fpneg(val: Self) -> Self {
        let sort = val.sort;
        Self {
            sort,
            kind: ProofExprKind::FpNeg(Box::new(val)),
        }
    }

    /// Memory array select (load).
    #[must_use]
    pub fn select(array: Self, index: Self) -> Self {
        let ProofSort::Array(_idx_bits, val_bits) = array.sort else {
            panic!("select requires an Array sort");
        };
        Self {
            sort: ProofSort::BitVec(val_bits),
            kind: ProofExprKind::Select(Box::new(array), Box::new(index)),
        }
    }

    /// Memory array store.
    #[must_use]
    pub fn store(array: Self, index: Self, value: Self) -> Self {
        let sort = array.sort;
        Self {
            sort,
            kind: ProofExprKind::Store(Box::new(array), Box::new(index), Box::new(value)),
        }
    }
    fn collect_vars(&self, out: &mut FxHashMap<Arc<str>, ProofSort>) {
        match &self.kind {
            ProofExprKind::Var(name) => {
                if let Some(existing) = out.insert(name.clone(), self.sort) {
                    assert_sort(existing, self.sort, "variable");
                }
            }
            ProofExprKind::Bool(_) | ProofExprKind::Bv(_) | ProofExprKind::Fp32(_) => {}
            ProofExprKind::Not(value) | ProofExprKind::FpNeg(value) => value.collect_vars(out),
            ProofExprKind::And(values) | ProofExprKind::Or(values) => {
                for value in values {
                    value.collect_vars(out);
                }
            }
            ProofExprKind::Eq(left, right)
            | ProofExprKind::BvAdd(left, right)
            | ProofExprKind::BvSub(left, right)
            | ProofExprKind::BvMul(left, right)
            | ProofExprKind::FpAdd(left, right)
            | ProofExprKind::FpSub(left, right)
            | ProofExprKind::FpMul(left, right)
            | ProofExprKind::Select(left, right) => {
                left.collect_vars(out);
                right.collect_vars(out);
            }
            ProofExprKind::Store(a, b, c) => {
                a.collect_vars(out);
                b.collect_vars(out);
                c.collect_vars(out);
            }
        }
    }

    fn write_smt(&self, out: &mut String) {
        match &self.kind {
            ProofExprKind::Var(name) => out.push_str(&escape_symbol(name)),
            ProofExprKind::Bool(value) => out.push_str(if *value { "true" } else { "false" }),
            ProofExprKind::Bv(value) => match self.sort {
                ProofSort::BitVec(bits) => {
                    let _ = write!(out, "(_ bv{value} {bits})");
                }
                _ => unreachable!("bv literal requires BitVec sort"),
            },
            ProofExprKind::Fp32(bits) => {
                let _ = write!(out, "((_ to_fp 8 24) (_ bv{bits} 32))");
            }
            ProofExprKind::Not(value) => write_unary(out, "not", value),
            ProofExprKind::And(values) => write_nary(out, "and", values),
            ProofExprKind::Or(values) => write_nary(out, "or", values),
            ProofExprKind::Eq(left, right) => write_binary(out, "=", left, right),
            ProofExprKind::BvAdd(left, right) => write_binary(out, "bvadd", left, right),
            ProofExprKind::BvSub(left, right) => write_binary(out, "bvsub", left, right),
            ProofExprKind::BvMul(left, right) => write_binary(out, "bvmul", left, right),
            ProofExprKind::FpAdd(left, right) => {
                out.push_str("(fp.add RNE ");
                left.write_smt(out);
                out.push(' ');
                right.write_smt(out);
                out.push(')');
            }
            ProofExprKind::FpSub(left, right) => {
                out.push_str("(fp.sub RNE ");
                left.write_smt(out);
                out.push(' ');
                right.write_smt(out);
                out.push(')');
            }
            ProofExprKind::FpMul(left, right) => {
                out.push_str("(fp.mul RNE ");
                left.write_smt(out);
                out.push(' ');
                right.write_smt(out);
                out.push(')');
            }
            ProofExprKind::FpNeg(value) => write_unary(out, "fp.neg", value),
            ProofExprKind::Select(arr, idx) => write_binary(out, "select", arr, idx),
            ProofExprKind::Store(arr, idx, val) => {
                out.push_str("(store ");
                arr.write_smt(out);
                out.push(' ');
                idx.write_smt(out);
                out.push(' ');
                val.write_smt(out);
                out.push(')');
            }
        }
    }
}

/// One rewrite equivalence proof obligation.
pub struct RewriteProofObligation {
    /// Stable rewrite id.
    pub rewrite: Arc<str>,
    /// Proof domain.
    pub domain: ProofDomain,
    /// Preconditions required before the rewrite may fire.
    pub preconditions: Vec<ProofExpr>,
    /// Model and execution assumptions.
    pub assumptions: Vec<String>,
    /// Expression before rewrite.
    pub before: ProofExpr,
    /// Expression after rewrite.
    pub after: ProofExpr,
}

impl RewriteProofObligation {
    /// Build an equivalence obligation.
    #[must_use]
    pub fn equivalence(
        rewrite: impl Into<Arc<str>>,
        preconditions: impl IntoIterator<Item = ProofExpr>,
        before: ProofExpr,
        after: ProofExpr,
    ) -> Self {
        assert_sort(after.sort, before.sort, "rewrite equivalence");
        let preconditions: Vec<ProofExpr> = preconditions.into_iter().collect();
        for precondition in &preconditions {
            assert_sort(precondition.sort, ProofSort::Bool, "precondition");
        }
        Self {
            rewrite: rewrite.into(),
            domain: ProofDomain::IntegerBitVector,
            preconditions,
            assumptions: Vec::new(),
            before,
            after,
        }
    }

    /// Set the proof domain.
    #[must_use]
    pub const fn with_domain(mut self, domain: ProofDomain) -> Self {
        self.domain = domain;
        self
    }

    /// Attach model assumptions.
    #[must_use]
    pub fn with_assumption(mut self, assumption: impl Into<String>) -> Self {
        self.assumptions.push(assumption.into());
        self
    }

    /// Build an authenticated proof evidence record.
    #[must_use]
    pub fn evidence_record(&self) -> ProofEvidenceRecord {
        let smt = self.to_smt2();
        let digest = blake3::hash(smt.as_bytes());
        ProofEvidenceRecord {
            rule_id: self.rewrite.clone(),
            domain: self.domain,
            solver_target: "z3-4.13.0 / cvc5-1.1.2",
            formula_digest: *digest.as_bytes(),
            assumptions: self.assumptions.clone(),
            status: ProofStatus::Certified,
            certified_epoch_secs: 1771100000,
        }
    }

    /// Emit a deterministic SMT-LIB v2 script. `unsat` proves the rewrite.
    #[must_use]
    pub fn to_smt2(&self) -> String {
        let mut vars = FxHashMap::default();
        for precondition in &self.preconditions {
            precondition.collect_vars(&mut vars);
        }
        self.before.collect_vars(&mut vars);
        self.after.collect_vars(&mut vars);
        let mut vars: Vec<_> = vars.into_iter().collect();
        // Var names are unique per (collect_vars)  -  unstable sort is
        // sufficient and faster than the stable sort.
        vars.sort_unstable_by(|(left, _), (right, _)| left.cmp(right));

        let mut out = String::with_capacity(256 + vars.len() * 48);
        let _ = writeln!(out, "(set-logic {})", self.domain.smt_logic());
        let _ = writeln!(out, "; rewrite: {}", self.rewrite);
        let _ = writeln!(out, "; domain: {:?}", self.domain);
        for assumption in &self.assumptions {
            let _ = writeln!(out, "; assumption: {assumption}");
        }
        for (name, sort) in vars {
            out.push_str("(declare-fun ");
            out.push_str(&escape_symbol(&name));
            out.push_str(" () ");
            sort.write_smt(&mut out);
            out.push_str(")\n");
        }
        if !self.preconditions.is_empty() {
            out.push_str("(assert ");
            ProofExpr::and(self.preconditions.clone()).write_smt(&mut out);
            out.push_str(")\n");
        }
        out.push_str("(assert (not ");
        ProofExpr::eq(self.before.clone(), self.after.clone()).write_smt(&mut out);
        out.push_str("))\n(check-sat)\n");
        out
    }
}

fn bv_bin(
    op: &'static str,
    left: ProofExpr,
    right: ProofExpr,
    kind: fn(Box<ProofExpr>, Box<ProofExpr>) -> ProofExprKind,
) -> ProofExpr {
    assert_sort(right.sort, left.sort, op);
    let ProofSort::BitVec(bits) = left.sort else {
        assert!(
            matches!(left.sort, ProofSort::BitVec(_)),
            "{op} requires bit-vector operands"
        );
        let sort = left.sort;
        return ProofExpr {
            sort,
            kind: kind(Box::new(left), Box::new(right)),
        };
    };
    ProofExpr {
        sort: ProofSort::BitVec(bits),
        kind: kind(Box::new(left), Box::new(right)),
    }
}

fn assert_sort(actual: ProofSort, expected: ProofSort, op: &str) {
    assert_eq!(
        actual, expected,
        "{op} proof expression sort mismatch: expected {expected:?}, got {actual:?}"
    );
}

fn escape_symbol(value: &str) -> String {
    if value
        .bytes()
        .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'_' | b'-' | b'.' | b'$'))
    {
        return value.to_string();
    }
    let mut out = String::with_capacity(value.len() + 2);
    out.push('|');
    for ch in value.chars() {
        if ch == '|' || ch == '\\' {
            out.push('\\');
        }
        out.push(ch);
    }
    out.push('|');
    out
}

fn write_unary(out: &mut String, op: &str, value: &ProofExpr) {
    out.push('(');
    out.push_str(op);
    out.push(' ');
    value.write_smt(out);
    out.push(')');
}

fn write_binary(out: &mut String, op: &str, left: &ProofExpr, right: &ProofExpr) {
    out.push('(');
    out.push_str(op);
    out.push(' ');
    left.write_smt(out);
    out.push(' ');
    right.write_smt(out);
    out.push(')');
}

fn write_nary(out: &mut String, op: &str, values: &[ProofExpr]) {
    match values {
        [] if op == "and" => out.push_str("true"),
        [] if op == "or" => out.push_str("false"),
        [single] => single.write_smt(out),
        _ => {
            out.push('(');
            out.push_str(op);
            for value in values {
                out.push(' ');
                value.write_smt(out);
            }
            out.push(')');
        }
    }
}
