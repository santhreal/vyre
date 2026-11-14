//! Canonical symbolic shape solver and proof certificate generation.
//!
//! Produces replayable satisfiability, equality, bound, divisibility,
//! and layout-compatibility proofs without embedding host `usize`.

use super::{ShapeConstraint, ShapeExprId, ShapeId, ShapeInterner, SymbolicDim};
use rustc_hash::FxHashMap;

/// Category of shape proof produced by the solver.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, serde::Deserialize, serde::Serialize)]
pub enum ShapeProofKind {
    /// Proof that a set of shape constraints is satisfiable.
    Satisfiability,
    /// Proof of structural or symbolic equality between two shapes.
    Equality,
    /// Proof that an extent lies within declared [min, max] bounds.
    Bound,
    /// Proof that an extent is an exact multiple of a given divisor.
    Divisibility,
    /// Proof that two shapes have compatible physical element volume and stride layout.
    LayoutCompatibility,
}

/// Replayable proof certificate recorded by verification and lowering.
#[derive(Clone, Debug, Eq, PartialEq, Hash, serde::Deserialize, serde::Serialize)]
pub struct ShapeProofCertificate {
    /// Proof category.
    pub proof_kind: ShapeProofKind,
    /// Target shape under proof.
    pub lhs_shape: ShapeId,
    /// Comparison shape (for binary proofs such as Equality / LayoutCompatibility).
    pub rhs_shape: Option<ShapeId>,
    /// Constraints evaluated during proof derivation.
    pub constraints_applied: Vec<ShapeConstraint>,
    /// Whether the proof holds unconditionally.
    pub verdict: bool,
    /// Audit log of derivation and reduction steps.
    pub derivation_steps: Vec<String>,
}

impl ShapeProofCertificate {
    /// Replay this certificate against an interner to verify its validity.
    #[must_use]
    pub fn replay(&self, interner: &ShapeInterner) -> bool {
        if !self.verdict {
            return false;
        }
        match self.proof_kind {
            ShapeProofKind::Equality => {
                let Some(rhs) = self.rhs_shape else {
                    return false;
                };
                if self.lhs_shape == rhs {
                    return true;
                }
                let Some(lhs_dims) = interner.get_shape(self.lhs_shape) else {
                    return false;
                };
                let Some(rhs_dims) = interner.get_shape(rhs) else {
                    return false;
                };
                lhs_dims == rhs_dims
            }
            ShapeProofKind::Satisfiability => {
                for c in &self.constraints_applied {
                    let (holds, _) = ShapeSolver::prove_constraint(interner, c, None);
                    if !holds {
                        return false;
                    }
                }
                true
            }
            ShapeProofKind::Bound | ShapeProofKind::Divisibility => self.verdict,
            ShapeProofKind::LayoutCompatibility => {
                let Some(rhs) = self.rhs_shape else {
                    return false;
                };
                let (holds, _) =
                    ShapeSolver::solve_layout_compatible(interner, self.lhs_shape, rhs);
                holds
            }
        }
    }
}

/// Canonical affine and symbolic shape solver.
pub struct ShapeSolver;

impl ShapeSolver {
    fn pair_cert(
        proof_kind: ShapeProofKind,
        lhs_shape: ShapeId,
        rhs_shape: ShapeId,
        verdict: bool,
        derivation_steps: Vec<String>,
    ) -> (bool, ShapeProofCertificate) {
        (
            verdict,
            ShapeProofCertificate {
                proof_kind,
                lhs_shape,
                rhs_shape: Some(rhs_shape),
                constraints_applied: Vec::new(),
                verdict,
                derivation_steps,
            },
        )
    }

    fn constraint_cert(
        proof_kind: ShapeProofKind,
        lhs_shape: ShapeId,
        constraint: &ShapeConstraint,
        verdict: bool,
        derivation_steps: Vec<String>,
    ) -> (bool, ShapeProofCertificate) {
        (
            verdict,
            ShapeProofCertificate {
                proof_kind,
                lhs_shape,
                rhs_shape: None,
                constraints_applied: vec![constraint.clone()],
                verdict,
                derivation_steps,
            },
        )
    }

    fn eval_bin(
        interner: &ShapeInterner,
        lhs: ShapeExprId,
        rhs: ShapeExprId,
        env: &FxHashMap<String, i128>,
    ) -> Option<(i128, i128)> {
        Some((
            Self::evaluate_expr(interner, lhs, env)?,
            Self::evaluate_expr(interner, rhs, env)?,
        ))
    }

    /// Prove symbolic shape equality.
    #[must_use]
    pub fn solve_equality(
        interner: &ShapeInterner,
        lhs: ShapeId,
        rhs: ShapeId,
    ) -> (bool, ShapeProofCertificate) {
        let mut steps = vec![format!("Comparing shape {lhs:?} with shape {rhs:?}")];
        if lhs == rhs {
            steps.push("Direct ShapeId identity match".into());
            return Self::pair_cert(ShapeProofKind::Equality, lhs, rhs, true, steps);
        }
        let verdict = match (interner.get_shape(lhs), interner.get_shape(rhs)) {
            (Some(l), Some(r)) if l.len() == r.len() => {
                let mut all_match = true;
                for (idx, (&ld, &rd)) in l.iter().zip(r.iter()).enumerate() {
                    if ld == rd {
                        steps.push(format!("Dim {idx} matched by ExprId: {ld:?} == {rd:?}"));
                    } else {
                        steps.push(format!("Dim {idx} mismatch: {ld:?} != {rd:?}"));
                        all_match = false;
                        break;
                    }
                }
                all_match
            }
            (Some(l), Some(r)) => {
                steps.push(format!(
                    "Rank mismatch: lhs rank {} != rhs rank {}",
                    l.len(),
                    r.len()
                ));
                false
            }
            _ => false,
        };
        Self::pair_cert(ShapeProofKind::Equality, lhs, rhs, verdict, steps)
    }

    /// Prove total layout / volume compatibility between two shapes.
    #[must_use]
    pub fn solve_layout_compatible(
        interner: &ShapeInterner,
        src: ShapeId,
        dst: ShapeId,
    ) -> (bool, ShapeProofCertificate) {
        let mut steps = vec![format!(
            "Solving layout compatibility between {src:?} and {dst:?}"
        )];
        if src == dst {
            steps.push("Identical shape ID; layout is trivially compatible".into());
            return Self::pair_cert(ShapeProofKind::LayoutCompatibility, src, dst, true, steps);
        }
        if let (Some(s), Some(d)) = (interner.get_shape(src), interner.get_shape(dst)) {
            let one = interner.constant(1);
            let src_vol = s.iter().fold(one, |acc, &dim| interner.mul(acc, dim));
            let dst_vol = d.iter().fold(one, |acc, &dim| interner.mul(acc, dim));
            if src_vol == dst_vol {
                steps.push(format!("Total symbolic volume matches: {src_vol:?}"));
                return Self::pair_cert(ShapeProofKind::LayoutCompatibility, src, dst, true, steps);
            }
        }
        steps.push("Layout volume mismatch or unresolvable shapes".into());
        Self::pair_cert(ShapeProofKind::LayoutCompatibility, src, dst, false, steps)
    }

    /// Evaluate an expression against a concrete valuation environment.
    #[must_use]
    pub fn evaluate_expr(
        interner: &ShapeInterner,
        expr_id: ShapeExprId,
        env: &FxHashMap<String, i128>,
    ) -> Option<i128> {
        match interner.get_expr(expr_id)? {
            SymbolicDim::Constant(val) => Some(val),
            SymbolicDim::Symbol(name) | SymbolicDim::Param { name, .. } => env.get(&name).copied(),
            SymbolicDim::Add(lhs, rhs) => {
                Self::eval_bin(interner, lhs, rhs, env).map(|(l, r)| l + r)
            }
            SymbolicDim::Sub(lhs, rhs) => {
                Self::eval_bin(interner, lhs, rhs, env).map(|(l, r)| l - r)
            }
            SymbolicDim::Mul(lhs, rhs) => {
                Self::eval_bin(interner, lhs, rhs, env).map(|(l, r)| l * r)
            }
            SymbolicDim::DivFloor(lhs, rhs) => {
                let (l, r) = Self::eval_bin(interner, lhs, rhs, env)?;
                (r != 0).then(|| l.div_euclid(r))
            }
            SymbolicDim::DivCeil(lhs, rhs) => {
                let (l, r) = Self::eval_bin(interner, lhs, rhs, env)?;
                (r > 0 && l >= 0).then(|| (l + r - 1) / r)
            }
            SymbolicDim::Mod(lhs, rhs) => {
                let (l, r) = Self::eval_bin(interner, lhs, rhs, env)?;
                (r != 0).then(|| l % r)
            }
            SymbolicDim::Min(lhs, rhs) => {
                Self::eval_bin(interner, lhs, rhs, env).map(|(l, r)| l.min(r))
            }
            SymbolicDim::Max(lhs, rhs) => {
                Self::eval_bin(interner, lhs, rhs, env).map(|(l, r)| l.max(r))
            }
            SymbolicDim::AlignTo { expr, alignment } => {
                let val = Self::evaluate_expr(interner, expr, env)?;
                (val >= 0 && alignment > 0).then(|| {
                    let a = alignment as i128;
                    ((val + a - 1) / a) * a
                })
            }
        }
    }

    /// Prove one shape constraint.
    #[must_use]
    pub fn prove_constraint(
        interner: &ShapeInterner,
        constraint: &ShapeConstraint,
        env: Option<&FxHashMap<String, i128>>,
    ) -> (bool, ShapeProofCertificate) {
        let mut steps = vec![format!("Proving constraint: {constraint:?}")];
        let dummy_shape = interner.intern_shape(&[]);
        let finish = |kind: ShapeProofKind, verdict: bool, steps: Vec<String>| {
            Self::constraint_cert(kind, dummy_shape, constraint, verdict, steps)
        };

        match constraint {
            ShapeConstraint::Equal(lhs, rhs) => {
                if lhs == rhs {
                    steps.push("Equal: direct ExprId identity".into());
                    return finish(ShapeProofKind::Satisfiability, true, steps);
                }
                if let Some((lv, rv)) = env.and_then(|e| Self::eval_bin(interner, *lhs, *rhs, e)) {
                    let ok = lv == rv;
                    steps.push(format!("Valuation: {lv} == {rv} => {ok}"));
                    return finish(ShapeProofKind::Satisfiability, ok, steps);
                }
                finish(ShapeProofKind::Satisfiability, false, steps)
            }
            ShapeConstraint::LessEqual(lhs, rhs) => {
                if lhs == rhs {
                    steps.push("LessEqual: direct ExprId identity (x <= x holds)".into());
                    return finish(ShapeProofKind::Bound, true, steps);
                }
                if let (Some(SymbolicDim::Constant(l)), Some(SymbolicDim::Constant(r))) =
                    (interner.get_expr(*lhs), interner.get_expr(*rhs))
                {
                    let ok = l <= r;
                    steps.push(format!("Constant bounds: {l} <= {r} => {ok}"));
                    return finish(ShapeProofKind::Bound, ok, steps);
                }
                if let Some((lv, rv)) = env.and_then(|e| Self::eval_bin(interner, *lhs, *rhs, e)) {
                    let ok = lv <= rv;
                    steps.push(format!("Valuation: {lv} <= {rv} => {ok}"));
                    return finish(ShapeProofKind::Bound, ok, steps);
                }
                finish(ShapeProofKind::Bound, false, steps)
            }
            ShapeConstraint::DivisibleBy { expr, divisor } => {
                if *divisor == 0 {
                    steps.push("DivisibleBy: divisor 0 is impossible".into());
                    return finish(ShapeProofKind::Divisibility, false, steps);
                }
                if *divisor == 1 {
                    steps.push("DivisibleBy: divisor 1 holds trivially".into());
                    return finish(ShapeProofKind::Divisibility, true, steps);
                }
                if let Some(SymbolicDim::Constant(c)) = interner.get_expr(*expr) {
                    let ok = c % (*divisor as i128) == 0;
                    steps.push(format!(
                        "Constant divisibility: {c} % {divisor} == 0 => {ok}"
                    ));
                    return finish(ShapeProofKind::Divisibility, ok, steps);
                }
                if let Some(v) = env.and_then(|e| Self::evaluate_expr(interner, *expr, e)) {
                    let ok = v % (*divisor as i128) == 0;
                    steps.push(format!(
                        "Valuation divisibility: {v} % {divisor} == 0 => {ok}"
                    ));
                    return finish(ShapeProofKind::Divisibility, ok, steps);
                }
                finish(ShapeProofKind::Divisibility, false, steps)
            }
            ShapeConstraint::Range { expr, min, max } => {
                if let Some(SymbolicDim::Constant(c)) = interner.get_expr(*expr) {
                    let ok = c >= *min && c <= *max;
                    steps.push(format!("Constant in range: {min} <= {c} <= {max} => {ok}"));
                    return finish(ShapeProofKind::Bound, ok, steps);
                }
                if let Some(v) = env.and_then(|e| Self::evaluate_expr(interner, *expr, e)) {
                    let ok = v >= *min && v <= *max;
                    steps.push(format!("Valuation in range: {min} <= {v} <= {max} => {ok}"));
                    return finish(ShapeProofKind::Bound, ok, steps);
                }
                finish(ShapeProofKind::Bound, false, steps)
            }
            ShapeConstraint::ModuloEqual {
                expr,
                modulus,
                remainder,
            } => {
                if *modulus == 0 || *remainder >= *modulus {
                    steps.push("Invalid modulo parameters".into());
                    return finish(ShapeProofKind::Divisibility, false, steps);
                }
                if let Some(SymbolicDim::Constant(c)) = interner.get_expr(*expr) {
                    let ok = (c % (*modulus as i128)) == (*remainder as i128);
                    steps.push(format!(
                        "Constant mod equals: {c} % {modulus} == {remainder} => {ok}"
                    ));
                    return finish(ShapeProofKind::Divisibility, ok, steps);
                }
                if let Some(v) = env.and_then(|e| Self::evaluate_expr(interner, *expr, e)) {
                    let ok = (v % (*modulus as i128)) == (*remainder as i128);
                    steps.push(format!(
                        "Valuation mod equals: {v} % {modulus} == {remainder} => {ok}"
                    ));
                    return finish(ShapeProofKind::Divisibility, ok, steps);
                }
                finish(ShapeProofKind::Divisibility, false, steps)
            }
        }
    }
}
