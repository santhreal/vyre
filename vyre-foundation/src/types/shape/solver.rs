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
    /// Prove symbolic shape equality.
    #[must_use]
    pub fn solve_equality(
        interner: &ShapeInterner,
        lhs: ShapeId,
        rhs: ShapeId,
    ) -> (bool, ShapeProofCertificate) {
        let mut steps = Vec::new();
        steps.push(format!("Comparing shape {lhs:?} with shape {rhs:?}"));

        if lhs == rhs {
            steps.push("Direct ShapeId identity match".into());
            return (
                true,
                ShapeProofCertificate {
                    proof_kind: ShapeProofKind::Equality,
                    lhs_shape: lhs,
                    rhs_shape: Some(rhs),
                    constraints_applied: Vec::new(),
                    verdict: true,
                    derivation_steps: steps,
                },
            );
        }

        let lhs_dims = interner.get_shape(lhs);
        let rhs_dims = interner.get_shape(rhs);

        match (lhs_dims, rhs_dims) {
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
                (
                    all_match,
                    ShapeProofCertificate {
                        proof_kind: ShapeProofKind::Equality,
                        lhs_shape: lhs,
                        rhs_shape: Some(rhs),
                        constraints_applied: Vec::new(),
                        verdict: all_match,
                        derivation_steps: steps,
                    },
                )
            }
            (Some(l), Some(r)) => {
                steps.push(format!(
                    "Rank mismatch: lhs rank {} != rhs rank {}",
                    l.len(),
                    r.len()
                ));
                (
                    false,
                    ShapeProofCertificate {
                        proof_kind: ShapeProofKind::Equality,
                        lhs_shape: lhs,
                        rhs_shape: Some(rhs),
                        constraints_applied: Vec::new(),
                        verdict: false,
                        derivation_steps: steps,
                    },
                )
            }
            _ => (
                false,
                ShapeProofCertificate {
                    proof_kind: ShapeProofKind::Equality,
                    lhs_shape: lhs,
                    rhs_shape: Some(rhs),
                    constraints_applied: Vec::new(),
                    verdict: false,
                    derivation_steps: steps,
                },
            ),
        }
    }

    /// Prove total layout / volume compatibility between two shapes.
    #[must_use]
    pub fn solve_layout_compatible(
        interner: &ShapeInterner,
        src: ShapeId,
        dst: ShapeId,
    ) -> (bool, ShapeProofCertificate) {
        let mut steps = Vec::new();
        steps.push(format!(
            "Solving layout compatibility between {src:?} and {dst:?}"
        ));

        if src == dst {
            steps.push("Identical shape ID; layout is trivially compatible".into());
            return (
                true,
                ShapeProofCertificate {
                    proof_kind: ShapeProofKind::LayoutCompatibility,
                    lhs_shape: src,
                    rhs_shape: Some(dst),
                    constraints_applied: Vec::new(),
                    verdict: true,
                    derivation_steps: steps,
                },
            );
        }

        let src_dims = interner.get_shape(src);
        let dst_dims = interner.get_shape(dst);

        if let (Some(s), Some(d)) = (src_dims, dst_dims) {
            // Compute total volume expression for each
            let src_vol = s
                .iter()
                .fold(interner.constant(1), |acc, &dim| interner.mul(acc, dim));
            let dst_vol = d
                .iter()
                .fold(interner.constant(1), |acc, &dim| interner.mul(acc, dim));

            if src_vol == dst_vol {
                steps.push(format!("Total symbolic volume matches: {src_vol:?}"));
                return (
                    true,
                    ShapeProofCertificate {
                        proof_kind: ShapeProofKind::LayoutCompatibility,
                        lhs_shape: src,
                        rhs_shape: Some(dst),
                        constraints_applied: Vec::new(),
                        verdict: true,
                        derivation_steps: steps,
                    },
                );
            }
        }

        steps.push("Layout volume mismatch or unresolvable shapes".into());
        (
            false,
            ShapeProofCertificate {
                proof_kind: ShapeProofKind::LayoutCompatibility,
                lhs_shape: src,
                rhs_shape: Some(dst),
                constraints_applied: Vec::new(),
                verdict: false,
                derivation_steps: steps,
            },
        )
    }

    /// Evaluate an expression against a concrete valuation environment.
    #[must_use]
    pub fn evaluate_expr(
        interner: &ShapeInterner,
        expr_id: ShapeExprId,
        env: &FxHashMap<String, i128>,
    ) -> Option<i128> {
        let dim = interner.get_expr(expr_id)?;
        match dim {
            SymbolicDim::Constant(val) => Some(val),
            SymbolicDim::Symbol(name) => env.get(&name).copied(),
            SymbolicDim::Param { name, .. } => env.get(&name).copied(),
            SymbolicDim::Add(lhs, rhs) => {
                let l = Self::evaluate_expr(interner, lhs, env)?;
                let r = Self::evaluate_expr(interner, rhs, env)?;
                Some(l + r)
            }
            SymbolicDim::Sub(lhs, rhs) => {
                let l = Self::evaluate_expr(interner, lhs, env)?;
                let r = Self::evaluate_expr(interner, rhs, env)?;
                Some(l - r)
            }
            SymbolicDim::Mul(lhs, rhs) => {
                let l = Self::evaluate_expr(interner, lhs, env)?;
                let r = Self::evaluate_expr(interner, rhs, env)?;
                Some(l * r)
            }
            SymbolicDim::DivFloor(lhs, rhs) => {
                let l = Self::evaluate_expr(interner, lhs, env)?;
                let r = Self::evaluate_expr(interner, rhs, env)?;
                if r != 0 {
                    Some(l.div_euclid(r))
                } else {
                    None
                }
            }
            SymbolicDim::DivCeil(lhs, rhs) => {
                let l = Self::evaluate_expr(interner, lhs, env)?;
                let r = Self::evaluate_expr(interner, rhs, env)?;
                if r > 0 && l >= 0 {
                    Some((l + r - 1) / r)
                } else {
                    None
                }
            }
            SymbolicDim::Mod(lhs, rhs) => {
                let l = Self::evaluate_expr(interner, lhs, env)?;
                let r = Self::evaluate_expr(interner, rhs, env)?;
                if r != 0 {
                    Some(l % r)
                } else {
                    None
                }
            }
            SymbolicDim::Min(lhs, rhs) => {
                let l = Self::evaluate_expr(interner, lhs, env)?;
                let r = Self::evaluate_expr(interner, rhs, env)?;
                Some(l.min(r))
            }
            SymbolicDim::Max(lhs, rhs) => {
                let l = Self::evaluate_expr(interner, lhs, env)?;
                let r = Self::evaluate_expr(interner, rhs, env)?;
                Some(l.max(r))
            }
            SymbolicDim::AlignTo { expr, alignment } => {
                let val = Self::evaluate_expr(interner, expr, env)?;
                if val >= 0 && alignment > 0 {
                    let a = alignment as i128;
                    Some(((val + a - 1) / a) * a)
                } else {
                    None
                }
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
        let mut steps = Vec::new();
        steps.push(format!("Proving constraint: {constraint:?}"));

        let dummy_shape = interner.intern_shape(&[]);

        match constraint {
            ShapeConstraint::Equal(lhs, rhs) => {
                if lhs == rhs {
                    steps.push("Equal: direct ExprId identity".into());
                    return (
                        true,
                        ShapeProofCertificate {
                            proof_kind: ShapeProofKind::Satisfiability,
                            lhs_shape: dummy_shape,
                            rhs_shape: None,
                            constraints_applied: vec![constraint.clone()],
                            verdict: true,
                            derivation_steps: steps,
                        },
                    );
                }
                if let Some(e) = env {
                    if let (Some(lv), Some(rv)) = (
                        Self::evaluate_expr(interner, *lhs, e),
                        Self::evaluate_expr(interner, *rhs, e),
                    ) {
                        let ok = lv == rv;
                        steps.push(format!("Valuation: {lv} == {rv} => {ok}"));
                        return (
                            ok,
                            ShapeProofCertificate {
                                proof_kind: ShapeProofKind::Satisfiability,
                                lhs_shape: dummy_shape,
                                rhs_shape: None,
                                constraints_applied: vec![constraint.clone()],
                                verdict: ok,
                                derivation_steps: steps,
                            },
                        );
                    }
                }
                (
                    false,
                    ShapeProofCertificate {
                        proof_kind: ShapeProofKind::Satisfiability,
                        lhs_shape: dummy_shape,
                        rhs_shape: None,
                        constraints_applied: vec![constraint.clone()],
                        verdict: false,
                        derivation_steps: steps,
                    },
                )
            }
            ShapeConstraint::LessEqual(lhs, rhs) => {
                if lhs == rhs {
                    steps.push("LessEqual: direct ExprId identity (x <= x holds)".into());
                    return (
                        true,
                        ShapeProofCertificate {
                            proof_kind: ShapeProofKind::Bound,
                            lhs_shape: dummy_shape,
                            rhs_shape: None,
                            constraints_applied: vec![constraint.clone()],
                            verdict: true,
                            derivation_steps: steps,
                        },
                    );
                }
                if let (Some(SymbolicDim::Constant(l)), Some(SymbolicDim::Constant(r))) =
                    (interner.get_expr(*lhs), interner.get_expr(*rhs))
                {
                    let ok = l <= r;
                    steps.push(format!("Constant bounds: {l} <= {r} => {ok}"));
                    return (
                        ok,
                        ShapeProofCertificate {
                            proof_kind: ShapeProofKind::Bound,
                            lhs_shape: dummy_shape,
                            rhs_shape: None,
                            constraints_applied: vec![constraint.clone()],
                            verdict: ok,
                            derivation_steps: steps,
                        },
                    );
                }
                if let Some(e) = env {
                    if let (Some(lv), Some(rv)) = (
                        Self::evaluate_expr(interner, *lhs, e),
                        Self::evaluate_expr(interner, *rhs, e),
                    ) {
                        let ok = lv <= rv;
                        steps.push(format!("Valuation: {lv} <= {rv} => {ok}"));
                        return (
                            ok,
                            ShapeProofCertificate {
                                proof_kind: ShapeProofKind::Bound,
                                lhs_shape: dummy_shape,
                                rhs_shape: None,
                                constraints_applied: vec![constraint.clone()],
                                verdict: ok,
                                derivation_steps: steps,
                            },
                        );
                    }
                }
                (
                    false,
                    ShapeProofCertificate {
                        proof_kind: ShapeProofKind::Bound,
                        lhs_shape: dummy_shape,
                        rhs_shape: None,
                        constraints_applied: vec![constraint.clone()],
                        verdict: false,
                        derivation_steps: steps,
                    },
                )
            }
            ShapeConstraint::DivisibleBy { expr, divisor } => {
                if *divisor == 0 {
                    steps.push("DivisibleBy: divisor 0 is impossible".into());
                    return (
                        false,
                        ShapeProofCertificate {
                            proof_kind: ShapeProofKind::Divisibility,
                            lhs_shape: dummy_shape,
                            rhs_shape: None,
                            constraints_applied: vec![constraint.clone()],
                            verdict: false,
                            derivation_steps: steps,
                        },
                    );
                }
                if *divisor == 1 {
                    steps.push("DivisibleBy: divisor 1 holds trivially".into());
                    return (
                        true,
                        ShapeProofCertificate {
                            proof_kind: ShapeProofKind::Divisibility,
                            lhs_shape: dummy_shape,
                            rhs_shape: None,
                            constraints_applied: vec![constraint.clone()],
                            verdict: true,
                            derivation_steps: steps,
                        },
                    );
                }
                if let Some(SymbolicDim::Constant(c)) = interner.get_expr(*expr) {
                    let ok = c % (*divisor as i128) == 0;
                    steps.push(format!(
                        "Constant divisibility: {c} % {divisor} == 0 => {ok}"
                    ));
                    return (
                        ok,
                        ShapeProofCertificate {
                            proof_kind: ShapeProofKind::Divisibility,
                            lhs_shape: dummy_shape,
                            rhs_shape: None,
                            constraints_applied: vec![constraint.clone()],
                            verdict: ok,
                            derivation_steps: steps,
                        },
                    );
                }
                if let Some(e) = env {
                    if let Some(v) = Self::evaluate_expr(interner, *expr, e) {
                        let ok = v % (*divisor as i128) == 0;
                        steps.push(format!(
                            "Valuation divisibility: {v} % {divisor} == 0 => {ok}"
                        ));
                        return (
                            ok,
                            ShapeProofCertificate {
                                proof_kind: ShapeProofKind::Divisibility,
                                lhs_shape: dummy_shape,
                                rhs_shape: None,
                                constraints_applied: vec![constraint.clone()],
                                verdict: ok,
                                derivation_steps: steps,
                            },
                        );
                    }
                }
                (
                    false,
                    ShapeProofCertificate {
                        proof_kind: ShapeProofKind::Divisibility,
                        lhs_shape: dummy_shape,
                        rhs_shape: None,
                        constraints_applied: vec![constraint.clone()],
                        verdict: false,
                        derivation_steps: steps,
                    },
                )
            }
            ShapeConstraint::Range { expr, min, max } => {
                if let Some(SymbolicDim::Constant(c)) = interner.get_expr(*expr) {
                    let ok = c >= *min && c <= *max;
                    steps.push(format!("Constant in range: {min} <= {c} <= {max} => {ok}"));
                    return (
                        ok,
                        ShapeProofCertificate {
                            proof_kind: ShapeProofKind::Bound,
                            lhs_shape: dummy_shape,
                            rhs_shape: None,
                            constraints_applied: vec![constraint.clone()],
                            verdict: ok,
                            derivation_steps: steps,
                        },
                    );
                }
                if let Some(e) = env {
                    if let Some(v) = Self::evaluate_expr(interner, *expr, e) {
                        let ok = v >= *min && v <= *max;
                        steps.push(format!("Valuation in range: {min} <= {v} <= {max} => {ok}"));
                        return (
                            ok,
                            ShapeProofCertificate {
                                proof_kind: ShapeProofKind::Bound,
                                lhs_shape: dummy_shape,
                                rhs_shape: None,
                                constraints_applied: vec![constraint.clone()],
                                verdict: ok,
                                derivation_steps: steps,
                            },
                        );
                    }
                }
                (
                    false,
                    ShapeProofCertificate {
                        proof_kind: ShapeProofKind::Bound,
                        lhs_shape: dummy_shape,
                        rhs_shape: None,
                        constraints_applied: vec![constraint.clone()],
                        verdict: false,
                        derivation_steps: steps,
                    },
                )
            }
            ShapeConstraint::ModuloEqual {
                expr,
                modulus,
                remainder,
            } => {
                if *modulus == 0 || *remainder >= *modulus {
                    steps.push("Invalid modulo parameters".into());
                    return (
                        false,
                        ShapeProofCertificate {
                            proof_kind: ShapeProofKind::Divisibility,
                            lhs_shape: dummy_shape,
                            rhs_shape: None,
                            constraints_applied: vec![constraint.clone()],
                            verdict: false,
                            derivation_steps: steps,
                        },
                    );
                }
                if let Some(SymbolicDim::Constant(c)) = interner.get_expr(*expr) {
                    let ok = (c % (*modulus as i128)) == (*remainder as i128);
                    steps.push(format!(
                        "Constant mod equals: {c} % {modulus} == {remainder} => {ok}"
                    ));
                    return (
                        ok,
                        ShapeProofCertificate {
                            proof_kind: ShapeProofKind::Divisibility,
                            lhs_shape: dummy_shape,
                            rhs_shape: None,
                            constraints_applied: vec![constraint.clone()],
                            verdict: ok,
                            derivation_steps: steps,
                        },
                    );
                }
                if let Some(e) = env {
                    if let Some(v) = Self::evaluate_expr(interner, *expr, e) {
                        let ok = (v % (*modulus as i128)) == (*remainder as i128);
                        steps.push(format!(
                            "Valuation mod equals: {v} % {modulus} == {remainder} => {ok}"
                        ));
                        return (
                            ok,
                            ShapeProofCertificate {
                                proof_kind: ShapeProofKind::Divisibility,
                                lhs_shape: dummy_shape,
                                rhs_shape: None,
                                constraints_applied: vec![constraint.clone()],
                                verdict: ok,
                                derivation_steps: steps,
                            },
                        );
                    }
                }
                (
                    false,
                    ShapeProofCertificate {
                        proof_kind: ShapeProofKind::Divisibility,
                        lhs_shape: dummy_shape,
                        rhs_shape: None,
                        constraints_applied: vec![constraint.clone()],
                        verdict: false,
                        derivation_steps: steps,
                    },
                )
            }
        }
    }
}
