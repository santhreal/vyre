//! Interned symbolic shape expressions, canonical interning, and constraints.
//!
//! Extents are represented as interned symbolic expressions.
//! Zero (`Constant(0)`) is a valid extent and never a sentinel for unknown.
//! Structurally equal expressions and shapes canonicalize to identical [`ShapeExprId`]
//! and [`ShapeId`] instances.

pub(crate) mod solver;

use rustc_hash::FxHashMap;
use std::sync::RwLock;

/// Interned identifier for a canonical symbolic dimension expression.
#[derive(
    Clone, Copy, Debug, Eq, PartialEq, Hash, PartialOrd, Ord, serde::Deserialize, serde::Serialize,
)]
pub struct ShapeExprId(pub u32);

/// Interned identifier for an ordered multidimensional tensor shape.
#[derive(
    Clone, Copy, Debug, Eq, PartialEq, Hash, PartialOrd, Ord, serde::Deserialize, serde::Serialize,
)]
pub struct ShapeId(pub u32);

/// Symbolic dimension expression node.
#[derive(Clone, Debug, Eq, PartialEq, Hash, serde::Deserialize, serde::Serialize)]
pub enum SymbolicDim {
    /// Exact integer constant (0 is valid!).
    Constant(i128),
    /// Named runtime parameter / symbol (e.g. "batch", "seq_len").
    Symbol(String),
    /// Parameter with explicit index and debug name.
    Param {
        /// Parameter index.
        index: u32,
        /// Parameter identifier.
        name: String,
    },
    /// Addition of two expressions: `lhs + rhs`.
    Add(ShapeExprId, ShapeExprId),
    /// Subtraction: `lhs - rhs`.
    Sub(ShapeExprId, ShapeExprId),
    /// Multiplication: `lhs * rhs`.
    Mul(ShapeExprId, ShapeExprId),
    /// Floor division: `floor(lhs / rhs)`.
    DivFloor(ShapeExprId, ShapeExprId),
    /// Ceiling division: `ceil(lhs / rhs)`.
    DivCeil(ShapeExprId, ShapeExprId),
    /// Modulo / remainder: `lhs % rhs`.
    Mod(ShapeExprId, ShapeExprId),
    /// Minimum: `min(lhs, rhs)`.
    Min(ShapeExprId, ShapeExprId),
    /// Maximum: `max(lhs, rhs)`.
    Max(ShapeExprId, ShapeExprId),
    /// Align expression up to alignment boundary: `((expr + align - 1) / align) * align`.
    AlignTo {
        /// Expression to align.
        expr: ShapeExprId,
        /// Alignment in elements.
        alignment: u64,
    },
}

/// Shape constraint over symbolic expressions.
#[derive(Clone, Debug, Eq, PartialEq, Hash, serde::Deserialize, serde::Serialize)]
pub enum ShapeConstraint {
    /// Expressions must be identical: `lhs == rhs`.
    Equal(ShapeExprId, ShapeExprId),
    /// Upper bound inequality: `lhs <= rhs`.
    LessEqual(ShapeExprId, ShapeExprId),
    /// Strict divisibility: `expr % divisor == 0`.
    DivisibleBy {
        /// Expression subject to divisibility constraint.
        expr: ShapeExprId,
        /// Required integer divisor.
        divisor: u64,
    },
    /// Inclusive range constraint: `min <= expr <= max`.
    Range {
        /// Expression to constrain.
        expr: ShapeExprId,
        /// Inclusive lower bound.
        min: i128,
        /// Inclusive upper bound.
        max: i128,
    },
    /// Modular equality: `expr % modulus == remainder`.
    ModuloEqual {
        /// Expression to constrain.
        expr: ShapeExprId,
        /// Modulus divisor.
        modulus: u64,
        /// Required remainder.
        remainder: u64,
    },
}

/// Thread-safe canonical symbolic shape interner.
///
/// Guaranteed properties:
/// 1. Two structurally equal expressions canonicalize to the exact same [`ShapeExprId`].
/// 2. Commutative operations (`Add(a, b)` and `Add(b, a)`) produce the exact same [`ShapeExprId`].
/// 3. Constant expressions are folded eagerly.
/// 4. Two structurally equal shape tuples intern to the exact same [`ShapeId`].
/// 5. Distinct expressions/shapes intern to distinct identifiers.
#[derive(Debug, Default)]
pub struct ShapeInterner {
    expr_to_id: RwLock<FxHashMap<SymbolicDim, ShapeExprId>>,
    id_to_expr: RwLock<Vec<SymbolicDim>>,
    shape_to_id: RwLock<FxHashMap<Vec<ShapeExprId>, ShapeId>>,
    id_to_shape: RwLock<Vec<Vec<ShapeExprId>>>,
}

impl ShapeInterner {
    /// Create a new, empty shape interner.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Intern a constant integer extent.
    pub fn constant(&self, value: i128) -> ShapeExprId {
        self.intern_expr_raw(SymbolicDim::Constant(value))
    }

    /// Intern a named symbolic extent.
    pub fn symbol(&self, name: impl Into<String>) -> ShapeExprId {
        self.intern_expr_raw(SymbolicDim::Symbol(name.into()))
    }

    /// Intern a parameter.
    pub fn param(&self, index: u32, name: impl Into<String>) -> ShapeExprId {
        self.intern_expr_raw(SymbolicDim::Param {
            index,
            name: name.into(),
        })
    }

    /// Intern an addition expression with canonical commutativity and constant folding.
    pub fn add(&self, lhs: ShapeExprId, rhs: ShapeExprId) -> ShapeExprId {
        // Constant folding
        if let (Some(SymbolicDim::Constant(a)), Some(SymbolicDim::Constant(b))) =
            (self.get_expr(lhs), self.get_expr(rhs))
        {
            return self.constant(a + b);
        }
        // Identity: x + 0 => x
        if let Some(SymbolicDim::Constant(0)) = self.get_expr(lhs) {
            return rhs;
        }
        if let Some(SymbolicDim::Constant(0)) = self.get_expr(rhs) {
            return lhs;
        }
        // Canonical commutative ordering: smaller ShapeExprId on LHS
        let (canonical_lhs, canonical_rhs) = if lhs.0 <= rhs.0 {
            (lhs, rhs)
        } else {
            (rhs, lhs)
        };
        self.intern_expr_raw(SymbolicDim::Add(canonical_lhs, canonical_rhs))
    }

    /// Intern a subtraction expression.
    pub fn sub(&self, lhs: ShapeExprId, rhs: ShapeExprId) -> ShapeExprId {
        if let (Some(SymbolicDim::Constant(a)), Some(SymbolicDim::Constant(b))) =
            (self.get_expr(lhs), self.get_expr(rhs))
        {
            return self.constant(a - b);
        }
        if let Some(SymbolicDim::Constant(0)) = self.get_expr(rhs) {
            return lhs;
        }
        if lhs == rhs {
            return self.constant(0);
        }
        self.intern_expr_raw(SymbolicDim::Sub(lhs, rhs))
    }

    /// Intern a multiplication expression with canonical commutativity and constant folding.
    pub fn mul(&self, lhs: ShapeExprId, rhs: ShapeExprId) -> ShapeExprId {
        // Constant folding
        if let (Some(SymbolicDim::Constant(a)), Some(SymbolicDim::Constant(b))) =
            (self.get_expr(lhs), self.get_expr(rhs))
        {
            return self.constant(a * b);
        }
        // Annihilator: x * 0 => 0
        if let Some(SymbolicDim::Constant(0)) = self.get_expr(lhs) {
            return self.constant(0);
        }
        if let Some(SymbolicDim::Constant(0)) = self.get_expr(rhs) {
            return self.constant(0);
        }
        // Identity: x * 1 => x
        if let Some(SymbolicDim::Constant(1)) = self.get_expr(lhs) {
            return rhs;
        }
        if let Some(SymbolicDim::Constant(1)) = self.get_expr(rhs) {
            return lhs;
        }
        // Canonical commutative ordering
        let (canonical_lhs, canonical_rhs) = if lhs.0 <= rhs.0 {
            (lhs, rhs)
        } else {
            (rhs, lhs)
        };
        self.intern_expr_raw(SymbolicDim::Mul(canonical_lhs, canonical_rhs))
    }

    /// Intern a floor division expression.
    pub fn div_floor(&self, lhs: ShapeExprId, rhs: ShapeExprId) -> ShapeExprId {
        if let (Some(SymbolicDim::Constant(a)), Some(SymbolicDim::Constant(b))) =
            (self.get_expr(lhs), self.get_expr(rhs))
        {
            if b != 0 {
                return self.constant(a.div_euclid(b));
            }
        }
        if let Some(SymbolicDim::Constant(1)) = self.get_expr(rhs) {
            return lhs;
        }
        self.intern_expr_raw(SymbolicDim::DivFloor(lhs, rhs))
    }

    /// Intern a ceiling division expression.
    pub fn div_ceil(&self, lhs: ShapeExprId, rhs: ShapeExprId) -> ShapeExprId {
        if let (Some(SymbolicDim::Constant(a)), Some(SymbolicDim::Constant(b))) =
            (self.get_expr(lhs), self.get_expr(rhs))
        {
            if b > 0 && a >= 0 {
                return self.constant((a + b - 1) / b);
            }
        }
        if let Some(SymbolicDim::Constant(1)) = self.get_expr(rhs) {
            return lhs;
        }
        self.intern_expr_raw(SymbolicDim::DivCeil(lhs, rhs))
    }

    /// Intern a modulo expression.
    pub fn modulo(&self, lhs: ShapeExprId, rhs: ShapeExprId) -> ShapeExprId {
        if let (Some(SymbolicDim::Constant(a)), Some(SymbolicDim::Constant(b))) =
            (self.get_expr(lhs), self.get_expr(rhs))
        {
            if b != 0 {
                return self.constant(a % b);
            }
        }
        self.intern_expr_raw(SymbolicDim::Mod(lhs, rhs))
    }

    /// Intern a min expression.
    pub fn min(&self, lhs: ShapeExprId, rhs: ShapeExprId) -> ShapeExprId {
        if lhs == rhs {
            return lhs;
        }
        if let (Some(SymbolicDim::Constant(a)), Some(SymbolicDim::Constant(b))) =
            (self.get_expr(lhs), self.get_expr(rhs))
        {
            return self.constant(a.min(b));
        }
        let (canonical_lhs, canonical_rhs) = if lhs.0 <= rhs.0 {
            (lhs, rhs)
        } else {
            (rhs, lhs)
        };
        self.intern_expr_raw(SymbolicDim::Min(canonical_lhs, canonical_rhs))
    }

    /// Intern a max expression.
    pub fn max(&self, lhs: ShapeExprId, rhs: ShapeExprId) -> ShapeExprId {
        if lhs == rhs {
            return lhs;
        }
        if let (Some(SymbolicDim::Constant(a)), Some(SymbolicDim::Constant(b))) =
            (self.get_expr(lhs), self.get_expr(rhs))
        {
            return self.constant(a.max(b));
        }
        let (canonical_lhs, canonical_rhs) = if lhs.0 <= rhs.0 {
            (lhs, rhs)
        } else {
            (rhs, lhs)
        };
        self.intern_expr_raw(SymbolicDim::Max(canonical_lhs, canonical_rhs))
    }

    /// Intern an alignment expression.
    pub fn align_to(&self, expr: ShapeExprId, alignment: u64) -> ShapeExprId {
        if alignment <= 1 {
            return expr;
        }
        if let Some(SymbolicDim::Constant(val)) = self.get_expr(expr) {
            if val >= 0 {
                let align = alignment as i128;
                let aligned = ((val + align - 1) / align) * align;
                return self.constant(aligned);
            }
        }
        self.intern_expr_raw(SymbolicDim::AlignTo { expr, alignment })
    }

    /// Intern a full multi-dimensional shape (list of dimension expressions).
    pub fn intern_shape(&self, dims: &[ShapeExprId]) -> ShapeId {
        let shape_vec = dims.to_vec();
        {
            let map = self.shape_to_id.read().unwrap();
            if let Some(&id) = map.get(&shape_vec) {
                return id;
            }
        }
        let mut map = self.shape_to_id.write().unwrap();
        if let Some(&id) = map.get(&shape_vec) {
            return id;
        }
        let mut id_list = self.id_to_shape.write().unwrap();
        let id = ShapeId(id_list.len() as u32);
        id_list.push(shape_vec.clone());
        map.insert(shape_vec, id);
        id
    }

    /// Look up an interned dimension expression by ID.
    pub fn get_expr(&self, id: ShapeExprId) -> Option<SymbolicDim> {
        let list = self.id_to_expr.read().unwrap();
        list.get(id.0 as usize).cloned()
    }

    /// Look up an interned multi-dimensional shape by ID.
    pub fn get_shape(&self, id: ShapeId) -> Option<Vec<ShapeExprId>> {
        let list = self.id_to_shape.read().unwrap();
        list.get(id.0 as usize).cloned()
    }

    fn intern_expr_raw(&self, dim: SymbolicDim) -> ShapeExprId {
        {
            let map = self.expr_to_id.read().unwrap();
            if let Some(&id) = map.get(&dim) {
                return id;
            }
        }
        let mut map = self.expr_to_id.write().unwrap();
        if let Some(&id) = map.get(&dim) {
            return id;
        }
        let mut id_list = self.id_to_expr.write().unwrap();
        let id = ShapeExprId(id_list.len() as u32);
        id_list.push(dim.clone());
        map.insert(dim, id);
        id
    }
}
