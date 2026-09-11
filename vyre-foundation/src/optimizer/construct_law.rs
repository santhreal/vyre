//! Which declarative law families each IR construct exposes.
//!
//! A derivation mechanism that reads laws instead of recipes needs one fact per
//! construct: which families of law may be cited over it, or, when none may,
//! why it is deliberately opaque. Recording that here is what keeps the
//! derivation closed. A construct added to the AST registry appears in the
//! generated variant tables immediately, and the closure test compares those
//! tables against these rows, so a new construct is red until its laws or its
//! opacity are recorded.
//!
//! Opacity is a decision, not a gap: `Expr::Atomic` exposes no value law
//! because a read-modify-write is ordered against concurrent invocations, and
//! recording that is what stops a later derivation from treating it as
//! arithmetic.

use vyre_spec::RegionLawFamily::{self, Algebraic, Layout, Numerical, Recurrence, Reduction};

use crate::ir::{expr_variant_name, node_variant_name, Expr, Node};

/// The law families one IR construct exposes, or the reason it exposes none.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ConstructLaws {
    /// Declared variant name, matching the generated AST variant tables.
    pub construct: &'static str,
    /// Families a law over this construct may belong to.
    pub families: &'static [RegionLawFamily],
    /// Why this construct exposes no law, when it exposes none.
    pub opacity: Option<&'static str>,
}

impl ConstructLaws {
    /// Whether a law in `family` may be cited over this construct.
    #[must_use]
    pub fn admits(&self, family: RegionLawFamily) -> bool {
        self.families.contains(&family)
    }
}

const fn laws(construct: &'static str, families: &'static [RegionLawFamily]) -> ConstructLaws {
    ConstructLaws {
        construct,
        families,
        opacity: None,
    }
}

const fn opaque(construct: &'static str, reason: &'static str) -> ConstructLaws {
    ConstructLaws {
        construct,
        families: &[],
        opacity: Some(reason),
    }
}

/// Law families each expression construct exposes.
pub const EXPR_CONSTRUCT_LAWS: &[ConstructLaws] = &[
    laws("LitU32", &[Algebraic]),
    laws("LitI32", &[Algebraic]),
    laws("LitF32", &[Algebraic, Numerical]),
    laws("LitBool", &[Algebraic]),
    opaque(
        "Var",
        "a name exposes the laws of the value bound to it, so the binding carries them",
    ),
    opaque(
        "BufferRef",
        "a buffer handle has no value algebra; the accesses that use it carry the layout laws",
    ),
    laws("Load", &[Layout]),
    opaque(
        "BufLen",
        "a buffer length is a launch fact, not a value a law may rewrite",
    ),
    laws("InvocationId", &[Layout]),
    laws("LogicalIndex", &[Layout]),
    laws("LogicalTileId", &[Layout]),
    laws("LogicalWithinTileId", &[Layout]),
    laws("WorkgroupId", &[Layout]),
    laws("LocalId", &[Layout]),
    laws("BinOp", &[Algebraic, Numerical]),
    laws("UnOp", &[Algebraic, Numerical]),
    opaque(
        "Call",
        "per-operation laws are declared in the frozen operation registry and verified by the \
         algebra checker, so a law over a call is a law over its op id",
    ),
    laws("Select", &[Algebraic]),
    laws("Cast", &[Numerical]),
    laws("Fma", &[Algebraic, Numerical]),
    opaque(
        "Atomic",
        "a read-modify-write is ordered against every concurrent invocation, so no value law may \
         duplicate, drop, or reorder it",
    ),
    opaque(
        "SubgroupBallot",
        "the result depends on the set of lanes that reach it, which no value law preserves",
    ),
    opaque(
        "SubgroupShuffle",
        "the result depends on the lane the value is read from, which no value law preserves",
    ),
    laws("SubgroupReduce", &[Reduction]),
    opaque("SubgroupLocalId", "a lane identity is a launch fact"),
    opaque("SubgroupSize", "a subgroup width is a target fact"),
    opaque(
        "Opaque",
        "an extension declares its own semantics, so the registry derives no law for it",
    ),
];

/// Law families each statement construct exposes.
pub const NODE_CONSTRUCT_LAWS: &[ConstructLaws] = &[
    laws("Let", &[Algebraic]),
    opaque(
        "Assign",
        "a rebinding is ordered against every later read of the name, so substitution laws apply \
         to the binding form instead",
    ),
    laws("Store", &[Layout]),
    laws("If", &[Algebraic]),
    laws("Loop", &[Recurrence]),
    opaque(
        "IndirectDispatch",
        "the dispatch count is read from a buffer at submission, so no law knows it",
    ),
    opaque(
        "AsyncLoad",
        "an asynchronous transaction is ordered by its tag, and a law that moved it would move the \
         wait with it",
    ),
    opaque(
        "AsyncStore",
        "an asynchronous transaction is ordered by its tag, and a law that moved it would move the \
         wait with it",
    ),
    opaque(
        "AsyncWait",
        "a wait is the ordering itself, so no law may move or drop it",
    ),
    opaque(
        "Trap",
        "a trap transfers control out of the region, so no law may reorder work across it",
    ),
    opaque("Resume", "a resume re-enters a trapped region at its tag"),
    laws("AllReduce", &[Reduction]),
    laws("AllGather", &[Layout]),
    laws("ReduceScatter", &[Reduction, Layout]),
    laws("Broadcast", &[Layout]),
    opaque(
        "Return",
        "termination is control structure, not a value a law may rewrite",
    ),
    opaque(
        "Barrier",
        "a barrier is the ordering itself; the synchronization passes state when one may move",
    ),
    opaque(
        "LogicalBarrier",
        "a logical barrier is the ordering itself; the synchronization passes state when one may \
         move",
    ),
    laws("Block", &[Layout]),
    laws("Region", &[Layout]),
    laws("TileLoad", &[Layout]),
    laws("TileStore", &[Layout]),
    laws("TileMatmul", &[Reduction, Layout]),
    laws("TileReduce", &[Reduction]),
    laws("TileElementwise", &[Algebraic]),
    laws("TileDecl", &[Layout]),
    opaque(
        "Opaque",
        "an extension declares its own semantics, so the registry derives no law for it",
    ),
];

/// Laws the construct named `construct` exposes, if it is a declared
/// expression construct.
#[must_use]
pub fn expr_construct_laws(construct: &str) -> Option<&'static ConstructLaws> {
    EXPR_CONSTRUCT_LAWS
        .iter()
        .find(|entry| entry.construct == construct)
}

/// Laws the construct named `construct` exposes, if it is a declared statement
/// construct.
#[must_use]
pub fn node_construct_laws(construct: &str) -> Option<&'static ConstructLaws> {
    NODE_CONSTRUCT_LAWS
        .iter()
        .find(|entry| entry.construct == construct)
}

/// Laws `expr` exposes.
///
/// The closure test proves every declared variant has a row, so this answers
/// for every expression the IR can hold.
///
/// # Panics
///
/// Panics when the expression's variant has no row, which means a variant
/// reached the AST registry without its laws or its deliberate opacity being
/// recorded. Failing closed is the contract: a construct whose laws are unknown
/// must not be treated as exposing none.
#[must_use]
pub fn laws_of(expr: &Expr) -> &'static ConstructLaws {
    expr_construct_laws(expr_variant_name(expr)).expect(
        "every declared expression variant has a construct-law row. Fix: record laws or \
             deliberate opacity for the new variant in EXPR_CONSTRUCT_LAWS.",
    )
}

/// Laws `node` exposes.
///
/// # Panics
///
/// Panics when the statement's variant has no row, for the reason
/// [`laws_of`] states.
#[must_use]
pub fn laws_of_node(node: &Node) -> &'static ConstructLaws {
    node_construct_laws(node_variant_name(node)).expect(
        "every declared statement variant has a construct-law row. Fix: record laws or \
             deliberate opacity for the new variant in NODE_CONSTRUCT_LAWS.",
    )
}
