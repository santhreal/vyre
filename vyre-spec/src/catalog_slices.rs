const EXPR_VARIANTS: &[&str] = &[
    "LitU32",
    "LitI32",
    "LitF32",
    "LitBool",
    "Var",
    "BufferRef",
    "Load",
    "BufLen",
    "InvocationId",
    "WorkgroupId",
    "LocalId",
    "BinOp",
    "UnOp",
    "Call",
    "Select",
    "Cast",
    "Fma",
    "Atomic",
    "SubgroupBallot",
    "SubgroupShuffle",
    "SubgroupReduce",
    "SubgroupLocalId",
    "SubgroupSize",
    "Opaque",
];

/// Return the frozen catalog of core `Expr` variant names.
#[must_use]
pub fn expr_variants() -> &'static [&'static str] {
    EXPR_VARIANTS
}

/// Return the catalog of all algebraic-law variant fingerprints.
///
/// Derived from [`crate::LawFamily`], which is closed against
/// [`crate::AlgebraicLaw`]. The former hand-written list was two members
/// behind that enum, and both categorical laws were unreachable through this
/// catalog while the check on it compared one hand list's length to another's.
#[must_use]
pub fn law_catalog() -> &'static [&'static str] {
    crate::law_family::law_family_names()
}
