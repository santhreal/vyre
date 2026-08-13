//! Dual CPU references for `primitive.bitwise.not`.

define_unary_bitwise_dual!(
    NotDualReference,
    "primitive.bitwise.not",
    |value| !value,
    |value| !value
);
