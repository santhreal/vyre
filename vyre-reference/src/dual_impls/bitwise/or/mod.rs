//! Dual CPU references for `primitive.bitwise.or`.

define_binary_bitwise_dual!(
    OrDualReference,
    "primitive.bitwise.or",
    |left, right| left | right,
    |left, right| left || right
);
