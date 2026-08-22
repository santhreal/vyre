//! Dual CPU references for `primitive.bitwise.xor`.

define_binary_bitwise_dual!(
    XorDualReference,
    "primitive.bitwise.xor",
    |left, right| left ^ right,
    |left, right| left != right
);
