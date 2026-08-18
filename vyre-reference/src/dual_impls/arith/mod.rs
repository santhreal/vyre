/// docs
macro_rules! define_arith_dual_reference {
    ($marker:ident, $direct:path, $independent:path) => {
        define_dual_reference_impl!(
            $marker,
            |input| crate::dual_impls::evaluator::binary_direct(input, $direct),
            $independent
        );
    };
}

/// docs
pub mod add;
mod bit_walk_reference;
/// docs
pub mod mul;
