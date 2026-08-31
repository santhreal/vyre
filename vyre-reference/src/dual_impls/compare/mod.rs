macro_rules! define_compare_dual_reference {
    ($marker:ident, $direct:expr, $independent:path) => {
        define_dual_reference_impl!(
            $marker,
            |input| crate::dual_impls::evaluator::binary_direct_predicate(input, $direct),
            $independent
        );
    };
}

mod byte_walk_reference;
/// docs
pub mod eq;
/// docs
pub mod lt;
