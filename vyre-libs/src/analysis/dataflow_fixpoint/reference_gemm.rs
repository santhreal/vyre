//! Telemetered host entry point for the semiring matrix product.

#[cfg(test)]
mod tests {
    use vyre_foundation::pass_substrate::semiring_closure as foundation_dataflow;
    use vyre_reference::composition_witness::semiring_gemm_witness;
    use vyre_spec::Semiring;

    #[test]
    fn telemetered_gemm_matches_the_owner_and_reuses_the_output() {
        let left = vec![1, 2, 3, 4, 5, 6];
        let right = vec![7, 8, 9, 10, 11, 12];
        let out = semiring_gemm_witness(&left, &right, 2, 2, 3, Semiring::Real);
        let mut expected = Vec::new();
        foundation_dataflow::reference_semiring_gemm_into(
            &left,
            &right,
            2,
            2,
            3,
            Semiring::Real,
            &mut expected,
        );
        assert_eq!(out, expected);
    }
}
