use super::*;

#[test]
fn gpu_matches_cpu_for_precedence_fixtures() {
    assert_expression_shape_parity(&[
        shift_precedence_fixture(),
        relational_precedence_fixture(),
        equality_precedence_fixture(),
        equality_left_assoc_fixture(),
        compound_assignment_fixture(),
        ternary_looser_than_assignment_fixture(),
        ternary_right_assoc_fixture(),
        comma_boundary_fixture(),
        full_precedence_ladder_fixture(),
    ]);
}
