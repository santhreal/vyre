use super::*;

/// Every declarator-matrix case, on the GPU, through all four stages.
///
/// One test rather than one per case and per stage. The case list is
/// `declarator_matrix_constructs::CASES`, which the CPU arm in `vyre-libs/tests`
/// iterates as well, so a construct cannot reach one arm and miss the other. The
/// per-case `#[test]` functions this replaces named eight of the ten cases for
/// the builder, annotator and classifier and only four of the ten for the
/// property-graph lowerer, and named `gnu_restrict_qualifier` nowhere at all.
#[test]
fn gpu_parity_declarator_matrix_cases() {
    assert_family_parity(&GpuArm, declarator_matrix_constructs::CASES);
}

/// The abstract-declarator cast, per stage, with the classifier fed the RAW
/// VAST.
///
/// `(const int (*)(void))p;` carries no typedef name, so the annotated rows and
/// the raw rows are equal and the family test above classifies the annotated
/// copy. Classifying the raw copy is the separate contract that the classifier
/// does not depend on having been handed annotated rows.
#[test]
fn gpu_parity_abstract_declarator_classifier_on_raw_vast() {
    let fix = fixture_abstract_declarator_with_qualifiers();
    let raw_cpu = reference_c11_build_vast_nodes(&fix.tok_types, &fix.tok_starts, &fix.tok_lens);
    let raw_gpu = run_gpu_vast_builder_from_parts(&fix.tok_types, &fix.tok_starts, &fix.tok_lens);
    assert_words_eq(
        &raw_gpu,
        &raw_cpu,
        "abstract_declarator_with_qualifiers: VAST builder GPU/CPU parity",
    );
    assert_words_eq(
        &run_gpu_classifier(&raw_gpu),
        &reference_c11_classify_vast_node_kinds(&raw_cpu),
        "abstract_declarator_with_qualifiers: classifier GPU/CPU parity on raw VAST",
    );
}
