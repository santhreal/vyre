//! Contract tests verifying all submodules and canonical items in `vyre-spec` are public to maintain SemVer compatibility.

#[test]
fn spec_submodules_and_items_are_public() {
    // category
    fn accepts_backend_availability<T: vyre_spec::category::BackendAvailability>() {}
    accepts_backend_availability::<vyre_spec::category::BackendAvailabilityPredicate>();
    let _cat: Option<vyre_spec::category::Category> = None;
    let _pred: Option<vyre_spec::category::BackendAvailabilityPredicate> = None;

    // op_contract
    let _contract: Option<vyre_spec::op_contract::OperationContract> = None;
    let _cap: Option<vyre_spec::op_contract::CapabilityId> = None;

    // op_metadata & op_signature
    let _op_meta: Option<vyre_spec::op_metadata::OpMetadata> = None;
    let _op_sig: Option<vyre_spec::op_signature::OpSignature> = None;

    // golden_sample & kat_vector
    let _golden: Option<vyre_spec::golden_sample::GoldenSample> = None;
    let _kat: Option<vyre_spec::kat_vector::KatVector> = None;

    // invariant & invariant_category
    let _inv: Option<vyre_spec::invariant::Invariant> = None;
    let _inv_cat: Option<vyre_spec::invariant_category::InvariantCategory> = None;

    // data_type
    let _dt: vyre_spec::data_type::DataType = vyre_spec::data_type::DataType::F32;

    // by_category & by_id
    let _by_cat = vyre_spec::by_category::by_category(vyre_spec::InvariantCategory::Execution);
    let _by_id = vyre_spec::by_id::by_id(vyre_spec::EngineInvariant::I1);

    // test_descriptor
    let _td: Option<vyre_spec::test_descriptor::TestDescriptor> = None;

    // bin_op, un_op, ternary_op, atomic_op, collective_op
    let _bin: Option<vyre_spec::bin_op::BinOp> = None;
    let _un: Option<vyre_spec::un_op::UnOp> = None;
    let _ter: Option<vyre_spec::ternary_op::TernaryOp> = None;
    let _atom: Option<vyre_spec::atomic_op::AtomicOp> = None;
    let _coll: Option<vyre_spec::collective_op::CollectiveOp> = None;

    // intrinsic_descriptor
    let _id: Option<vyre_spec::intrinsic_descriptor::IntrinsicDescriptor> = None;
    let _backend_id: Option<vyre_spec::intrinsic_descriptor::BackendId> = None;

    // domain-owning submodules
    let _ext: Option<vyre_spec::extension::ExtensionDataTypeId> = None;
    let _tok = vyre_spec::c11_token::TOK_EOF;
    let _expr_tok = vyre_spec::c11_expr_token::TOK_EOF;
    let _go_tok = vyre_spec::go_token::TOK_NONE;
    let _py_tok = vyre_spec::python_token::TOK_NONE;
    let _analysis: Option<vyre_spec::analysis::AnalysisFactRecord> = None;
    let _soundness: Option<vyre_spec::soundness::Soundness> = None;
}
