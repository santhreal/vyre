//! Contract tests verifying all canonical items in `vyre-spec` are reachable at the crate root
//! and domain-owning submodules remain public without duplication.

#[test]
fn spec_canonical_items_are_public() {
    // category
    fn accepts_backend_availability<T: vyre_spec::BackendAvailability>() {}
    accepts_backend_availability::<vyre_spec::BackendAvailabilityPredicate>();
    let _cat: Option<vyre_spec::Category> = None;
    let _pred: Option<vyre_spec::BackendAvailabilityPredicate> = None;

    // op_contract
    let _contract: Option<vyre_spec::OperationContract> = None;
    let _cap: Option<vyre_spec::CapabilityId> = None;

    // golden_sample
    let _golden: Option<vyre_spec::GoldenSample> = None;

    // invariant & invariant_category
    let _inv: Option<vyre_spec::Invariant> = None;
    let _inv_cat: Option<vyre_spec::InvariantCategory> = None;

    // data_type
    let _dt: vyre_spec::DataType = vyre_spec::DataType::F32;

    // by_category & by_id
    let _by_cat = vyre_spec::by_category(vyre_spec::InvariantCategory::Execution);
    let _by_id = vyre_spec::by_id(vyre_spec::EngineInvariant::I1);

    // test_descriptor
    let _td: Option<vyre_spec::TestDescriptor> = None;

    // bin_op, un_op, ternary_op, atomic_op, collective_op
    let _bin: Option<vyre_spec::BinOp> = None;
    let _un: Option<vyre_spec::UnOp> = None;
    let _ter: Option<vyre_spec::TernaryOp> = None;
    let _atom: Option<vyre_spec::AtomicOp> = None;
    let _coll: Option<vyre_spec::CollectiveOp> = None;

    // intrinsic_descriptor
    let _id: Option<vyre_spec::IntrinsicDescriptor> = None;

    // domain-owning submodules
    let _ext: Option<vyre_spec::extension::ExtensionDataTypeId> = None;
    let _tok = vyre_spec::c11_token::TOK_EOF;
    let _expr_tok = vyre_spec::c11_expr_token::TOK_EOF;
    let _go_tok = vyre_spec::go_token::TOK_NONE;
    let _py_tok = vyre_spec::python_token::TOK_NONE;
    let _analysis: Option<vyre_spec::analysis::AnalysisFactRecord> = None;
    let _soundness: Option<vyre_spec::soundness::Soundness> = None;
}
