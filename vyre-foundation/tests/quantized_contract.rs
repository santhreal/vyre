//! What a quantized value states, and what a graph that reads one is priced at.
//!
//! WHY: a quantized value is one logical contract rather than a private
//! convention each wrapper restates. Three defects motivate the cases here. A packing law restated by each builder lets two consumers read
//! the same bytes differently with nothing in the program saying so. A
//! quantized region priced through its storage format is priced as exact
//! integer arithmetic, so its dequantization error never reaches a graph
//! budget. A layout no producer can write is only discovered when its bytes
//! are misread.
//!
//! What these cases do not prove: what a device computes, or that the IR a
//! builder emits from the contract evaluates to the value the contract states.
//! The conformance suite in `vyre-conform` executes that IR against the
//! reference decode.

use std::collections::BTreeMap;

use vyre_foundation::ir::{
    BufferAccess, BufferDecl, DataType, Expr, GraphInput, GraphOutput, Node, Program, ProgramGraph,
    ValueLifetime,
};
use vyre_foundation::logical::LogicalProgramGraph;
use vyre_foundation::numeric::{
    CalibrationIdentity, ErrorMeasure, GroupAxis, NumericContract, PackingOrder, QuantizedContract,
    QuantizedRefusal, Reassociation, ScalarFormat,
};
use vyre_spec::{QuantizationScale, QuantizationZeroPoint, RoundingMode, SaturationBehavior};
use vyre_test_support::graph_values::typed_vector as vector;

/// The INT4 contract most cases start from.
fn packed_i4() -> QuantizedContract {
    QuantizedContract::symmetric(ScalarFormat::I4, ScalarFormat::F32, ScalarFormat::U32)
}

/// Every refusal, named.
///
/// The match has no catch-all arm, so a new refusal stops this suite from
/// compiling until the case table below produces it.
fn refusal_label(refusal: &QuantizedRefusal) -> &'static str {
    match refusal {
        QuantizedRefusal::StorageNotPacked { .. } => "storage-not-packed",
        QuantizedRefusal::LogicalNotWider { .. } => "logical-not-wider",
        QuantizedRefusal::SignednessDisagrees { .. } => "signedness-disagrees",
        QuantizedRefusal::ContainerNotUnsigned { .. } => "container-not-unsigned",
        QuantizedRefusal::ContainerNotWhole { .. } => "container-not-whole",
        QuantizedRefusal::GroupExtentZero { .. } => "group-extent-zero",
        QuantizedRefusal::GroupAxisRepeated { .. } => "group-axis-repeated",
        QuantizedRefusal::GroupingDisagreesWithScale { .. } => "grouping-disagrees",
        QuantizedRefusal::ZeroPointDisagreesWithScale { .. } => "zero-point-disagrees",
        QuantizedRefusal::AlignmentNotContainerMultiple { .. } => "alignment-not-multiple",
        QuantizedRefusal::AccumulatorNarrowerThanLogical { .. } => "accumulator-narrower",
        QuantizedRefusal::RoundingDisagrees { .. } => "rounding-disagrees",
        QuantizedRefusal::SaturationDisagrees { .. } => "saturation-disagrees",
        QuantizedRefusal::LayoutsDisagree { .. } => "layouts-disagree",
    }
}

#[test]
fn every_quantized_storage_family_states_a_contract_that_reads_its_bytes() {
    let families = DataType::SCALAR_LEAVES
        .iter()
        .filter(|dtype| dtype.is_quantized_storage())
        .collect::<Vec<_>>();
    assert!(
        !families.is_empty(),
        "the spec declares at least one quantized storage family"
    );
    let by_format = ScalarFormat::ALL
        .iter()
        .filter(|format| format.data_type().is_quantized_storage())
        .count();
    assert_eq!(
        families.len(),
        by_format,
        "every quantized storage leaf must name a scalar format"
    );

    for dtype in families {
        let storage = ScalarFormat::of(dtype).expect("a scalar leaf names a scalar format");
        let contract = QuantizedContract::symmetric(storage, ScalarFormat::F32, ScalarFormat::U32);
        contract
            .check()
            .unwrap_or_else(|refusal| panic!("{storage} states an unreadable layout: {refusal}"));
        let fields = contract.fields_per_container();
        assert_eq!(fields, 32 / storage.bit_width(), "{storage}");
        let last = contract.field(u64::from(fields) - 1);
        assert_eq!(last.word, 0, "{storage}");
        assert_eq!(
            last.shift_bits + storage.bit_width(),
            32,
            "the last field of a word ends at the container's top bit for {storage}"
        );
        assert_eq!(contract.field(u64::from(fields)).word, 1, "{storage}");
        assert_eq!(
            contract.container_words(u64::from(fields) + 1),
            2,
            "{storage}"
        );
    }
}

#[test]
fn a_packed_field_lands_where_the_packing_order_states() {
    let low = packed_i4();
    let high = packed_i4().packed(PackingOrder::HighFieldFirst);
    assert_eq!(low.fields_per_container(), 8);

    let low_third = low.field(3);
    assert_eq!(
        (low_third.word, low_third.shift_bits, low_third.mask),
        (0, 12, 0xF)
    );
    let high_third = high.field(3);
    assert_eq!(
        (high_third.word, high_third.shift_bits, high_third.mask),
        (0, 16, 0xF)
    );

    assert_eq!(low.field(8).word, 1);
    assert_eq!(high.field(8).word, 1);
    assert_eq!(low.field(8).shift_bits, 0);
    assert_eq!(high.field(8).shift_bits, 28);
    assert!(
        !low.propagates_to(&high),
        "two packing orders read the same word differently"
    );
}

#[test]
fn a_group_that_does_not_divide_the_axis_keeps_a_short_tail() {
    let contract = packed_i4().grouped_by(vec![GroupAxis {
        axis: 0,
        extent: 32,
    }]);
    contract.check().expect("a grouped INT4 layout is readable");
    assert_eq!(contract.group_elements(), 32);
    assert_eq!(contract.groups_over(96), 3);
    assert_eq!(contract.tail_elements(96), 32);
    assert_eq!(contract.groups_over(100), 4);
    assert_eq!(contract.tail_elements(100), 4);
    assert_eq!(contract.groups_over(0), 0);
    assert_eq!(contract.tail_elements(0), 0);
    assert_eq!(contract.groups_over(1), 1);
    assert_eq!(contract.tail_elements(1), 1);
}

#[test]
fn a_two_dimensional_group_states_both_axes() {
    let contract = packed_i4().grouped_by(vec![
        GroupAxis {
            axis: 0,
            extent: 16,
        },
        GroupAxis { axis: 1, extent: 4 },
    ]);
    contract
        .check()
        .expect("a block-grouped layout is readable");
    assert_eq!(contract.group_elements(), 64);
    assert_eq!(
        contract.scale,
        QuantizationScale::PerGroup { group_size: 64 },
        "the scale source states what the axes cover"
    );
}

#[test]
fn a_layout_no_producer_can_write_is_refused() {
    let mut ungrouped_with_axes = packed_i4();
    ungrouped_with_axes.grouping = vec![GroupAxis { axis: 0, extent: 8 }];
    let mut split_group = packed_i4().grouped_by(vec![GroupAxis {
        axis: 0,
        extent: 32,
    }]);
    split_group.zero_point = QuantizationZeroPoint::PerGroup { group_size: 16 };
    let mut wrong_signedness = packed_i4();
    wrong_signedness.signed = false;

    let cases: Vec<(&str, QuantizedContract)> = vec![
        (
            "storage-not-packed",
            QuantizedContract::symmetric(ScalarFormat::F32, ScalarFormat::F64, ScalarFormat::U32),
        ),
        (
            "logical-not-wider",
            QuantizedContract::symmetric(ScalarFormat::I8, ScalarFormat::I8, ScalarFormat::U32),
        ),
        ("signedness-disagrees", wrong_signedness),
        (
            "container-not-unsigned",
            QuantizedContract::symmetric(ScalarFormat::I4, ScalarFormat::F32, ScalarFormat::I32),
        ),
        (
            "container-not-whole",
            QuantizedContract::symmetric(ScalarFormat::U16, ScalarFormat::F32, ScalarFormat::U8),
        ),
        (
            "group-extent-zero",
            packed_i4().grouped_by(vec![GroupAxis { axis: 0, extent: 0 }]),
        ),
        (
            "group-axis-repeated",
            packed_i4().grouped_by(vec![
                GroupAxis { axis: 0, extent: 4 },
                GroupAxis { axis: 0, extent: 8 },
            ]),
        ),
        ("grouping-disagrees", ungrouped_with_axes),
        ("zero-point-disagrees", split_group),
        ("alignment-not-multiple", packed_i4().aligned_to(3)),
        (
            "accumulator-narrower",
            packed_i4().accumulating_in(ScalarFormat::F16),
        ),
        ("rounding-disagrees", {
            let mut exact = packed_i4();
            exact.rounding = RoundingMode::Exact;
            exact
        }),
        ("rounding-disagrees", {
            let mut quantile = packed_i4();
            quantile.rounding = RoundingMode::NearestQuantile;
            quantile
        }),
        ("rounding-disagrees", {
            let mut nearest = QuantizedContract::symmetric(
                ScalarFormat::NF4,
                ScalarFormat::F32,
                ScalarFormat::U32,
            );
            nearest.rounding = RoundingMode::RoundToNearestEven;
            nearest
        }),
        ("saturation-disagrees", {
            let mut unit = packed_i4();
            unit.saturation = SaturationBehavior::ClampsToUnitInterval;
            unit
        }),
    ];

    for (expected, contract) in cases {
        let refusal = contract
            .check()
            .expect_err(&format!("{expected} must be refused"));
        assert_eq!(refusal_label(&refusal), expected, "{refusal}");
    }
}

#[test]
fn a_grid_states_the_step_it_quantizes_to() {
    let signed_four = packed_i4().step_fraction();
    assert!(
        (signed_four - 1.0 / 7.0).abs() < f64::EPSILON,
        "a signed four-bit grid spends one code on the sign: {signed_four}"
    );
    let unsigned_eight =
        QuantizedContract::symmetric(ScalarFormat::U8, ScalarFormat::F32, ScalarFormat::U32)
            .step_fraction();
    assert!(
        (unsigned_eight - 1.0 / 255.0).abs() < f64::EPSILON,
        "an unsigned eight-bit grid spends none: {unsigned_eight}"
    );
    let signed_sixteen =
        QuantizedContract::symmetric(ScalarFormat::I16, ScalarFormat::F32, ScalarFormat::U32)
            .step_fraction();
    assert!(
        signed_sixteen < signed_four / 4000.0,
        "a wider grid steps far more finely: {signed_sixteen}"
    );
    let codebook =
        QuantizedContract::symmetric(ScalarFormat::NF4, ScalarFormat::F32, ScalarFormat::U32)
            .step_fraction();
    assert!(
        codebook > signed_four,
        "a normal-float codebook has a wider widest gap than a uniform grid: {codebook}"
    );

    assert_eq!(
        packed_i4().dequantization_measure(),
        ErrorMeasure::relative(1.0 / 14.0),
        "nearest rounding lands within half a step"
    );
    let mut truncating = packed_i4();
    truncating.rounding = RoundingMode::TruncateTowardsZero;
    assert_eq!(
        truncating.dequantization_measure(),
        ErrorMeasure::relative(1.0 / 7.0),
        "truncation lands within a whole one"
    );
}

#[test]
fn a_quantized_value_is_priced_at_its_logical_format() {
    let contract = packed_i4();
    let numeric = contract.numeric();
    assert_eq!(numeric.storage, ScalarFormat::F32);
    assert_eq!(numeric.accumulator, ScalarFormat::F32);
    assert_eq!(numeric.measure, contract.dequantization_measure());
    assert!(
        !numeric.measure.is_exact(),
        "reading a quantized value costs the step it was quantized to"
    );
    assert!(
        NumericContract::of(ScalarFormat::I4).measure.is_exact(),
        "the storage format on its own is exact, which is the defect this closes"
    );
}

#[test]
fn a_quantized_region_admits_a_reordering_binary32_alone_forbids() {
    assert_eq!(
        NumericContract::of(ScalarFormat::F32).reassociation,
        Reassociation::Forbidden
    );
    assert_eq!(
        packed_i4().numeric().reassociation,
        Reassociation::WithinBudget,
        "a value already quantized to a step admits an order change inside that step"
    );
}

#[test]
fn saturation_and_mixed_precision_reach_the_numeric_contract() {
    let mut wrapping = packed_i4();
    wrapping.saturation = SaturationBehavior::None;
    assert_eq!(
        wrapping.numeric().overflow,
        NumericContract::of(ScalarFormat::F32).overflow,
        "a layout that does not saturate keeps the logical format's own overflow"
    );
    assert_eq!(
        packed_i4().numeric().overflow,
        vyre_spec::OverflowBehavior::SaturateToFiniteRange
    );

    let mixed =
        QuantizedContract::symmetric(ScalarFormat::I8, ScalarFormat::F16, ScalarFormat::U32)
            .accumulating_in(ScalarFormat::F32);
    mixed.check().expect("f16 read of int8 codes is readable");
    let numeric = mixed.numeric();
    assert_eq!(numeric.storage, ScalarFormat::F16);
    assert_eq!(numeric.accumulator, ScalarFormat::F32);
}

#[test]
fn a_conversion_between_two_layouts_is_priced_rather_than_assumed() {
    let low = packed_i4();
    let high = packed_i4().packed(PackingOrder::HighFieldFirst);
    let grouped = packed_i4().grouped_by(vec![GroupAxis {
        axis: 0,
        extent: 32,
    }]);
    let wider =
        QuantizedContract::symmetric(ScalarFormat::I8, ScalarFormat::F32, ScalarFormat::U32);

    assert!(low.conversion(&low.clone()).is_free());
    assert!(low.propagates_to(&low.clone()));

    let repack = low.conversion(&high);
    assert!(repack.repacks && !repack.rescales && !repack.requantizes);
    assert_eq!(repack.steps(), 1);
    assert_eq!(repack.measure(&high), ErrorMeasure::Exact);

    let rescale = low.conversion(&grouped);
    assert!(rescale.rescales && !rescale.requantizes);

    let requantize = low.conversion(&wider);
    assert!(requantize.requantizes);
    assert_eq!(
        requantize.measure(&wider),
        wider.dequantization_measure(),
        "landing on another grid costs that grid's step"
    );
    assert!(
        requantize.steps() >= 1,
        "a conversion the search inserts is never free"
    );
}

#[test]
fn a_calibration_identity_authenticates_the_payload_it_was_taken_over() {
    let payload = b"scale=0.0139,zero=0";
    let identity = CalibrationIdentity::of(payload);
    assert!(identity.authenticates(payload));
    assert!(!identity.authenticates(b"scale=0.0140,zero=0"));
    assert_eq!(identity.hex().len(), 64);
    assert_ne!(
        identity,
        CalibrationIdentity::of(b"scale=0.0140,zero=0"),
        "two calibrations of the same weights are two identities"
    );

    let calibrated = packed_i4().calibrated_by(identity);
    assert_ne!(
        calibrated,
        packed_i4(),
        "a contract produced under a calibration is not the uncalibrated one"
    );
    assert!(
        !calibrated.propagates_to(&packed_i4()),
        "two calibrations do not share one packed layout"
    );
}

#[test]
fn a_frozen_quantized_type_states_the_contract_its_bytes_follow() {
    let dtype = DataType::Quantized {
        storage: Box::new(DataType::I4),
        scale: QuantizationScale::PerGroup { group_size: 32 },
        zero_point: QuantizationZeroPoint::PerGroup { group_size: 32 },
    };
    let contract = QuantizedContract::of(&dtype).expect("a quantized type states a contract");
    contract.check().expect("the derived contract is readable");
    assert_eq!(contract.storage, ScalarFormat::I4);
    assert_eq!(contract.logical, ScalarFormat::F32);
    assert_eq!(contract.group_elements(), 32);
    assert_eq!(
        contract.zero_point,
        QuantizationZeroPoint::PerGroup { group_size: 32 }
    );
    assert!(QuantizedContract::of(&DataType::F32).is_none());
    assert_eq!(
        QuantizedContract::of(&DataType::Vec {
            element: Box::new(dtype),
            count: 4,
        })
        .map(|nested| nested.storage),
        Some(ScalarFormat::I4),
        "a composite carries its element's contract"
    );
}

/// A graph whose one region reads `inputs` and writes one binary32 vector.
fn reading_graph(inputs: &[DataType]) -> ProgramGraph {
    let count = 64;
    let output = vector(
        count,
        DataType::F32,
        BufferAccess::WriteOnly,
        ValueLifetime::Output,
    );
    let mut graph = ProgramGraph::new();
    let mut buffers = Vec::new();
    let mut graph_inputs = Vec::new();
    let mut sum = Expr::f32(0.0);
    for (slot, dtype) in inputs.iter().enumerate() {
        let name = format!("input{slot}");
        let contract = vector(
            count,
            dtype.clone(),
            BufferAccess::ReadOnly,
            ValueLifetime::Invocation,
        );
        let value = graph
            .add_external_value(&name, contract.clone())
            .expect("fixture external value must be valid");
        buffers.push(
            BufferDecl::read(
                &name,
                u32::try_from(slot).expect("slot fits"),
                dtype.clone(),
            )
            .with_count(count),
        );
        graph_inputs.push(GraphInput {
            buffer: name.clone(),
            value,
            contract,
        });
        sum = Expr::add(
            sum,
            Expr::cast(DataType::F32, Expr::load(&name, Expr::gid_x())),
        );
    }
    buffers.push(
        BufferDecl::output(
            "output",
            u32::try_from(inputs.len()).expect("slot fits"),
            DataType::F32,
        )
        .with_count(count),
    );
    graph
        .add_node(
            "read",
            Program::wrapped(
                buffers,
                [64, 1, 1],
                vec![Node::store("output", Expr::gid_x(), sum)],
            ),
            graph_inputs,
            vec![GraphOutput {
                buffer: "output".into(),
                name: "output".into(),
                contract: output,
                retained_successor_of: None,
            }],
        )
        .expect("fixture node must be valid");
    graph
}

/// The contract the one output of `graph` carries.
fn output_budget(graph: &ProgramGraph) -> NumericContract {
    let logical = LogicalProgramGraph::validate(graph, &BTreeMap::new())
        .expect("fixture logical stage must validate");
    let budgets = logical
        .output_budgets()
        .expect("outputs must state a budget");
    budgets[0].1
}

/// A quantized INT4 type with the sidecar sources `scale` states.
fn quantized(storage: DataType, scale: QuantizationScale) -> DataType {
    DataType::Quantized {
        storage: Box::new(storage),
        scale,
        zero_point: QuantizationZeroPoint::Absent,
    }
}

#[test]
fn a_graph_reading_a_quantized_value_carries_its_dequantization_error() {
    let quantized_budget = output_budget(&reading_graph(&[quantized(
        DataType::I4,
        QuantizationScale::PerTensor,
    )]));
    let exact_budget = output_budget(&reading_graph(&[DataType::I4]));

    assert_eq!(quantized_budget.storage, ScalarFormat::F32);
    assert_eq!(
        exact_budget.storage,
        ScalarFormat::F32,
        "both graphs write binary32, so both are priced over it"
    );
    let quantized_error = quantized_budget.relative_error().expect("a readable bound");
    let exact_error = exact_budget.relative_error().expect("a readable bound");
    assert!(
        quantized_error > exact_error,
        "the same program reading quantized data carries a wider bound: {quantized_error} against {exact_error}"
    );
    assert!(
        quantized_error >= packed_i4().dequantization_measure().magnitude(),
        "the graph budget is at least the step the value was quantized to"
    );
}

#[test]
fn one_region_reading_two_packings_is_refused() {
    let same = reading_graph(&[
        quantized(DataType::I4, QuantizationScale::PerTensor),
        quantized(DataType::I4, QuantizationScale::PerTensor),
    ]);
    LogicalProgramGraph::validate(&same, &BTreeMap::new())
        .expect("one packing read twice is one lane law");

    let mixed = reading_graph(&[
        quantized(DataType::I4, QuantizationScale::PerTensor),
        quantized(DataType::I8, QuantizationScale::PerTensor),
    ]);
    let refusal = LogicalProgramGraph::validate(&mixed, &BTreeMap::new())
        .expect_err("two packings in one region cannot share a lane law");
    assert!(
        refusal.to_string().contains("not the same packing"),
        "the refusal names the disagreement: {refusal}"
    );
}

/// A graph whose one region reads `input` and writes `output`.
fn converting_graph(input: &DataType, output: &DataType) -> ProgramGraph {
    let count = 64;
    let written = vector(
        count,
        output.clone(),
        BufferAccess::WriteOnly,
        ValueLifetime::Output,
    );
    let read = vector(
        count,
        input.clone(),
        BufferAccess::ReadOnly,
        ValueLifetime::Invocation,
    );
    let mut graph = ProgramGraph::new();
    let value = graph
        .add_external_value("input", read.clone())
        .expect("fixture external value must be valid");
    graph
        .add_node(
            "convert",
            Program::wrapped(
                vec![
                    BufferDecl::read("input", 0, input.clone()).with_count(count),
                    BufferDecl::output("output", 1, output.clone()).with_count(count),
                ],
                [64, 1, 1],
                vec![Node::store(
                    "output",
                    Expr::gid_x(),
                    Expr::load("input", Expr::gid_x()),
                )],
            ),
            vec![GraphInput {
                buffer: "input".into(),
                value,
                contract: read,
            }],
            vec![GraphOutput {
                buffer: "output".into(),
                name: "output".into(),
                contract: written,
                retained_successor_of: None,
            }],
        )
        .expect("fixture node must be valid");
    graph
}

/// WHY: a region that writes its value onto a different grid quantizes it a
/// second time, and the caller checking a stated ceiling has to see that step.
/// Pricing only the read reported a requantizing chain as costing what a
/// dequantizing one costs, so a ceiling that admitted the read admitted the
/// round trip too. A conversion that only moves fields inside their container
/// changes no value and stays free, which is what keeps the term a measurement
/// rather than a penalty for touching quantized data.
#[test]
fn a_region_that_writes_a_different_grid_prices_the_second_quantization() {
    let four_bit = quantized(DataType::I4, QuantizationScale::PerTensor);
    let byte = quantized(DataType::I8, QuantizationScale::PerTensor);

    let held = output_budget(&converting_graph(&byte, &byte));
    let requantized = output_budget(&converting_graph(&byte, &four_bit));
    let held_error = held.relative_error().expect("a readable bound");
    let requantized_error = requantized.relative_error().expect("a readable bound");
    assert!(
        requantized_error > held_error,
        "placing the value on a coarser grid is a second quantization: {requantized_error} against {held_error}"
    );
    assert!(
        requantized_error
            >= QuantizedContract::of(&four_bit)
                .expect("a quantized type states a contract")
                .dequantization_measure()
                .magnitude(),
        "the bound is at least the step of the grid the value was written to"
    );

    let quantizing = output_budget(&converting_graph(&DataType::F32, &four_bit))
        .relative_error()
        .expect("a readable bound");
    assert!(
        quantizing
            >= QuantizedContract::of(&four_bit)
                .expect("a quantized type states a contract")
                .dequantization_measure()
                .magnitude(),
        "placing an unquantized value on a grid for the first time costs that grid's step"
    );
}
