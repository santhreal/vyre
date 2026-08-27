//! One fixture per `DataType` variant, so a tag table can be tested against
//! the whole enum instead of against the variants its author remembered.
//!
//! # Why this exists in a shared crate
//!
//! `DataType` is the identity of a buffer element, and several tables key off
//! it: the VIR0 wire tag, the driver's specialization cache key, and the AOT
//! artifact name. When each table carried its own hand-written fixture list,
//! each list was complete on the day it was written and silently partial after
//! that, and a missing variant does not read as missing. It reads as a variant
//! that shares a cache key with every other variant nobody named, which serves
//! one element type's compiled shader for another.
//!
//! # How a new variant fails closed
//!
//! The member set is not written here. [`declared_data_type_variants`] reads
//! the `pub enum DataType` declaration in `vyre-spec` at run time and returns
//! the variant names it finds, and [`assert_covers_every_data_type_variant`]
//! holds the fixtures to it. Adding a variant to the spec turns every suite
//! built on these fixtures RED until a fixture exists for it, and writing that
//! fixture forces a decision about each table above.
//!
//! The enumeration reads source as TEXT and never compiles it, so it reports
//! the same variant set whichever features the runner selects. Its failure mode
//! is finding nothing, which is why [`assert_covers_every_data_type_variant`]
//! refuses a variant set smaller than [`DECLARED_VARIANT_FLOOR`].

use std::collections::BTreeSet;

use vyre_spec::DataType;

use crate::monorepo::vyre_workspace_root;

/// Fewest `DataType` variants a working source enumeration can find.
///
/// A scan that matched nothing would report a trivially covered empty set. The
/// floor is well below the current count, so it catches a broken scan without
/// needing an edit every time the spec grows.
pub const DECLARED_VARIANT_FLOOR: usize = 25;

/// The variant names `vyre-spec` declares for `DataType`, read from source.
///
/// # Panics
///
/// Panics when the declaration cannot be located or parsed, which is a broken
/// enumeration rather than an empty enum.
#[must_use]
pub fn declared_data_type_variants() -> BTreeSet<String> {
    let path = vyre_workspace_root().join("vyre-spec/src/data_type/mod.rs");
    let source = crate::read_source_file_bounded(&path).unwrap_or_else(|err| {
        panic!("Fix: cannot read the DataType declaration at {path:?}: {err}")
    });
    let body = crate::braced_body(&source, "pub enum DataType {").unwrap_or_else(|| {
        panic!("Fix: no `pub enum DataType` declaration in {path:?}; update this enumeration")
    });
    crate::top_level_variant_names(body)
}

/// The flat `DataType` forms a buffer declaration can carry as its element.
///
/// The flat leaves come from [`crate::data_type_elements`], so the two tables
/// cannot disagree about which flat forms exist. Parameterised variants get the
/// smallest well-formed payload: the tables under test key off the outer
/// discriminant, and a fixture that varied the payload would test the payload
/// rather than the table.
#[must_use]
pub fn data_type_variant_samples() -> Vec<DataType> {
    let mut samples = crate::data_type_elements::flat_buffer_element_types(1);
    samples.extend([
        DataType::Handle(vyre_spec::TypeId(0)),
        DataType::Vec {
            element: Box::new(DataType::U32),
            count: 1,
        },
        DataType::TensorShaped {
            element: Box::new(DataType::U32),
            shape: smallvec::smallvec![1],
        },
        DataType::SparseCsr {
            element: Box::new(DataType::U32),
        },
        DataType::SparseCoo {
            element: Box::new(DataType::U32),
        },
        DataType::SparseBsr {
            element: Box::new(DataType::U32),
            block_rows: 1,
            block_cols: 1,
        },
        DataType::F8E4M3,
        DataType::F8E5M2,
        DataType::I4,
        DataType::FP4,
        DataType::NF4,
        DataType::DeviceMesh {
            axes: smallvec::smallvec![1],
        },
        DataType::Quantized {
            storage: Box::new(DataType::I8),
            scale: vyre_spec::QuantizationScale::PerTensor,
            zero_point: vyre_spec::QuantizationZeroPoint::Absent,
        },
        DataType::Opaque(vyre_spec::extension::ExtensionDataTypeId::from_name(
            "vyre.test_support.fixture_data_type",
        )),
    ]);
    samples
}

/// The fixtures name every variant the spec declares, exactly once.
///
/// # Panics
///
/// Panics naming the variants that have no fixture, or the fixtures that name
/// a variant the spec no longer declares.
pub fn assert_covers_every_data_type_variant(samples: &[DataType]) {
    let declared = declared_data_type_variants();
    assert!(
        declared.len() >= DECLARED_VARIANT_FLOOR,
        "Fix: the DataType source enumeration found only {} variants, below the floor of \
         {DECLARED_VARIANT_FLOOR}; the scan is broken, not the enum",
        declared.len()
    );

    let covered: BTreeSet<String> = samples.iter().map(variant_name).collect();

    let missing: Vec<&String> = declared.difference(&covered).collect();
    assert!(
        missing.is_empty(),
        "Fix: add a DataType fixture for each of {missing:?} in \
         vyre_test_support::data_type_variants::data_type_variant_samples; every tag table \
         keyed on DataType is untested for them until you do"
    );

    let unknown: Vec<&String> = covered.difference(&declared).collect();
    assert!(
        unknown.is_empty(),
        "Fix: these fixtures name DataType variants vyre-spec no longer declares: {unknown:?}"
    );
}

/// The declared variant name of `value`, from its `Debug` rendering.
///
/// `Debug` prints the variant name first for every shape a variant can have
/// (unit, tuple, struct), and it is derived, so it cannot drift from the
/// declaration the way a hand-written mapping would.
#[must_use]
pub fn variant_name(value: &DataType) -> String {
    let rendered = format!("{value:?}");
    let end = rendered
        .find(|c: char| !c.is_ascii_alphanumeric() && c != '_')
        .unwrap_or(rendered.len());
    rendered[..end].to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The fixtures cover the enum as declared today.
    #[test]
    fn the_fixture_set_covers_every_declared_data_type_variant() {
        assert_covers_every_data_type_variant(&data_type_variant_samples());
    }

    /// The source enumeration reads variant names, not doc text or payloads.
    ///
    /// Without this the coverage assertion above could pass on an enumeration
    /// that returned an empty set for a reason the floor does not catch.
    #[test]
    fn the_source_enumeration_reads_variants_out_of_a_declaration() {
        let source = "\
/// doc
#[non_exhaustive]
pub enum DataType {
    /// A unit variant.
    U8,
    // A line comment naming Nothing.
    Array {
        /// Payload field that must not be read as a variant.
        ElementSize: usize,
    },
    Handle(TypeId),
}
";
        let body =
            crate::braced_body(source, "pub enum DataType {").expect("the declaration is present");
        let names = crate::top_level_variant_names(body);
        assert_eq!(
            names,
            ["Array", "Handle", "U8"]
                .into_iter()
                .map(str::to_string)
                .collect::<BTreeSet<String>>()
        );
    }

    /// The variant name comes from `Debug`, for every payload shape.
    #[test]
    fn variant_names_cover_unit_tuple_and_struct_shapes() {
        assert_eq!(variant_name(&DataType::U8), "U8");
        assert_eq!(
            variant_name(&DataType::Handle(vyre_spec::TypeId(0))),
            "Handle"
        );
        assert_eq!(variant_name(&DataType::Array { element_size: 1 }), "Array");
    }
}
