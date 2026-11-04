//! What a quantized value is, stated once.
//!
//! A quantized buffer is bytes plus a law for reading them: which storage
//! family the fields belong to, what logical value they stand for, whether the
//! codes are signed, where the scale and zero point come from, how many logical
//! elements share one, the order fields occupy inside a container word, what
//! alignment the layout assumes, and what the arithmetic saturates to, rounds
//! under and accumulates in. A consumer that restates any of those reinterprets
//! the same bytes, so the contract states all of them together and refuses a
//! combination that cannot describe a buffer.
//!
//! The contract is also what prices a quantized region. [`ScalarFormat::of`]
//! answers the storage format of a quantized value, and an integer storage
//! format is exact, so a region read through storage alone contributes no error
//! to a graph budget. [`QuantizedContract::numeric`] states the contract of the
//! logical value instead, which carries the dequantization step as a relative
//! measure and admits reordering inside the declared budget.

mod calibration;
mod packing;

use serde::{Deserialize, Serialize};
use vyre_spec::{
    DataType, NumericFormat, OverflowBehavior, QuantizationScale, QuantizationZeroPoint,
    RoundingMode, SaturationBehavior, NF4_QUANTILE_TABLE,
};

use super::contract::{ErrorMeasure, NumericContract, Reassociation};
use super::format::ScalarFormat;

pub use calibration::{CalibrationIdentity, CALIBRATION_IDENTITY_VERSION};
pub use packing::{FieldTarget, PackedField};

/// Version of the quantized contract shape.
///
/// A recorded contract is read back by the schedule and the artifact that
/// consumed it, so the shape carries its own version.
pub const QUANTIZED_CONTRACT_VERSION: u32 = 1;

/// One axis a scale or zero point is shared along.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Deserialize, Serialize)]
pub struct GroupAxis {
    /// The tensor axis the group runs along.
    pub axis: u32,
    /// Logical elements per group along that axis.
    pub extent: u32,
}

/// The order packed fields occupy inside one container word.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Deserialize, Serialize)]
pub enum PackingOrder {
    /// Element zero occupies the least significant field of the word.
    LowFieldFirst,
    /// Element zero occupies the most significant field of the word.
    HighFieldFirst,
}

/// Why a quantized contract cannot describe a buffer.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Deserialize, Serialize)]
pub enum QuantizedRefusal {
    /// The storage format is not one a quantized value is packed in.
    StorageNotPacked {
        /// The stated storage format.
        storage: ScalarFormat,
    },
    /// The logical format is no wider than the storage it stands for, so
    /// nothing is being dequantized.
    LogicalNotWider {
        /// The stated storage format.
        storage: ScalarFormat,
        /// The stated logical format.
        logical: ScalarFormat,
    },
    /// The stated signedness is not the signedness of the storage family.
    SignednessDisagrees {
        /// The stated storage format.
        storage: ScalarFormat,
        /// What the contract states.
        stated: bool,
    },
    /// The container is not an unsigned integer, so a field shifted out of it
    /// would carry a sign.
    ContainerNotUnsigned {
        /// The stated container format.
        container: ScalarFormat,
    },
    /// The container does not hold a whole number of fields.
    ContainerNotWhole {
        /// The stated storage format.
        storage: ScalarFormat,
        /// The stated container format.
        container: ScalarFormat,
    },
    /// A group states no elements, so no scale covers the axis.
    GroupExtentZero {
        /// The axis that states a zero extent.
        axis: u32,
    },
    /// Two groups state the same axis, so an element sits in two groups.
    GroupAxisRepeated {
        /// The repeated axis.
        axis: u32,
    },
    /// The grouping axes and the scale source describe different groups.
    GroupingDisagreesWithScale {
        /// Logical elements the grouping axes cover.
        grouped_elements: u64,
        /// What the scale source states.
        scale: String,
    },
    /// The zero point is shared over a different grouping than the scale, so
    /// one element's scale and zero point come from different groups.
    ZeroPointDisagreesWithScale {
        /// What the scale source states.
        scale: String,
        /// What the zero-point source states.
        zero_point: String,
    },
    /// The alignment does not hold a whole number of container words.
    AlignmentNotContainerMultiple {
        /// The stated alignment.
        alignment_bytes: u32,
        /// Bytes one container word occupies.
        container_bytes: u32,
    },
    /// The accumulator is narrower than the logical value it accumulates.
    AccumulatorNarrowerThanLogical {
        /// The stated accumulator format.
        accumulator: ScalarFormat,
        /// The stated logical format.
        logical: ScalarFormat,
    },
    /// The rounding is not the rounding the storage grid has.
    RoundingDisagrees {
        /// The stated storage format.
        storage: ScalarFormat,
        /// What the contract states.
        stated: RoundingMode,
    },
    /// The saturation is not one the storage grid has.
    SaturationDisagrees {
        /// The stated storage format.
        storage: ScalarFormat,
        /// What the contract states.
        stated: SaturationBehavior,
    },
    /// Two quantized values meeting in one region are packed differently, so
    /// one lane law cannot read both.
    LayoutsDisagree {
        /// What the first value states.
        first: String,
        /// What the second value states.
        second: String,
    },
}

impl core::fmt::Display for QuantizedRefusal {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::StorageNotPacked { storage } => {
                write!(formatter, "{storage} is not a quantized storage format")
            }
            Self::LogicalNotWider { storage, logical } => write!(
                formatter,
                "logical {logical} is no wider than storage {storage}"
            ),
            Self::SignednessDisagrees { storage, stated } => write!(
                formatter,
                "contract states signed={stated} over {storage} codes"
            ),
            Self::ContainerNotUnsigned { container } => {
                write!(formatter, "container {container} is not an unsigned integer")
            }
            Self::ContainerNotWhole { storage, container } => write!(
                formatter,
                "container {container} does not hold a whole number of {storage} fields"
            ),
            Self::GroupExtentZero { axis } => {
                write!(formatter, "axis {axis} states a group extent of zero")
            }
            Self::GroupAxisRepeated { axis } => {
                write!(formatter, "axis {axis} is grouped twice")
            }
            Self::GroupingDisagreesWithScale {
                grouped_elements,
                scale,
            } => write!(
                formatter,
                "grouping covers {grouped_elements} element(s) and the scale is {scale}"
            ),
            Self::ZeroPointDisagreesWithScale { scale, zero_point } => write!(
                formatter,
                "the scale is {scale} and the zero point is {zero_point}"
            ),
            Self::AlignmentNotContainerMultiple {
                alignment_bytes,
                container_bytes,
            } => write!(
                formatter,
                "alignment {alignment_bytes} is not a multiple of the {container_bytes}-byte container"
            ),
            Self::AccumulatorNarrowerThanLogical {
                accumulator,
                logical,
            } => write!(
                formatter,
                "accumulator {accumulator} is narrower than logical {logical}"
            ),
            Self::RoundingDisagrees { storage, stated } => {
                write!(formatter, "{storage} is not quantized under {stated:?}")
            }
            Self::SaturationDisagrees { storage, stated } => {
                write!(formatter, "{storage} does not saturate as {stated:?}")
            }
            Self::LayoutsDisagree { first, second } => write!(
                formatter,
                "one region reads {first} and {second}, which are not the same packing"
            ),
        }
    }
}

impl core::error::Error for QuantizedRefusal {}

/// What an explicit conversion between two quantized layouts does.
///
/// A conversion is a cost the search pays, not a reinterpretation it performs
/// for free. Each fact it changes is one pass over the value.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Deserialize, Serialize)]
pub struct QuantizedConversion {
    /// Whether fields move inside their container word.
    pub repacks: bool,
    /// Whether the scale or zero point is read from a different source.
    pub rescales: bool,
    /// Whether the value is placed on a different grid.
    pub requantizes: bool,
}

impl QuantizedConversion {
    /// Whether the two layouts are the same, so nothing is converted.
    #[must_use]
    pub const fn is_free(&self) -> bool {
        !self.repacks && !self.rescales && !self.requantizes
    }

    /// Passes over one element the conversion costs.
    #[must_use]
    pub const fn steps(&self) -> u32 {
        self.repacks as u32 + self.rescales as u32 + self.requantizes as u32
    }

    /// The error the conversion adds to a value read as `target`.
    ///
    /// Moving fields and reading a different sidecar do not change the value.
    /// Placing it on another grid quantizes it again, which costs what that
    /// grid's step costs.
    #[must_use]
    pub fn measure(&self, target: &QuantizedContract) -> ErrorMeasure {
        if self.requantizes {
            target.dequantization_measure()
        } else {
            ErrorMeasure::Exact
        }
    }
}

/// What one quantized value is, and what reading it costs.
///
/// Every fact a consumer needs to turn bytes into numbers is stated here, so a
/// second consumer reads the same bytes the same way or states a different
/// contract and is refused where the two meet.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Deserialize, Serialize)]
pub struct QuantizedContract {
    /// Version of the contract shape.
    pub version: u32,
    /// The packed element the buffer holds.
    pub storage: ScalarFormat,
    /// The value one packed element stands for once it is dequantized.
    pub logical: ScalarFormat,
    /// Whether a stored code carries a sign.
    pub signed: bool,
    /// Where the scale that dequantizes a code comes from.
    pub scale: QuantizationScale,
    /// Where the zero point that centers a code comes from.
    pub zero_point: QuantizationZeroPoint,
    /// The axes a group runs along, and how many elements share one scale.
    pub grouping: Vec<GroupAxis>,
    /// The order packed fields occupy inside a container word.
    pub packing: PackingOrder,
    /// The word packed fields are read out of.
    pub container: ScalarFormat,
    /// Bytes a group's first container word is aligned to.
    pub alignment_bytes: u32,
    /// What happens to a value that leaves the storage grid.
    pub saturation: SaturationBehavior,
    /// How a value is placed on the storage grid.
    pub rounding: RoundingMode,
    /// The format a dequantized product accumulates in.
    pub accumulator: ScalarFormat,
    /// Which calibration produced the scales, when one is recorded.
    pub calibration: Option<CalibrationIdentity>,
}

impl QuantizedContract {
    /// A symmetric per-tensor contract over `storage`, read as `logical` out of
    /// `container` words.
    ///
    /// Every remaining fact takes the value the storage grid has: the
    /// signedness of its family, its own rounding, saturation to its finite
    /// range, one field per storage width with element zero low, alignment of
    /// one container word, and accumulation in the logical format.
    #[must_use]
    pub fn symmetric(
        storage: ScalarFormat,
        logical: ScalarFormat,
        container: ScalarFormat,
    ) -> Self {
        Self {
            version: QUANTIZED_CONTRACT_VERSION,
            storage,
            logical,
            signed: storage_is_signed(storage),
            scale: QuantizationScale::PerTensor,
            zero_point: QuantizationZeroPoint::Absent,
            grouping: Vec::new(),
            packing: PackingOrder::LowFieldFirst,
            container,
            alignment_bytes: container.bit_width() / 8,
            saturation: SaturationBehavior::ClampsToFiniteRange,
            rounding: grid_rounding(storage),
            accumulator: logical,
            calibration: None,
        }
    }

    /// The same contract with its scale shared over `axes`.
    ///
    /// The scale source becomes one group of the product of the extents, which
    /// is what the check compares the axes against.
    #[must_use]
    pub fn grouped_by(mut self, axes: Vec<GroupAxis>) -> Self {
        let group_size = axes
            .iter()
            .try_fold(1u32, |product, axis| product.checked_mul(axis.extent))
            .unwrap_or(u32::MAX);
        self.scale = QuantizationScale::PerGroup { group_size };
        self.grouping = axes;
        self
    }

    /// The same contract with an affine zero point from the same source as the
    /// scale.
    #[must_use]
    pub fn affine(mut self) -> Self {
        self.zero_point = match &self.scale {
            QuantizationScale::PerTensor => QuantizationZeroPoint::PerTensor,
            QuantizationScale::PerChannel { axis } => {
                QuantizationZeroPoint::PerChannel { axis: *axis }
            }
            QuantizationScale::PerGroup { group_size } => QuantizationZeroPoint::PerGroup {
                group_size: *group_size,
            },
        };
        self
    }

    /// The same contract with its scale shared along `axis`.
    #[must_use]
    pub fn per_channel(mut self, axis: u32) -> Self {
        self.scale = QuantizationScale::PerChannel { axis };
        self.grouping = Vec::new();
        self
    }

    /// The same contract packing fields in `packing` order.
    #[must_use]
    pub fn packed(mut self, packing: PackingOrder) -> Self {
        self.packing = packing;
        self
    }

    /// The same contract aligning a group to `alignment_bytes`.
    #[must_use]
    pub fn aligned_to(mut self, alignment_bytes: u32) -> Self {
        self.alignment_bytes = alignment_bytes;
        self
    }

    /// The same contract accumulating in `accumulator`.
    #[must_use]
    pub fn accumulating_in(mut self, accumulator: ScalarFormat) -> Self {
        self.accumulator = accumulator;
        self
    }

    /// The same contract produced under `calibration`.
    #[must_use]
    pub fn calibrated_by(mut self, calibration: CalibrationIdentity) -> Self {
        self.calibration = Some(calibration);
        self
    }

    /// The contract a [`DataType::Quantized`] states.
    ///
    /// The frozen data type carries the storage family and the two sidecar
    /// sources. Everything else is what a producer that states nothing further
    /// implies: the value is read as binary32 out of 32-bit containers with
    /// element zero in the low field, a contiguous group runs along the fastest
    /// axis, and the grid keeps its own rounding. A producer that packs
    /// differently states its own contract, and the two are refused where they
    /// meet rather than silently reinterpreted.
    #[must_use]
    pub fn of(dtype: &DataType) -> Option<Self> {
        match dtype {
            DataType::Quantized {
                storage,
                scale,
                zero_point,
            } => {
                let format = ScalarFormat::of(storage)?;
                let mut contract = Self::symmetric(format, ScalarFormat::F32, ScalarFormat::U32);
                contract.grouping = match scale {
                    QuantizationScale::PerGroup { group_size } => vec![GroupAxis {
                        axis: 0,
                        extent: *group_size,
                    }],
                    QuantizationScale::PerTensor | QuantizationScale::PerChannel { .. } => {
                        Vec::new()
                    }
                };
                contract.scale = scale.clone();
                contract.zero_point = zero_point.clone();
                Some(contract)
            }
            DataType::Vec { element, .. }
            | DataType::TensorShaped { element, .. }
            | DataType::SparseCsr { element }
            | DataType::SparseCoo { element }
            | DataType::SparseBsr { element, .. } => Self::of(element),
            _ => None,
        }
    }

    /// Whether a fused chain may keep this packed layout where `other` is read.
    ///
    /// Two contracts propagate when their bytes mean the same thing: the same
    /// storage family, signedness, container, packing order, grouping, sidecar
    /// sources, alignment and calibration. A wider logical or accumulator
    /// format is not a reinterpretation, so it does not stop propagation.
    #[must_use]
    pub fn propagates_to(&self, other: &Self) -> bool {
        self.storage == other.storage
            && self.signed == other.signed
            && self.container == other.container
            && self.packing == other.packing
            && self.grouping == other.grouping
            && self.scale == other.scale
            && self.zero_point == other.zero_point
            && self.alignment_bytes == other.alignment_bytes
            && self.calibration == other.calibration
    }

    /// What converting a value under this contract to `target` costs.
    ///
    /// A conversion the search inserts is explicit and priced: fields move
    /// inside their container when the packing or the container changes, the
    /// sidecars are read again when the scale source moves, and the value is
    /// placed on a new grid when the storage family changes. Two contracts that
    /// propagate convert for nothing.
    #[must_use]
    pub fn conversion(&self, target: &Self) -> QuantizedConversion {
        QuantizedConversion {
            repacks: self.packing != target.packing
                || self.container != target.container
                || self.alignment_bytes != target.alignment_bytes,
            rescales: self.scale != target.scale
                || self.zero_point != target.zero_point
                || self.grouping != target.grouping
                || self.calibration != target.calibration,
            requantizes: self.storage != target.storage || self.signed != target.signed,
        }
    }

    /// Whether the contract describes a buffer that can be read.
    ///
    /// # Errors
    ///
    /// Returns the first fact that cannot hold, so a layout that no producer
    /// can write is refused where it is stated rather than where its bytes are
    /// misread.
    pub fn check(&self) -> Result<(), QuantizedRefusal> {
        if !self.storage.data_type().is_quantized_storage() {
            return Err(QuantizedRefusal::StorageNotPacked {
                storage: self.storage,
            });
        }
        if self.logical.bit_width() <= self.storage.bit_width() {
            return Err(QuantizedRefusal::LogicalNotWider {
                storage: self.storage,
                logical: self.logical,
            });
        }
        if self.signed != storage_is_signed(self.storage) {
            return Err(QuantizedRefusal::SignednessDisagrees {
                storage: self.storage,
                stated: self.signed,
            });
        }
        if self.container.semantics().format != NumericFormat::UnsignedInteger {
            return Err(QuantizedRefusal::ContainerNotUnsigned {
                container: self.container,
            });
        }
        let container_bits = self.container.bit_width();
        let storage_bits = self.storage.bit_width();
        if container_bits < storage_bits || container_bits % storage_bits != 0 {
            return Err(QuantizedRefusal::ContainerNotWhole {
                storage: self.storage,
                container: self.container,
            });
        }
        let container_bytes = container_bits / 8;
        if self.alignment_bytes == 0 || self.alignment_bytes % container_bytes != 0 {
            return Err(QuantizedRefusal::AlignmentNotContainerMultiple {
                alignment_bytes: self.alignment_bytes,
                container_bytes,
            });
        }
        if self.accumulator.bit_width() < self.logical.bit_width() {
            return Err(QuantizedRefusal::AccumulatorNarrowerThanLogical {
                accumulator: self.accumulator,
                logical: self.logical,
            });
        }
        self.check_grouping()?;
        self.check_grid()
    }

    /// Whether the grouping axes, the scale and the zero point agree.
    fn check_grouping(&self) -> Result<(), QuantizedRefusal> {
        let mut seen: Vec<u32> = Vec::with_capacity(self.grouping.len());
        for axis in &self.grouping {
            if axis.extent == 0 {
                return Err(QuantizedRefusal::GroupExtentZero { axis: axis.axis });
            }
            if seen.contains(&axis.axis) {
                return Err(QuantizedRefusal::GroupAxisRepeated { axis: axis.axis });
            }
            seen.push(axis.axis);
        }
        let grouped_elements = self.group_elements();
        match &self.scale {
            QuantizationScale::PerTensor | QuantizationScale::PerChannel { .. } => {
                if !self.grouping.is_empty() {
                    return Err(QuantizedRefusal::GroupingDisagreesWithScale {
                        grouped_elements,
                        scale: scale_name(&self.scale),
                    });
                }
            }
            QuantizationScale::PerGroup { group_size } => {
                if self.grouping.is_empty() || grouped_elements != u64::from(*group_size) {
                    return Err(QuantizedRefusal::GroupingDisagreesWithScale {
                        grouped_elements,
                        scale: scale_name(&self.scale),
                    });
                }
            }
        }
        let agrees = match (&self.scale, &self.zero_point) {
            (_, QuantizationZeroPoint::Absent | QuantizationZeroPoint::PerTensor) => true,
            (
                QuantizationScale::PerChannel { axis },
                QuantizationZeroPoint::PerChannel { axis: shared },
            ) => axis == shared,
            (
                QuantizationScale::PerGroup { group_size },
                QuantizationZeroPoint::PerGroup { group_size: shared },
            ) => group_size == shared,
            _ => false,
        };
        if agrees {
            Ok(())
        } else {
            Err(QuantizedRefusal::ZeroPointDisagreesWithScale {
                scale: scale_name(&self.scale),
                zero_point: zero_point_name(&self.zero_point),
            })
        }
    }

    /// Whether the rounding and saturation are the ones the storage grid has.
    fn check_grid(&self) -> Result<(), QuantizedRefusal> {
        let codebook = self.storage.semantics().format == NumericFormat::NormalFloat;
        if self.rounding == RoundingMode::Exact
            || codebook != (self.rounding == RoundingMode::NearestQuantile)
        {
            return Err(QuantizedRefusal::RoundingDisagrees {
                storage: self.storage,
                stated: self.rounding,
            });
        }
        if self.saturation == SaturationBehavior::ClampsToUnitInterval && !codebook {
            return Err(QuantizedRefusal::SaturationDisagrees {
                storage: self.storage,
                stated: self.saturation,
            });
        }
        Ok(())
    }

    /// Logical elements that share one scale.
    ///
    /// An ungrouped contract shares one scale over the whole buffer or one per
    /// channel, and answers one element per group so a tail block is compared
    /// against the same unit either way.
    #[must_use]
    pub fn group_elements(&self) -> u64 {
        self.grouping
            .iter()
            .map(|axis| u64::from(axis.extent))
            .product::<u64>()
            .max(1)
    }

    /// Groups an axis of `extent` elements needs, including a short tail block.
    #[must_use]
    pub fn groups_over(&self, extent: u64) -> u64 {
        extent.div_ceil(self.group_elements())
    }

    /// Elements the last group of an axis of `extent` elements covers.
    ///
    /// A group that divides the axis has a full last group; one that does not
    /// has a short tail whose scale still covers only the elements present.
    #[must_use]
    pub fn tail_elements(&self, extent: u64) -> u64 {
        let group = self.group_elements();
        match extent % group {
            0 if extent == 0 => 0,
            0 => group,
            remainder => remainder,
        }
    }

    /// One quantization step as a fraction of the largest magnitude the grid
    /// represents.
    ///
    /// A signed integer grid spends one code on the sign, an unsigned grid
    /// spends none, a float grid steps by its own unit in the last place, and a
    /// codebook steps by its widest gap.
    #[must_use]
    pub fn step_fraction(&self) -> f64 {
        let bits = self.storage.bit_width();
        match self.storage.semantics().format {
            NumericFormat::SignedInteger => {
                let codes = f64::from(1u32 << bits.saturating_sub(1)) - 1.0;
                1.0 / codes
            }
            NumericFormat::UnsignedInteger => {
                let codes = 2.0_f64.powi(i32::try_from(bits).unwrap_or(i32::MAX)) - 1.0;
                1.0 / codes
            }
            NumericFormat::NormalFloat => codebook_step(),
            _ => self.storage.ulp_fraction().unwrap_or(1.0),
        }
    }

    /// How far a dequantized value may sit from the value it was quantized
    /// from, as a fraction of the group's largest magnitude.
    ///
    /// Nearest rounding lands within half a step of the grid and truncation
    /// within a whole one.
    #[must_use]
    pub fn dequantization_measure(&self) -> ErrorMeasure {
        let step = self.step_fraction();
        let fraction = match self.rounding {
            RoundingMode::Exact => 0.0,
            RoundingMode::RoundToNearestEven | RoundingMode::NearestQuantile => step / 2.0,
            RoundingMode::TruncateTowardsZero => step,
        };
        if fraction == 0.0 {
            ErrorMeasure::Exact
        } else {
            ErrorMeasure::relative(fraction)
        }
    }

    /// The numeric contract of the dequantized value.
    ///
    /// This is what a region holding a quantized value states to the graph
    /// budget. The storage format is exact, so a schedule that priced the value
    /// through storage alone would read a quantized region as contributing no
    /// error; the logical contract carries the dequantization step instead.
    /// Reordering is admitted inside the budget rather than forbidden, because
    /// the order a dequantized sum lands in moves the result by less than the
    /// step the value was already quantized to, and the graph composition
    /// proves that rather than assuming it.
    #[must_use]
    pub fn numeric(&self) -> NumericContract {
        let mut contract = NumericContract::of(self.logical)
            .with_measure(self.dequantization_measure())
            .accumulating_in(self.accumulator);
        contract.overflow = match self.saturation {
            SaturationBehavior::None => contract.overflow,
            SaturationBehavior::ClampsToFiniteRange | SaturationBehavior::ClampsToUnitInterval => {
                OverflowBehavior::SaturateToFiniteRange
            }
        };
        if !self.logical.is_exact() {
            contract.reassociation = Reassociation::WithinBudget;
        }
        contract
    }
}

impl core::fmt::Display for QuantizedContract {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let order = match self.packing {
            PackingOrder::LowFieldFirst => "low field first",
            PackingOrder::HighFieldFirst => "high field first",
        };
        write!(
            formatter,
            "{} as {} packed {} per {}, {order}, scale {}",
            self.storage,
            self.logical,
            self.fields_per_container(),
            self.container,
            scale_name(&self.scale)
        )
    }
}

/// Whether a stored code in `storage` carries a sign.
fn storage_is_signed(storage: ScalarFormat) -> bool {
    storage.semantics().format != NumericFormat::UnsignedInteger
}

/// The rounding the storage grid places a value under.
fn grid_rounding(storage: ScalarFormat) -> RoundingMode {
    if storage.semantics().format == NumericFormat::NormalFloat {
        RoundingMode::NearestQuantile
    } else {
        RoundingMode::RoundToNearestEven
    }
}

/// The widest gap in the normal-float codebook, as a fraction of its peak.
fn codebook_step() -> f64 {
    let mut previous = f64::from(NF4_QUANTILE_TABLE[0]);
    let mut widest = 0.0_f64;
    let mut peak = previous.abs();
    for quantile in NF4_QUANTILE_TABLE.iter().skip(1) {
        let value = f64::from(*quantile);
        widest = widest.max((value - previous).abs());
        peak = peak.max(value.abs());
        previous = value;
    }
    if peak == 0.0 {
        1.0
    } else {
        widest / peak
    }
}

/// What a scale source states, as prose an error can carry.
fn scale_name(scale: &QuantizationScale) -> String {
    match scale {
        QuantizationScale::PerTensor => "per tensor".to_string(),
        QuantizationScale::PerChannel { axis } => format!("per channel on axis {axis}"),
        QuantizationScale::PerGroup { group_size } => format!("per group of {group_size}"),
    }
}

/// What a zero-point source states, as prose an error can carry.
fn zero_point_name(zero_point: &QuantizationZeroPoint) -> String {
    match zero_point {
        QuantizationZeroPoint::Absent => "absent".to_string(),
        QuantizationZeroPoint::PerTensor => "per tensor".to_string(),
        QuantizationZeroPoint::PerChannel { axis } => format!("per channel on axis {axis}"),
        QuantizationZeroPoint::PerGroup { group_size } => format!("per group of {group_size}"),
    }
}
