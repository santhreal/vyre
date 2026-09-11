//! What one numeric result is allowed to be.

use serde::{Deserialize, Serialize};
use vyre_spec::{InfinityBehavior, NanBehavior, OverflowBehavior, RoundingMode, SubnormalBehavior};

use super::format::ScalarFormat;

/// Version of the numeric contract shape.
///
/// A recorded contract is read back by the schedule that consumed it, so the
/// shape carries its own version rather than borrowing the artifact's.
pub const NUMERIC_CONTRACT_VERSION: u32 = 1;

/// How far a computed value may sit from the exact result.
///
/// A measure is a ceiling, not an observation: it states what a caller may rely
/// on, and a schedule that cannot prove it stays under the ceiling is refused
/// rather than measured.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Deserialize, Serialize)]
pub enum ErrorMeasure {
    /// The exact result, bit for bit. No other value is admitted.
    Exact,
    /// Within `count` units in the last place of the declared storage format.
    Ulp {
        /// Admitted distance in units in the last place.
        count: u32,
    },
    /// Within an absolute distance of the exact result.
    Absolute {
        /// The distance as `f64` bits, so the measure stays hashable.
        bits: u64,
    },
    /// Within a fraction of the exact magnitude.
    Relative {
        /// The fraction as `f64` bits, so the measure stays hashable.
        bits: u64,
    },
}

impl ErrorMeasure {
    /// An absolute bound of `distance`.
    #[must_use]
    pub const fn absolute(distance: f64) -> Self {
        Self::Absolute {
            bits: distance.to_bits(),
        }
    }

    /// A relative bound of `fraction`.
    #[must_use]
    pub const fn relative(fraction: f64) -> Self {
        Self::Relative {
            bits: fraction.to_bits(),
        }
    }

    /// The bound as a number, or zero for the exact measure.
    ///
    /// A ULP count reads as its own count: converting it to a fraction needs
    /// the storage format, which a measure alone does not carry.
    #[must_use]
    pub fn magnitude(self) -> f64 {
        match self {
            Self::Exact => 0.0,
            Self::Ulp { count } => f64::from(count),
            Self::Absolute { bits } | Self::Relative { bits } => f64::from_bits(bits),
        }
    }

    /// Whether this measure admits no error at all.
    #[must_use]
    pub const fn is_exact(self) -> bool {
        matches!(self, Self::Exact)
    }
}

/// Whether a transform may change the order in which values are combined.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Deserialize, Serialize)]
pub enum Reassociation {
    /// The stated order is the contract. A reordered schedule is refused.
    Forbidden,
    /// Reordering is exact, so any order produces the same bits.
    Exact,
    /// Reordering is admitted while the composed error stays under the measure.
    WithinBudget,
}

/// Whether two runs of the same schedule on the same input agree bit for bit.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Deserialize, Serialize)]
pub enum Determinism {
    /// Two runs agree bit for bit.
    Deterministic,
    /// Two runs may disagree within the measure, because the combine order is
    /// decided by the device rather than the schedule.
    RunToRunVariable,
}

/// Whether the result depends on the order in which atomics land.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Deserialize, Serialize)]
pub enum AtomicOrderSensitivity {
    /// The combine is exact and order-free, so the landing order is invisible.
    Insensitive,
    /// The result depends on the order the atomics land in.
    Sensitive,
}

/// An approximate native instruction the lowering may select.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Deserialize, Serialize)]
pub enum Approximation {
    /// Only correctly rounded operations are admitted.
    Refused,
    /// A native approximate instruction is admitted and contributes `measure`.
    Native {
        /// What the approximation contributes to the composed error.
        measure: ErrorMeasure,
    },
}

/// Why a contract, a composition or a transform was refused.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Deserialize, Serialize)]
pub enum ContractRefusal {
    /// A field states behavior the storage format does not have.
    FormatDisagrees {
        /// The field that disagrees.
        field: &'static str,
        /// What the contract states.
        stated: String,
        /// What the format's semantics state.
        format: String,
    },
    /// An absolute bound met a relative or ULP bound with no magnitude bound to
    /// convert between them.
    UnboundedMagnitude {
        /// The measure that needed a magnitude.
        measure: ErrorMeasure,
    },
    /// Two contracts describe different storage formats.
    FormatMismatch {
        /// The format the first contract states.
        first: ScalarFormat,
        /// The format the second contract states.
        second: ScalarFormat,
    },
    /// The composed error exceeds the declared measure.
    BudgetExceeded {
        /// What the graph declared.
        declared: ErrorMeasure,
        /// What composition proved.
        composed: ErrorMeasure,
    },
    /// A transform needs a reassociation the contract does not admit.
    ReassociationRefused {
        /// What the contract admits.
        stated: Reassociation,
    },
    /// An approximate instruction was selected under a contract that refuses one.
    ApproximationRefused,
    /// A magnitude proof does not cover the choice it was offered for.
    RangeUnproven {
        /// What the choice needed proven.
        needed: String,
        /// What the range proves.
        proven: String,
    },
}

impl core::fmt::Display for ContractRefusal {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::FormatDisagrees {
                field,
                stated,
                format,
            } => write!(
                formatter,
                "the contract states {field} {stated} and the storage format states {format}"
            ),
            Self::UnboundedMagnitude { measure } => write!(
                formatter,
                "an absolute bound composed with {measure:?} needs a proven magnitude"
            ),
            Self::FormatMismatch { first, second } => write!(
                formatter,
                "one contract stores {first} and the next stores {second}"
            ),
            Self::BudgetExceeded { declared, composed } => write!(
                formatter,
                "composition proves {composed:?} and the graph declares {declared:?}"
            ),
            Self::ReassociationRefused { stated } => {
                write!(formatter, "the contract admits reassociation {stated:?}")
            }
            Self::ApproximationRefused => {
                write!(formatter, "the contract admits no approximate instruction")
            }
            Self::RangeUnproven { needed, proven } => write!(
                formatter,
                "the choice needs {needed} proven and range analysis proves {proven}"
            ),
        }
    }
}

impl core::error::Error for ContractRefusal {}

/// What a numeric operation or region is allowed to produce.
///
/// Storage, intermediate and accumulator formats are stated separately because
/// a region may hold a value in one format, compute in a second and accumulate
/// in a third: a reduction that stores `f16` and accumulates `f32` has a
/// different error than one that does neither, and the difference decides which
/// schedules are legal.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Deserialize, Serialize)]
pub struct NumericContract {
    /// How far the result may sit from the exact one.
    pub measure: ErrorMeasure,
    /// Whether a transform may reorder the combines.
    pub reassociation: Reassociation,
    /// The format a value is held in between regions.
    pub storage: ScalarFormat,
    /// The format one operation computes in.
    pub intermediate: ScalarFormat,
    /// The format a reduction accumulates in.
    pub accumulator: ScalarFormat,
    /// The rounding the result is produced under.
    pub rounding: RoundingMode,
    /// What happens when a value leaves the representable range.
    pub overflow: OverflowBehavior,
    /// What happens to a NaN.
    pub nan: NanBehavior,
    /// What happens to an infinity.
    pub infinity: InfinityBehavior,
    /// What happens to a subnormal.
    pub subnormal: SubnormalBehavior,
    /// Whether two runs agree bit for bit.
    pub determinism: Determinism,
    /// Whether the result depends on atomic landing order.
    pub atomic_order: AtomicOrderSensitivity,
    /// Whether an approximate native instruction is admitted.
    pub approximation: Approximation,
}

impl NumericContract {
    /// The exact contract over the fundamental device word.
    ///
    /// A registration states this when its result is the exact one, which is
    /// every integer operation and every operation whose output is compared
    /// byte for byte.
    pub const EXACT: Self = Self::exact_word();

    /// The exact contract over the fundamental device word.
    #[must_use]
    pub const fn exact_word() -> Self {
        Self {
            measure: ErrorMeasure::Exact,
            reassociation: Reassociation::Exact,
            storage: ScalarFormat::U32,
            intermediate: ScalarFormat::U32,
            accumulator: ScalarFormat::U32,
            rounding: RoundingMode::Exact,
            overflow: OverflowBehavior::WrapTwoComplement,
            nan: NanBehavior::NotApplicable,
            infinity: InfinityBehavior::Unsupported,
            subnormal: SubnormalBehavior::Unsupported,
            determinism: Determinism::Deterministic,
            atomic_order: AtomicOrderSensitivity::Insensitive,
            approximation: Approximation::Refused,
        }
    }

    /// The IEEE binary32 contract admitting `ulp` units in the last place.
    ///
    /// Reassociation is forbidden: binary32 addition is not associative, so a
    /// schedule that reorders it must widen the bound and say so rather than
    /// inherit one measured under the stated order.
    #[must_use]
    pub const fn ieee_f32(ulp: u32) -> Self {
        Self {
            measure: ErrorMeasure::Ulp { count: ulp },
            reassociation: Reassociation::Forbidden,
            storage: ScalarFormat::F32,
            intermediate: ScalarFormat::F32,
            accumulator: ScalarFormat::F32,
            rounding: RoundingMode::RoundToNearestEven,
            overflow: OverflowBehavior::SignedInfinity,
            nan: NanBehavior::CanonicalQuietNan,
            infinity: InfinityBehavior::SignedInfinity,
            subnormal: SubnormalBehavior::PreservedIEEE,
            determinism: Determinism::Deterministic,
            atomic_order: AtomicOrderSensitivity::Insensitive,
            approximation: Approximation::Refused,
        }
    }

    /// The contract an exact format states by itself.
    ///
    /// Every field is read from the format's own semantics, so an integer
    /// contract cannot state a rounding mode the format does not have.
    #[must_use]
    pub fn of(storage: ScalarFormat) -> Self {
        let semantics = storage.semantics();
        let exact = storage.is_exact();
        Self {
            measure: if exact {
                ErrorMeasure::Exact
            } else {
                ErrorMeasure::Ulp { count: 0 }
            },
            reassociation: if exact {
                Reassociation::Exact
            } else {
                Reassociation::Forbidden
            },
            intermediate: storage,
            accumulator: storage,
            storage,
            rounding: semantics.rounding,
            overflow: semantics.overflow,
            nan: semantics.nan,
            infinity: semantics.infinity,
            subnormal: semantics.subnormal,
            determinism: Determinism::Deterministic,
            atomic_order: AtomicOrderSensitivity::Insensitive,
            approximation: Approximation::Refused,
        }
    }

    /// The same contract admitting `count` units in the last place.
    #[must_use]
    pub const fn within_ulp(mut self, count: u32) -> Self {
        self.measure = ErrorMeasure::Ulp { count };
        self
    }

    /// The same contract under `measure`.
    #[must_use]
    pub const fn with_measure(mut self, measure: ErrorMeasure) -> Self {
        self.measure = measure;
        self
    }

    /// The same contract accumulating in `accumulator`.
    #[must_use]
    pub const fn accumulating_in(mut self, accumulator: ScalarFormat) -> Self {
        self.accumulator = accumulator;
        self
    }

    /// The same contract computing in `intermediate`.
    #[must_use]
    pub const fn computing_in(mut self, intermediate: ScalarFormat) -> Self {
        self.intermediate = intermediate;
        self
    }

    /// The same contract under `reassociation`.
    #[must_use]
    pub const fn reassociating(mut self, reassociation: Reassociation) -> Self {
        self.reassociation = reassociation;
        self
    }

    /// The same contract under `determinism`.
    #[must_use]
    pub const fn under(mut self, determinism: Determinism) -> Self {
        self.determinism = determinism;
        self
    }

    /// The same contract under `sensitivity` to atomic landing order.
    #[must_use]
    pub const fn sensitive_to(mut self, sensitivity: AtomicOrderSensitivity) -> Self {
        self.atomic_order = sensitivity;
        self
    }

    /// The same contract admitting `approximation`.
    #[must_use]
    pub const fn approximating(mut self, approximation: Approximation) -> Self {
        self.approximation = approximation;
        self
    }

    /// The same contract flushing subnormals to zero.
    #[must_use]
    pub const fn flushing_subnormals(mut self) -> Self {
        self.subnormal = SubnormalBehavior::FlushedToSignedZero;
        self
    }

    /// Whether the contract states behavior its formats have.
    ///
    /// # Errors
    ///
    /// Returns the first field whose stated behavior the storage format does
    /// not have, so a contract claiming exactness over a floating format or
    /// preserved subnormals over a format without them is refused where it is
    /// declared rather than where it is read.
    pub fn check(&self) -> Result<(), ContractRefusal> {
        let semantics = self.storage.semantics();
        if self.measure.is_exact() && !self.storage.is_exact() {
            return Err(ContractRefusal::FormatDisagrees {
                field: "measure",
                stated: "exact".into(),
                format: format!("{} arithmetic rounds", self.storage),
            });
        }
        if self.reassociation == Reassociation::Exact && !self.storage.is_exact() {
            return Err(ContractRefusal::FormatDisagrees {
                field: "reassociation",
                stated: "exact".into(),
                format: format!("{} arithmetic rounds", self.storage),
            });
        }
        if self.subnormal == SubnormalBehavior::PreservedIEEE
            && semantics.subnormal == SubnormalBehavior::Unsupported
        {
            return Err(ContractRefusal::FormatDisagrees {
                field: "subnormal",
                stated: "preserved".into(),
                format: format!("{} has no subnormals", self.storage),
            });
        }
        if self.infinity == InfinityBehavior::SignedInfinity
            && semantics.infinity == InfinityBehavior::Unsupported
        {
            return Err(ContractRefusal::FormatDisagrees {
                field: "infinity",
                stated: "signed infinity".into(),
                format: format!("{} has no infinity", self.storage),
            });
        }
        if matches!(self.approximation, Approximation::Native { measure } if measure.is_exact()) {
            return Err(ContractRefusal::FormatDisagrees {
                field: "approximation",
                stated: "exact native approximation".into(),
                format: "an approximate instruction contributes error".into(),
            });
        }
        Ok(())
    }

    /// The relative fraction one unit in the last place of `format` spans.
    ///
    /// An exact format has no such fraction: every value it holds is the value
    /// itself, so a ULP bound over it is a count of exact steps.
    #[must_use]
    pub fn ulp_fraction(format: ScalarFormat) -> Option<f64> {
        format.ulp_fraction()
    }

    /// This contract's measure as a relative fraction of the exact magnitude.
    ///
    /// # Errors
    ///
    /// Returns [`ContractRefusal::UnboundedMagnitude`] for an absolute bound,
    /// which cannot be read as a fraction without a proven magnitude.
    pub fn relative_error(&self) -> Result<f64, ContractRefusal> {
        match self.measure {
            ErrorMeasure::Exact => Ok(0.0),
            ErrorMeasure::Relative { bits } => Ok(f64::from_bits(bits)),
            ErrorMeasure::Ulp { count } => Ok(f64::from(count)
                * Self::ulp_fraction(self.storage).ok_or(ContractRefusal::UnboundedMagnitude {
                    measure: self.measure,
                })?),
            ErrorMeasure::Absolute { .. } => Err(ContractRefusal::UnboundedMagnitude {
                measure: self.measure,
            }),
        }
    }

    /// The comparison budget in units in the last place of the storage format.
    ///
    /// An absolute bound has no ULP reading without a proven magnitude, so it
    /// answers `None` rather than a number a buffer comparison would trust.
    #[must_use]
    pub fn ulp_budget(&self) -> Option<u32> {
        match self.measure {
            ErrorMeasure::Exact => Some(0),
            ErrorMeasure::Ulp { count } => Some(count),
            ErrorMeasure::Relative { bits } => {
                let fraction = self.storage.ulp_fraction()?;
                let spans = f64::from_bits(bits) / fraction;
                Some(saturating_scale(1, spans))
            }
            ErrorMeasure::Absolute { .. } => None,
        }
    }

    /// The contract a value carries after `self` is followed by `next`.
    ///
    /// Errors add, the weaker determinism wins, and the composed contract keeps
    /// the second region's formats because that is what the value is held in
    /// once the second region has run.
    ///
    /// # Errors
    ///
    /// Returns [`ContractRefusal::FormatMismatch`] when the second region reads
    /// a format the first does not produce, and
    /// [`ContractRefusal::UnboundedMagnitude`] when an absolute bound meets a
    /// relative one with no magnitude proof to convert between them.
    pub fn compose(&self, next: &Self) -> Result<Self, ContractRefusal> {
        if self.storage != next.intermediate && self.storage != next.storage {
            return Err(ContractRefusal::FormatMismatch {
                first: self.storage,
                second: next.intermediate,
            });
        }
        let measure = self.compose_measure(next)?;
        let mut composed = *next;
        composed.measure = measure;
        composed.reassociation = weaker_reassociation(self.reassociation, next.reassociation);
        composed.determinism = weaker_determinism(self.determinism, next.determinism);
        composed.atomic_order = weaker_atomic_order(self.atomic_order, next.atomic_order);
        composed.approximation = weaker_approximation(self.approximation, next.approximation);
        Ok(composed)
    }

    /// The measure a value carries after `self` is followed by `next`.
    fn compose_measure(&self, next: &Self) -> Result<ErrorMeasure, ContractRefusal> {
        let with_approximation = |contract: &Self| match contract.approximation {
            Approximation::Refused => contract.measure,
            Approximation::Native { measure } => wider(contract.measure, measure),
        };
        let first = with_approximation(self);
        let second = with_approximation(next);
        match (first, second) {
            (ErrorMeasure::Exact, other) | (other, ErrorMeasure::Exact) => Ok(other),
            (ErrorMeasure::Ulp { count: left }, ErrorMeasure::Ulp { count: right })
                if self.storage == next.storage =>
            {
                Ok(ErrorMeasure::Ulp {
                    count: left.saturating_add(right),
                })
            }
            (ErrorMeasure::Absolute { bits: left }, ErrorMeasure::Absolute { bits: right }) => Ok(
                ErrorMeasure::absolute(f64::from_bits(left) + f64::from_bits(right)),
            ),
            (ErrorMeasure::Absolute { .. }, other) | (other, ErrorMeasure::Absolute { .. }) => {
                Err(ContractRefusal::UnboundedMagnitude { measure: other })
            }
            (left, right) => {
                let left = fraction_of(left, self.storage)?;
                let right = fraction_of(right, next.storage)?;
                Ok(ErrorMeasure::relative(left + right + left * right))
            }
        }
    }

    /// The contract of a reduction of `terms` values under `self`.
    ///
    /// A pairwise or tree reduction of `n` terms accumulates the per-step error
    /// `log2(n)` times, and a sequential one accumulates it `n - 1` times. The
    /// stated order decides which of the two applies: a contract that forbids
    /// reassociation is held to the sequential count it asked for.
    ///
    /// A reduction accumulating in a format narrower than its storage rounds
    /// every partial sum to the accumulator, so the per-step error is that
    /// format's step and not the declared bound. An `f16` accumulator over
    /// `f32` storage rounds to one part in 2^10 where the storage holds one
    /// part in 2^23, and the reduction is priced at the wider of the two.
    ///
    /// # Errors
    ///
    /// Returns [`ContractRefusal::UnboundedMagnitude`] when the measure cannot
    /// be read as a fraction of the exact magnitude.
    pub fn over_reduction(&self, terms: u32) -> Result<Self, ContractRefusal> {
        let steps = match self.reassociation {
            Reassociation::Forbidden => f64::from(terms.saturating_sub(1)),
            Reassociation::Exact | Reassociation::WithinBudget => {
                f64::from(terms.max(2).ilog2().max(1))
            }
        };
        let accumulator_step = if self.accumulator == self.storage {
            0.0
        } else {
            self.accumulator.ulp_fraction().unwrap_or(0.0)
        };
        let mut reduced = *self;
        reduced.measure = match self.measure {
            ErrorMeasure::Exact if accumulator_step == 0.0 => ErrorMeasure::Exact,
            ErrorMeasure::Absolute { bits } => ErrorMeasure::absolute(f64::from_bits(bits) * steps),
            ErrorMeasure::Ulp { count } if accumulator_step == 0.0 => ErrorMeasure::Ulp {
                count: saturating_scale(count, steps),
            },
            measure => {
                let stated = fraction_of(measure, self.storage)?;
                ErrorMeasure::relative(stated.max(accumulator_step) * steps)
            }
        };
        Ok(reduced)
    }

    /// The contract of a recurrence advanced `steps` times under `self`.
    ///
    /// Each step reads what the step before it wrote, so a relative error `e`
    /// compounds: after `n` steps the result sits within `(1 + e)^n - 1` of the
    /// exact one. A reduction is not the same shape, because its partial sums
    /// are independent until they meet; a recurrence has one chain and no
    /// order to choose, so the result forbids reassociation whatever the
    /// contract stated.
    ///
    /// A compounding bound reaches the representable ceiling for a long enough
    /// chain. The measure saturates there rather than wrapping, so a recurrence
    /// nothing can bound states an error nothing admits instead of a small one.
    ///
    /// # Errors
    ///
    /// Returns [`ContractRefusal::UnboundedMagnitude`] when the measure cannot
    /// be read as a fraction of the exact magnitude.
    pub fn over_recurrence(&self, steps: u32) -> Result<Self, ContractRefusal> {
        let mut advanced = *self;
        advanced.reassociation = if self.storage.is_exact() {
            Reassociation::Exact
        } else {
            Reassociation::Forbidden
        };
        if steps == 0 {
            return Ok(advanced);
        }
        advanced.measure = match self.measure {
            ErrorMeasure::Exact => ErrorMeasure::Exact,
            ErrorMeasure::Absolute { bits } => {
                ErrorMeasure::absolute(f64::from_bits(bits) * f64::from(steps))
            }
            measure => {
                let step = fraction_of(measure, self.storage)?;
                let compounded = (1.0 + step).powi(i32::try_from(steps).unwrap_or(i32::MAX)) - 1.0;
                if compounded.is_finite() {
                    ErrorMeasure::relative(compounded)
                } else {
                    ErrorMeasure::relative(f64::MAX)
                }
            }
        };
        Ok(advanced)
    }

    /// Whether `composed` stays inside the measure this contract declares.
    ///
    /// # Errors
    ///
    /// Returns [`ContractRefusal::BudgetExceeded`] when the composed error is
    /// wider than the declared one, and [`ContractRefusal::UnboundedMagnitude`]
    /// when the two measures cannot be compared without a magnitude proof.
    pub fn admits(&self, composed: &ErrorMeasure) -> Result<(), ContractRefusal> {
        let declared = fraction_of(self.measure, self.storage)?;
        let proven = fraction_of(*composed, self.storage)?;
        if proven > declared {
            return Err(ContractRefusal::BudgetExceeded {
                declared: self.measure,
                composed: *composed,
            });
        }
        Ok(())
    }

    /// Whether a transform may reorder the combines this contract covers.
    ///
    /// # Errors
    ///
    /// Returns [`ContractRefusal::ReassociationRefused`] when the contract
    /// states the order is the contract.
    pub fn permits_reassociation(&self) -> Result<(), ContractRefusal> {
        match self.reassociation {
            Reassociation::Forbidden => Err(ContractRefusal::ReassociationRefused {
                stated: self.reassociation,
            }),
            Reassociation::Exact | Reassociation::WithinBudget => Ok(()),
        }
    }
}

/// A measure as a fraction of the exact magnitude.
fn fraction_of(measure: ErrorMeasure, storage: ScalarFormat) -> Result<f64, ContractRefusal> {
    match measure {
        ErrorMeasure::Exact => Ok(0.0),
        ErrorMeasure::Relative { bits } => Ok(f64::from_bits(bits)),
        ErrorMeasure::Ulp { count } => storage
            .ulp_fraction()
            .map(|fraction| f64::from(count) * fraction)
            .ok_or(ContractRefusal::UnboundedMagnitude { measure }),
        ErrorMeasure::Absolute { .. } => Err(ContractRefusal::UnboundedMagnitude { measure }),
    }
}

/// The wider of two measures of the same kind, keeping the first on a tie.
fn wider(left: ErrorMeasure, right: ErrorMeasure) -> ErrorMeasure {
    match (left, right) {
        (ErrorMeasure::Exact, other) | (other, ErrorMeasure::Exact) => other,
        (ErrorMeasure::Ulp { count: first }, ErrorMeasure::Ulp { count: second }) => {
            ErrorMeasure::Ulp {
                count: first.max(second),
            }
        }
        _ => {
            if right.magnitude() > left.magnitude() {
                right
            } else {
                left
            }
        }
    }
}

/// A ULP count scaled by a step count, saturating at the count ceiling.
fn saturating_scale(count: u32, steps: f64) -> u32 {
    let scaled = f64::from(count).max(1.0) * steps;
    if scaled >= f64::from(u32::MAX) {
        u32::MAX
    } else {
        // Fix: the ceiling above keeps the cast inside the u32 range.
        scaled.ceil() as u32
    }
}

/// The reassociation two composed regions admit together.
fn weaker_reassociation(first: Reassociation, second: Reassociation) -> Reassociation {
    match (first, second) {
        (Reassociation::Forbidden, _) | (_, Reassociation::Forbidden) => Reassociation::Forbidden,
        (Reassociation::WithinBudget, _) | (_, Reassociation::WithinBudget) => {
            Reassociation::WithinBudget
        }
        (Reassociation::Exact, Reassociation::Exact) => Reassociation::Exact,
    }
}

/// The determinism two composed regions have together.
fn weaker_determinism(first: Determinism, second: Determinism) -> Determinism {
    if first == Determinism::RunToRunVariable || second == Determinism::RunToRunVariable {
        Determinism::RunToRunVariable
    } else {
        Determinism::Deterministic
    }
}

/// The atomic-order sensitivity two composed regions have together.
fn weaker_atomic_order(
    first: AtomicOrderSensitivity,
    second: AtomicOrderSensitivity,
) -> AtomicOrderSensitivity {
    if first == AtomicOrderSensitivity::Sensitive || second == AtomicOrderSensitivity::Sensitive {
        AtomicOrderSensitivity::Sensitive
    } else {
        AtomicOrderSensitivity::Insensitive
    }
}

/// The approximation two composed regions admit together.
fn weaker_approximation(first: Approximation, second: Approximation) -> Approximation {
    match (first, second) {
        (Approximation::Refused, other) | (other, Approximation::Refused) => other,
        (Approximation::Native { measure: left }, Approximation::Native { measure: right }) => {
            Approximation::Native {
                measure: wider(left, right),
            }
        }
    }
}
