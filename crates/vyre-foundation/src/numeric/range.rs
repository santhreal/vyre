//! What magnitudes a value takes, and which schedule choices a bound enables.
//!
//! A numeric contract states how far a result may sit from the exact one. Most
//! of the choices a schedule wants to make are legal only over a value whose
//! magnitude is bounded: storing in a narrower format needs the value to fit,
//! accumulating in a narrower one needs the partial sums to fit, converting an
//! absolute bound into a relative one needs a magnitude to divide by, and
//! reassociating a sum needs the terms not to cancel. Range analysis proves the
//! bound, and each proof is attached to the one choice it enables so a schedule
//! cannot inherit a proof that was carried out for something else.
//!
//! Three shapes recur and are answered here directly: an exponential, whose
//! bound is the exponential of the bound; an inverse, which is bounded only away
//! from zero; and an affine recurrence, which is bounded when its gain is under
//! one and grows without limit when it is not.

use super::contract::{ContractRefusal, ErrorMeasure, NumericContract};
use super::format::ScalarFormat;

/// A proven bound on the values one graph value takes.
///
/// The endpoints are held as `f64` bits so a range stays hashable and can be
/// recorded beside the schedule choice it proved.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct MagnitudeRange {
    low: u64,
    high: u64,
}

impl MagnitudeRange {
    /// The range `low ..= high`, or `None` when it is not a bound.
    ///
    /// A NaN endpoint and an inverted interval are not bounds: neither states a
    /// value the analysis may divide by or compare a format's ceiling against.
    #[must_use]
    pub fn new(low: f64, high: f64) -> Option<Self> {
        if low.is_nan() || high.is_nan() || low > high {
            return None;
        }
        Some(Self {
            low: low.to_bits(),
            high: high.to_bits(),
        })
    }

    /// The range of a value known exactly.
    #[must_use]
    pub fn point(value: f64) -> Option<Self> {
        Self::new(value, value)
    }

    /// The range `-bound ..= bound`.
    #[must_use]
    pub fn symmetric(bound: f64) -> Option<Self> {
        Self::new(-bound.abs(), bound.abs())
    }

    /// The lower endpoint.
    #[must_use]
    pub fn low(self) -> f64 {
        f64::from_bits(self.low)
    }

    /// The upper endpoint.
    #[must_use]
    pub fn high(self) -> f64 {
        f64::from_bits(self.high)
    }

    /// The largest magnitude any value in the range takes.
    #[must_use]
    pub fn peak(self) -> f64 {
        self.low().abs().max(self.high().abs())
    }

    /// The smallest magnitude any value in the range takes.
    ///
    /// A range that straddles zero contains zero, so its smallest magnitude is
    /// zero and nothing may be divided by it.
    #[must_use]
    pub fn floor(self) -> f64 {
        if self.contains_zero() {
            0.0
        } else {
            self.low().abs().min(self.high().abs())
        }
    }

    /// Whether zero is inside the range.
    #[must_use]
    pub fn contains_zero(self) -> bool {
        self.low() <= 0.0 && self.high() >= 0.0
    }

    /// Whether every value in the range has the same sign.
    ///
    /// A sum of same-signed terms does not cancel, which is what makes a
    /// reassociated reduction over it stay inside a per-step bound.
    #[must_use]
    pub fn single_signed(self) -> bool {
        self.low() > 0.0 || self.high() < 0.0
    }

    /// Whether `format` represents every value in the range as a finite number.
    #[must_use]
    pub fn fits(self, format: ScalarFormat) -> bool {
        let semantics = format.semantics();
        self.low() >= semantics.min_finite && self.high() <= semantics.max_finite
    }

    /// The range of `exp(x)` over this range.
    ///
    /// `None` where the upper endpoint exponentiates past what `f64` holds: an
    /// exponential nothing bounds is what makes a narrower storage format an
    /// overflow rather than a rounding.
    #[must_use]
    pub fn exponential(self) -> Option<Self> {
        let low = self.low().exp();
        let high = self.high().exp();
        if !low.is_finite() || !high.is_finite() {
            return None;
        }
        Self::new(low, high)
    }

    /// The range of `1 / x` over this range.
    ///
    /// `None` where the range contains zero, because the inverse is unbounded
    /// there whatever the endpoints are.
    #[must_use]
    pub fn reciprocal(self) -> Option<Self> {
        if self.contains_zero() {
            return None;
        }
        let first = 1.0 / self.low();
        let second = 1.0 / self.high();
        Self::new(first.min(second), first.max(second))
    }

    /// The range `s` reaches under `s := gain * s + offset`, applied `steps`
    /// times from this range.
    ///
    /// A gain under one contracts: the state converges on `offset / (1 - gain)`
    /// and never leaves the interval spanned by that limit and where it started.
    /// A gain of one or more grows the state geometrically, which is bounded
    /// only for a step count small enough that the growth stays finite.
    #[must_use]
    pub fn affine_recurrence(self, gain: f64, offset: f64, steps: u32) -> Option<Self> {
        if !gain.is_finite() || !offset.is_finite() {
            return None;
        }
        let magnitude = gain.abs();
        let bound = if magnitude < 1.0 {
            self.peak() + offset.abs() / (1.0 - magnitude)
        } else {
            let growth = magnitude.powi(i32::try_from(steps).unwrap_or(i32::MAX));
            let accumulated = if (magnitude - 1.0).abs() < f64::EPSILON {
                offset.abs() * f64::from(steps)
            } else {
                offset.abs() * (growth - 1.0) / (magnitude - 1.0)
            };
            self.peak().mul_add(growth, accumulated)
        };
        if !bound.is_finite() {
            return None;
        }
        Self::symmetric(bound)
    }

    /// This range widened to cover `other` as well.
    #[must_use]
    pub fn join(self, other: Self) -> Self {
        Self {
            low: self.low().min(other.low()).to_bits(),
            high: self.high().max(other.high()).to_bits(),
        }
    }

    /// `measure` as a relative fraction over this range.
    ///
    /// # Errors
    ///
    /// Returns [`ContractRefusal::RangeUnproven`] for an absolute bound over a
    /// range that reaches zero, where no fraction of the exact value is bounded.
    pub fn relative_of(
        self,
        measure: ErrorMeasure,
        storage: ScalarFormat,
    ) -> Result<f64, ContractRefusal> {
        match measure {
            ErrorMeasure::Exact => Ok(0.0),
            ErrorMeasure::Relative { bits } => Ok(f64::from_bits(bits)),
            ErrorMeasure::Ulp { count } => storage
                .ulp_fraction()
                .map(|fraction| f64::from(count) * fraction)
                .ok_or_else(|| ContractRefusal::RangeUnproven {
                    needed: format!("a rounding step for {storage}"),
                    proven: format!("magnitudes in {self}"),
                }),
            ErrorMeasure::Absolute { bits } => {
                let floor = self.floor();
                if floor <= 0.0 {
                    return Err(ContractRefusal::RangeUnproven {
                        needed: "a magnitude bounded away from zero".into(),
                        proven: format!("magnitudes in {self}"),
                    });
                }
                Ok(f64::from_bits(bits) / floor)
            }
        }
    }
}

impl core::fmt::Display for MagnitudeRange {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(formatter, "[{}, {}]", self.low(), self.high())
    }
}

/// One numeric decision a schedule wants to make.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum NumericChoice {
    /// Hold the value in a narrower format between regions.
    StoreAs(ScalarFormat),
    /// Accumulate a reduction of `terms` values in `format`.
    AccumulateIn {
        /// The accumulator format.
        format: ScalarFormat,
        /// Values combined into one output point.
        terms: u32,
    },
    /// Select a native instruction contributing `measure`.
    Approximate {
        /// What the instruction contributes.
        measure: ErrorMeasure,
    },
    /// Combine `terms` values in an order the program did not state.
    Reassociate {
        /// Values combined into one output point.
        terms: u32,
    },
    /// Reduce `terms` values as sequential chunks of `chunk`, combined as a tree.
    ChunkReduction {
        /// Values combined into one output point.
        terms: u32,
        /// Values combined sequentially inside one chunk.
        chunk: u32,
    },
}

/// One schedule choice, the range that proved it, and what it costs.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct RangeProof {
    /// The choice this proof enables.
    pub choice: NumericChoice,
    /// The magnitudes the proof covers.
    pub range: MagnitudeRange,
    /// What the choice contributes to the composed error.
    pub measure: ErrorMeasure,
}

/// Whether `range` proves `choice` legal under `contract`.
///
/// `contract` is the declared ceiling for the value, not what the graph has
/// already accumulated: composition states the accumulated error and this
/// states what one choice costs on top of the format it is made in. Every arm
/// answers two questions, whether the values stay representable under the
/// choice and whether its cost stays inside the declared measure. A choice that
/// passes both is returned with the proof attached, so a later stage records
/// which bound it was selected under rather than that it was selected.
///
/// # Errors
///
/// Returns [`ContractRefusal::RangeUnproven`] where the magnitudes do not cover
/// the choice, [`ContractRefusal::ReassociationRefused`] where the contract
/// states the order is the contract, [`ContractRefusal::ApproximationRefused`]
/// where it admits no approximate instruction, and
/// [`ContractRefusal::BudgetExceeded`] where the choice costs more than the
/// declared measure.
pub fn prove(
    range: MagnitudeRange,
    contract: &NumericContract,
    choice: NumericChoice,
) -> Result<RangeProof, ContractRefusal> {
    let measure = match choice {
        NumericChoice::StoreAs(format) => {
            require_fits(range, format, "every value")?;
            step_cost(format, 1)
        }
        NumericChoice::AccumulateIn { format, terms } => {
            let sums =
                accumulated_range(range, terms).ok_or_else(|| ContractRefusal::RangeUnproven {
                    needed: format!("a bound on {terms} partial sums"),
                    proven: format!("magnitudes in {range}"),
                })?;
            require_fits(sums, format, "every partial sum")?;
            step_cost(format, terms.saturating_sub(1))
        }
        NumericChoice::Approximate { measure } => {
            if contract.approximation == super::contract::Approximation::Refused {
                return Err(ContractRefusal::ApproximationRefused);
            }
            measure
        }
        NumericChoice::Reassociate { terms } => {
            contract.permits_reassociation()?;
            if !range.single_signed() {
                return Err(ContractRefusal::RangeUnproven {
                    needed: "terms of one sign, which do not cancel".into(),
                    proven: format!("magnitudes in {range}"),
                });
            }
            step_cost(contract.accumulator, terms.max(2).ilog2())
        }
        NumericChoice::ChunkReduction { terms, chunk } => {
            if chunk == 0 || chunk > terms {
                return Err(ContractRefusal::RangeUnproven {
                    needed: format!("a chunk of at most {terms} values"),
                    proven: format!("a chunk of {chunk}"),
                });
            }
            contract.permits_reassociation()?;
            step_cost(contract.accumulator, chunked_steps(terms, chunk))
        }
    };
    contract.admits(&measure)?;
    Ok(RangeProof {
        choice,
        range,
        measure,
    })
}

/// The number of rounding steps a reduction run as sequential chunks performs.
///
/// Each chunk rounds `chunk - 1` times in sequence, and the chunk results are
/// combined as a tree over the chunk count. One long chunk is the sequential
/// order and one value per chunk is the tree, so both fall out of the same
/// count.
fn chunked_steps(terms: u32, chunk: u32) -> u32 {
    let chunks = terms.div_ceil(chunk);
    chunk
        .saturating_sub(1)
        .saturating_add(chunks.max(2).ilog2())
}

/// The range the partial sums of `terms` values from `range` take.
fn accumulated_range(range: MagnitudeRange, terms: u32) -> Option<MagnitudeRange> {
    let scale = f64::from(terms.max(1));
    MagnitudeRange::new(range.low() * scale, range.high() * scale)
}

/// Whether `format` holds every value in `range`.
fn require_fits(
    range: MagnitudeRange,
    format: ScalarFormat,
    subject: &str,
) -> Result<(), ContractRefusal> {
    if range.fits(format) {
        return Ok(());
    }
    Err(ContractRefusal::RangeUnproven {
        needed: format!("{subject} inside {format}"),
        proven: format!("magnitudes in {range}"),
    })
}

/// What `steps` roundings in `format` contribute.
///
/// An exact format has no rounding step, so any number of steps in it costs
/// nothing: an integer reduction is the same value in every order.
fn step_cost(format: ScalarFormat, steps: u32) -> ErrorMeasure {
    match format.ulp_fraction() {
        None => ErrorMeasure::Exact,
        Some(fraction) => ErrorMeasure::relative(fraction * f64::from(steps)),
    }
}
