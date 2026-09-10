//! Executable proof for the algebraic laws an operation declares.
//!
//! A declared law used to be a string in a registration that nothing executed.
//! The certificate carried a `laws_verified` list its caller supplied, and the
//! only prover in this crate checked a host closure passed in by the caller, so
//! a label asserted a property of whatever the caller wrote rather than of the
//! registered program.
//!
//! Every proof here runs the operation's own program through the reference
//! oracle. The witness is derived from the program's host-input buffers, so a
//! new operation declaring a law is proved by the same code that proves the
//! existing ones, and a law whose statement cannot be exercised through the
//! declared buffer shape reports that instead of passing.
//!
//! What each witness establishes:
//!
//! - `ExchangedInputs` proves the full statement of commutativity for a binary
//!   operation: the two host inputs are exchanged and every output must be
//!   byte-identical.
//! - `PermutedElements` reverses the element order of the one read-only input of
//!   a reduction. Order invariance is a necessary condition of commutativity and
//!   of associativity for a reduction, not the whole statement; a refutation is
//!   still a refutation of the declared law.
//! - `RepeatedInput` proves idempotence of a binary operation as `f(a, a) == a`.
//! - `Reapplied` proves idempotence of a map as `f(f(a)) == f(a)`, and
//!   self-inverse as `f(f(a)) == a`.
//! - `Chained` proves associativity as `f(f(a, b), c) == f(a, f(b, c))` and
//!   self-inverse of a binary operation as `f(f(a, b), b) == a`.

use vyre_foundation::ir::{BufferAccess, BufferDecl, Program};
use vyre_foundation::operation::SemanticOperation;
use vyre_reference::value::Value;
use vyre_reference::{is_reference_input, is_reference_output, reference_eval};

/// Why a declared law carries no executable proof.
///
/// Every one of these says the law name is a bare string where its statement
/// needs a payload: which element is the identity, which operator it
/// distributes over, which operation inverts it, which order it is monotone in.
/// Recording the payload is what turns a row here into a proof, so the roster
/// of pairs carrying one only ever shrinks.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnprovenKind {
    /// The law names an element the declaration does not carry: an identity, an
    /// absorbing element, or a bound.
    MissingElement,
    /// The law relates two operations and the declaration names only one:
    /// distributivity, De Morgan duality, an inverse partner.
    MissingPartner,
    /// The law is an order relation and the declaration names no order.
    MissingOrder,
    /// The law is stated over a shape the declared buffers cannot compose: an
    /// output that cannot be fed back, or a reduction whose statement needs a
    /// different input length.
    ShapeNotComposable,
    /// The law is `custom` and the declaration carries no check to run.
    NoCheckDeclared,
    /// The registration carries no program builder, no fixture inputs, or zero
    /// fixture cases, so nothing can be executed for any law it declares.
    NothingToExecute,
}

impl UnprovenKind {
    /// Every kind, so a reader of the decision table is judged against the
    /// whole vocabulary rather than the members someone remembered.
    pub const ALL: &'static [Self] = &[
        Self::MissingElement,
        Self::MissingPartner,
        Self::MissingOrder,
        Self::ShapeNotComposable,
        Self::NoCheckDeclared,
        Self::NothingToExecute,
    ];

    /// Stable name recorded in the decision table.
    #[must_use]
    pub const fn name(self) -> &'static str {
        // Exhaustive with no catch-all: a kind added above fails to compile
        // here until it has a recorded name.
        match self {
            Self::MissingElement => "missing-element",
            Self::MissingPartner => "missing-partner",
            Self::MissingOrder => "missing-order",
            Self::ShapeNotComposable => "shape-not-composable",
            Self::NoCheckDeclared => "no-check-declared",
            Self::NothingToExecute => "nothing-to-execute",
        }
    }

    /// The kind `name` spells, or `None` when the table names one that does not
    /// exist.
    #[must_use]
    pub fn parse(name: &str) -> Option<Self> {
        Self::ALL.iter().copied().find(|kind| kind.name() == name)
    }

    /// The payload the statement of `law` needs and a bare law name cannot
    /// carry, or `None` for a law this module proves by execution.
    ///
    /// Every name in `vyre_spec::law_catalog()` is either in
    /// [`PROVABLE_LAWS`] or classified here. A law added to that catalog and to
    /// neither reaches no arm, which the conformance suite reports by name.
    #[must_use]
    pub fn for_law(law: &str) -> Option<Self> {
        match law {
            "identity" | "left-identity" | "right-identity" | "absorbing" | "left-absorbing"
            | "right-absorbing" | "bounded" | "complement" | "zero-product" => {
                Some(Self::MissingElement)
            }
            // A categorical law relates the arrow to the arrows it composes
            // with: the identity arrow for `f ∘ id = id ∘ f = f`, two further
            // arrows for `(h ∘ g) ∘ f = h ∘ (g ∘ f)`. A registration names one
            // arrow, so the payload the statement needs is the partner's
            // registration id, exactly as it is for a distributive pair.
            "de-morgan" | "distributive" | "lattice-absorption" | "inverse-of"
            | "categorical-identity" | "categorical-associative" => Some(Self::MissingPartner),
            "monotone" | "monotonic" | "trichotomy" => Some(Self::MissingOrder),
            "custom" => Some(Self::NoCheckDeclared),
            _ => None,
        }
    }
}

/// The laws this module proves by executing the operation.
///
/// A name here reaches an arm of `Shape::plan`, so its verdict is a proof or a
/// refutation whenever the declared buffer shape admits the witness.
pub const PROVABLE_LAWS: &[&str] = &[
    "commutative",
    "associative",
    "idempotent",
    "self-inverse",
    "involution",
];

/// What was executed to judge a declared law.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LawWitness {
    /// The two read-only host inputs were exchanged.
    ExchangedInputs,
    /// The element order of the one read-only host input was reversed.
    PermutedElements,
    /// The one read-only host input was supplied to both operands.
    RepeatedInput,
    /// The operation was applied to its own output.
    Reapplied,
    /// The operation was composed with itself in both groupings.
    Chained,
}

impl LawWitness {
    /// Stable name recorded in a certificate.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::ExchangedInputs => "exchanged-inputs",
            Self::PermutedElements => "permuted-elements",
            Self::RepeatedInput => "repeated-input",
            Self::Reapplied => "reapplied",
            Self::Chained => "chained",
        }
    }
}

/// The judgment on one declared law of one operation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LawVerdict {
    /// Every fixture case satisfied the witness.
    Holds {
        /// Number of fixture cases exercised, never zero.
        cases: usize,
    },
    /// The oracle refuted the law on a fixture case.
    Refuted {
        /// Zero-based fixture case index.
        case: usize,
        /// What differed, naming the output and the first differing byte.
        detail: String,
    },
    /// No witness was executed, and the reason is a property of the
    /// declaration rather than a failure.
    ///
    /// This is not a pass. Every pair reporting it has to carry a recorded
    /// decision naming the same kind, which is what keeps an unexercised label
    /// from reading as a proven one.
    Unproven {
        /// The payload the law's statement needs, or the shape that admits no
        /// witness.
        kind: UnprovenKind,
        /// Why no witness ran, in the terms the shape reports.
        reason: String,
    },
    /// The proof was attempted and could not complete: the oracle failed, or a
    /// fixture case does not match the program's buffer arity.
    ///
    /// No decision can excuse this; it is a defect in the operation or its
    /// fixtures.
    Unrunnable {
        /// What the oracle reported, or which fixture case disagreed.
        reason: String,
    },
}

/// One law of one operation, and what proving it produced.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LawProof {
    /// Stable operation identifier.
    pub op_id: &'static str,
    /// Declared law name, from the frozen law catalog.
    pub law: &'static str,
    /// What was executed, when a witness was derivable.
    pub witness: Option<LawWitness>,
    /// The judgment.
    pub verdict: LawVerdict,
}

impl LawProof {
    /// Whether this proof refutes the declared law.
    #[must_use]
    pub fn is_refuted(&self) -> bool {
        matches!(self.verdict, LawVerdict::Refuted { .. })
    }

    /// Whether this proof establishes the declared law on every fixture case.
    #[must_use]
    pub fn holds(&self) -> bool {
        matches!(self.verdict, LawVerdict::Holds { .. })
    }
}

/// Prove every law `entry` declares.
///
/// The returned rows are in declaration order, one per declared law. An
/// operation declaring no law returns an empty vector.
#[must_use]
pub fn prove_declared_laws(entry: &SemanticOperation) -> Vec<LawProof> {
    entry.laws.iter().map(|law| prove_law(entry, law)).collect()
}

/// Prove one law of one operation, whether or not the operation declares it.
///
/// Proving an undeclared law is how the prover itself is tested: an operation
/// that is not commutative must be refuted when asked for commutativity.
#[must_use]
pub fn prove_law(entry: &SemanticOperation, law: &'static str) -> LawProof {
    let proof = |witness: Option<LawWitness>, verdict: LawVerdict| LawProof {
        op_id: entry.id,
        law,
        witness,
        verdict,
    };
    let unproven = |kind: UnprovenKind, reason: String| LawProof {
        op_id: entry.id,
        law,
        witness: None,
        verdict: LawVerdict::Unproven { kind, reason },
    };
    let Some(build) = entry.build else {
        return unproven(
            UnprovenKind::NothingToExecute,
            "the registration carries no program builder".to_string(),
        );
    };
    let Some(fixtures) = entry.test_inputs else {
        return unproven(
            UnprovenKind::NothingToExecute,
            "the registration carries no fixture inputs".to_string(),
        );
    };
    let program = build().with_entry_op_id(entry.id);
    let shape = match Shape::read(&program) {
        Ok(shape) => shape,
        Err(reason) => return unproven(UnprovenKind::ShapeNotComposable, reason),
    };
    let cases = fixtures();
    if cases.is_empty() {
        return unproven(
            UnprovenKind::NothingToExecute,
            "the registration supplies zero fixture cases".to_string(),
        );
    }
    let Some(plan) = shape.plan(law, &cases) else {
        // A law outside the executable families is unproven because its
        // statement needs a payload; one inside them is unproven because this
        // shape admits no witness for it.
        return unproven(
            UnprovenKind::for_law(law).unwrap_or(UnprovenKind::ShapeNotComposable),
            shape.describe(law),
        );
    };
    let mut exercised = 0usize;
    let mut rejected: Option<String> = None;
    for (case, inputs) in cases.iter().enumerate() {
        if inputs.len() != shape.inputs.len() {
            return proof(
                Some(plan.witness),
                LawVerdict::Unrunnable {
                    reason: format!(
                        "fixture case {case} supplies {} value(s) for {} host input buffer(s)",
                        inputs.len(),
                        shape.inputs.len()
                    ),
                },
            );
        }
        if !(plan.admits)(&shape, inputs) {
            continue;
        }
        match (plan.check)(&program, &shape, inputs, &cases) {
            Ok(None) => exercised += 1,
            Ok(Some(detail)) => {
                return proof(Some(plan.witness), LawVerdict::Refuted { case, detail })
            }
            // The witness feeds the program an input the operation's own
            // declared range precondition rejects, so this case is outside the
            // domain the law is stated over. A registration whose own fixtures
            // the oracle cannot run is a parity defect and is reported there.
            Err(reason) => rejected = Some(reason),
        }
    }
    if exercised == 0 {
        if let Some(reason) = rejected {
            return unproven(UnprovenKind::ShapeNotComposable, reason);
        }
    }
    proof(Some(plan.witness), LawVerdict::Holds { cases: exercised })
}

/// The host-input and output structure a witness is derived from.
struct Shape {
    /// Every host-input buffer, in `Program::buffers` order.
    inputs: Vec<BufferDecl>,
    /// Positions within `inputs` whose buffer is read-only.
    read_only: Vec<usize>,
    /// Element width in bytes of each host input.
    element_bytes: Vec<usize>,
    /// Number of buffers `reference_eval` returns.
    outputs: usize,
    /// Total declared element count across every output buffer.
    output_elements: usize,
    /// Declared byte extent of each output buffer, in declaration order.
    output_bytes: Vec<usize>,
}

impl Shape {
    fn read(program: &Program) -> Result<Self, String> {
        let inputs: Vec<BufferDecl> = program
            .buffers()
            .iter()
            .filter(|decl| is_reference_input(decl))
            .cloned()
            .collect();
        if inputs.is_empty() {
            return Err("the program declares no host input buffer".to_string());
        }
        let mut element_bytes = Vec::with_capacity(inputs.len());
        for decl in &inputs {
            let element = decl.element();
            let sized = |count: usize| -> Result<Option<usize>, String> {
                element.packed_size_bytes(count).map_err(|error| {
                    format!(
                        "host input `{}` has no fixed element width: {error}",
                        decl.name()
                    )
                })
            };
            let width = sized(1)?.ok_or_else(|| {
                format!(
                    "host input `{}` carries a variable-width element type",
                    decl.name()
                )
            })?;
            if width == 0 {
                return Err(format!(
                    "host input `{}` reports a zero-byte element width",
                    decl.name()
                ));
            }
            // A sub-byte type packs several elements into one byte, so a
            // byte-granular permutation would reorder blocks rather than
            // elements. Refusing it here is what keeps `PermutedElements`
            // exactly the permutation its name states.
            if sized(2)? != Some(width * 2) {
                return Err(format!(
                    "host input `{}` packs more than one element per byte",
                    decl.name()
                ));
            }
            element_bytes.push(width);
        }
        let read_only = inputs
            .iter()
            .enumerate()
            .filter(|(_, decl)| decl.access() == BufferAccess::ReadOnly)
            .map(|(index, _)| index)
            .collect();
        let output_decls: Vec<&BufferDecl> = program
            .buffers()
            .iter()
            .filter(|decl| is_reference_output(decl))
            .collect();
        if output_decls.is_empty() {
            return Err("the program returns no output buffer".to_string());
        }
        let mut output_elements = 0usize;
        let mut output_bytes = Vec::with_capacity(output_decls.len());
        for decl in &output_decls {
            let count = usize::try_from(decl.count()).unwrap_or(usize::MAX);
            output_elements = output_elements.saturating_add(count);
            let bytes = decl
                .element()
                .packed_size_bytes(count)
                .map_err(|error| format!("output `{}` has no fixed extent: {error}", decl.name()))?
                .ok_or_else(|| {
                    format!(
                        "output `{}` carries a variable-width element type",
                        decl.name()
                    )
                })?;
            output_bytes.push(bytes);
        }
        Ok(Self {
            inputs,
            read_only,
            element_bytes,
            outputs: output_decls.len(),
            output_elements,
            output_bytes,
        })
    }

    /// The two read-only inputs a commutativity exchange needs, when the shape
    /// has exactly two and they carry the same element type.
    fn exchange_pair(&self, inputs: &[Vec<u8>]) -> Option<(usize, usize)> {
        let [first, second] = self.read_only[..] else {
            return None;
        };
        let same_element = self.inputs[first].element() == self.inputs[second].element();
        let same_len = inputs[first].len() == inputs[second].len();
        (same_element && same_len).then_some((first, second))
    }

    /// The one read-only input of a reduction, whose element order a witness may
    /// reverse: one read-only input holding more elements than every output
    /// returns.
    ///
    /// Answered from the declared output extent rather than from an evaluated
    /// output, so an infeasible witness is never selected and reported as a
    /// failed run.
    fn reduced_input(&self, inputs: &[Vec<u8>]) -> Option<usize> {
        let [only] = self.read_only[..] else {
            return None;
        };
        let input_elements = inputs[only].len() / self.element_bytes[only];
        (input_elements > self.output_elements && input_elements > 1).then_some(only)
    }

    /// The read-only input a permutation witness may reverse: a reduction input
    /// whose reversal is a different byte sequence, so the witness distinguishes
    /// two orders instead of comparing a value against itself.
    fn permutable_input(&self, inputs: &[Vec<u8>]) -> Option<usize> {
        let only = self.reduced_input(inputs)?;
        let reversed = reverse_elements(&inputs[only], self.element_bytes[only]);
        (reversed != inputs[only]).then_some(only)
    }

    /// The one read-only input of a map, whose output has the same byte length:
    /// the shape `f(f(a))` needs.
    fn mapped_input(&self, inputs: &[Vec<u8>]) -> Option<usize> {
        let [only] = self.read_only[..] else {
            return None;
        };
        let [output_bytes] = self.output_bytes[..] else {
            return None;
        };
        (output_bytes == inputs[only].len()).then_some(only)
    }

    /// Why `law` has no witness under this shape.
    fn describe(&self, law: &str) -> String {
        format!(
            "law `{law}` has no witness for {} host input buffer(s) ({} read-only) and {} output buffer(s)",
            self.inputs.len(),
            self.read_only.len(),
            self.outputs
        )
    }

    /// Whether this case holds an exchangeable pair of read-only inputs.
    fn admits_exchange(&self, inputs: &[Vec<u8>]) -> bool {
        self.exchange_pair(inputs).is_some()
    }

    /// Whether this case holds a reduction input whose element order a witness
    /// can reverse into a different value.
    fn admits_permutation(&self, inputs: &[Vec<u8>]) -> bool {
        self.permutable_input(inputs).is_some()
    }

    /// Whether this case holds a read-only input the single output can be fed
    /// back into.
    fn admits_mapping(&self, inputs: &[Vec<u8>]) -> bool {
        self.mapped_input(inputs).is_some()
    }

    /// Whether this case holds an exchangeable operand pair and a single output
    /// with the first operand's byte extent: the shape a witness needs to feed
    /// the output back into that operand, or to compare it against one.
    fn admits_operand_feedback(&self, inputs: &[Vec<u8>]) -> bool {
        let Some((first, _)) = self.exchange_pair(inputs) else {
            return false;
        };
        matches!(self.output_bytes[..], [only] if only == inputs[first].len())
    }

    /// The witness this shape admits for `law`, or `None` when no fixture case
    /// admits one.
    ///
    /// Feasibility is decided here rather than inside the check, so a shape that
    /// carries no witness is reported as unproven and recordable instead of as a
    /// proof that failed to run. A run that fails is the oracle or the fixture
    /// arity, which no decision excuses.
    ///
    /// A witness is selected when at least one case admits it, and the returned
    /// plan carries the predicate that says which. A case that cannot carry the
    /// witness proves nothing and refutes nothing: a single-element buffer has
    /// one element order, and an output whose extent is not the input's cannot be
    /// fed back. Skipping such a case keeps the proof the other cases give,
    /// which abandoning the pair would throw away.
    fn plan(&self, law: &str, cases: &[Vec<Vec<u8>>]) -> Option<Plan> {
        let (binary, unary) = match law {
            "commutative" => (
                Plan {
                    witness: LawWitness::ExchangedInputs,
                    check: check_exchanged,
                    admits: Self::admits_exchange,
                },
                Plan {
                    witness: LawWitness::PermutedElements,
                    check: check_permuted,
                    admits: Self::admits_permutation,
                },
            ),
            "associative" => (
                Plan {
                    witness: LawWitness::Chained,
                    check: check_associative,
                    admits: Self::admits_operand_feedback,
                },
                Plan {
                    witness: LawWitness::PermutedElements,
                    check: check_permuted,
                    admits: Self::admits_permutation,
                },
            ),
            "idempotent" => (
                Plan {
                    witness: LawWitness::RepeatedInput,
                    check: check_repeated,
                    admits: Self::admits_operand_feedback,
                },
                Plan {
                    witness: LawWitness::Reapplied,
                    check: check_reapplied_equals_once,
                    admits: Self::admits_mapping,
                },
            ),
            "self-inverse" | "involution" => (
                Plan {
                    witness: LawWitness::Chained,
                    check: check_self_inverse_binary,
                    admits: Self::admits_operand_feedback,
                },
                Plan {
                    witness: LawWitness::Reapplied,
                    check: check_reapplied_equals_input,
                    admits: Self::admits_mapping,
                },
            ),
            _ => return None,
        };
        [binary, unary]
            .into_iter()
            .find(|plan| self.admits_a_case(cases, plan.admits))
    }

    /// Whether any fixture case admits the witness `admits` derives.
    ///
    /// A case whose value count disagrees with the declared buffer count is
    /// skipped: that is a fixture defect, and the run reports it by name rather
    /// than it reading as a shape that carries no witness.
    fn admits_a_case(&self, cases: &[Vec<Vec<u8>>], admits: fn(&Self, &[Vec<u8>]) -> bool) -> bool {
        cases
            .iter()
            .filter(|inputs| inputs.len() == self.inputs.len())
            .any(|inputs| admits(self, inputs))
    }
}

/// A derived witness, the check that executes it, and which cases admit it.
struct Plan {
    witness: LawWitness,
    check: CheckFn,
    admits: fn(&Shape, &[Vec<u8>]) -> bool,
}

/// `Ok(None)` when the case satisfied the witness, `Ok(Some(detail))` when the
/// oracle refuted it, `Err` when the case could not be run at all.
type CheckFn = fn(
    program: &Program,
    shape: &Shape,
    inputs: &[Vec<u8>],
    cases: &[Vec<Vec<u8>>],
) -> Result<Option<String>, String>;

fn run(program: &Program, inputs: &[Vec<u8>]) -> Result<Vec<Vec<u8>>, String> {
    let values: Vec<Value> = inputs
        .iter()
        .map(|bytes| Value::Bytes(bytes.as_slice().into()))
        .collect();
    reference_eval(program, &values)
        .map(|outputs| outputs.into_iter().map(|value| value.to_bytes()).collect())
        .map_err(|error| format!("the reference oracle failed: {error}"))
}

/// First differing output, named by position and first differing byte.
fn difference(left: &[Vec<u8>], right: &[Vec<u8>], what: &str) -> Option<String> {
    if left.len() != right.len() {
        return Some(format!(
            "{what}: {} output buffer(s) against {}",
            left.len(),
            right.len()
        ));
    }
    for (index, (lhs, rhs)) in left.iter().zip(right.iter()).enumerate() {
        if lhs == rhs {
            continue;
        }
        let byte = lhs
            .iter()
            .zip(rhs.iter())
            .position(|(left_byte, right_byte)| left_byte != right_byte);
        return Some(match byte {
            Some(byte) => format!(
                "{what}: output {index} differs at byte {byte} ({} against {})",
                lhs[byte], rhs[byte]
            ),
            None => format!(
                "{what}: output {index} has length {} against {}",
                lhs.len(),
                rhs.len()
            ),
        });
    }
    None
}

fn check_exchanged(
    program: &Program,
    shape: &Shape,
    inputs: &[Vec<u8>],
    _cases: &[Vec<Vec<u8>>],
) -> Result<Option<String>, String> {
    let (first, second) = shape
        .exchange_pair(inputs)
        .ok_or_else(|| "the fixture case does not admit an input exchange".to_string())?;
    let mut exchanged = inputs.to_vec();
    exchanged.swap(first, second);
    let base = run(program, inputs)?;
    let swapped = run(program, &exchanged)?;
    Ok(difference(
        &base,
        &swapped,
        "exchanging the two host inputs",
    ))
}

fn check_permuted(
    program: &Program,
    shape: &Shape,
    inputs: &[Vec<u8>],
    _cases: &[Vec<Vec<u8>>],
) -> Result<Option<String>, String> {
    let only = shape.permutable_input(inputs).ok_or_else(|| {
        "the fixture case does not admit an element permutation that changes the input".to_string()
    })?;
    let mut permuted = inputs.to_vec();
    permuted[only] = reverse_elements(&inputs[only], shape.element_bytes[only]);
    let base = run(program, inputs)?;
    let reversed = run(program, &permuted)?;
    Ok(difference(
        &base,
        &reversed,
        "reversing the input element order",
    ))
}

fn check_repeated(
    program: &Program,
    shape: &Shape,
    inputs: &[Vec<u8>],
    _cases: &[Vec<Vec<u8>>],
) -> Result<Option<String>, String> {
    let (first, second) = shape
        .exchange_pair(inputs)
        .ok_or_else(|| "the fixture case does not admit a repeated operand".to_string())?;
    let mut repeated = inputs.to_vec();
    repeated[second] = repeated[first].clone();
    let outputs = run(program, &repeated)?;
    if outputs.len() != 1 {
        return Err(format!(
            "idempotence compares one output against the operand, but the program returns {}",
            outputs.len()
        ));
    }
    Ok(difference(
        &outputs,
        std::slice::from_ref(&inputs[first]),
        "applying the operation to one operand twice",
    ))
}

fn check_reapplied_equals_once(
    program: &Program,
    shape: &Shape,
    inputs: &[Vec<u8>],
    _cases: &[Vec<Vec<u8>>],
) -> Result<Option<String>, String> {
    let (once, twice, _) = reapply(program, shape, inputs)?;
    Ok(difference(&twice, &once, "applying the operation twice"))
}

fn check_reapplied_equals_input(
    program: &Program,
    shape: &Shape,
    inputs: &[Vec<u8>],
    _cases: &[Vec<Vec<u8>>],
) -> Result<Option<String>, String> {
    let (_, twice, only) = reapply(program, shape, inputs)?;
    Ok(difference(
        &twice,
        std::slice::from_ref(&inputs[only]),
        "applying the operation twice",
    ))
}

/// `f(a)`, `f(f(a))`, and which input carried `a`.
fn reapply(
    program: &Program,
    shape: &Shape,
    inputs: &[Vec<u8>],
) -> Result<(Vec<Vec<u8>>, Vec<Vec<u8>>, usize), String> {
    let only = shape
        .mapped_input(inputs)
        .ok_or_else(|| "the output does not have the shape of the read-only input".to_string())?;
    let once = run(program, inputs)?;
    let mut fed = inputs.to_vec();
    fed[only] = once[0].clone();
    let twice = run(program, &fed)?;
    Ok((once, twice, only))
}

fn check_associative(
    program: &Program,
    shape: &Shape,
    inputs: &[Vec<u8>],
    cases: &[Vec<Vec<u8>>],
) -> Result<Option<String>, String> {
    let (first, second) = shape
        .exchange_pair(inputs)
        .ok_or_else(|| "the fixture case does not admit a chained composition".to_string())?;
    let third = third_operand(shape, inputs, cases, first);
    let left = compose(program, inputs, first, second, &third, Grouping::Left)?;
    let right = compose(program, inputs, first, second, &third, Grouping::Right)?;
    Ok(difference(
        &left,
        &right,
        "regrouping a three-operand composition",
    ))
}

fn check_self_inverse_binary(
    program: &Program,
    shape: &Shape,
    inputs: &[Vec<u8>],
    _cases: &[Vec<Vec<u8>>],
) -> Result<Option<String>, String> {
    let (first, second) = shape
        .exchange_pair(inputs)
        .ok_or_else(|| "the fixture case does not admit a chained composition".to_string())?;
    let once = run(program, inputs)?;
    if once.len() != 1 || once[0].len() != inputs[first].len() {
        return Err(
            "self-inverse composes the output back into the first operand, which this output shape does not admit"
                .to_string(),
        );
    }
    let mut fed = inputs.to_vec();
    fed[first] = once[0].clone();
    fed[second] = inputs[second].clone();
    let twice = run(program, &fed)?;
    Ok(difference(
        &twice,
        std::slice::from_ref(&inputs[first]),
        "applying the operation to its own output with the same second operand",
    ))
}

/// Which grouping of `f(a, b, c)` a composition evaluates.
enum Grouping {
    /// `f(f(a, b), c)`
    Left,
    /// `f(a, f(b, c))`
    Right,
}

/// The third operand of an associativity witness: the next fixture case's first
/// operand when the fixtures supply one, otherwise the first operand reversed.
fn third_operand(
    shape: &Shape,
    inputs: &[Vec<u8>],
    cases: &[Vec<Vec<u8>>],
    first: usize,
) -> Vec<u8> {
    let other = cases
        .iter()
        .find(|case| case.len() == inputs.len() && case[first] != inputs[first]);
    match other {
        Some(case) if case[first].len() == inputs[first].len() => case[first].clone(),
        _ => reverse_elements(&inputs[first], shape.element_bytes[first]),
    }
}

fn compose(
    program: &Program,
    inputs: &[Vec<u8>],
    first: usize,
    second: usize,
    third: &[u8],
    grouping: Grouping,
) -> Result<Vec<Vec<u8>>, String> {
    let (inner_first, inner_second, outer_position) = match grouping {
        Grouping::Left => (inputs[first].clone(), inputs[second].clone(), second),
        Grouping::Right => (inputs[second].clone(), third.to_vec(), first),
    };
    let mut inner = inputs.to_vec();
    inner[first] = inner_first;
    inner[second] = inner_second;
    let inner_output = run(program, &inner)?;
    if inner_output.len() != 1 || inner_output[0].len() != inputs[first].len() {
        return Err(
            "associativity composes the output back into an operand, which this output shape does not admit"
                .to_string(),
        );
    }
    let mut outer = inputs.to_vec();
    match grouping {
        Grouping::Left => {
            outer[first] = inner_output[0].clone();
            outer[outer_position] = third.to_vec();
        }
        Grouping::Right => {
            outer[outer_position] = inputs[first].clone();
            outer[second] = inner_output[0].clone();
        }
    }
    run(program, &outer)
}

/// `bytes` with its `width`-byte elements in reverse order.
///
/// A trailing partial element is left in place: the element count is what the
/// witness permutes, and truncating the tail would change the buffer length the
/// program declares.
fn reverse_elements(bytes: &[u8], width: usize) -> Vec<u8> {
    let count = bytes.len() / width;
    let mut reversed = Vec::with_capacity(bytes.len());
    for index in (0..count).rev() {
        reversed.extend_from_slice(&bytes[index * width..(index + 1) * width]);
    }
    reversed.extend_from_slice(&bytes[count * width..]);
    reversed
}

#[cfg(test)]
mod tests {
    use super::*;
    use vyre_foundation::ir::DataType;

    /// One read-only input and one read-write output, the shape a map declares.
    ///
    /// A plain read-write buffer is both a host input and a returned output, so
    /// a fixture case for this program supplies two values.
    fn one_in_one_out() -> Program {
        Program::wrapped(
            vec![
                BufferDecl::read("a", 0, DataType::U32).with_count(4),
                BufferDecl::read_write("out", 1, DataType::U32).with_count(1),
            ],
            [1, 1, 1],
            Vec::new(),
        )
    }

    fn bytes(count: usize) -> Vec<u8> {
        vec![0; count * 4]
    }

    /// A fixture case supplying `count` input elements and one output element.
    fn case(count: usize) -> Vec<Vec<u8>> {
        vec![bytes(count), bytes(1)]
    }

    /// WHY: witness selection used to read the first fixture case only, so a
    /// pair whose first case cannot carry the witness reported that the shape
    /// admits none, and every later case that could have proved the law was
    /// never reached. The registry has no operation supplying both kinds of
    /// case, so the contract is held here against a declared shape rather than
    /// against whichever fixture set a library op happens to ship.
    ///
    /// What this does not catch: whether the selected check is the right
    /// statement of the law. `plan` names the witness and the checks above own
    /// the statement.
    #[test]
    fn a_witness_no_earlier_case_admits_is_still_selected() {
        let program = one_in_one_out();
        let shape = Shape::read(&program)
            .expect("Fix: the fixture program declares one read-only input and one output.");
        let wider_than_the_output = vec![case(4)];
        let matching_the_output = vec![case(1)];
        let first_admits_nothing = vec![case(4), case(1)];

        assert!(
            shape.plan("idempotent", &wider_than_the_output).is_none(),
            "Fix: a case whose input is wider than the output cannot be fed back, so no witness exists for it."
        );
        assert!(
            shape.plan("idempotent", &matching_the_output).is_some(),
            "Fix: a case whose input matches the output extent admits the reapplication witness."
        );
        assert_eq!(
            shape
                .plan("idempotent", &first_admits_nothing)
                .map(|plan| plan.witness),
            Some(LawWitness::Reapplied),
            "Fix: selection must read every case; stopping at the first one loses the proof the later case gives."
        );
    }
}
