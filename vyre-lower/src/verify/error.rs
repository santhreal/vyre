//! Descriptor verification failures.
//!
//! Owns the shape a verification failure is reported in: the location, the
//! enumerated cause, the collection type the walk fills, and the short
//! rendering a backend puts in a diagnostic. It runs no check.

use serde::{Deserialize, Serialize};

/// Result type  -  `Ok(())` if every invariant holds; `Err(Vec)` lists
/// every violation found, not just the first.
pub type VerifyResult = Result<(), Vec<VerifyError>>;

/// Format the first descriptor verification errors for backend diagnostics.
#[must_use]
pub fn format_verify_errors(errors: &[VerifyError]) -> String {
    let mut out = String::new();
    for (index, error) in errors.iter().take(4).enumerate() {
        if index != 0 {
            out.push_str("; ");
        }
        out.push_str(&format!("{error:?}"));
    }
    if errors.len() > 4 {
        out.push_str("; ...");
    }
    out
}

/// One descriptor verification failure location and cause.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct VerifyError {
    /// Path from the root descriptor body to the failing body.
    pub body_path: Vec<usize>,
    /// Operation index within the failing body.
    pub op_index: usize,
    /// Verification invariant that failed.
    pub kind: VerifyErrorKind,
}

/// Descriptor invariant violated during verification.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum VerifyErrorKind {
    /// A body assigns one result identifier more than once.
    DuplicateResultId(u32),
    /// An operand references a result without a visible producer.
    DanglingResultRef {
        /// Operand position containing the reference.
        operand_pos: usize,
        /// Missing result identifier.
        ref_id: u32,
    },
    /// A literal operand indexes beyond the body literal pool.
    LiteralPoolOutOfRange {
        /// Operand position containing the pool index.
        operand_pos: usize,
        /// Invalid literal-pool index.
        pool_idx: u32,
        /// Number of available literals.
        pool_size: usize,
    },
    /// A structured operation indexes beyond the child-body table.
    ChildBodyIndexOutOfRange {
        /// Operand position containing the body index.
        operand_pos: usize,
        /// Invalid child-body index.
        body_idx: u32,
        /// Number of available child bodies.
        child_count: usize,
    },
    /// Two ops in DIFFERENT bodies of the same descriptor assign the
    /// same result id. `vyre-lower` allocates result ids globally, and
    /// backends rely on that: the PTX emitter keeps one flat
    /// result-id → register map for the whole kernel, so a reused id
    /// silently resolves to whichever producer the emitter walked last.
    /// A `GlobalInvocationId` store index that collided with a sibling
    /// body's `Literal(1)` was emitted as the constant address
    /// `[%rd4+4]`, making every thread write element 1.
    ResultIdReusedAcrossBodies {
        /// Reused result identifier.
        result: u32,
        /// Body path of the first op that assigned this id. The error's
        /// own `body_path`/`op_index` locate the second one.
        first_body_path: Vec<usize>,
    },
    /// A literal operation has no literal-pool operand.
    LiteralOpMissingPoolOperand,
    /// An operation provides fewer operands than its kind requires.
    OperandCountTooShort {
        /// Minimum required operand count.
        expected_min: usize,
        /// Received operand count.
        got: usize,
    },
    /// `dispatch.workgroup_size[axis]` is zero. A kernel with a zero
    /// dim never runs  -  almost certainly a host-side bug.
    DispatchZeroDim {
        /// Zero-valued dispatch axis.
        axis: u8,
    },
    /// Two `BindingSlot` entries share the same `.slot` field. The
    /// emitters look up bindings by `.slot`; duplicates make the
    /// lookup ambiguous.
    DuplicateBindingSlotId {
        /// Duplicated binding slot.
        slot: u32,
    },
    /// A host-bound binding (`Global` / `Constant` / `Uniform`) sits
    /// in the workgroup-reserved slot range (`>= 1<<24`). Backend
    /// bind-group layouts cap at 1000 bindings on wgpu and similar
    /// limits elsewhere; a host slot in the reserved range fails
    /// layout creation with a "binding index N greater than maximum"
    /// validator error. Earlier rewrites should have allocated the
    /// new slot in the host range.
    HostBindingInWorkgroupRange {
        /// Host binding slot in the reserved workgroup range.
        slot: u32,
    },
    /// A workgroup binding (`Shared` / `Scratch`) sits in the
    /// host-bindable slot range (`< 1<<24`). The host dispatch path
    /// addresses host bindings by slot id; a workgroup binding in
    /// that range can collide with a Global binding's slot id and
    /// silently steer load/store ops to the wrong memory class.
    WorkgroupBindingInHostRange {
        /// Workgroup binding slot in the host-visible range.
        slot: u32,
    },
}
