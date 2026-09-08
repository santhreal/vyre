//! Launch coverage a node's own program requires.
//!
//! A logical region states its domain from one declared value, and a launch
//! sized from that domain runs exactly as many invocations as that value has
//! elements. Three constructs make a program's result depend on how many
//! invocations ran rather than only on which elements each one touched, so for
//! those the declared input span is the domain and a narrower launch computes a
//! different number: an atomic, a subgroup collective, and a workgroup-scoped
//! buffer. `vyre_foundation::launch_covers_full_input_span` states them, and
//! the below-admission dispatch path derives its span from the same analysis.
//!
//! An atomic reduction over 4096 elements accumulating into one is the case that
//! separates the two. Its region domain is the accumulator, one point, so the
//! selected geometry launched one workgroup and the sum came back as that
//! workgroup's lane count instead of the element count.

use vyre_foundation::ir::{BufferAccess, MemoryKind, Program};

use crate::{failure, CompileError, CompilerFailureKind, GeometryRecord};

/// Logical points a launch of `program` must cover, or zero when the domain the
/// node's region states is already the launch domain.
///
/// Zero constrains nothing rather than stating an empty launch. Widening a
/// program whose guards do bound its effects would fire invocations past the
/// domain it stores into, which writes past the value it declares rather than
/// leaving a lane idle.
pub(crate) fn required_coverage(program: &Program) -> u64 {
    if !vyre_foundation::launch_covers_full_input_span(program) {
        return 0;
    }
    u64::from(vyre_foundation::admitted_logical_span(
        program,
        declared_span(program),
    ))
}

/// Widest logical element count `program` declares outside workgroup scope.
///
/// Workgroup scratch is one allocation per group rather than a domain, so its
/// count states nothing about how many invocations run. A declaration whose
/// count is resolved per dispatch contributes zero and leaves the region domain
/// in force.
fn declared_span(program: &Program) -> u32 {
    program
        .buffers()
        .iter()
        .filter(|buffer| {
            buffer.kind() != MemoryKind::Shared && buffer.access() != BufferAccess::Workgroup
        })
        .map(|buffer| buffer.count())
        .max()
        .unwrap_or(0)
}

/// Reject a launch record covering fewer logical points than its program needs.
///
/// Every geometry record an artifact carries is minted through one call site, so
/// refusing here is what makes an under-covering artifact impossible rather
/// than something a device reports as a wrong number. The refusal is
/// [`CompilerFailureKind::InvalidProgram`] because the coverage a phase carries
/// is this crate's own selection: a record that reaches this check and fails it
/// is a defect in schedule derivation, not in the request.
pub(crate) fn admit_coverage(
    record: &GeometryRecord,
    program: &Program,
) -> Result<(), CompileError> {
    let required = required_coverage(program);
    let covered = record
        .logical_coverage
        .iter()
        .copied()
        .fold(1u64, u64::saturating_mul);
    if covered >= required {
        return Ok(());
    }
    Err(failure(
        CompilerFailureKind::InvalidProgram,
        format!("artifact.geometry[{}].logical_coverage", record.node.0),
        format!(
            "the selected launch covers {covered} logical points and the program states {required}"
        ),
        "derive the phase coverage from the input span the program's atomics, subgroup \
         collectives and workgroup-scoped buffers depend on",
    ))
}

// Inline: `launch_span` is `pub(crate)`, so no integration test can name either
// the derivation or the refusal directly.
#[cfg(test)]
mod tests {
    use super::*;

    use vyre_foundation::ir::{BufferAccess, BufferDecl, DataType, Expr, Node, Program};

    /// An atomic reduction of `count` elements into a one-element accumulator.
    fn atomic_sum(count: u32) -> Program {
        Program::wrapped(
            vec![
                BufferDecl::storage("sum", 0, BufferAccess::ReadWrite, DataType::U32).with_count(1),
                BufferDecl::read("values", 1, DataType::U32).with_count(count),
            ],
            [32, 1, 1],
            vec![
                Node::let_bind("idx", Expr::gid_x()),
                Node::if_then(
                    Expr::lt(Expr::var("idx"), Expr::u32(count)),
                    vec![Node::let_bind(
                        "prior",
                        Expr::atomic_add(
                            "sum",
                            Expr::u32(0),
                            Expr::load("values", Expr::var("idx")),
                        ),
                    )],
                ),
            ],
        )
    }

    /// A guarded copy of `count` elements, with no construct that makes the
    /// result depend on how many invocations ran.
    fn guarded_copy(count: u32) -> Program {
        Program::wrapped(
            vec![
                BufferDecl::read("src", 0, DataType::U32).with_count(count),
                BufferDecl::output("dst", 1, DataType::U32).with_count(count),
            ],
            [32, 1, 1],
            vec![
                Node::let_bind("idx", Expr::gid_x()),
                Node::if_then(
                    Expr::lt(Expr::var("idx"), Expr::u32(count)),
                    vec![Node::store(
                        "dst",
                        Expr::var("idx"),
                        Expr::load("src", Expr::var("idx")),
                    )],
                ),
            ],
        )
    }

    /// WHY: the accumulator is one element and the input is the domain, so a
    /// coverage read off the accumulator leaves every element but one lane's
    /// worth unreduced. The required span has to be the input span, whatever the
    /// narrowest value the program declares.
    #[test]
    fn an_atomic_reduction_requires_its_whole_input_span() {
        assert_eq!(required_coverage(&atomic_sum(4096)), 4096);
        assert_eq!(required_coverage(&atomic_sum(17)), 17);
    }

    /// WHY: a program the guards bound has a domain the region already states,
    /// and widening it would store past the value it declares. This analysis
    /// constrains such a program with nothing rather than with a floor.
    #[test]
    fn a_guarded_program_leaves_its_region_domain_in_force() {
        assert_eq!(required_coverage(&guarded_copy(4096)), 0);
    }

    /// WHY: this is the choke point. A record whose coverage is below the span
    /// its program needs has to be refused where it is minted, because the only
    /// other place the shortfall shows up is a wrong number a device computed.
    #[test]
    fn a_launch_below_the_required_span_is_refused_at_the_record() {
        let record = crate::geometry_fixtures::geometry(3, 0, [32, 1, 1]);
        let covered = record
            .logical_coverage
            .iter()
            .copied()
            .fold(1u64, u64::saturating_mul);
        let covered = u32::try_from(covered).expect("the fixture coverage fits a declared count");
        let wider = covered.saturating_mul(8);

        admit_coverage(&record, &atomic_sum(covered))
            .expect("a launch covering the whole input span is admitted");
        admit_coverage(&record, &guarded_copy(wider))
            .expect("a guarded program is admitted at its region domain");

        let error = admit_coverage(&record, &atomic_sum(wider))
            .expect_err("a launch below the required span must be refused");
        assert_eq!(
            error
                .diagnostic
                .location
                .as_ref()
                .and_then(|location| location.path.clone()),
            Some("artifact.geometry[3].logical_coverage".to_string()),
            "the refusal must name the coverage field that fell short"
        );
    }
}
