//! Operand classification contracts.

use vyre_lower::operand_class::{classify_operand, operand_is_result_reference, OperandClass};
use vyre_lower::KernelOpKind;

#[test]
fn operand_classifier_separates_indices_from_result_ids() {
    assert!(!operand_is_result_reference(&KernelOpKind::Literal, 0));
    assert!(!operand_is_result_reference(&KernelOpKind::LoadGlobal, 0));
    assert!(operand_is_result_reference(&KernelOpKind::LoadGlobal, 1));
    assert!(!operand_is_result_reference(
        &KernelOpKind::StructuredForLoop {
            loop_var: "i".into(),
        },
        2,
    ));
    assert!(operand_is_result_reference(
        &KernelOpKind::StructuredForLoop {
            loop_var: "i".into(),
        },
        1,
    ));
    assert!(operand_is_result_reference(
        &KernelOpKind::AsyncStore { tag: "copy".into() },
        2,
    ));
    assert!(!operand_is_result_reference(
        &KernelOpKind::IndirectDispatch { count_offset: 0 },
        0,
    ));
}

/// Two tables once answered this question and disagreed: one treated an
/// operand past a structured op's contract as metadata, the owner kept
/// it as an SSA reference. The owner's answer is the one that cannot
/// miscompile, because a use that is not counted makes a live value look
/// dead to elimination and hoisting.
#[test]
fn out_of_contract_operands_stay_result_references() {
    let loop_kind = KernelOpKind::StructuredForLoop {
        loop_var: "i".into(),
    };
    assert_eq!(classify_operand(&loop_kind, 2), OperandClass::ChildBodyIdx);
    assert_eq!(classify_operand(&loop_kind, 3), OperandClass::ResultRef);
    assert!(operand_is_result_reference(&loop_kind, 3));

    let carrier = KernelOpKind::LoopCarrier { name: "acc".into() };
    assert!(operand_is_result_reference(&carrier, 0));
    assert!(operand_is_result_reference(&carrier, 1));
}
