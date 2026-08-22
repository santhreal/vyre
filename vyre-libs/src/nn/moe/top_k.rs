//! Top-K selection: indices of the K largest elements.
//!
//! Category-A composition. Sequential implementation for the reference
//! oracle; parallel bitonic top-k is not built yet.

use super::topk_selection::{
    copy_top_k_indices, init_top_k_slots, insert_top_k_candidate, BEST_IDXS, BEST_VALS,
};
use vyre_foundation::composition::{trap_program, wrap_anonymous_region};
use vyre_foundation::ir::{BufferAccess, BufferDecl, DataType, Expr, Node, Program};

/// Build a Program that finds the indices of the `k` largest elements in `input`.
/// `input`: `n`, `output_indices`: `k`.
///
/// Uses a sequential insertion-sort-into-slots algorithm: maintains `k` best
/// (value, index) pairs in descending order, updating on every new element.
#[must_use]
pub fn top_k(input: &str, output_indices: &str, n: u32, k: u32) -> Program {
    if k == 0 {
        return trap_program(
            "vyre-libs::nn::top_k",
            Some((output_indices, DataType::U32)),
            "Fix: top_k requires k > 0 so the selection scratch has at least one slot.".to_string(),
        );
    }
    let mut body = init_top_k_slots(k);

    // For each input element i:
    //   val = input[i]
    //   Scan j=0..k: if val > best_vals[j], shift j..k-1 down and insert at j
    body.push(Node::loop_for(
        "i",
        Expr::u32(0),
        Expr::u32(n),
        vec![
            Node::let_bind("val", Expr::load(input, Expr::var("i"))),
            Node::let_bind("idx", Expr::var("i")),
            Node::Block(insert_top_k_candidate(
                k,
                Expr::var("val"),
                Expr::var("idx"),
            )),
        ],
    ));

    body.extend(copy_top_k_indices(output_indices, k));

    Program::wrapped(
        vec![
            BufferDecl::storage(input, 0, BufferAccess::ReadOnly, DataType::F32).with_count(n),
            BufferDecl::output(output_indices, 1, DataType::U32).with_count(k),
            // Internal scratch buffers
            BufferDecl::read_write(BEST_VALS, 2, DataType::F32).with_count(k),
            BufferDecl::read_write(BEST_IDXS, 3, DataType::U32).with_count(k),
        ],
        [1, 1, 1],
        // The selection scan is serial over `n` and keeps its running best in
        // read-write scratch, so one workgroup owns it. The grid a backend
        // derives from the output length would give every workgroup its own
        // pass over the same scratch and the same output words.
        vec![wrap_anonymous_region(
            "vyre-libs::nn::top_k",
            vec![Node::if_then(Expr::is_first_workgroup(), body)],
        )],
    )
}

#[cfg(test)]
mod tests {
    use super::super::topk_selection::u32_from_bytes;
    use super::*;
    use crate::fixture_bytes::eval_bytes;
    use crate::fixture_bytes::f32_bytes;

    #[test]
    fn top_k_descending_input() {
        let scores: Vec<f32> = (1..=8).map(|i| i as f32).collect();
        let program = top_k("input", "output", 8, 2);
        let outputs = eval_bytes(
            "top_k",
            &program,
            vec![f32_bytes(&scores), vec![0u8; 2 * 4], vec![0u8; 2 * 4]],
        );
        let indices = u32_from_bytes(&outputs[0]);
        assert_eq!(indices[0], 7); // max = 8.0 at index 7
        assert_eq!(indices[1], 6); // second = 7.0 at index 6
    }

    #[test]
    fn top_k_ascending_input() {
        let scores: Vec<f32> = (1..=8).rev().map(|i| i as f32).collect();
        let program = top_k("input", "output", 8, 2);
        let outputs = eval_bytes(
            "top_k",
            &program,
            vec![f32_bytes(&scores), vec![0u8; 2 * 4], vec![0u8; 2 * 4]],
        );
        let indices = u32_from_bytes(&outputs[0]);
        assert_eq!(indices[0], 0); // max = 8.0 at index 0
        assert_eq!(indices[1], 1); // second = 7.0 at index 1
    }

    #[test]
    fn top_k_with_duplicates() {
        let scores = vec![3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0];
        let program = top_k("input", "output", 8, 3);
        let outputs = eval_bytes(
            "top_k",
            &program,
            vec![f32_bytes(&scores), vec![0u8; 3 * 4], vec![0u8; 3 * 4]],
        );
        let indices = u32_from_bytes(&outputs[0]);
        // 9.0(5), 6.0(7), 5.0(4), 4.0(2), 3.0(0), 2.0(6), 1.0(1), 1.0(3)
        assert_eq!(indices[0], 5);
        assert_eq!(indices[1], 7);
        assert_eq!(indices[2], 4);
    }
}

const EXPECTED_TOP_K_INDICES_BYTES: [u8; 8] = [0x07, 0x00, 0x00, 0x00, 0x06, 0x00, 0x00, 0x00];
const EXPECTED_TOP_K_VALUES_BYTES: [u8; 8] = [0x00, 0x00, 0x00, 0x41, 0x00, 0x00, 0xE0, 0x40];

inventory::submit! {
    vyre_foundation::operation::OperationRegistration::library(
        "vyre-libs::nn::top_k",
        || top_k("input", "output", 8, 2),
        Some(|| {
            let scores: [f32; 8] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
            let input_bytes = vyre_primitives::wire::pack_f32_slice(&scores);
            vec![vec![
                input_bytes,
                vec![0u8; 4 * 2],
                vec![0u8; 4 * 2],
            ]]
        }),
        Some(|| {
            vec![vec![
                EXPECTED_TOP_K_INDICES_BYTES.to_vec(),
                EXPECTED_TOP_K_VALUES_BYTES.to_vec(),
                EXPECTED_TOP_K_INDICES_BYTES.to_vec(),
            ]]
        }),
    )
    .with_category("nn")
}
