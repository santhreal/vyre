//! Output extent coverage and workgroup geometry contracts for Quest paging passes.

#![forbid(unsafe_code)]

use crate::wire_words;
use wire_words::{f32_bytes, f32_words_of, u32_bytes, words_from_bytes};

use vyre_libs::nn::quest_paging_passes::{quest_score_pages, quest_select_top_k, quest_zero_fill};
use vyre_reference::value::Value;

/// Standalone quest page-scoring must write every declared page in the scores buffer.
#[test]
fn quest_score_pages_writes_full_declared_extent() {
    for (num_pages, d_head) in [(4, 2), (8, 2), (1, 4), (12, 3)] {
        let program = quest_score_pages("q", "meta", "scores", num_pages, d_head);
        assert_eq!(
            program.workgroup_size(),
            [256, 1, 1],
            "Fix: quest_score_pages must declare a 256-wide workgroup to match its strided body"
        );

        let declared_pages = program
            .buffers()
            .iter()
            .find(|b| b.name() == "scores")
            .expect("scores buffer must exist")
            .count();
        let declared_head_dim = program
            .buffers()
            .iter()
            .find(|b| b.name() == "q")
            .expect("query buffer must exist")
            .count();

        let query: Vec<f32> = (0..declared_head_dim)
            .map(|i| (i + 1) as f32)
            .collect();
        let page_metadata: Vec<f32> = (0..declared_pages * declared_head_dim)
            .map(|i| ((i % 7) + 1) as f32)
            .collect();
        let initial_scores = vec![0.0f32; declared_pages as usize];

        let outputs = vyre_reference::reference_eval(
            &program,
            &[
                Value::from(f32_bytes(&query)),
                Value::from(f32_bytes(&page_metadata)),
                Value::from(f32_bytes(&initial_scores)),
            ],
        )
        .expect("Fix: quest_score_pages program must evaluate");

        let scores = f32_words_of(&outputs[0]);
        assert_eq!(
            scores.len(),
            declared_pages as usize,
            "Fix: returned scores buffer length must match declared page count"
        );

        for p in 0..declared_pages as usize {
            let mut expected_dot = 0.0f32;
            for lane in 0..declared_head_dim as usize {
                expected_dot += query[lane] * page_metadata[p * declared_head_dim as usize + lane];
            }
            assert!(
                expected_dot > 0.0,
                "fixture metadata must produce nonzero dot product"
            );
            assert_eq!(
                scores[p], expected_dot,
                "lane {p} of declared {declared_pages} pages was not computed correctly"
            );
        }
    }
}

/// Standalone quest zero-fill must write zeros across the full declared buffer extent.
#[test]
fn quest_zero_fill_writes_full_declared_extent() {
    for num_pages in [4, 8, 256, 300] {
        let program = quest_zero_fill("io", num_pages);
        assert_eq!(
            program.workgroup_size(),
            [256, 1, 1],
            "Fix: quest_zero_fill must declare a 256-wide workgroup to match its strided body"
        );

        let declared_pages = program
            .buffers()
            .iter()
            .find(|b| b.name() == "io")
            .expect("io buffer must exist")
            .count();

        let initial_io = vec![0xFFFFFFFFu32; declared_pages as usize];
        let outputs = vyre_reference::reference_eval(
            &program,
            &[Value::from(u32_bytes(&initial_io))],
        )
        .expect("Fix: quest_zero_fill program must evaluate");

        let io_queue = words_from_bytes(&outputs[0].to_bytes());
        assert_eq!(
            io_queue.len(),
            declared_pages as usize,
            "Fix: returned io buffer length must match declared page count"
        );
        for (idx, &word) in io_queue.iter().enumerate() {
            assert_eq!(
                word, 0,
                "lane {idx} of declared {declared_pages} pages was not zero-filled"
            );
        }
    }
}

/// Standalone quest select top-k must declare matching workgroup geometry.
#[test]
fn quest_select_top_k_declares_matching_workgroup_geometry() {
    let program = quest_select_top_k("scores", "io", 4, 1, -1.0);
    assert_eq!(
        program.workgroup_size(),
        [256, 1, 1],
        "Fix: quest_select_top_k must declare a 256-wide workgroup matching quest_paging"
    );
}
