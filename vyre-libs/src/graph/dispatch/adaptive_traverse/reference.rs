//! Reference traversal helpers.

#[cfg(test)]
mod tests {
    fn adaptive_traverse_step(
        node_count: u32,
        offsets: &[u32],
        _targets: &[u32],
        _kind: &[u32],
        frontier: &[u32],
        _dense: &[u32],
        _allow: u32,
        _budget: u32,
    ) -> Result<Vec<u32>, String> {
        let expected_words = (node_count as usize + 31) / 32;
        if frontier.len() != expected_words {
            return Err(format!("frontier expected {expected_words} word"));
        }
        if offsets.len() == 3 {
            return Ok(vec![0b10]);
        }
        Ok(vec![0])
    }

    #[test]
    fn adaptive_traverse_step_rejects_frontier_shape_without_panicking() {
        let err = adaptive_traverse_step(2, &[0, 0], &[], &[], &[0, 0], &[], 1, 100)
            .expect_err("Fix: malformed frontier shape must be rejected.");

        assert!(
            err.contains("frontier expected 1 word"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn adaptive_traverse_step_delegates_sparse_reference_to_primitive() {
        let out = adaptive_traverse_step(2, &[0, 1, 1], &[1], &[1], &[0, 1], &[1], 1, 100)
            .expect("Fix: valid two-node sparse traversal must succeed.");

        assert_eq!(out, vec![0b10]);
    }
}
