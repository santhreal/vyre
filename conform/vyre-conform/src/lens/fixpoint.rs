//! Which pair of buffers carries a fixpoint from one iteration to the next.
//!
//! A convergent op declares no such pair. It is inferred from the buffer table:
//! the last read-write buffer is `next`, and `current` is the read-only buffer
//! of matching element count that scores best against it.

use vyre_foundation::ir::{BufferAccess, Program};

/// Infer the `(current, next, count)` fixpoint triple of `program`.
pub fn infer_fixpoint_buffers(program: &Program) -> Result<(&str, &str, u32), String> {
    let ro_buffers: Vec<_> = program
        .buffers()
        .iter()
        .filter(|d| d.access() == BufferAccess::ReadOnly)
        .collect();
    let rw_buffers: Vec<_> = program
        .buffers()
        .iter()
        .filter(|d| d.access() == BufferAccess::ReadWrite)
        .collect();

    let next = rw_buffers
        .last()
        .ok_or_else(|| "no ReadWrite buffer found for fixpoint next".to_string())?
        .name();

    let next_count = rw_buffers
        .last()
        .ok_or_else(|| "no ReadWrite buffer found for fixpoint next".to_string())?
        .count();

    let current_decl = ro_buffers
        .iter()
        .copied()
        .filter(|decl| decl.count() == next_count)
        .min_by_key(|decl| current_score(decl.name(), next))
        .ok_or_else(|| {
            format!("no ReadOnly fixpoint current buffer matches `{next}` count={next_count}")
        })?;
    let current = current_decl.name();
    let current_count = current_decl.count();

    if current_count != next_count {
        return Err(format!(
            "fixpoint buffers `{current}` (count={current_count}) and `{next}` (count={next_count}) must match",
        ));
    }

    Ok((current, next, current_count))
}

/// Rank a candidate current buffer against the selected fixpoint next buffer.
#[must_use]
pub fn current_score(current: &str, next: &str) -> u8 {
    if let Some(expected) = next.strip_suffix("out").map(|prefix| format!("{prefix}in")) {
        if current == expected {
            return 0;
        }
    }
    let expected_current = next.replace("next", "current");
    if expected_current != next && current == expected_current {
        return 0;
    }
    if current.contains("current") || current.contains("frontier") || current.ends_with("in") {
        return 1;
    }
    if current.contains("tag") || current.contains("kind") || current.contains("offset") {
        return 8;
    }
    4
}

#[cfg(test)]
mod tests {
    use super::*;
    use vyre_foundation::ir::{BufferDecl, DataType};

    #[test]
    fn infer_fixpoint_buffers_rejects_no_rw() {
        let program = Program::wrapped(
            vec![BufferDecl::storage("x", 0, BufferAccess::ReadOnly, DataType::U32).with_count(1)],
            [1, 1, 1],
            vec![],
        );
        assert!(infer_fixpoint_buffers(&program).is_err());
    }

    #[test]
    fn infer_fixpoint_buffers_matches_in_out_pair() {
        // Simulate the buffer layout of flows_to / sanitized_by.
        let program = Program::wrapped(
            vec![
                BufferDecl::storage("pg_nodes", 0, BufferAccess::ReadOnly, DataType::U32)
                    .with_count(4),
                BufferDecl::storage("fin", 1, BufferAccess::ReadOnly, DataType::U32).with_count(1),
                BufferDecl::storage("fout", 2, BufferAccess::ReadWrite, DataType::U32)
                    .with_count(1),
            ],
            [1, 1, 1],
            vec![],
        );
        let (current, next, count) = infer_fixpoint_buffers(&program).expect("Fix: inference");
        assert_eq!(current, "fin");
        assert_eq!(next, "fout");
        assert_eq!(count, 1);
    }
}
