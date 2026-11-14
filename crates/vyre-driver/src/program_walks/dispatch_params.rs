//! Dispatch ABI parameter derivation from binding plans.

use vyre_foundation::ir::Program;

use crate::binding::{Binding, BindingRole};

/// Derive the dispatch element count from a binding plan and Program body.
///
/// Atomics and subgroup collectives execute over the logical input span even
/// when they compact into a smaller output. Using the compact output length
/// would suppress participating lanes before the atomic or collective runs.
///
/// Every other program is capped by the domain its own guards admit, so a
/// scatter that moves a small source into a large destination launches over the
/// source. This is the same narrowing target lowering applies, through the same
/// analysis, so a below-admission dispatch and a compiled artifact agree on the
/// span without either caller publishing a grid.
///
/// The widest binding is taken over the buffers the program references, not
/// over every buffer the plan declares. A grid-sync segment carries the whole
/// program's buffer table so one resident resource slice binds to every
/// segment, so the pass that scans a 4096-element block-total buffer declares
/// the 1048576-element input beside it and launched one lane per input element,
/// 256 times its own domain. A buffer no statement of the program names is
/// touched by no lane, so its declared width states nothing about how many
/// lanes this program needs.
#[must_use]
pub fn dispatch_element_count_for_program(program: &Program, bindings: &[Binding]) -> u32 {
    let full_span = vyre_foundation::launch_covers_full_input_span(program);
    let referenced = vyre_foundation::visit::referenced_buffers(program);
    let names = |binding: &Binding| {
        referenced
            .iter()
            .any(|name| name.as_ref() == &*binding.name)
    };
    let count = match dispatch_element_count_over(bindings, full_span, &names) {
        // A plan whose names never meet the program's carries no narrowing, so
        // the whole plan decides it as it did before.
        0 => dispatch_element_count_inner(bindings, full_span),
        narrowed => narrowed,
    };
    vyre_foundation::admitted_logical_span(program, count)
}

fn dispatch_element_count_inner(bindings: &[Binding], force_full_span: bool) -> u32 {
    dispatch_element_count_over(bindings, force_full_span, &|_| true)
}

/// The dispatch width over the bindings `named` admits.
///
/// Returns 0 when `named` admits no binding, which is the one answer the
/// policy below never produces, so a caller can tell an empty selection from a
/// width and decide it on the whole plan instead.
fn dispatch_element_count_over(
    bindings: &[Binding],
    force_full_span: bool,
    named: &dyn Fn(&Binding) -> bool,
) -> u32 {
    // Single pass over bindings: collect every fact the dispatch
    // policy needs (any-shared / max non-shared / max output) in one
    // scan. Previously up to three independent .iter() passes
    // traversed the same slice  -  for launch shapes that carry 60+
    // bindings each pass is real work.
    let mut any_named = false;
    let mut any_shared = false;
    let mut max_non_shared: u32 = 0;
    let mut max_output: u32 = 0;
    for binding in bindings {
        if !named(binding) {
            continue;
        }
        any_named = true;
        if binding.role == BindingRole::Shared {
            any_shared = true;
            continue;
        }
        if binding.element_count > max_non_shared {
            max_non_shared = binding.element_count;
        }
        if matches!(binding.role, BindingRole::Output | BindingRole::InputOutput)
            && binding.element_count > max_output
        {
            max_output = binding.element_count;
        }
    }
    if !any_named {
        return 0;
    }
    if any_shared || force_full_span {
        return max_non_shared.max(1);
    }
    if max_output > 0 {
        return max_output;
    }
    max_non_shared.max(1)
}

/// Build per-buffer element-count parameter words into caller-owned storage
/// with explicit allocation and ABI-width errors.
pub fn try_dispatch_param_words_into(
    bindings: &[Binding],
    element_count: u32,
    words: &mut Vec<u32>,
) -> Result<(), String> {
    let word_len = dispatch_param_word_len_for_bindings(bindings)?;
    reserve_dispatch_param_words(words, word_len)?;
    words.clear();
    words.resize(word_len, 0);
    words[0] = element_count;
    for binding in bindings {
        if binding.role == BindingRole::Shared {
            continue;
        }
        let slot = dispatch_param_word_slot(binding)?;
        words[slot] = if binding.element_count == 0 {
            element_count
        } else {
            binding.element_count
        };
    }
    Ok(())
}

fn dispatch_param_word_len_for_bindings(bindings: &[Binding]) -> Result<usize, String> {
    let mut word_len = dispatch_param_word_len_checked(bindings.len())?;
    for binding in bindings {
        if binding.role == BindingRole::Shared {
            continue;
        }
        let required = dispatch_param_word_slot(binding)?
            .checked_add(1)
            .ok_or_else(|| {
                format!(
                    "dispatch binding slot {} overflows ABI parameter word count. Fix: split the Program before launch-parameter planning.",
                    binding.binding
                )
            })?;
        if required > word_len {
            word_len = required;
        }
    }
    Ok(word_len)
}

fn dispatch_param_word_slot(binding: &Binding) -> Result<usize, String> {
    let slot = usize::try_from(binding.binding).map_err(|error| {
        format!(
            "dispatch binding slot {} does not fit host usize ({error}). Fix: split the Program before launch-parameter planning.",
            binding.binding
        )
    })?;
    slot.checked_add(1).ok_or_else(|| {
        format!(
            "dispatch binding slot {} overflows ABI parameter slot. Fix: split the Program before launch-parameter planning.",
            binding.binding
        )
    })
}

fn dispatch_param_word_len_checked(binding_count: usize) -> Result<usize, String> {
    binding_count.checked_add(1).ok_or_else(|| {
        format!(
            "dispatch binding count {binding_count} overflows ABI parameter word count. Fix: split the Program before launch-parameter planning."
        )
    })
}

fn reserve_dispatch_param_words(words: &mut Vec<u32>, word_len: usize) -> Result<(), String> {
    crate::allocation::try_reserve_vec_to_capacity(words, word_len).map_err(|error| {
        format!(
            "dispatch parameter staging could not reserve {word_len} u32 word(s): {error}. Fix: split the Program before launch-parameter planning."
        )
    })
}

// Inline: `vyre_driver::program_walks` is `pub(crate)`, so no integration test can reach what this
// suite exercises.
#[cfg(test)]
mod tests {
    use super::*;
    use crate::binding::BindingRole;
    use std::sync::Arc;

    fn binding(buffer_index: usize, element_count: u32) -> Binding {
        Binding {
            name: Arc::from("test"),
            binding: u32::try_from(buffer_index).expect("Fix: test binding index fits u32"),
            buffer_index,
            role: BindingRole::Input,
            element_size: 4,
            preferred_alignment: 4,
            element_count,
            static_byte_len: None,
            input_index: Some(0),
            output_index: None,
        }
    }

    /// Subgroup collectives require every logical input lane to participate.
    /// A compact bitmap output cannot define the launch span because the
    /// ballot needs all 32 lanes before lane zero writes one output word.
    #[test]
    fn subgroup_collective_dispatch_uses_full_input_span() {
        let mut output = binding(0, 16);
        output.role = BindingRole::Output;
        output.output_index = Some(0);
        output.input_index = None;
        let input = binding(1, 512);
        let program = vyre_foundation::ir::Program::wrapped(
            Vec::new(),
            [256, 1, 1],
            vec![vyre_foundation::ir::Node::let_bind(
                "mask",
                vyre_foundation::ir::Expr::subgroup_ballot(vyre_foundation::ir::Expr::bool(true)),
            )],
        );

        assert_eq!(
            dispatch_element_count_for_program(&program, &[output, input]),
            512
        );
    }

    // Reproducing test for: dispatch-param-words-silent-empty-return
    // Before fix: `dispatch_param_words` (infallible) existed and returned Vec::new() on error,
    // silently producing a zero-length param buffer. After fix: function is removed; callers
    // must use `try_dispatch_param_words` which returns a Result.
    // Reproducing test for: dispatch-param-words-into-silent-clear
    // Before fix: `dispatch_param_words_into` returned () and silently cleared `words` on error.
    // After fix: it returns `Result<(), String>` so callers can observe and propagate the error.
}
