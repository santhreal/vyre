//! A shared buffer holds zero when a kernel body starts.
//!
//! The reference evaluator allocates workgroup memory zeroed and re-zeros it
//! between runs, and the SPIR-V writer requests
//! `ZeroInitializeWorkgroupMemoryMode::Polyfill`. PTX `.shared` storage keeps
//! whatever the previous CTA on that SM left behind, so without an entry
//! prologue a program that reads an element it has not written returns a value
//! that depends on which kernel ran before it. Conformance compares every
//! operation against the reference, so that divergence reaches any operation
//! whose declared workgroup extent exceeds the part a case writes.
//!
//! The case list is read from `vyre_lower::emit_adversarial_corpus` at run
//! time, so a new descriptor that declares a shared binding is checked without
//! editing this file.
//!
//! What this does not catch: a body that reads a shared element written by
//! another lane without a barrier between the write and the read. That is a
//! data race, and the reference reports it through race exploration.

use vyre_foundation::ir::DataType;
use vyre_lower::descriptor_builder::{body, descriptor, shared_rw};
use vyre_lower::{emit_adversarial_corpus, KernelDescriptor};

/// `.shared .align 4 .b8 <symbol>[<bytes>];`
fn shared_declarations(ptx: &str) -> Vec<(String, u32)> {
    ptx.lines()
        .filter_map(|line| {
            let rest = line.trim().strip_prefix(".shared .align 4 .b8 ")?;
            let (symbol, rest) = rest.split_once('[')?;
            let bytes = rest.strip_suffix("];")?;
            Some((symbol.to_string(), bytes.parse().expect("byte length")))
        })
        .collect()
}

/// Text from the zero prologue through the barrier that closes it.
fn zero_prologue(ptx: &str) -> Option<&str> {
    let start = ptx.find("    // Workgroup memory holds zero at entry.\n")?;
    let tail = &ptx[start..];
    let end = tail.find("    bar.sync 0;\n")?;
    Some(&tail[..end])
}

/// Offset of the first instruction that addresses shared memory outside the
/// prologue's own stores.
fn first_body_shared_access(ptx: &str) -> Option<usize> {
    let after = zero_prologue(ptx).map_or(0, |prologue| {
        ptx.find(prologue).expect("prologue is a slice of ptx") + prologue.len()
    });
    ["ld.shared", "st.shared", "atom.shared", "red.shared"]
        .iter()
        .filter_map(|opcode| ptx[after..].find(opcode).map(|at| after + at))
        .min()
}

fn emit(case: &emit_adversarial_corpus::EmitAdversarialCase) -> String {
    vyre_emit_ptx::emit(&case.descriptor)
        .unwrap_or_else(|error| panic!("case `{}` must emit PTX: {error:?}", case.id))
}

/// Every declared shared byte is stored before the body reads any of it.
#[test]
fn every_declared_shared_buffer_is_zeroed_before_the_body() {
    let mut cases_with_shared_memory = 0usize;
    for case in emit_adversarial_corpus::success_cases() {
        let ptx = emit(&case);
        let declarations = shared_declarations(&ptx);
        if declarations.is_empty() {
            assert!(
                zero_prologue(&ptx).is_none(),
                "Fix: case `{}` declares no shared buffer yet emits a zero prologue.",
                case.id
            );
            continue;
        }
        cases_with_shared_memory += 1;
        let prologue = zero_prologue(&ptx).unwrap_or_else(|| {
            panic!(
                "Fix: case `{}` declares {} shared buffer(s) and emits no zero prologue.",
                case.id,
                declarations.len()
            )
        });
        for (symbol, byte_len) in &declarations {
            assert_eq!(
                byte_len % 4,
                0,
                "Fix: case `{}` declares `{symbol}` at {byte_len} bytes; the word store loop \
                 needs a whole-word extent.",
                case.id
            );
            assert!(
                prologue.contains(&format!(", {symbol};\n")),
                "Fix: case `{}` never takes the address of `{symbol}` in its zero prologue.",
                case.id
            );
            assert!(
                prologue.contains(&format!(", {};\n", byte_len / 4)),
                "Fix: case `{}` does not bound a zero loop at the {} words `{symbol}` declares.",
                case.id,
                byte_len / 4
            );
        }
        assert_eq!(
            prologue.matches("st.shared.u32").count(),
            declarations.len(),
            "Fix: case `{}` zeroes {} buffer(s) and declares {}.",
            case.id,
            prologue.matches("st.shared.u32").count(),
            declarations.len()
        );
    }
    assert!(
        cases_with_shared_memory > 0,
        "Fix: no success case declares a shared buffer, so this contract checked nothing."
    );
}

/// The prologue stores through a whole-CTA cursor, so it covers the buffer
/// whatever the launched block shape is, and it terminates: each lane advances
/// by the CTA lane count and the loop exits once the cursor passes the word
/// count.
#[test]
fn the_zero_loop_strides_by_the_whole_cta_and_terminates() {
    for case in emit_adversarial_corpus::success_cases() {
        let ptx = emit(&case);
        let Some(prologue) = zero_prologue(&ptx) else {
            continue;
        };
        assert!(
            prologue.contains("mul.lo.u32    %r"),
            "Fix: case `{}` does not derive a lane count for its zero loop.",
            case.id
        );
        let declarations = shared_declarations(&ptx);
        assert_eq!(
            prologue.matches("setp.ge.u32").count(),
            declarations.len(),
            "Fix: case `{}` must bound one zero loop per declared shared buffer.",
            case.id
        );
        assert_eq!(
            prologue.matches("bra $L_zero_shared").count(),
            declarations.len() * 2,
            "Fix: case `{}` must both exit and repeat each zero loop.",
            case.id
        );
    }
}

/// A lane that left the kernel can never arrive at the prologue barrier, so a
/// descriptor with shared memory must take the all-lanes-live entry.
#[test]
fn a_kernel_with_shared_memory_keeps_every_lane_live_through_the_prologue() {
    for case in emit_adversarial_corpus::success_cases() {
        let ptx = emit(&case);
        if shared_declarations(&ptx).is_empty() {
            continue;
        }
        let barrier = ptx.find("    bar.sync 0;\n").expect("prologue barrier");
        assert!(
            !ptx[..barrier].contains("bra $L_exit"),
            "Fix: case `{}` exits lanes before the zero prologue's barrier.",
            case.id
        );
    }
}

/// The barrier is what makes the zeroed bytes visible to the lanes that did
/// not write them.
#[test]
fn the_body_reads_shared_memory_only_after_the_prologue_barrier() {
    for case in emit_adversarial_corpus::success_cases() {
        let ptx = emit(&case);
        if shared_declarations(&ptx).is_empty() {
            continue;
        }
        let barrier = ptx.find("    bar.sync 0;\n").expect("prologue barrier");
        let Some(access) = first_body_shared_access(&ptx) else {
            continue;
        };
        assert!(
            access > barrier,
            "Fix: case `{}` addresses shared memory before the zero prologue's barrier.",
            case.id
        );
    }
}

/// A shared extent that is not a whole number of words.
///
/// `st.shared.u32` covers four bytes, so a buffer whose declared extent ends
/// mid-word leaves its last bytes holding whatever the previous CTA wrote
/// unless the declaration itself is padded out to a word. No corpus case
/// declares a sub-word buffer, so this builds one.
fn sub_word_shared_buffer(count: u32) -> KernelDescriptor {
    descriptor("sub_word_shared")
        .slot(shared_rw(0, DataType::U8, count, "s"))
        .dispatch(64, 1, 1)
        .body(body())
        .build()
}

#[test]
fn a_sub_word_shared_extent_is_declared_and_zeroed_at_word_granularity() {
    for (count, expected_bytes) in [(1u32, 4u32), (5, 8), (7, 8), (8, 8), (9, 12)] {
        let ptx = vyre_emit_ptx::emit(&sub_word_shared_buffer(count))
            .unwrap_or_else(|error| panic!("{count}-byte shared buffer must emit: {error:?}"));
        assert_eq!(
            shared_declarations(&ptx),
            vec![("shared_buf_0".to_string(), expected_bytes)],
            "Fix: a {count}-byte shared buffer must be declared at {expected_bytes} bytes so \
             the word store loop covers it."
        );
        let prologue = zero_prologue(&ptx).expect("zero prologue");
        assert!(
            prologue.contains(&format!(", {};\n", expected_bytes / 4)),
            "Fix: a {count}-byte shared buffer must zero {} word(s).",
            expected_bytes / 4
        );
    }
}
