//! Every global store carries its own buffer's bounds predicate.
//!
//! `memory::clamp_index_to_buffer_length` folds an out-of-range element index
//! to 0 before forming the address. That is sound for a load, whose value is
//! discarded, and it is corruption for a store: element 0 receives whatever
//! value the out-of-range lane carried. The kernel-wide exit a kernel without
//! shared memory emits does not cover it, because that exit compares the global
//! id against the dispatch element count, which is the longest buffer's length,
//! and says nothing about a shorter buffer in the same program. The reference
//! interpreter discards the store, so an unpredicated one is a parity defect.
//!
//! The population is the shared adversarial success corpus, read at run time,
//! so a corpus case added later is checked here without being named.

use vyre_emit_ptx::TRAP_SIDECAR_SYMBOL;
use vyre_lower::descriptor_builder::{body, descriptor, effect, lit};
use vyre_lower::{KernelOpKind, LiteralValue};

/// Registers holding the trap sidecar base, as `mov.u64 %rdN, <symbol>;`.
///
/// The sidecar is a fixed-length module-scope array written at compile-time
/// word offsets under a CAS that admits one lane, so its stores are in range by
/// construction and carry no index to bound.
fn sidecar_bases(ptx: &str) -> Vec<String> {
    ptx.lines()
        .filter_map(|line| {
            let rest = line.trim().strip_prefix("mov.u64")?;
            let (register, symbol) = rest.trim_end_matches(';').split_once(',')?;
            (symbol.trim() == TRAP_SIDECAR_SYMBOL).then(|| register.trim().to_owned())
        })
        .collect()
}

/// Global-store lines that carry no `@` predicate and do not write the trap
/// sidecar.
fn unpredicated_global_stores(ptx: &str) -> Vec<&str> {
    let bases = sidecar_bases(ptx);
    ptx.lines()
        .map(str::trim)
        .filter(|line| line.starts_with("st.global"))
        .filter(|line| {
            !bases.iter().any(|base| {
                line.contains(&format!("[{base} ")) || line.contains(&format!("[{base}]"))
            })
        })
        .collect()
}

#[test]
fn the_shared_success_corpus_emits_no_unpredicated_global_store() {
    let cases = vyre_lower::emit_adversarial_corpus::success_cases();
    assert!(
        !cases.is_empty(),
        "Fix: the shared success corpus is empty, so this contract checks nothing."
    );
    for case in cases {
        let descriptor = vyre_lower::verify_descriptor(&case.descriptor)
            .unwrap_or_else(|error| panic!("{}: corpus case must verify: {error:?}", case.id));
        let ptx = vyre_emit_ptx::emit(&descriptor)
            .unwrap_or_else(|error| panic!("{}: corpus case must emit: {error:?}", case.id));
        let unpredicated = unpredicated_global_stores(&ptx);
        assert!(
            unpredicated.is_empty(),
            "{}: {} global store(s) carry no bounds predicate: {unpredicated:?}. \
             Fix: issue the store through `store_guard_for_index`.",
            case.id,
            unpredicated.len()
        );
    }
}

#[test]
fn the_trap_sidecar_exclusion_names_a_store_that_exists() {
    // The exclusion above is only sound while the sidecar path really is the
    // one unpredicated global store. If the trap path stops emitting one, the
    // exclusion is dead and hides whatever takes its place.
    let kernel = descriptor("trap_sidecar")
        .body(
            body()
                .ops([
                    lit(0, 0),
                    effect(
                        KernelOpKind::Trap {
                            tag: "bounds".into(),
                        },
                        [0],
                    ),
                ])
                .literal(LiteralValue::U32(0)),
        )
        .build();
    let ptx = vyre_emit_ptx::emit(&kernel).expect("trap descriptor must emit");
    let bases = sidecar_bases(&ptx);
    assert!(
        !bases.is_empty(),
        "the trap path must load the sidecar base with `mov.u64`"
    );
    let sidecar_stores = ptx
        .lines()
        .map(str::trim)
        .filter(|line| line.starts_with("st.global"))
        .filter(|line| bases.iter().any(|base| line.contains(&format!("[{base} "))))
        .count();
    assert!(
        sidecar_stores > 0,
        "the trap path must write the sidecar, otherwise the exclusion covers nothing"
    );
    assert!(
        unpredicated_global_stores(&ptx).is_empty(),
        "only the sidecar may store unpredicated: {:?}",
        unpredicated_global_stores(&ptx)
    );
}
