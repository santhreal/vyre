//! Test: analysis fixture corpuses.
use vyre_foundation::ir::DataType;
use vyre_lower::analyses::vec_pack;
use vyre_lower::descriptor_builder::{body, descriptor, fixed_global_ro, lit, load_global};
use vyre_lower::LiteralValue;

#[test]
fn a16_vec_pack_corpus_detects_adjacent_load_chain() {
    let desc = descriptor("a16_vec_pack_fixture")
        .slot(fixed_global_ro(0, DataType::U32, 4096, "input"))
        .dispatch(256, 1, 1)
        .body(
            body()
                .literals([
                    LiteralValue::U32(64),
                    LiteralValue::U32(65),
                    LiteralValue::U32(66),
                    LiteralValue::U32(67),
                ])
                .op(lit(0, 1))
                .op(lit(1, 2))
                .op(lit(2, 3))
                .op(lit(3, 4))
                .op(load_global(0, 1, 10))
                .op(load_global(0, 2, 11))
                .op(load_global(0, 3, 12))
                .op(load_global(0, 4, 13)),
        )
        .build();

    let report = vec_pack::analyze(&desc);
    assert!(report.has_chains());
    assert_eq!(report.chains.len(), 1);
    assert_eq!(report.chains[0].slot, 0);
    assert_eq!(report.chains[0].start_index, 64);
    assert_eq!(report.chains[0].pack_width(), 4);
    assert_eq!(report.total_ops_eliminated, 3);
}
