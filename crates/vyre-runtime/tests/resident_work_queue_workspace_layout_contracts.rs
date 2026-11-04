//! Contracts for `vyre_runtime::resident_work_queue::workspace_layout`.
//!
//! Every item under test is public API, so the suite reaches the crate the way
//! a consumer does.

use vyre_runtime::resident_work_queue::workspace_layout::{
    build_workspace_regions, first_workspace_region, next_record_workspace_region,
    next_workspace_region, workspace_record_words, ResidentWorkspaceLayoutError,
    ResidentWorkspaceRegion, ResidentWorkspaceRegionSpec,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Region {
    Header,
    Rows,
    Work,
}

#[test]
fn generated_workspace_regions_are_contiguous_for_many_capacities() {
    for rows in [1_u32, 2, 7, 8, 31, 32, 1024] {
        for work in [1_u32, 3, 64, 4096] {
            let header = first_workspace_region(Region::Header, 16, 1, 16);
            let rows = next_record_workspace_region(header, Region::Rows, 5, rows)
                .expect("Fix: generated row region should fit");
            let work = next_record_workspace_region(rows, Region::Work, 2, work)
                .expect("Fix: generated work region should fit");

            assert_eq!(header.offset_words, 0);
            assert_eq!(rows.offset_words, header.end_words().unwrap());
            assert_eq!(work.offset_words, rows.end_words().unwrap());
        }
    }
}

#[test]
fn record_region_word_overflow_is_reported_before_offset_overflow() {
    let header = first_workspace_region(Region::Header, 16, 1, 16);

    assert_eq!(
        workspace_record_words(u32::MAX, 2),
        None,
        "record sizing must catch multiplication overflow"
    );
    assert_eq!(
        next_record_workspace_region(header, Region::Rows, u32::MAX, 2),
        None,
        "record-backed layout must reject overflowing record arenas"
    );
}

#[test]
fn next_region_rejects_offset_overflow() {
    let previous = ResidentWorkspaceRegion {
        id: Region::Header,
        offset_words: u32::MAX,
        words: 1,
        record_words: 1,
        capacity_records: 1,
    };

    assert_eq!(
        next_workspace_region(previous, Region::Rows, 1, 1, 1),
        None,
        "contiguous layout must reject overflowing end offsets"
    );
}

#[test]
fn bulk_workspace_region_builder_matches_chained_layout_for_generated_capacities() {
    for rows in [1_u32, 2, 7, 8, 31, 32, 1024] {
        for work in [1_u32, 3, 64, 4096] {
            let specs = [
                ResidentWorkspaceRegionSpec::fixed(Region::Header, 16, 1, 16),
                ResidentWorkspaceRegionSpec::record(Region::Rows, 5, rows),
                ResidentWorkspaceRegionSpec::record(Region::Work, 2, work),
            ];
            let bulk = build_workspace_regions(&specs)
                .expect("Fix: generated bulk workspace layout should fit");

            let header = first_workspace_region(Region::Header, 16, 1, 16);
            let rows = next_record_workspace_region(header, Region::Rows, 5, rows)
                .expect("Fix: generated row region should fit");
            let work = next_record_workspace_region(rows, Region::Work, 2, work)
                .expect("Fix: generated work region should fit");

            assert_eq!(bulk, vec![header, rows, work]);
        }
    }
}

#[test]
fn bulk_workspace_region_builder_reports_record_and_offset_overflow_separately() {
    let record = [ResidentWorkspaceRegionSpec::record(
        Region::Rows,
        u32::MAX,
        2,
    )];
    assert_eq!(
        build_workspace_regions(&record),
        Err(ResidentWorkspaceLayoutError::RecordWordsOverflow {
            region: Region::Rows
        })
    );

    let offset = [
        ResidentWorkspaceRegionSpec::fixed(Region::Header, u32::MAX, 1, u32::MAX),
        ResidentWorkspaceRegionSpec::fixed(Region::Rows, 1, 1, 1),
    ];
    assert_eq!(
        build_workspace_regions(&offset),
        Err(ResidentWorkspaceLayoutError::OffsetOverflow {
            region: Region::Rows
        })
    );
}
