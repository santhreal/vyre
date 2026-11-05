//! Pattern oracles and wire access for this crate's tests.

pub(crate) use vyre_primitives::wire::decode_u32_le_bytes_all as decode_u32_words;

#[cfg(feature = "pattern")]
pub(crate) fn reference_dedup_regions(
    regions: Vec<vyre_libs_pattern::pattern::RegionTriple>,
) -> Vec<vyre_libs_pattern::pattern::RegionTriple> {
    let input: Vec<(u32, u32, u32)> = regions.iter().map(|r| (r.pid, r.start, r.end)).collect();
    let deduped = vyre_reference::composition_witness::dedup_regions_witness(input);
    deduped
        .into_iter()
        .map(|(pid, start, end)| vyre_libs_pattern::pattern::RegionTriple::new(pid, start, end))
        .collect()
}

#[cfg(feature = "pattern")]
pub(crate) fn reference_dedup_regions_in_place(
    regions: &mut Vec<vyre_libs_pattern::pattern::RegionTriple>,
) {
    let deduped = reference_dedup_regions(std::mem::take(regions));
    *regions = deduped;
}
