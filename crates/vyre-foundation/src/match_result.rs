//! Domain-neutral tagged byte ranges shared by scan and source-processing products.

/// A tagged, half-open byte range `[start, end)`.
///
/// `tag` is a producer-chosen 32-bit identifier. A matching dialect can pass a
/// pattern id, a decoder can pass an encoding id, and a source-span producer
/// can pass a node kind. Foundation does not interpret the field.
#[repr(C)]
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ByteRange {
    /// Producer-chosen 32-bit identifier.
    pub tag: u32,
    /// Inclusive byte start offset.
    pub start: u32,
    /// Exclusive byte end offset.
    pub end: u32,
}

impl ByteRange {
    /// Construct a range. Reversed ranges fail loudly because accepting them
    /// corrupts every downstream range predicate.
    #[must_use]
    pub const fn new(tag: u32, start: u32, end: u32) -> Self {
        assert!(
            end >= start,
            "ByteRange::new requires end >= start. Fix: pass half-open byte ranges as [start, end)."
        );
        Self { tag, start, end }
    }

    /// Length of the range in bytes.
    #[must_use]
    pub const fn len(&self) -> u32 {
        self.end - self.start
    }

    /// True when the range has zero length.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.end == self.start
    }

    /// True when `self` contains `other`.
    #[must_use]
    pub const fn contains(&self, other: &ByteRange) -> bool {
        self.start <= other.start && other.end <= self.end
    }

    /// True when `self` ends at or before `other` starts.
    #[must_use]
    pub const fn ends_before(&self, other: &ByteRange) -> bool {
        self.end <= other.start
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    #[test]
    fn construction() {
        let range = ByteRange::new(1, 10, 20);
        assert_eq!(range.tag, 1);
        assert_eq!(range.start, 10);
        assert_eq!(range.end, 20);
    }

    #[test]
    fn ordering() {
        let a = ByteRange::new(0, 5, 10);
        let b = ByteRange::new(0, 15, 20);
        let c = ByteRange::new(1, 0, 5);
        let mut ranges = [c, a, b];
        ranges.sort();
        assert_eq!(ranges[0].start, 5);
        assert_eq!(ranges[1].start, 15);
        assert_eq!(ranges[2].tag, 1);
    }

    #[test]
    fn hash_consistency() {
        let mut set = HashSet::new();
        let range = ByteRange::new(1, 0, 10);
        set.insert(range);
        assert!(set.contains(&ByteRange::new(1, 0, 10)));
        assert!(!set.contains(&ByteRange::new(2, 0, 10)));
    }

    #[test]
    #[should_panic(expected = "ByteRange::new requires end >= start")]
    fn reversed_ranges_fail_at_construction() {
        let _ = ByteRange::new(1, 10, 9);
    }

    #[test]
    fn range_predicates_cover_boundaries() {
        let outer = ByteRange::new(0, 0, 100);
        let inner = ByteRange::new(1, 10, 90);
        let adjacent = ByteRange::new(2, 100, 100);

        assert!(outer.contains(&inner));
        assert!(outer.contains(&outer));
        assert!(!inner.contains(&outer));
        assert!(outer.ends_before(&adjacent));
        assert_eq!(inner.len(), 80);
        assert!(adjacent.is_empty());
    }

    #[test]
    fn layout_matches_packed_triple_abi() {
        assert_eq!(std::mem::size_of::<ByteRange>(), 12);
        assert_eq!(std::mem::align_of::<ByteRange>(), 4);
    }
}
