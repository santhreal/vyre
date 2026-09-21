//! Packing a snapshot into one uploadable slab.
//!
//! A backend takes a single `u32` buffer, so every column is appended to one
//! word vector and the layout records where each one starts and how long it
//! is. The span table is what a kernel indexes with; nothing here reads an
//! e-graph.

/// Contiguous span inside a packed GPU e-graph device image.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct GpuEGraphDeviceSpan {
    offset: usize,
    len: usize,
}

impl GpuEGraphDeviceSpan {
    const fn new(offset: usize, len: usize) -> Self {
        Self { offset, len }
    }

    /// Word offset of the span inside the packed u32 slab.
    #[must_use]
    pub const fn offset(&self) -> usize {
        self.offset
    }

    /// Number of u32 words in the span.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.len
    }

    /// `true` iff the span contains no words.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }

    fn slice<'a>(&self, words: &'a [u32]) -> &'a [u32] {
        &words[self.offset..self.offset + self.len]
    }
}

/// Segment table for a packed GPU e-graph device image.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct GpuEGraphDeviceLayout {
    pub(super) row_count: usize,
    pub(super) child_count: usize,
    pub(super) eclass_group_count: usize,
    pub(super) row_eclass_ids: GpuEGraphDeviceSpan,
    pub(super) row_language_op_ids: GpuEGraphDeviceSpan,
    pub(super) row_children_offsets: GpuEGraphDeviceSpan,
    pub(super) row_children_lens: GpuEGraphDeviceSpan,
    pub(super) row_signatures: GpuEGraphDeviceSpan,
    pub(super) children: GpuEGraphDeviceSpan,
    pub(super) group_eclass_ids: GpuEGraphDeviceSpan,
    pub(super) group_offsets: GpuEGraphDeviceSpan,
    pub(super) group_rows: GpuEGraphDeviceSpan,
}

impl GpuEGraphDeviceLayout {
    /// Number of snapshot rows packed into the device image.
    #[must_use]
    pub const fn row_count(&self) -> usize {
        self.row_count
    }

    /// Number of child references packed into the device image.
    #[must_use]
    pub const fn child_count(&self) -> usize {
        self.child_count
    }

    /// Number of e-class row groups in the device image.
    #[must_use]
    pub const fn eclass_group_count(&self) -> usize {
        self.eclass_group_count
    }

    /// Span containing one e-class id per snapshot row.
    #[must_use]
    pub const fn row_eclass_ids(&self) -> GpuEGraphDeviceSpan {
        self.row_eclass_ids
    }

    /// Span containing one language op id per snapshot row.
    #[must_use]
    pub const fn row_language_op_ids(&self) -> GpuEGraphDeviceSpan {
        self.row_language_op_ids
    }

    /// Span containing one child-column offset per snapshot row.
    #[must_use]
    pub const fn row_children_offsets(&self) -> GpuEGraphDeviceSpan {
        self.row_children_offsets
    }

    /// Span containing one child count per snapshot row.
    #[must_use]
    pub const fn row_children_lens(&self) -> GpuEGraphDeviceSpan {
        self.row_children_lens
    }

    /// Span containing one structural row signature per snapshot row.
    #[must_use]
    pub const fn row_signatures(&self) -> GpuEGraphDeviceSpan {
        self.row_signatures
    }

    /// Span containing the flat child e-class column.
    #[must_use]
    pub const fn children(&self) -> GpuEGraphDeviceSpan {
        self.children
    }

    /// Span containing sorted e-class ids for row groups.
    #[must_use]
    pub const fn group_eclass_ids(&self) -> GpuEGraphDeviceSpan {
        self.group_eclass_ids
    }

    /// Span containing prefix offsets into [`Self::group_rows`].
    #[must_use]
    pub const fn group_offsets(&self) -> GpuEGraphDeviceSpan {
        self.group_offsets
    }

    /// Span containing row indices grouped by e-class.
    #[must_use]
    pub const fn group_rows(&self) -> GpuEGraphDeviceSpan {
        self.group_rows
    }
}

/// Validated, single-slab u32 image ready for backend upload.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct GpuEGraphDeviceImage {
    pub(super) words: Vec<u32>,
    pub(super) layout: GpuEGraphDeviceLayout,
}

impl GpuEGraphDeviceImage {
    /// Packed u32 words. Backends can upload this slab with one host-to-device copy.
    #[must_use]
    pub fn words(&self) -> &[u32] {
        &self.words
    }

    /// Segment table describing the packed word slab.
    #[must_use]
    pub const fn layout(&self) -> GpuEGraphDeviceLayout {
        self.layout
    }

    /// One e-class id per snapshot row.
    #[must_use]
    pub fn row_eclass_ids(&self) -> &[u32] {
        self.layout.row_eclass_ids.slice(&self.words)
    }

    /// One language op id per snapshot row.
    #[must_use]
    pub fn row_language_op_ids(&self) -> &[u32] {
        self.layout.row_language_op_ids.slice(&self.words)
    }

    /// One child-column offset per snapshot row.
    #[must_use]
    pub fn row_children_offsets(&self) -> &[u32] {
        self.layout.row_children_offsets.slice(&self.words)
    }

    /// One child count per snapshot row.
    #[must_use]
    pub fn row_children_lens(&self) -> &[u32] {
        self.layout.row_children_lens.slice(&self.words)
    }

    /// One structural signature per snapshot row.
    #[must_use]
    pub fn row_signatures(&self) -> &[u32] {
        self.layout.row_signatures.slice(&self.words)
    }

    /// Flat child e-class column.
    #[must_use]
    pub fn children(&self) -> &[u32] {
        self.layout.children.slice(&self.words)
    }

    /// Sorted e-class ids for row groups.
    #[must_use]
    pub fn group_eclass_ids(&self) -> &[u32] {
        self.layout.group_eclass_ids.slice(&self.words)
    }

    /// Prefix offsets into [`Self::group_rows`].
    #[must_use]
    pub fn group_offsets(&self) -> &[u32] {
        self.layout.group_offsets.slice(&self.words)
    }

    /// Row indices grouped by e-class.
    #[must_use]
    pub fn group_rows(&self) -> &[u32] {
        self.layout.group_rows.slice(&self.words)
    }
}

/// Append `values` to `words` and return the span they occupy.
pub(super) fn append_words<I>(words: &mut Vec<u32>, values: I) -> GpuEGraphDeviceSpan
where
    I: IntoIterator<Item = u32>,
{
    let offset = words.len();
    words.extend(values);
    GpuEGraphDeviceSpan::new(offset, words.len() - offset)
}
