//! The fixed-capacity list two objective records share.
//!
//! [`MetricSequence`](super::MetricSequence) and
//! [`WorkloadProfile`](super::WorkloadProfile) are both an ordered list of a
//! `Copy` element with a compile-time capacity. The capacity is what keeps the
//! whole objective `Copy`, and `Copy` is what lets request identity hash the
//! objective by value. Both need the same operations, and two copies of them can
//! disagree about what a full list does with one more element, so the operations
//! are emitted once for a record that stores its elements in a fixed array
//! beside a `u8` length.

/// Emit the fixed-capacity list operations for `$type`.
///
/// `$items` is the array field and `$item` its element type. The record declares
/// its own `CAPACITY`, so one list holds four tie breakers and another four
/// workload classes without the operations knowing either number.
macro_rules! fixed_capacity_list {
    ($type:ty, $items:ident, $item:ty) => {
        impl $type {
            /// Append `item`, or return the list unchanged when it is full.
            ///
            /// A list past its capacity would silently drop what a caller
            /// stated, so a full list reports itself through [`Self::is_full`]
            /// rather than through a length that stopped growing.
            #[must_use]
            pub const fn pushed(mut self, item: $item) -> Self {
                if self.len as usize == Self::CAPACITY {
                    return self;
                }
                self.$items[self.len as usize] = item;
                self.len += 1;
                self
            }

            /// Stated elements, in order.
            #[must_use]
            pub fn as_slice(&self) -> &[$item] {
                &self.$items[..self.len as usize]
            }

            /// Number of elements stated.
            #[must_use]
            pub const fn len(&self) -> usize {
                self.len as usize
            }

            /// Whether no element is stated.
            #[must_use]
            pub const fn is_empty(&self) -> bool {
                self.len == 0
            }

            /// Whether the list cannot hold another element.
            #[must_use]
            pub const fn is_full(&self) -> bool {
                self.len as usize == Self::CAPACITY
            }
        }
    };
}

pub(crate) use fixed_capacity_list;
