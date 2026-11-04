//! Property contracts for nested value-array equality and byte encoding.

use proptest::prelude::*;
use vyre_reference::value::Value;

fn u32_array(values: &[u32]) -> Value {
    Value::Array(values.iter().copied().map(Value::U32).collect())
}

fn flattened_u32_bytes(values: &[u32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

proptest! {
    #[test]
    fn generated_u32_arrays_flatten_in_element_order(values in prop::collection::vec(any::<u32>(), 0..64)) {
        let value = u32_array(&values);

        prop_assert_eq!(value.to_bytes(), flattened_u32_bytes(&values));
        prop_assert_eq!(value.truthy(), !values.is_empty());
        prop_assert_eq!(value.try_as_u32(), None);
        prop_assert_eq!(value.try_as_u64(), None);
    }

    #[test]
    fn generated_u32_array_width_encoding_matches_extend_api(
        prefix in prop::collection::vec(any::<u8>(), 0..32),
        values in prop::collection::vec(any::<u32>(), 0..64),
        width in 0usize..96,
    ) {
        let value = u32_array(&values);
        let mut encoded = prefix.clone();
        value
            .extend_bytes_width(width, &mut encoded)
            .expect("generated array byte extension must not overflow for bounded inputs");

        let mut expected = prefix;
        expected.extend(value.to_bytes_width(width));

        prop_assert_eq!(encoded, expected);
    }
}

proptest! {
    /// An array value equals itself, its clone, and any array with the same
    /// elements, and differs from every array that differs in one element.
    ///
    /// # The class closed here
    ///
    /// [`Value`] hand-writes `PartialEq` and also claims `Eq`, which requires
    /// reflexivity. `Array` had no arm of its own and fell through to the
    /// catch-all that answers `false`, so an array was unequal to itself: the
    /// oracle's own output comparison reported a mismatch for every array
    /// result, and any map keyed by a `Value` could not find an array key it
    /// had just inserted. A variant added to this enum without an arm lands in
    /// the same catch-all, which is why the contract is stated as a law over
    /// the value rather than as one case.
    ///
    /// # What it does not catch
    ///
    /// Equality of the other variants, which
    /// `value_encoding_contract` covers, and ordering, which `Value` does not
    /// define.
    #[test]
    fn generated_u32_arrays_are_equal_exactly_to_arrays_with_the_same_elements(
        values in prop::collection::vec(any::<u32>(), 0..64),
        index in 0usize..64,
    ) {
        let value = u32_array(&values);

        prop_assert_eq!(&value, &value);
        prop_assert_eq!(value.clone(), u32_array(&values));

        if !values.is_empty() {
            let mut perturbed = values.clone();
            let at = index % perturbed.len();
            perturbed[at] = perturbed[at].wrapping_add(1);
            prop_assert_ne!(&value, &u32_array(&perturbed));
        }

        let mut longer = values.clone();
        longer.push(0);
        prop_assert_ne!(&value, &u32_array(&longer));
    }
}
