pub fn dec_len<T: num::Integer + num::ToPrimitive>(value: &T) -> usize {
    if value.is_zero() {
        return 1;
    }
    let v = value.to_f64().expect("Unable to conver to i128").abs();
    v.log10().floor() as usize + 1
}

pub fn max_dec_len<T: num::Integer + num::ToPrimitive + num::Bounded>() -> usize {
    return std::cmp::max(dec_len(&T::min_value()), dec_len(&T::max_value()));
}

#[cfg(test)]
mod power_tests {
    use super::*;

    #[test]
    fn dec_len_test() {
        assert_eq!(dec_len(&(1 as u8)), 1);
        assert_eq!(dec_len(&(0 as i8)), 1);
        assert_eq!(dec_len(&(-11 as i16)), 2);
        assert_eq!(dec_len(&(101 as u16)), 3);
        assert_eq!(dec_len(&(10000 as i32)), 5);
        assert_eq!(dec_len(&(99999 as u32)), 5);
        assert_eq!(dec_len(&(-1199999 as i64)), 7);
        assert_eq!(dec_len(&(100000000034 as u64)), 12);
        assert_eq!(dec_len(&(100000000034 as u64)), 12);
        assert_eq!(
            dec_len(&(10_000_000_000_000_000_000_000_000_000_000_000_000 as i128)),
            38
        );
        assert_eq!(
            dec_len(&(10_000_000_000_000_000_000_000_000_000_000_010_034 as u128)),
            38
        );
    }
}
