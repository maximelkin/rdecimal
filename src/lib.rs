extern crate num;

#[cfg(feature = "postgres")]
#[macro_use]
extern crate postgres;
#[cfg(feature = "postgres")]
mod postgres_types;
#[cfg(feature = "serde")]
mod serde_types;
mod utils;

use std::{
    cmp::Ordering,
    convert::From,
    fmt,
    ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign},
    str::FromStr,
};

use utils::{dec_len, max_dec_len};

// TODO: Div, DivAssign

pub type D32 = Decimal<i32>;
pub type D64 = Decimal<i64>;
pub type D128 = Decimal<i128>;

#[derive(Clone, Debug)]
pub struct Decimal<T>
where
    T: num::Integer + num::FromPrimitive + num::ToPrimitive + Clone + Eq,
{
    pub mantissa: T,
    pub exponent: i8,
}

// ######################### ORDER ############################################

fn eq_impl<T>(first: &&Decimal<T>, second: &&Decimal<T>) -> bool
where
    T: num::Integer + num::FromPrimitive + num::ToPrimitive + Clone + Eq,
{
    let n_first = first.normalize();
    let n_second = second.normalize();
    return n_first.mantissa == n_second.mantissa && n_first.exponent == n_second.exponent;
}

impl<T: num::Integer + num::FromPrimitive + num::ToPrimitive + Clone + Eq> PartialEq
    for Decimal<T>
{
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        eq_impl(&self, &other)
    }
}

impl<T: num::Integer + num::FromPrimitive + num::ToPrimitive + Clone + Eq> PartialEq<&Decimal<T>>
    for Decimal<T>
{
    #[inline]
    fn eq(&self, other: &&Decimal<T>) -> bool {
        eq_impl(&self, other)
    }
}

impl<T: num::Integer + num::FromPrimitive + num::ToPrimitive + Clone + Eq> PartialEq<Decimal<T>>
    for &Decimal<T>
{
    #[inline]
    fn eq(&self, other: &Decimal<T>) -> bool {
        eq_impl(&self, &other)
    }
}

impl<T: num::Integer + num::FromPrimitive + num::ToPrimitive + Clone + Eq> Eq for Decimal<T> {}

fn cmp_impl<T>(first: &&Decimal<T>, second: &&Decimal<T>) -> Ordering
where
    T: num::Integer + num::FromPrimitive + num::ToPrimitive + Clone + Eq,
{
    // TODO: incorrect comparsion on mantissa overflow
    let min_exp = std::cmp::min(first.exponent, second.exponent);
    let m_first = first.to_exponent(min_exp);
    let m_second = second.to_exponent(min_exp);

    m_first.mantissa.cmp(&m_second.mantissa)
}

impl<T: num::Integer + num::FromPrimitive + num::ToPrimitive + Clone + Eq> Ord for Decimal<T> {
    #[inline]
    fn cmp(&self, other: &Self) -> Ordering {
        cmp_impl(&self, &other)
    }
}

impl<T: num::Integer + num::FromPrimitive + num::ToPrimitive + Clone + Eq> PartialOrd
    for Decimal<T>
{
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(cmp_impl(&self, &other))
    }
}

impl<T: num::Integer + num::FromPrimitive + num::ToPrimitive + Clone + Eq> PartialOrd<&Decimal<T>>
    for Decimal<T>
{
    #[inline]
    fn partial_cmp(&self, other: &&Decimal<T>) -> Option<Ordering> {
        Some(cmp_impl(&self, &other))
    }
}

impl<T: num::Integer + num::FromPrimitive + num::ToPrimitive + Clone + Eq> PartialOrd<Decimal<T>>
    for &Decimal<T>
{
    #[inline]
    fn partial_cmp(&self, other: &Decimal<T>) -> Option<Ordering> {
        Some(cmp_impl(&self, &other))
    }
}

// ######################### CONVERT ##########################################

impl<T> From<T> for Decimal<T>
where
    T: num::Integer + num::FromPrimitive + num::ToPrimitive + Clone + Eq,
{
    #[inline]
    fn from(value: T) -> Self {
        Self::new(value, 0).normalize()
    }
}

impl<T> FromStr for Decimal<T>
where
    T: num::Integer + num::FromPrimitive + num::ToPrimitive + Clone + Eq,
{
    type Err = T::FromStrRadixErr;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let v: Vec<&str> = s.split('.').collect();
        let mantissa: T;
        let exponent: i8;
        match v.len() {
            1 => {
                exponent = 0;
                mantissa = T::from_str_radix(v[0], 10)?;
            }
            2 => {
                exponent = -(v[1].len() as i8);
                let qt_mantissa = T::from_str_radix(v[0], 10)?;
                let multiplier = num::pow(Self::num_10(), exponent.abs() as usize);
                if v[0].starts_with("-") {
                    mantissa = qt_mantissa * multiplier - T::from_str_radix(v[1], 10)?;
                } else {
                    mantissa = qt_mantissa * multiplier + T::from_str_radix(v[1], 10)?;
                }
            }
            _ => {
                exponent = 0;
                mantissa = T::from_str_radix(".", 10)?;
            }
        }
        Ok(Self::new(mantissa, exponent).normalize())
    }
}

impl<T> fmt::Display for Decimal<T>
where
    T: num::Integer + num::FromPrimitive + num::ToPrimitive + Clone + Eq,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.mantissa.is_zero() {
            return write!(f, "0");
        }

        if self.exponent >= 0 {
            let value = self.mantissa.to_i128().unwrap();
            let mut suffix = String::with_capacity(self.exponent.abs() as usize);
            for _ in 0..self.exponent.abs() {
                suffix.push('0');
            }
            write!(f, "{}{}", value, suffix)
        } else {
            let sign = if self.mantissa < T::zero() { "-" } else { "" };
            let norm = self.normalize();
            let multiplier = num::pow(Self::num_10(), norm.exponent.abs() as usize);
            let (qt, rem) = norm.mantissa.div_rem(&multiplier);
            write!(
                f,
                "{}{}.{:0width$}",
                sign,
                qt.to_i128().unwrap().abs(),
                rem.to_i128().unwrap().abs(),
                width = norm.exponent.abs() as usize,
            )
        }
    }
}

// ######################### INIT #############################################

impl<T> num::Zero for Decimal<T>
where
    T: num::Integer + num::FromPrimitive + num::ToPrimitive + Clone + Eq,
{
    fn is_zero(&self) -> bool {
        self.mantissa.is_zero()
    }

    fn zero() -> Self {
        Self::new(T::from_u8(0).unwrap(), 0)
    }
}

impl<T> num::One for Decimal<T>
where
    T: num::Integer + num::FromPrimitive + num::ToPrimitive + Clone + Eq,
{
    fn one() -> Self {
        Self::new(T::from_u8(1).unwrap(), 0)
    }
}

// ######################### ARITHMETICS ######################################
impl<T> Neg for Decimal<T>
where
    T: num::Integer + num::FromPrimitive + num::ToPrimitive + Clone + Eq + num::Signed,
{
    type Output = Self;

    fn neg(self) -> Self::Output {
        Self::new(-self.mantissa, self.exponent)
    }
}

impl<T> AddAssign for Decimal<T>
where
    T: num::Integer + num::FromPrimitive + num::ToPrimitive + Clone + Eq,
{
    fn add_assign(&mut self, other: Self) {
        let min_exp = std::cmp::min(self.exponent, other.exponent);
        self.inpl_to_exponent(min_exp);
        let m_other = other.to_exponent(min_exp);

        self.mantissa = self.mantissa.clone() + m_other.mantissa;
        self.inpl_normalize();
    }
}

impl<'a, T> AddAssign<&'a Decimal<T>> for Decimal<T>
where
    T: num::Integer + num::FromPrimitive + num::ToPrimitive + Clone + Eq,
{
    fn add_assign(&mut self, other: &'a Self) {
        let min_exp = std::cmp::min(self.exponent, other.exponent);
        self.inpl_to_exponent(min_exp);
        let m_other = other.to_exponent(min_exp);

        self.mantissa = self.mantissa.clone() + m_other.mantissa;
        self.inpl_normalize();
    }
}

impl<T> Add for Decimal<T>
where
    T: num::Integer + num::FromPrimitive + num::ToPrimitive + Clone + Eq,
{
    type Output = Self;

    fn add(self, other: Self) -> Self {
        self.add(&other)
    }
}

impl<'a, T> Add<&'a Decimal<T>> for Decimal<T>
where
    T: num::Integer + num::FromPrimitive + num::ToPrimitive + Clone + Eq,
{
    type Output = Self;

    fn add(mut self, other: &Self) -> Self {
        self.add_assign(other);
        self
    }
}

impl<'a, T> Add<Decimal<T>> for &'a Decimal<T>
where
    T: num::Integer + num::FromPrimitive + num::ToPrimitive + Clone + Eq,
{
    type Output = Decimal<T>;

    fn add(self, other: Decimal<T>) -> Self::Output {
        self.add(&other)
    }
}

impl<'a, T> Add for &'a Decimal<T>
where
    T: num::Integer + num::FromPrimitive + num::ToPrimitive + Clone + Eq,
{
    type Output = Decimal<T>;

    fn add(self, other: &Decimal<T>) -> Self::Output {
        let mut res = self.clone();
        res.add_assign(other);
        res
    }
}

impl<T> SubAssign for Decimal<T>
where
    T: num::Integer + num::FromPrimitive + num::ToPrimitive + Clone + Eq,
{
    fn sub_assign(&mut self, other: Self) {
        let min_exp = std::cmp::min(self.exponent, other.exponent);
        self.inpl_to_exponent(min_exp);
        let m_other = other.to_exponent(min_exp);

        self.mantissa = self.mantissa.clone() - m_other.mantissa;
        self.inpl_normalize();
    }
}

impl<'a, T> SubAssign<&'a Decimal<T>> for Decimal<T>
where
    T: num::Integer + num::FromPrimitive + num::ToPrimitive + Clone + Eq,
{
    fn sub_assign(&mut self, other: &'a Self) {
        let min_exp = std::cmp::min(self.exponent, other.exponent);
        self.inpl_to_exponent(min_exp);
        let m_other = other.to_exponent(min_exp);

        self.mantissa = self.mantissa.clone() - m_other.mantissa;
        self.inpl_normalize();
    }
}

impl<T> Sub for Decimal<T>
where
    T: num::Integer + num::FromPrimitive + num::ToPrimitive + Clone + Eq,
{
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        self.sub(&other)
    }
}

impl<'a, T> Sub<&'a Decimal<T>> for Decimal<T>
where
    T: num::Integer + num::FromPrimitive + num::ToPrimitive + Clone + Eq,
{
    type Output = Self;

    fn sub(mut self, other: &Self) -> Self {
        self.sub_assign(other);
        self
    }
}

impl<'a, T> Sub<Decimal<T>> for &'a Decimal<T>
where
    T: num::Integer + num::FromPrimitive + num::ToPrimitive + Clone + Eq,
{
    type Output = Decimal<T>;

    fn sub(self, other: Decimal<T>) -> Self::Output {
        self.sub(&other)
    }
}

impl<'a, T> Sub for &'a Decimal<T>
where
    T: num::Integer + num::FromPrimitive + num::ToPrimitive + Clone + Eq,
{
    type Output = Decimal<T>;

    fn sub(self, other: &Decimal<T>) -> Self::Output {
        let mut res = self.clone();
        res.sub_assign(other);
        res
    }
}

impl<T> MulAssign for Decimal<T>
where
    T: num::Integer + num::FromPrimitive + num::ToPrimitive + Clone + Eq,
{
    fn mul_assign(&mut self, other: Self) {
        self.mantissa = self.mantissa.clone() * other.mantissa;
        self.exponent = self.exponent + &other.exponent;
        self.inpl_normalize();
    }
}

impl<'a, T> MulAssign<&'a Decimal<T>> for Decimal<T>
where
    T: num::Integer + num::FromPrimitive + num::ToPrimitive + Clone + Eq,
{
    fn mul_assign(&mut self, other: &'a Self) {
        self.mantissa = self.mantissa.clone() * other.mantissa.clone();
        self.exponent = self.exponent + &other.exponent;
        self.inpl_normalize();
    }
}

impl<T> Mul for Decimal<T>
where
    T: num::Integer + num::FromPrimitive + num::ToPrimitive + Clone + Eq,
{
    type Output = Self;

    fn mul(self, other: Self) -> Self {
        self.mul(&other)
    }
}

impl<'a, T> Mul<&'a Decimal<T>> for Decimal<T>
where
    T: num::Integer + num::FromPrimitive + num::ToPrimitive + Clone + Eq,
{
    type Output = Self;

    fn mul(mut self, other: &Self) -> Self {
        self.mul_assign(other);
        self
    }
}

impl<'a, T> Mul<Decimal<T>> for &'a Decimal<T>
where
    T: num::Integer + num::FromPrimitive + num::ToPrimitive + Clone + Eq,
{
    type Output = Decimal<T>;

    fn mul(self, other: Decimal<T>) -> Self::Output {
        self.mul(&other)
    }
}

impl<'a, T> Mul for &'a Decimal<T>
where
    T: num::Integer + num::FromPrimitive + num::ToPrimitive + Clone + Eq,
{
    type Output = Decimal<T>;

    fn mul(self, other: &Decimal<T>) -> Self::Output {
        let mut res = self.clone();
        res.mul_assign(other);
        res
    }
}

impl<T> DivAssign for Decimal<T>
where
    T: num::Integer + num::Bounded + num::FromPrimitive + num::ToPrimitive + Clone + Eq,
{
    #[inline]
    fn div_assign(&mut self, other: Self) {
        self.inpl_div_assign(&other);
    }
}

impl<'a, T> DivAssign<&'a Decimal<T>> for Decimal<T>
where
    T: num::Integer + num::Bounded + num::FromPrimitive + num::ToPrimitive + Clone + Eq,
{
    #[inline]
    fn div_assign(&mut self, other: &'a Self) {
        self.inpl_div_assign(other);
    }
}

impl<T> Div for Decimal<T>
where
    T: num::Integer + num::Bounded + num::FromPrimitive + num::ToPrimitive + Clone + Eq,
{
    type Output = Self;

    #[inline]
    fn div(self, other: Self) -> Self {
        self.div(&other)
    }
}

impl<'a, T> Div<&'a Decimal<T>> for Decimal<T>
where
    T: num::Integer + num::Bounded + num::FromPrimitive + num::ToPrimitive + Clone + Eq,
{
    type Output = Self;

    #[inline]
    fn div(mut self, other: &Self) -> Self {
        self.div_assign(other);
        self
    }
}

impl<'a, T> Div<Decimal<T>> for &'a Decimal<T>
where
    T: num::Integer + num::Bounded + num::FromPrimitive + num::ToPrimitive + Clone + Eq,
{
    type Output = Decimal<T>;

    #[inline]
    fn div(self, other: Decimal<T>) -> Self::Output {
        self.div(&other)
    }
}

impl<'a, T> Div for &'a Decimal<T>
where
    T: num::Integer + num::Bounded + num::FromPrimitive + num::ToPrimitive + Clone + Eq,
{
    type Output = Decimal<T>;

    #[inline]
    fn div(self, other: &Decimal<T>) -> Self::Output {
        let mut res = self.clone();
        res.div_assign(other);
        res
    }
}

// ######################### IMPL #############################################

enum Round {
    Floor,
    Ceil,
}

impl<T> Decimal<T>
where
    T: num::Integer + num::FromPrimitive + num::ToPrimitive + Clone,
{
    #[inline]
    pub fn new(mantissa: T, exponent: i8) -> Self {
        Decimal { mantissa, exponent }
    }

    pub fn from_dec<U>(other: Decimal<U>) -> Self
    where
        U: num::Integer + num::FromPrimitive + num::ToPrimitive + Clone + Into<T>,
    {
        Decimal {
            mantissa: other.mantissa.into(),
            exponent: other.exponent,
        }
    }

    pub fn inpl_normalize(&mut self) {
        if self.mantissa.is_zero() {
            self.exponent = 0;
            return;
        }
        loop {
            let (qt, rem) = self.mantissa.div_rem(&Self::num_10());
            if rem.is_zero() {
                self.mantissa = qt;
                self.exponent += 1;
            } else {
                break;
            }
        }
    }

    pub fn normalize(&self) -> Self {
        let mut res = self.clone();
        res.inpl_normalize();
        res
    }

    pub fn inpl_to_exponent(&mut self, new_exp: i8) {
        let exp_delta = new_exp - self.exponent;
        let multiplier = num::pow(Self::num_10(), exp_delta.abs() as usize);
        if exp_delta >= 0 {
            self.mantissa = self.mantissa.clone() / multiplier;
        } else {
            self.mantissa = self.mantissa.clone() * multiplier;
        }
        self.exponent = new_exp;
    }

    pub fn to_exponent(&self, new_exp: i8) -> Self {
        let mut res = self.clone();
        res.inpl_to_exponent(new_exp);
        res
    }

    pub fn round_ceil(&self, new_exp: i8) -> Self {
        let mut res = self.clone();
        res.inpl_to_exponent_round(new_exp, Round::Ceil);
        res
    }

    pub fn round_floor(&self, new_exp: i8) -> Self {
        let mut res = self.clone();
        res.inpl_to_exponent_round(new_exp, Round::Floor);
        res
    }

    #[inline]
    pub fn inpl_floor(&mut self) {
        if self.exponent < 0 {
            self.inpl_to_exponent_round(0, Round::Floor);
        }
    }

    pub fn floor(&self) -> Self {
        let mut res = self.clone();
        res.inpl_floor();
        res
    }

    #[inline]
    pub fn inpl_ceil(&mut self) {
        if self.exponent < 0 {
            self.inpl_to_exponent_round(0, Round::Ceil);
        }
    }

    pub fn ceil(&self) -> Self {
        let mut res = self.clone();
        res.inpl_ceil();
        res
    }

    fn inpl_to_exponent_round(&mut self, new_exp: i8, round: Round) {
        let exp_delta = new_exp - self.exponent;
        let multiplier = num::pow(Self::num_10(), exp_delta.abs() as usize);
        if exp_delta < 0 {
            self.mantissa = self.mantissa.clone() * multiplier;
        } else {
            let (qt, rem) = self.mantissa.div_rem(&multiplier);
            self.mantissa = match (self.mantissa >= T::zero(), round) {
                (true, Round::Floor) | (false, Round::Ceil) => qt,
                (true, Round::Ceil) => {
                    qt + if rem == T::zero() {
                        T::zero()
                    } else {
                        T::one()
                    }
                }
                (false, Round::Floor) => {
                    qt - if rem == T::zero() {
                        T::zero()
                    } else {
                        T::one()
                    }
                }
            }
        }
        self.exponent = new_exp;
    }

    #[inline]
    fn num_10() -> T {
        T::from_u8(10).unwrap()
    }
}

impl<T> Decimal<T>
where
    T: num::Integer + num::Bounded + num::FromPrimitive + num::ToPrimitive + Clone,
{
    fn inpl_div_assign(&mut self, other: &Self) {
        let max_len = max_dec_len::<T>();
        let (mut qt, mut rem) = self.mantissa.div_rem(&other.mantissa);
        let mut exp_delta: i8 = 0;

        while rem != T::zero() && dec_len(&qt) < max_len - 1 {
            let pow = 1;
            // TODO: Fix iterations number with no overflow error
            // TODO: Don't forget about bug if qty starts with 0
            // std::cmp::min(
            //     (max_len - dec_len(&rem) - 2) as i8,
            //     ((dec_len(&other.mantissa) - dec_len(&rem) + 1) as isize
            //         * (max_len as isize - dec_len(&qt) as isize)) as i8,
            // );
            rem = rem * num::pow(Self::num_10(), pow as usize);
            exp_delta -= pow;

            let (n_qt, n_rem) = rem.div_rem(&other.mantissa);
            rem = n_rem;
            qt = qt * num::pow(Self::num_10(), dec_len(&n_qt)) + n_qt;
        }

        self.mantissa = qt;
        self.exponent = self.exponent - &other.exponent + exp_delta;
        self.inpl_normalize();
    }
}

#[cfg(test)]
mod decimal_tests {
    use super::*;

    #[test]
    fn eq_test() {
        assert_eq!(Decimal::new(10, 0), Decimal::new(10, 0));
        assert_eq!(Decimal::new(10, 0), Decimal::new(100, -1));
        assert_eq!(Decimal::new(1, 1), Decimal::new(10, 0));

        assert_eq!(&Decimal::new(10, 0), Decimal::new(10, 0));
        assert_eq!(Decimal::new(10, 0), &Decimal::new(100, -1));
        assert_eq!(&Decimal::new(1, 1), &Decimal::new(10, 0));

        assert_ne!(Decimal::new(11, 0), Decimal::new(10, 0));
    }

    #[test]
    fn normalize_test() {
        let mut dec = Decimal::<i32>::new(5000, -2);
        dec.inpl_normalize();
        assert_eq!(dec.mantissa, 5);
        assert_eq!(dec.exponent, 1);
        dec = Decimal::<i32>::new(5050, -2);
        dec.inpl_normalize();
        assert_eq!(dec.mantissa, 505);
        assert_eq!(dec.exponent, -1);
    }

    #[test]
    fn to_exp_test() {
        let dec = Decimal::new(5000, -2);
        let to_p1 = dec.to_exponent(1);
        assert_eq!(to_p1.mantissa, 5);
        assert_eq!(to_p1.exponent, 1);

        let to_m5 = dec.to_exponent(-5);

        assert_eq!(to_m5.mantissa, 5000000);
        assert_eq!(to_m5.exponent, -5);
    }

    #[test]
    fn order_test() {
        assert!(Decimal::new(5, -1) < Decimal::new(75, -2));
        assert!(Decimal::new(5, -1) > Decimal::new(75, -3));

        assert!(Decimal::new(5, -1) < &Decimal::new(75, -2));
        assert!(&Decimal::new(5, -1) > Decimal::new(75, -3));
        assert!(&Decimal::new(5, -1) > &Decimal::new(75, -3));
    }

    #[test]
    fn from_test() {
        assert_eq!(Decimal::from(5), Decimal::new(5, 0));
        assert_eq!(Decimal::from(5 as i8), Decimal::new(5, 0));
    }

    #[test]
    fn arithmetics_test() {
        assert_eq!(Decimal::new(5, 0) + Decimal::new(5, 0), Decimal::new(10, 0));
        assert_eq!(
            Decimal::new(5, -1) + Decimal::new(5, -2),
            Decimal::new(55, -2)
        );

        assert_eq!(
            Decimal::new(5, -1) - Decimal::new(5, -2),
            Decimal::new(45, -2)
        );
        assert_eq!(
            Decimal::new(5, -1) - Decimal::new(-5, -2),
            Decimal::new(55, -2)
        );

        assert_eq!(
            Decimal::new(5, -1) * Decimal::new(5, -2),
            Decimal::new(25, -3)
        );
        assert_eq!(
            Decimal::new(5, 1) * Decimal::new(5, -2),
            Decimal::new(25, -1)
        );
        assert_eq!(
            Decimal::new(5, -1) * Decimal::new(-5, -2),
            Decimal::new(-25, -3)
        );
        assert_eq!(Decimal::from(5) / Decimal::new(5, -2), Decimal::from(100));
        assert_eq!(Decimal::from(5) / Decimal::new(-5, -2), Decimal::from(-100));
        assert_eq!(Decimal::from(-5) / Decimal::new(5, -2), Decimal::from(-100));
        assert_eq!(
            Decimal::from(2) / Decimal::new(3, -1),
            Decimal::new(666666666, -8)
        );
        assert_eq!(
            Decimal::from(-2) / Decimal::new(3, -1),
            Decimal::new(-666666666, -8)
        );
        assert_eq!(
            Decimal::from(1) / Decimal::new(689395, -2),
            Decimal::new(145054721, -12)
        );
    }

    #[test]
    fn from_str_test() {
        assert_eq!(Decimal::from_str("11").unwrap(), Decimal::new(11, 0));
        assert_eq!(Decimal::from_str("11.01").unwrap(), Decimal::new(1101, -2));
        assert_eq!(
            Decimal::from_str("-11.02").unwrap(),
            Decimal::new(-1102, -2)
        );
        assert_eq!(Decimal::from_str("-0.5").unwrap(), Decimal::new(-5, -1));
        assert!(D64::from_str("str").is_err());
        assert!(D64::from_str("11.00.11").is_err());
    }

    #[test]
    fn to_string_test() {
        assert_eq!(Decimal::new(5, -1).to_string(), "0.5");
        assert_eq!(Decimal::new(5, 3).to_string(), "5000");
        assert_eq!(Decimal::new(-5, -3).to_string(), "-0.005");
    }

    #[test]
    fn convert_test() {
        let dec = D64::new(5000, -2);
        assert_eq!(D128::new(5000, -2), D128::from_dec(dec));
    }

    #[test]
    fn floor_test() {
        assert_eq!(Decimal::new(37, -1).floor(), Decimal::new(3, 0));
        assert_eq!(Decimal::new(30, -1).floor(), Decimal::new(3, 0));
        assert_eq!(Decimal::new(-37, -1).floor(), Decimal::new(-4, 0));

        assert_eq!(Decimal::new(1, -6).round_floor(-3), Decimal::new(0, 0));
        assert_eq!(Decimal::new(-1, -6).round_floor(-3), Decimal::new(-1, -3));
    }

    #[test]
    fn ceil_test() {
        assert_eq!(Decimal::new(301, -2).ceil(), Decimal::new(4, 0));
        assert_eq!(Decimal::new(30, -1).ceil(), Decimal::new(3, 0));
        assert_eq!(Decimal::new(-37, -1).ceil(), Decimal::new(-3, 0));

        assert_eq!(Decimal::new(1, -6).round_ceil(-3), Decimal::new(1, -3));
        assert_eq!(Decimal::new(-1, -6).round_ceil(-3), Decimal::new(0, 0));
    }

    #[test]
    fn fmt_test() {
        assert_eq!(format!("{}", Decimal::from(1).div(Decimal::from(30000))), "0.0000333333333");
    }
}
