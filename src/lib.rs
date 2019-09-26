extern crate num;

#[cfg(feature = "serde")]
mod serde_types;

use std::cmp::Ordering;
use std::convert::From;
use std::fmt;
use std::ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign};
use std::str::FromStr;

// TODO: Div, DivAssign

pub type D32 = Decimal<i32>;
pub type D64 = Decimal<i64>;
pub type D128 = Decimal<i128>;

#[derive(Clone, Debug)]
pub struct Decimal<T>
where
    T: num::Integer + num::NumCast + Clone + Eq,
{
    pub mantissa: T,
    pub exponent: i8,
}

// ######################### ORDER ############################################

fn eq_impl<T>(first: &&Decimal<T>, second: &&Decimal<T>) -> bool
where
    T: num::Integer + num::NumCast + Clone + Eq,
{
    let n_first = first.normalize();
    let n_second = second.normalize();
    return n_first.mantissa == n_second.mantissa && n_first.exponent == n_second.exponent;
}

impl<T: num::Integer + num::NumCast + Clone + Eq> PartialEq for Decimal<T> {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        eq_impl(&self, &other)
    }
}

impl<T: num::Integer + num::NumCast + Clone + Eq> PartialEq<&Decimal<T>> for Decimal<T> {
    #[inline]
    fn eq(&self, other: &&Decimal<T>) -> bool {
        eq_impl(&self, other)
    }
}

impl<T: num::Integer + num::NumCast + Clone + Eq> PartialEq<Decimal<T>> for &Decimal<T> {
    #[inline]
    fn eq(&self, other: &Decimal<T>) -> bool {
        eq_impl(&self, &other)
    }
}

impl<T: num::Integer + num::NumCast + Clone + Eq> Eq for Decimal<T> where {}

fn cmp_impl<T>(first: &&Decimal<T>, second: &&Decimal<T>) -> Ordering
where
    T: num::Integer + num::NumCast + Clone + Eq,
{
    // TODO: incorrect comparsion on mantissa overflow
    let min_exp = std::cmp::min(first.exponent, second.exponent);
    let m_first = first.to_exponent(min_exp);
    let m_second = second.to_exponent(min_exp);

    m_first.mantissa.cmp(&m_second.mantissa)
}

impl<T: num::Integer + num::NumCast + Clone + Eq> Ord for Decimal<T> {
    #[inline]
    fn cmp(&self, other: &Self) -> Ordering {
        cmp_impl(&self, &other)
    }
}

impl<T: num::Integer + num::NumCast + Clone + Eq> PartialOrd for Decimal<T> {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(cmp_impl(&self, &other))
    }
}

impl<T: num::Integer + num::NumCast + Clone + Eq> PartialOrd<&Decimal<T>> for Decimal<T> {
    #[inline]
    fn partial_cmp(&self, other: &&Decimal<T>) -> Option<Ordering> {
        Some(cmp_impl(&self, &other))
    }
}

impl<T: num::Integer + num::NumCast + Clone + Eq> PartialOrd<Decimal<T>> for &Decimal<T> {
    #[inline]
    fn partial_cmp(&self, other: &Decimal<T>) -> Option<Ordering> {
        Some(cmp_impl(&self, &other))
    }
}

// ######################### CONVERT ##########################################

impl<U, T> From<U> for Decimal<T>
where
    U: num::ToPrimitive,
    T: num::Integer + num::NumCast + Clone + Eq,
{
    #[inline]
    fn from(value: U) -> Self {
        Self::new(T::from(value).unwrap(), 0).normalize()
    }
}

impl<T> FromStr for Decimal<T>
where
    T: num::Integer + num::NumCast + Clone + Eq,
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
    T: num::Integer + num::NumCast + Clone + Eq,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.exponent >= 0 {
            let value = self.mantissa.to_i128().unwrap();
            let mut suffix = String::with_capacity(self.exponent.abs() as usize);
            for _ in 0..self.exponent.abs() {
                suffix.push('0');
            }
            write!(f, "{}{}.", value, suffix)
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
    T: num::Integer + num::NumCast + Clone + Eq,
{
    fn is_zero(&self) -> bool {
        self.mantissa.is_zero()
    }

    fn zero() -> Self {
        Self::new(T::from(0).unwrap(), 0)
    }
}

impl<T> num::One for Decimal<T>
where
    T: num::Integer + num::NumCast + Clone + Eq,
{
    fn one() -> Self {
        Self::new(T::from(1).unwrap(), 0)
    }
}

// ######################### ARITHMETICS ######################################
impl<T> Neg for Decimal<T>
where
    T: num::Integer + num::NumCast + Clone + Eq + num::Signed,
{
    type Output = Self;

    fn neg(self) -> Self::Output {
        Self::new(-self.mantissa, self.exponent)
    }
}

impl<T> AddAssign for Decimal<T>
where
    T: num::Integer + num::NumCast + Clone + Eq,
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
    T: num::Integer + num::NumCast + Clone + Eq,
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
    T: num::Integer + num::NumCast + Clone + Eq,
{
    type Output = Self;

    fn add(self, other: Self) -> Self {
        self.add(&other)
    }
}

impl<'a, T> Add<&'a Decimal<T>> for Decimal<T>
where
    T: num::Integer + num::NumCast + Clone + Eq,
{
    type Output = Self;

    fn add(mut self, other: &Self) -> Self {
        self.add_assign(other);
        self
    }
}

impl<'a, T> Add<Decimal<T>> for &'a Decimal<T>
where
    T: num::Integer + num::NumCast + Clone + Eq,
{
    type Output = Decimal<T>;

    fn add(self, other: Decimal<T>) -> Self::Output {
        self.add(&other)
    }
}

impl<'a, T> Add for &'a Decimal<T>
where
    T: num::Integer + num::NumCast + Clone + Eq,
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
    T: num::Integer + num::NumCast + Clone + Eq,
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
    T: num::Integer + num::NumCast + Clone + Eq,
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
    T: num::Integer + num::NumCast + Clone + Eq,
{
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        self.sub(&other)
    }
}

impl<'a, T> Sub<&'a Decimal<T>> for Decimal<T>
where
    T: num::Integer + num::NumCast + Clone + Eq,
{
    type Output = Self;

    fn sub(mut self, other: &Self) -> Self {
        self.sub_assign(other);
        self
    }
}

impl<'a, T> Sub<Decimal<T>> for &'a Decimal<T>
where
    T: num::Integer + num::NumCast + Clone + Eq,
{
    type Output = Decimal<T>;

    fn sub(self, other: Decimal<T>) -> Self::Output {
        self.sub(&other)
    }
}

impl<'a, T> Sub for &'a Decimal<T>
where
    T: num::Integer + num::NumCast + Clone + Eq,
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
    T: num::Integer + num::NumCast + Clone + Eq,
{
    fn mul_assign(&mut self, other: Self) {
        self.mantissa = self.mantissa.clone() * other.mantissa;
        self.exponent = self.exponent * &other.exponent;
        self.inpl_normalize();
    }
}

impl<'a, T> MulAssign<&'a Decimal<T>> for Decimal<T>
where
    T: num::Integer + num::NumCast + Clone + Eq,
{
    fn mul_assign(&mut self, other: &'a Self) {
        self.mantissa = self.mantissa.clone() * other.mantissa.clone();
        self.exponent = self.exponent * &other.exponent;
        self.inpl_normalize();
    }
}

impl<T> Mul for Decimal<T>
where
    T: num::Integer + num::NumCast + Clone + Eq,
{
    type Output = Self;

    fn mul(self, other: Self) -> Self {
        self.mul(&other)
    }
}

impl<'a, T> Mul<&'a Decimal<T>> for Decimal<T>
where
    T: num::Integer + num::NumCast + Clone + Eq,
{
    type Output = Self;

    fn mul(mut self, other: &Self) -> Self {
        self.mul_assign(other);
        self
    }
}

impl<'a, T> Mul<Decimal<T>> for &'a Decimal<T>
where
    T: num::Integer + num::NumCast + Clone + Eq,
{
    type Output = Decimal<T>;

    fn mul(self, other: Decimal<T>) -> Self::Output {
        self.mul(&other)
    }
}

impl<'a, T> Mul for &'a Decimal<T>
where
    T: num::Integer + num::NumCast + Clone + Eq,
{
    type Output = Decimal<T>;

    fn mul(self, other: &Decimal<T>) -> Self::Output {
        let mut res = self.clone();
        res.mul_assign(other);
        res
    }
}

// ######################### IMPL #############################################

impl<T> Decimal<T>
where
    T: num::Integer + num::NumCast + Clone,
{
    #[inline]
    pub fn new(mantissa: T, exponent: i8) -> Self {
        Decimal { mantissa, exponent }
    }

    pub fn from_dec<U>(other: Decimal<U>) -> Self
    where
        U: num::Integer + num::NumCast + Clone + Into<T>,
    {
        Decimal {
            mantissa: other.mantissa.into(),
            exponent: other.exponent,
        }
    }

    pub fn inpl_normalize(&mut self) {
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

    pub fn inpl_floor(&mut self) {
        if self.exponent >= 0 {
            return;
        }
        if self.mantissa >= T::zero() {
            self.inpl_to_exponent(0);
            return;
        }
        let multiplier = num::pow(Self::num_10(), self.exponent.abs() as usize);
        let (qt, rem) = self.mantissa.div_rem(&multiplier);

        self.exponent = 0;
        self.mantissa = qt;
        if rem != T::zero() {
            self.mantissa = self.mantissa.clone() - T::one();
        }
    }

    pub fn floor(&self) -> Self {
        let mut res = self.clone();
        res.inpl_floor();
        res
    }

    pub fn inpl_ceil(&mut self) {
        if self.exponent >= 0 {
            return;
        }
        if self.mantissa <= T::zero() {
            self.inpl_to_exponent(0);
            return;
        }
        let multiplier = num::pow(Self::num_10(), self.exponent.abs() as usize);
        let (qt, rem) = self.mantissa.div_rem(&multiplier);

        self.exponent = 0;
        self.mantissa = qt;
        if rem != T::zero() {
            self.mantissa = self.mantissa.clone() + T::one();
        }
    }

    pub fn ceil(&self) -> Self {
        let mut res = self.clone();
        res.inpl_ceil();
        res
    }

    #[inline]
    fn num_10() -> T {
        T::from(10).unwrap()
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
        assert_eq!(Decimal::new(5, 3).to_string(), "5000.");
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
    }

    #[test]
    fn ceil_test() {
        assert_eq!(Decimal::new(301, -2).ceil(), Decimal::new(4, 0));
        assert_eq!(Decimal::new(30, -1).ceil(), Decimal::new(3, 0));
        assert_eq!(Decimal::new(-37, -1).ceil(), Decimal::new(-3, 0));
    }
}
