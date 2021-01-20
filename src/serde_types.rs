use crate::Decimal;
use std::{fmt, marker::PhantomData, str::FromStr};

use serde::{self, de::Unexpected};

impl<T> serde::Serialize for Decimal<T>
where
    T: num::Integer + num::FromPrimitive + num::ToPrimitive + Clone + Eq,
{
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        serializer.serialize_str(&self.to_string())
    }
}

struct DecimalVisitor<T>
where
    T: num::Integer + num::FromPrimitive + num::ToPrimitive + Clone + Eq,
{
    data: PhantomData<T>,
}

impl<'a, T> serde::de::Visitor<'a> for DecimalVisitor<T>
where
    T: num::Integer + num::FromPrimitive + num::ToPrimitive + Clone + Eq,
{
    type Value = Decimal<T>;

    fn visit_f64<E>(self, value: f64) -> Result<Decimal<T>, E>
    where
        E: serde::de::Error,
    {
        Decimal::from_str(&value.to_string())
            .map_err(|_| E::invalid_value(Unexpected::Float(value), &self))
    }

    fn visit_i64<E>(self, value: i64) -> Result<Decimal<T>, E>
    where
        E: serde::de::Error,
    {
        match T::from_i64(value) {
            Some(value) => Ok(Decimal::from(value)),
            None => Err(serde::de::Error::custom("i64 parsing error")),
        }
    }

    fn visit_u64<E>(self, value: u64) -> Result<Decimal<T>, E>
    where
        E: serde::de::Error,
    {
        match T::from_u64(value) {
            Some(value) => Ok(Decimal::from(value)),
            None => Err(serde::de::Error::custom("u64 parsing error")),
        }
    }

    fn visit_str<E>(self, value: &str) -> Result<Decimal<T>, E>
    where
        E: serde::de::Error,
    {
        Decimal::from_str(value).map_err(|_| E::invalid_value(Unexpected::Str(value), &self))
    }

    fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        write!(
            formatter,
            "a Decimal type representing a fixed-point number"
        )
    }
}

impl<'a, T> serde::Deserialize<'a> for Decimal<T>
where
    T: num::Integer + num::FromPrimitive + num::ToPrimitive + Clone + Eq,
{
    fn deserialize<D>(deserializer: D) -> Result<Decimal<T>, D::Error>
    where
        D: serde::de::Deserializer<'a>,
    {
        deserializer.deserialize_any(DecimalVisitor::<T> { data: PhantomData })
    }
}

#[cfg(test)]
mod test {
    use serde_derive::{Deserialize, Serialize};

    use crate::D64;

    #[derive(Serialize, Deserialize, Debug)]
    struct TestData {
        data: D64,
    }

    #[test]
    fn deserialize_valid_decimal() {
        let data = [
            ("{\"data\":\"1.234\"}", "1.234"),
            ("{\"data\":-10}", "-10"),
            ("{\"data\":0.012}", "0.012"),
            ("{\"data\":\"-0.005\"}", "-0.005"),
        ];
        for &(serialized, value) in data.iter() {
            let result = serde_json::from_str(serialized);
            assert!(result.is_ok());
            let test: TestData = result.unwrap();
            assert_eq!(value, test.data.to_string(),);
        }
    }

    #[test]
    #[should_panic]
    fn deserialize_invalid_decimal() {
        let serialized = "{\"data\":\"abacaba\"}";
        let _: TestData = serde_json::from_str(serialized).unwrap();
    }

    #[test]
    fn serialize_decimal() {
        let tests = [
            (
                TestData {
                    data: D64::new(1234, -3),
                },
                "1.234",
            ),
            (
                TestData {
                    data: D64::new(10, -3),
                },
                "0.01",
            ),
            (
                TestData {
                    data: D64::new(-5, -3),
                },
                "-0.005",
            ),
        ];
        for (data, str_number) in tests.iter() {
            let serialized = serde_json::to_string(&data).unwrap();
            assert_eq!(format!("{{\"data\":\"{}\"}}", str_number), serialized);
        }
    }
}
