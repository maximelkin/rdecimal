use std::fmt;

use crate::Decimal;

use byteorder::{BigEndian, ReadBytesExt, WriteBytesExt};
use postgres::types::{FromSql, IsNull, ToSql, Type, NUMERIC};

const HEADER_SIZE: usize = 8;

#[derive(Debug)]
enum ConversionError {
    BadHeader,
    WrongSize,
    IncorrectType,
}

impl fmt::Display for ConversionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ConversionError::BadHeader => write!(f, "Incorrect data header"),
            ConversionError::WrongSize => write!(f, "Data len and header.len missmatch"),
            ConversionError::IncorrectType => write!(f, "Not possible convert PGType into Decimal"),
        }
    }
}

impl std::error::Error for ConversionError {}

impl<T> FromSql for Decimal<T>
where
    T: num::Integer + num::FromPrimitive + num::ToPrimitive + Clone + Eq,
{
    fn from_sql(
        ty: &Type,
        raw: &[u8],
    ) -> Result<Self, Box<dyn std::error::Error + 'static + Sync + Send>> {
        match *ty {
            NUMERIC => {
                if raw.len() < HEADER_SIZE {
                    return Err(Box::new(ConversionError::BadHeader));
                }
                let mut cursor = std::io::Cursor::new(raw);
                let len = cursor.read_i16::<BigEndian>().unwrap();
                let scale = cursor.read_i16::<BigEndian>().unwrap();
                let sign = cursor.read_i8().unwrap();
                let _ = cursor.read_i8().unwrap();
                let neg = match sign {
                    0 => false,
                    64 => true,
                    _ => return Err(Box::new(ConversionError::BadHeader)),
                };
                let _low_len = cursor.read_i16::<BigEndian>().unwrap();

                if raw.len() != HEADER_SIZE + len as usize * 2 {
                    return Err(Box::new(ConversionError::WrongSize));
                }
                let mut mantissa = T::zero();
                let mantissa_part_mul = T::from_u16(10000).unwrap();
                for _ in 0..len as usize {
                    let mantissa_part = cursor.read_i16::<BigEndian>().unwrap();

                    mantissa =
                        mantissa * mantissa_part_mul.clone() + T::from_i16(mantissa_part).unwrap();
                }
                if neg {
                    mantissa = T::zero() - mantissa;
                }
                let exponent = 4 * (scale - len + 1) as i8;
                return Ok(Decimal::new(mantissa, exponent).normalize());
            }
            _ => {
                return Err(Box::new(ConversionError::IncorrectType));
            }
        }
    }

    fn accepts(ty: &Type) -> bool {
        match *ty {
            NUMERIC => true,
            _ => false,
        }
    }
}

impl<T> ToSql for Decimal<T>
where
    T: num::Integer + num::FromPrimitive + num::ToPrimitive + Clone + Eq + std::fmt::Debug,
{
    to_sql_checked!();

    fn to_sql(
        &self,
        ty: &Type,
        out: &mut Vec<u8>,
    ) -> Result<IsNull, Box<dyn std::error::Error + 'static + Sync + Send>> {
        match *ty {
            NUMERIC => {
                let exp_to_write = if self.exponent >= 0 {
                    self.exponent - self.exponent % 4
                } else {
                    self.exponent - (self.exponent % 4 + 4) % 4
                };
                let dec_to_write = self.to_exponent(exp_to_write);
                let mut mantissa = if dec_to_write.mantissa >= T::zero() {
                    dec_to_write.mantissa
                } else {
                    T::zero() - dec_to_write.mantissa
                };

                let mut mantissa_parts: Vec<i16> = vec![];
                let mantissa_part_mul = T::from_u16(10000).unwrap();
                loop {
                    let (qt, rem) = mantissa.div_rem(&mantissa_part_mul);
                    mantissa_parts.push(rem.to_i16().unwrap());
                    mantissa = qt;

                    if mantissa.is_zero() {
                        break;
                    }
                }

                out.write_i16::<BigEndian>(mantissa_parts.len() as i16)
                    .unwrap();
                let scale = exp_to_write as i16 / 4 + mantissa_parts.len() as i16 - 1;
                out.write_i16::<BigEndian>(scale).unwrap();
                out.write_i8(if self.mantissa >= T::zero() { 0 } else { 64 })
                    .unwrap();
                out.write_i8(0).unwrap();
                let low_exp = std::cmp::max(0, -self.exponent) as i16;
                out.write_i16::<BigEndian>(low_exp).unwrap();
                for i in mantissa_parts.iter().rev() {
                    out.write_i16::<BigEndian>(*i).unwrap();
                }

                Ok(IsNull::No)
            }
            _ => Err(Box::new(ConversionError::IncorrectType)),
        }
    }

    fn accepts(ty: &Type) -> bool {
        match *ty {
            NUMERIC => true,
            _ => false,
        }
    }
}

#[cfg(test)]
mod test {
    use postgres::{
        types::{FromSql, IsNull, ToSql, Type, NUMERIC},
        Connection, TlsMode,
    };

    use crate::D64;

    #[test]
    fn direct_conversion_test() {
        let test_data = vec![
            D64::new(10, 0),
            D64::new(10, 5),
            D64::new(-10, 0),
            D64::new(11, 0),
            D64::new(111, 0),
            D64::new(1111, 0),
            D64::new(11, -1),
            D64::new(12, -1),
            D64::new(1111111111, -5),
            D64::new(65536, 0),
            D64::new(65536, -4),
            D64::new(1, -1),
            D64::new(1, -5),
        ];
        for i in &test_data {
            let mut out: Vec<u8> = vec![];
            assert!(i.to_sql(&NUMERIC, &mut out).is_ok());
            let parsed = D64::from_sql(&NUMERIC, &out);
            match &parsed {
                Ok(_) => {
                    assert_eq!(parsed.unwrap(), i);
                }
                Err(e) => {
                    println!("Error during parse: {}", e);
                    assert!(false);
                }
            }
        }
    }

    #[test]
    fn from_sql_test() {
        let test_data = vec![
            D64::new(10, 0),
            D64::new(10, 5),
            D64::new(-10, 0),
            D64::new(11, 0),
            D64::new(111, 0),
            D64::new(1111, 0),
            D64::new(11, -1),
            D64::new(12, -1),
            D64::new(1111111111, -5),
            D64::new(65536, 0),
            D64::new(65536, -4),
            D64::new(1, -1),
            D64::new(1, -5),
        ];
        let conn =
            Connection::connect("postgresql://postgres@localhost:5432", TlsMode::None).unwrap();
        let trans = conn.transaction().unwrap();
        trans
            .execute("CREATE TEMP TABLE temp(val NUMERIC(40, 15));", &[])
            .unwrap();
        for i in &test_data {
            trans
                .execute("INSERT INTO temp VALUES ($1);", &[&i])
                .unwrap();
        }
        let mut iter = test_data.iter();
        for row in &conn.query("SELECT val FROM temp;", &[]).unwrap() {
            let res: D64 = row.get(0);
            assert_eq!(res, iter.next().unwrap());
        }
        trans.commit().unwrap();
    }
}
