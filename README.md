# RDecimal

Naive implementation for decimal numbers in Rust.

Inspired by https://github.com/paupino/rust-decimal but suppots different mantissa type.

WIP

## Usage

```rust
use rdecimal::{Decimal, D64, D128};

let dec = D64::new(10, -3); // 0.01

let f_str = D128::from_str("0.02").unwrap();

let f_num = D64::from(10); // 10.
```

## TODO:

* division
* docs

