# fin_decimal

## Project Overview
`fin_decimal` is a high-performance, `#![no_std]` compatible Rust library for decimal fixed-point arithmetic. It is tailored specifically for financial computations, tax calculations, and exchange rates, where strict decimal rounding is critical but extreme number ranges are unnecessary.

The library uses signed integers under the hood to represent values implicitly multiplied by a power of 10. This design allows common operations like addition and subtraction to be performed natively and extremely efficiently in a few CPU instructions, without rounding errors. Operations like multiplication and division use double-width intermediates and specific rounding logic to ensure strict decimal accuracy.

**Core idea — power-of-10 scaling on a native integer:**
The in-memory value is a machine integer that is the *true* value scaled by `10^SCALE` (the mantissa). Addition and subtraction need no adjustment — they are exact native integer ops. Multiplication and division must re-adjust the scale (divide or multiply by `10^SCALE`), which is where decimal rounding rules are applied. The backing integer comes in three widths sharing the same decimal semantics:

| Backing | Type | Aliases | Range at 4 digits |
|---|---|---|---|
| `i64` | `Decimal<DIGITS>` | `Amount64` (4), `Rate64` (8) | ~±922 trillion |
| `i128` | `Decimal128<DIGITS>` | `Amount128` (4), `Rate128` (8) | ~±1.7 * 10^34 |
| 256-bit `I256` | `Decimal256<DIGITS>` | `Amount256` (4), `Rate256` (8) | ~±5.8 * 10^72 |

All three implement the same trait surface and rounding semantics; 20k-case differential tests pin `Decimal128` to `Decimal` and `Decimal256` to `Decimal128` across every operator and rounding mode.

**Main Technologies & Architecture:**
* **Rust (Edition 2024)** with `#![no_std]` support.
* **Minimal dependencies:** the crate avoids pulling in external crates; `serde` and `ufmt` are optional features. Where a primitive operation is missing, prefer a small, correct implementation (and inline assembly where it pays off) over adding a dependency. Even the benchmark harness (`benches/perf.rs`) is dependency-free.
* **Const Generics (`Decimal<const DIGITS: u8>`):** The core types are parameterized by the number of decimal digits (`DIGITS <= 19` for the wide types, so `10^DIGITS` fits one 64-bit limb). `DIGITS` is also exposed under the stable name `SCALE` (an `i32`).
* **One arithmetic engine (`src/limbs.rs`):** all three backings are thin sign-magnitude adapters over width-generic cores — `dec_mul`/`dec_div` (parameterized by `DIGITS` and limb count `W` = 1/2/4), one parser, one formatter. `MIN == -MAX` on every backing makes the overflow check uniformly "top bit of the top limb must be clear". Narrow-operand fast paths tier `Decimal256` work down to the two-limb core.
* **Shared trait impls (`src/common.rs`):** the `impl_decimal_common!` macro stamps out the identical trait impls (FromStr, assigns, Sum/Product, Display, serde, ufmt) for each type; a type only supplies `sign_mag4` and `from_str_rounded`. Arithmetic deliberately stays out of the macro.

## Optimization Goals
These are load-bearing design rules — verify them when touching arithmetic code:

* **Performance — division by the scale must never emit a division instruction.** Re-scaling by the compile-time constant `10^DIGITS` uses Möller–Granlund reciprocal multiplication with the reciprocal computed at compile time (`limbs::div_words_by_pow10`, `limbs::div_rem_u128_pow10`). Important hard-won fact: **LLVM does NOT strength-reduce 128-bit division by a constant** — writing `u128 / CONST` silently lowers to a `__udivti3` libcall, which is software on most non-x86 targets. `scripts/check_asm.sh` inspects generated assembly and fails if any constant re-scale path contains division instructions or 128-bit division builtins. Run it (both `scripts/check_asm.sh` and `scripts/check_asm.sh asm`) after changing hot paths.
* **Performance — division by a runtime divisor never calls `__udivti3`.** It goes through `limbs::div_2by1` (hardware `div` under the `asm` feature on `x86_64`, portable Knuth half-digit division otherwise) and algorithm D over significant limbs only. The `asm` feature affects *only* these runtime-divisor paths and the i64 `Decimal` mul/div/recip; measure before recommending it (on modern x86 it is roughly a wash).
* **No heap allocations — ever.** The crate is `no_std` without `alloc`. All buffers are fixed-size stack arrays (`[u64; N]` limbs, `[u8; 96/128]` string buffers). Formatting writes into caller/stack buffers; nothing returns owned strings. Allocations are only acceptable inside `#[cfg(test)]`.
* **Compile-time evaluation.** Hot cores and constructors are `const fn` (`checked_mul`, `mul_rounded`, `round_to`, `trunc`, `fract`, `powi`, `from_str_rounded`, `from_str_const`, `from_decimal_parts`, ...), so rates, fees, and whole derived constants evaluate at compile time: `const TAX: Amount64 = PRICE.mul_rounded(RATE, Rounding::HalfEven);`. When adding functionality, prefer `const fn` (while-loops instead of iterators, `split_at` instead of range indexing, `match` instead of `?`/`unwrap`) unless it would force a slower runtime path — division by runtime divisors stays non-const because its fast path uses inline asm under the `asm` feature, and constness must not depend on features (feature additivity).
* **Minimum panics.** Prefer `checked_*`/`Result` APIs; reserve panics for arithmetic contract violations (`/` by zero, `*`//`/` operator overflow on every backing — documented in each operator's `# Panics` section) and `from_str_const` on bad literals (a compile error in const context). Never introduce implicit panic paths (unchecked indexing/unwraps) in arithmetic code; write loops and bounds so the compiler can elide checks, and document invariants that make wrapping/unchecked math safe.

## Building and Running
The project is built and managed using Cargo.

* **Build the library:** `cargo build`
* **Run the test suite** (unit + doc tests; also run with features): `cargo test --all-features`, plus `cargo test` and `cargo test --features asm`
* **Performance tests** (self-contained, prints ns/op per case across operand/divisor sizes): `cargo bench` and `cargo bench --features asm`
* **Assembly verification** (codegen contract for hot paths): `./scripts/check_asm.sh` and `./scripts/check_asm.sh asm`; probes live in `examples/asm_probe.rs`
* **Lint / format:** `cargo clippy --all-features`, `cargo fmt`

## Development Conventions
* **Idiomatic Rust:** Standard Rust naming, formatting, and typing conventions apply. Always run `cargo fmt` and `cargo clippy` to ensure code quality.
* **`no_std` Compatibility:** The crate strictly adheres to `#![no_std]`, relying solely on `core` (e.g. `core::mem::ManuallyDrop`). Note `f64::floor` etc. do not exist in `core` — emulate with casts.
* **Testing:** All new functionality, traits, or bug fixes must include corresponding unit tests in the `tests` module of the relevant source file, or as documentation tests. Arithmetic changes must keep the randomized self-checks and the cross-type differential tests passing — they are the semantic contract between the three backings. `const fn` changes should extend the `test_const_eval` tests (CTFE runs with overflow checks always on, which catches bugs release builds miss).
* **Safety:** While performance is prioritized, undefined behavior must be strictly avoided. Arrays must be properly initialized (e.g. `[0u8; N]`) rather than using uninitialized memory assumptions (`MaybeUninit`).
* **Trait Implementations:** The library favors standard mathematical trait implementations (e.g., `Add`, `Sub`, `Mul`, `Div`, `Rem`, their `Assign` counterparts, `Sum`, and `Product`) to make custom types feel ergonomic.
* **Explicit Financial Logic:**
  * **Rounding:** In addition to implicit `HalfUp` operations, explicit rounding must be supported via `.round_to(mode)`, `.mul_rounded(rhs, mode)`, and `.div_rounded(rhs, mode)` to support strict tax compliance (e.g. `Rounding::HalfEven` or "Banker's Rounding"). On magnitudes, `Down`/`Up` are directional (floor/ceil); half-modes compare `2 * remainder` against the divisor; `HalfEven` ties only exist for even divisors.
  * **Checked Math:** For overflow prevention in critical financial systems, `checked_add`, `checked_sub`, `checked_mul`, and `checked_div` returning `Option<Self>` are standard. The range is symmetric (`MIN == -MAX`) on all backings.

* **Scale-agnostic (mantissa, exponent) interchange** (on `Decimal` and `Decimal128`; `Decimal256`'s mantissa exceeds `i128`, so it deliberately omits this API):
  * `SCALE: i32` — the fixed decimal scale (number of fractional digits), a backing-independent stable name for `DIGITS`.
  * `mantissa() -> i128` — the raw scaled value widened to `i128`; the signature is backing-independent.
  * `to_decimal_parts() -> (i128, i32)` — decompose to `(mantissa, exponent)`; the exponent is always `-SCALE`.
  * `from_decimal_parts(mantissa, exponent) -> Result<Self, AmountErrorKind>` — the inverse, and the single place the backing-range check lives. Scaling **up** is exact; scaling **down** is **exact-or-error**: surplus *trailing zeros* are dropped exactly (not an error); only genuinely non-zero dropped digits yield `AmountErrorKind::Inexact`.
* **Inexact intake with explicit rounding:** `from_decimal_parts_rounded(mantissa, exponent, mode)` never returns `Inexact` (only `Overflow`); `from_str_rounded(src, mode)` / `parse_decimal_i64_rounded(src, scale, mode)` give string intake with explicit rounding-mode control. `FromStr`/`From<&str>` delegate with `Rounding::HalfUp`. Rounding is IEEE-correct: first dropped digit + sticky flag + digit parity for `HalfEven`.
* **`AmountErrorKind::Inexact`:** distinct from `Overflow` — signals that a value carried more fractional precision than the target scale can represent and the exact (non-rounding) path was used.
