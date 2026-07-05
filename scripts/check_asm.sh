#!/usr/bin/env bash
# Verifies that the compiler produces the expected code for the hot paths by
# inspecting the assembly of examples/asm_probe.rs.
#
# Checks:
#   * re-scaling by the compile-time 10^DIGITS constant (all `mul`, `round_to`
#     and `trunc` probes) contains NO division instructions — LLVM must have
#     strength-reduced it to multiply sequences;
#   * no probe ever calls the compiler's 128-bit division builtins
#     (__udivti3 / __umodti3 / __divti3 / __modti3) — wide division always
#     goes through the crate's own limb algorithms;
#   (The `asm` feature only affects division by runtime divisors, through
#   div_2by1; multiplication is division-free in both modes.)
#
# Usage: scripts/check_asm.sh [asm]
set -euo pipefail
cd "$(dirname "$0")/.."

FEATURES="${1:-}"
ARGS=(--release --example asm_probe)
if [ -n "$FEATURES" ]; then
    ARGS+=(--features "$FEATURES")
fi

# A separate target dir keeps hashes stable and avoids clobbering normal builds.
export CARGO_TARGET_DIR=target/asm-check
cargo rustc "${ARGS[@]}" -- --emit asm -C codegen-units=1 >/dev/null 2>&1
ASM=$(ls -t "$CARGO_TARGET_DIR"/release/examples/asm_probe-*.s | head -1)

# Prints the body of one function from the .s file.
body() {
    awk -v fn="$1" '
        $0 == fn ":" { inside = 1; next }
        inside && /^\.Lfunc_end/ { exit }
        inside { print }
    ' "$ASM"
}

count() { # count <text> <mnemonic-regex>
    printf '%s\n' "$1" | grep -cE "^[[:space:]]+($2)[[:space:]]" || true
}

fail=0
report() { # report <probe> <status> <detail>
    printf '  %-28s %-6s %s\n' "$1" "$2" "$3"
}

echo "asm check: features='${FEATURES:-default}'  file=$ASM"

DIV_RE='divq|divl|idivq|idivl|div|idiv'
MUL_RE='mulq|mulxq|imulq|mull|imull'

# 1) Constant re-scale paths must be division-free.
for fn in probe_amount64_mul probe_amount128_mul probe_amount256_mul \
          probe_amount128_round probe_amount256_round \
          probe_amount128_trunc probe_amount256_trunc \
          probe_amount128_fract; do
    b=$(body "$fn")
    if [ -z "$b" ]; then
        report "$fn" "FAIL" "function not found in assembly"
        fail=1
        continue
    fi
    divs=$(count "$b" "$DIV_RE")
    muls=$(count "$b" "$MUL_RE")
    calls=$(printf '%s\n' "$b" | grep -cE 'call.*(udivti3|umodti3|divti3|modti3)' || true)
    if [ "$divs" -eq 0 ] && [ "$calls" -eq 0 ]; then
        report "$fn" "OK" "0 div, $muls mul (strength-reduced)"
    else
        report "$fn" "FAIL" "$divs div instructions, $calls builtin calls"
        fail=1
    fi
done

# 2) Runtime-divisor division: div instructions are fine, builtins are not.
for fn in probe_amount64_div probe_amount128_div probe_amount256_div \
          probe_amount128_rem; do
    b=$(body "$fn")
    calls=$(printf '%s\n' "$b" | grep -cE 'call.*(udivti3|umodti3|divti3|modti3)' || true)
    divs=$(count "$b" "$DIV_RE")
    if [ "$calls" -eq 0 ]; then
        report "$fn" "OK" "$divs div, no 128-bit builtin calls"
    else
        report "$fn" "FAIL" "calls 128-bit division builtins"
        fail=1
    fi
done

if [ "$fail" -ne 0 ]; then
    echo "ASM CHECK FAILED"
    exit 1
fi
echo "ASM CHECK PASSED"
