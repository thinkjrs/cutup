#!/usr/bin/env bash
# Builds the browser engine into public/wasm, where the engine's worker loads it: the wrapper
# for wasm32 with the size-optimized profile, wasm-bindgen's glue at the version in Cargo.lock,
# then a wasm-opt pass. Fetches whatever's missing, Rust through rustup and prebuilt wasm-bindgen
# and binaryen into .tools, so it runs on Vercel's build image as it does locally.
set -euo pipefail
cd "$(dirname "$0")/.."

BINARYEN=version_133
TOOLS="$PWD/.tools"
OUT=public/wasm

case "$(uname -s)-$(uname -m)" in
  Linux-x86_64) triple=x86_64-unknown-linux-musl binaryen_os=x86_64-linux ;;
  Linux-aarch64) triple=aarch64-unknown-linux-gnu binaryen_os=aarch64-linux ;;
  Darwin-arm64) triple=aarch64-apple-darwin binaryen_os=arm64-macos ;;
  Darwin-x86_64) triple=x86_64-apple-darwin binaryen_os=x86_64-macos ;;
  *) echo "no prebuilt tools for $(uname -s) $(uname -m)" >&2 && exit 1 ;;
esac

# Downloads a .tar.gz into .tools and unpacks it there, once. It's unpacked aside and moved in
# whole, so a download that breaks off leaves nothing a later run would take for the tool.
fetch() {
  local url=$1 dir=$2
  if [ ! -d "$TOOLS/$dir" ]; then
    mkdir -p "$TOOLS"
    local partial
    partial=$(mktemp -d "$TOOLS/.partial.XXXXXX")
    curl --proto '=https' --tlsv1.2 -sSfL "$url" | tar -xz -C "$partial"
    mv "$partial/$dir" "$TOOLS/$dir"
    rm -rf "$partial"
  fi
}

if ! command -v cargo >/dev/null; then
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
    | sh -s -- -y --profile minimal --target wasm32-unknown-unknown
  # shellcheck source=/dev/null
  source "$HOME/.cargo/env"
elif command -v rustup >/dev/null; then
  rustup target add wasm32-unknown-unknown >/dev/null
fi

# The CLI has to match the wasm-bindgen crate exactly.
version=$(awk '$0 == "name = \"wasm-bindgen\"" { getline; gsub(/"/, "", $3); print $3 }' Cargo.lock)
if [ "$(wasm-bindgen --version 2>/dev/null)" = "wasm-bindgen $version" ]; then
  bindgen=wasm-bindgen
else
  dir="wasm-bindgen-$version-$triple"
  fetch "https://github.com/wasm-bindgen/wasm-bindgen/releases/download/$version/$dir.tar.gz" "$dir"
  bindgen="$TOOLS/$dir/wasm-bindgen"
fi

fetch "https://github.com/WebAssembly/binaryen/releases/download/$BINARYEN/binaryen-$BINARYEN-$binaryen_os.tar.gz" \
  "binaryen-$BINARYEN"
opt="$TOOLS/binaryen-$BINARYEN/bin/wasm-opt"

cargo build --locked --profile wasm-release --target wasm32-unknown-unknown -p campaign-wasm
rm -rf "$OUT"
"$bindgen" --target web --no-typescript --out-dir "$OUT" \
  target/wasm32-unknown-unknown/wasm-release/campaign_wasm.wasm
# The features Rust's wasm32 target turns on, so wasm-opt reads what it's given.
"$opt" -Oz --enable-bulk-memory --enable-nontrapping-float-to-int --enable-sign-ext \
  --enable-reference-types --enable-multivalue \
  "$OUT/campaign_wasm_bg.wasm" -o "$OUT/campaign_wasm_bg.wasm"
echo "built $OUT: $(wc -c <"$OUT/campaign_wasm_bg.wasm") bytes of wasm"
