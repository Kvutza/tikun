# tikun

Tikun is consumed as a normal Rust crate. Downstream Cargo/Reindeer/Buck2
projects do not need Tikun's Buck2 prelude or toolchains.

## Local Buck2 development

The Buck2 prelude is intentionally not stored as a Git submodule. Bootstrap the
pinned development copy after cloning the repository:

```sh
./scripts/bootstrap-buck2.sh
```

The script downloads the expected Buck2 prelude revision into the ignored
`prelude/` directory. It is only needed to run Tikun's own Buck2 build; Cargo
and downstream Buck2 consumers do not use it.

```sh
# Bootstrap once after cloning
./scripts/bootstrap-buck2.sh

# Build workspace
./tikun build

# Run static verifier and workspace check
./tikun check

# Run tests
./tikun test
```

## Workspace Layout

- `crates/tikun-core`: PyTree metadata (`TreeDef`), Schedule IR, and static verifier.
- `crates/tikun-cpu`: Vectorized SIMD lowering engine.
- `crates/tikun-metal`: Metal compute shader lowering engine.
- `crates/tikun-py`: PyO3 CPython 3.14 module surface.
