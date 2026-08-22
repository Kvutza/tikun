platform(
    name = "macos-arm64-platform",
    constraint_values = [
        "prelude//cpu/constraints:arm64",
        "prelude//os/constraints:macos",
    ],
    visibility = ["PUBLIC"],
)

alias(
    name = "cpu",
    actual = ":tikun_cpu",
    visibility = ["PUBLIC"],
)

alias(
    name = "gpu",
    actual = ":tikun_metal",
    visibility = ["PUBLIC"],
)

alias(
    name = "dev",
    actual = ":tikun_dev",
    visibility = ["PUBLIC"],
)

alias(
    name = "core",
    actual = ":tikun_core",
    visibility = ["PUBLIC"],
)

alias(
    name = "py",
    actual = ":tikun_py",
    visibility = ["PUBLIC"],
)

load("@toolchains//:defs.bzl", "tikun_rust_library")

tikun_rust_library(
    name = "tikun_core",
    srcs = glob(["crates/tikun-core/src/**/*.rs"]),
    crate_root = "crates/tikun-core/src/lib.rs",
    deps = [
        "//third-party/rust:serde",
        "//third-party/rust:serde_json",
        "//third-party/rust:comfy-table",
        "//third-party/rust:ennx",
    ],
)

tikun_rust_library(
    name = "tikun_cpu",
    srcs = glob(["crates/tikun-cpu/src/**/*.rs"]),
    crate_root = "crates/tikun-cpu/src/lib.rs",
    linker_flags = [
        "-framework",
        "Accelerate",
    ],
    deps = [
        ":tikun_core",
        "//third-party/rust:rayon",
        "//third-party/rust:serde",
        "//third-party/rust:serde_json",
    ],
)

cxx_library(
    name = "metal_kernels",
    srcs = glob(["metal/*.metal"]),
    visibility = ["PUBLIC"],
)

genrule(
    name = "metallib",
    srcs = [
        "metal/adamw.metal",
    ],
    out = "adamw.metallib",
    cmd = "xcrun -sdk macosx metal -c $SRCDIR/metal/adamw.metal -o adamw.air && xcrun -sdk macosx metallib adamw.air -o $OUT",
    visibility = ["PUBLIC"],
)

tikun_rust_library(
    name = "tikun_metal",
    srcs = glob(["crates/tikun-metal/src/**/*.rs"]),
    crate_root = "crates/tikun-metal/src/lib.rs",
    linker_flags = [
        "-framework",
        "Metal",
    ],
    deps = [
        ":tikun_core",
        "//third-party/rust:metal",
        "//third-party/rust:objc",
    ],
)

rust_library(
    name = "tikun_py",
    srcs = glob(["crates/tikun-py/src/**/*.rs"]),
    crate_root = "crates/tikun-py/src/lib.rs",
    linker_flags = [
        "-undefined",
        "dynamic_lookup",
        "-framework",
        "Accelerate",
        "-framework",
        "Metal",
    ],
    deps = [
        ":tikun_core",
        ":tikun_cpu",
        ":tikun_metal",
        "//third-party/rust:pyo3",
        "//third-party/rust:numpy",
    ],
    visibility = ["PUBLIC"],
)

rust_binary(
    name = "tikun_dev",
    srcs = glob(["crates/tikun-dev/src/**/*.rs"]),
    crate_root = "crates/tikun-dev/src/main.rs",
    linker_flags = [
        "-framework",
        "Accelerate",
    ],
    deps = [
        ":tikun_core",
        ":tikun_cpu",
        "//third-party/rust:clap",
        "//third-party/rust:serde",
        "//third-party/rust:serde_json",
    ],
    visibility = ["PUBLIC"],
)

sh_test(
    name = "check",
    test = "tests/test_parity.sh",
    deps = [
        ":tikun_core",
        ":tikun_cpu",
    ],
    visibility = ["PUBLIC"],
)

test_suite(
    name = "all_tests",
    tests = [
        ":tikun_core_tests",
        ":tikun_cpu_tests",
        ":check",
    ],
    visibility = ["PUBLIC"],
)

alias(
    name = "rust_tests",
    actual = ":all_tests",
    visibility = ["PUBLIC"],
)

