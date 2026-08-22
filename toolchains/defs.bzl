# Reusable Buck2 Macros for Tikun Monorepo

def tikun_rust_library(name, crate_root, srcs, deps = [], linker_flags = [], visibility = ["PUBLIC"]):
    native.rust_library(
        name = name,
        srcs = srcs,
        crate_root = crate_root,
        deps = deps,
        linker_flags = linker_flags,
        visibility = visibility,
    )

    native.rust_test(
        name = name + "_tests",
        srcs = srcs,
        crate_root = crate_root,
        deps = deps,
        linker_flags = linker_flags,
        visibility = visibility,
    )
