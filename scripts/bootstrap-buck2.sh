#!/bin/sh

set -eu

PRELUDE_REPOSITORY="https://github.com/facebook/buck2-prelude.git"
PRELUDE_REVISION="03e5fc33baace1ab207f6c97d8bca3d1c88a4216"

repo_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
prelude_dir="$repo_root/prelude"

if [ -d "$prelude_dir/.git" ]; then
    current_revision=$(git -C "$prelude_dir" rev-parse HEAD)
    if [ "$current_revision" = "$PRELUDE_REVISION" ]; then
        echo "Buck2 prelude is already bootstrapped at $PRELUDE_REVISION"
        exit 0
    fi

    echo "error: $prelude_dir exists at revision $current_revision" >&2
    echo "remove it and rerun this script to install $PRELUDE_REVISION" >&2
    exit 1
fi

if [ -e "$prelude_dir" ]; then
    echo "error: $prelude_dir already exists and is not a Git checkout" >&2
    exit 1
fi

echo "Cloning Buck2 prelude..."
git clone "$PRELUDE_REPOSITORY" "$prelude_dir"
git -C "$prelude_dir" checkout --detach "$PRELUDE_REVISION"
echo "Buck2 prelude installed at $PRELUDE_REVISION"
