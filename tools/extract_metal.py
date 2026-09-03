#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Box3D Metal contributors
# SPDX-License-Identifier: MIT
"""One-shot Phase 5a extractor: move MSL out of metal_backend.m string literals.

Reads src/metal_backend.m, decodes the b3_metalSource / b3_contactSource C
string concatenations, and emits:

  src/metal/box3d_main.metal     decoded main-source MSL (structs removed)
  src/metal/box3d_contact.metal  decoded contact-source MSL (structs removed)
  src/metal_abi.h                shared header:
                                   MSL side  (#if __METAL_VERSION__): the union
                                     of struct definitions from both sources
                                   C side    (#else): the _Static_assert ABI
                                     block moved verbatim from metal_backend.m
and rewrites src/metal_backend.m in place:

  - every line starting with _Static_assert is deleted (moved to metal_abi.h)
  - both MSL string literals plus their overlength-strings pragma guards are
    deleted; #include "metal_abi.h" is inserted at the first deletion point
    (after all C struct definitions, so the moved asserts see complete types)

Duplicate struct definitions across the two sources must be layout-identical
after whitespace normalization; any mismatch aborts the extraction for manual
resolution. Kernel/helper function names must not collide across the two
files (they link into one metallib).

Usage:
    python3 tools/extract_metal.py --check   # report boundaries, write nothing
    python3 tools/extract_metal.py --extract # write files, rewrite metal_backend.m
"""

import re
import sys

SRC = "src/metal_backend.m"
MAIN_OUT = "src/metal/box3d_main.metal"
CONTACT_OUT = "src/metal/box3d_contact.metal"
ABI_OUT = "src/metal_abi.h"


def load():
    with open(SRC, "r", encoding="utf-8") as f:
        return f.read().splitlines()


def find_literal(lines, marker):
    """Return (first_content_line, last_content_line) 0-based inclusive."""
    found = [i for i, l in enumerate(lines) if l.startswith(marker)]
    if not found:
        print("marker not found (already extracted): %s" % marker)
        sys.exit(2)
    start = found[0]
    first = start + 1
    last = next(
        i for i in range(first, len(lines)) if lines[i].rstrip().endswith('";')
    )
    return first, last


def decode(lines, first, last):
    out = []
    for i in range(first, last + 1):
        line = lines[i]
        stripped = line.strip()
        assert stripped.startswith('"'), "unexpected literal line %d: %r" % (i + 1, line[:80])
        if i == last:
            assert stripped.endswith('";'), "last literal line %d missing terminator" % (i + 1)
            body = stripped[1:-2]
        else:
            assert stripped.endswith('"') and not stripped.endswith('";'), (
                "unexpected terminator at line %d" % (i + 1)
            )
            body = stripped[1:-1]
        # Decode C escapes the same way the compiler does for this ASCII text.
        out.append(body.encode("utf-8").decode("unicode_escape"))
    # Each literal carried its own trailing newline; re-split into true lines.
    text = "".join(out)
    result = text.split("\n")
    assert result[-1] == "", "decoded source must end with a newline"
    return result[:-1]


def split_structs(msl_lines):
    """Split MSL lines into (struct_blocks, rest). A struct block starts at a
    line whose stripped form starts with 'struct NAME {' and ends at the first
    line containing '};'. Returns dict name -> text and the remaining lines."""
    structs = {}
    rest = []
    i = 0
    n = len(msl_lines)
    while i < n:
        m = re.match(r"^struct\s+(\w+)\s*\{", msl_lines[i].strip())
        if m:
            name = m.group(1)
            j = i
            while "};" not in msl_lines[j]:
                j += 1
                assert j < n, "unterminated struct %s" % name
            assert name not in structs, "duplicate struct in one source: %s" % name
            structs[name] = "\n".join(msl_lines[i : j + 1])
            i = j + 1
        else:
            rest.append(msl_lines[i])
            i += 1
    return structs, rest


def norm_struct(text):
    # Token-level comparison: both sources compile today, so both bodies are
    # valid MSL; layout differences survive whitespace removal while
    # formatting-only differences (line breaks, ", " vs ",") do not.
    return re.sub(r"\s+", "", text).strip()


def kernel_names(msl_lines):
    names = set()
    for line in msl_lines:
        m = re.search(r"kernel\s+void\s+(\w+)\s*\(", line)
        if m:
            names.add(m.group(1))
    return names


def helper_names(msl_lines):
    """Top-level (non-kernel, non-struct) function definitions."""
    names = set()
    for line in msl_lines:
        s = line.strip()
        if (
            s
            and not s.startswith("struct")
            and not s.startswith("#")
            and not s.startswith("using")
            and not s.startswith("constant")
            and not s.startswith("}")
            and re.match(r"^[\w:]+\s+\w+\s*\(.*\)\s*\{", s)
        ):
            m = re.match(r"^[\w:]+\s+(\w+)\s*\(", s)
            if m and m.group(1) not in ("if", "for", "while", "switch", "return"):
                names.add(m.group(1))
    return names


def topo_order(blocks):
    names = list(blocks)
    deps = {}
    for name, block in blocks.items():
        body = block[len("struct " + name):]
        deps[name] = sorted(
            {other for other in names if other != name and re.search(r"\b%s\b" % other, body)}
        )
    ordered = []
    done = set()
    while len(ordered) < len(names):
        progress = False
        for name in names:
            if name not in done and all(d in done for d in deps[name]):
                ordered.append(name)
                done.add(name)
                progress = True
        assert progress, "struct dependency cycle: %s" % (set(names) - done)
    return ordered


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "--check"
    lines = load()

    # --- assert lines: every line starting with _Static_assert, anywhere ---
    assert_lines = [i for i, l in enumerate(lines) if l.startswith("_Static_assert")]
    if not assert_lines:
        print("no file-scope _Static_assert lines: sources already live in src/metal/*.metal")
        return 0
    print("assert lines: %d (first %d, last %d)" % (
        len(assert_lines), assert_lines[0] + 1, assert_lines[-1] + 1))

    m_first, m_last = find_literal(lines, "static const char* b3_metalSource =")
    c_first, c_last = find_literal(lines, "static const char* b3_contactSource =")

    print("main source lines %d..%d" % (m_first + 1, m_last + 1))
    print("contact source lines %d..%d" % (c_first + 1, c_last + 1))

    main_msl = decode(lines, m_first, m_last)
    contact_msl = decode(lines, c_first, c_last)
    print("decoded main lines: %d contact lines: %d" % (len(main_msl), len(contact_msl)))

    # Both sources start with stdlib include + using; strip (re-added by emitter).
    for label, msl in (("main", main_msl), ("contact", contact_msl)):
        assert msl[0] == "#include <metal_stdlib>", (label, msl[0])
        assert msl[1] == "using namespace metal;", (label, msl[1])
        del msl[0:2]

    main_structs, main_rest = split_structs(main_msl)
    contact_structs, contact_rest = split_structs(contact_msl)
    print(
        "main structs: %d contact structs: %d" % (len(main_structs), len(contact_structs))
    )

    overlap = set(main_structs) & set(contact_structs)
    print("overlapping structs: %d" % len(overlap))
    mismatched = [
        name
        for name in sorted(overlap)
        if norm_struct(main_structs[name]) != norm_struct(contact_structs[name])
    ]
    if mismatched:
        print("MISMATCHED struct layouts (resolve manually):")
        for name in mismatched:
            print("--- main %s ---\n%s" % (name, main_structs[name]))
            print("--- contact %s ---\n%s" % (name, contact_structs[name]))
        sys.exit(1)

    union_structs = dict(main_structs)
    for name in contact_structs:
        if name not in union_structs:
            union_structs[name] = contact_structs[name]
    # Declaration-before-use order for MSL (encounter order is already valid
    # since each source compiled standalone; topo-sort to be certain).
    struct_order = topo_order(union_structs)
    union_structs = {name: union_structs[name] for name in struct_order}
    print("union structs: %d" % len(union_structs))

    main_kernels = kernel_names(main_rest)
    contact_kernels = kernel_names(contact_rest)
    print(
        "main kernels: %d contact kernels: %d overlap: %s"
        % (len(main_kernels), len(contact_kernels), sorted(main_kernels & contact_kernels))
    )
    assert not (main_kernels & contact_kernels), "kernel name collision"

    main_helpers = helper_names(main_rest)
    contact_helpers = helper_names(contact_rest)
    collisions = (main_helpers & contact_helpers) | (
        main_helpers & contact_kernels
    ) | (contact_helpers & main_kernels)
    print("helper collisions across files: %s" % sorted(collisions))
    assert not collisions, "helper name collision"

    if mode != "--extract":
        print("check OK; rerun with --extract to write files")
        return 0

    header = "#include <metal_stdlib>\nusing namespace metal;\n#include \"metal_abi.h\"\n"
    with open(MAIN_OUT, "w", encoding="utf-8") as f:
        f.write(header + "\n".join(main_rest).strip() + "\n")
    with open(CONTACT_OUT, "w", encoding="utf-8") as f:
        f.write(header + "\n".join(contact_rest).strip() + "\n")

    with open(ABI_OUT, "w", encoding="utf-8") as f:
        f.write("// SPDX-FileCopyrightText: 2026 Box3D Metal contributors\n")
        f.write("// SPDX-License-Identifier: MIT\n\n")
        f.write("// Shared CPU/MSL ABI for the Metal backend (Phase 5a).\n")
        f.write("// Generated by tools/extract_metal.py from metal_backend.m; do not hand-edit\n")
        f.write("// the struct layouts. Regenerate after changing either side.\n")
        f.write("#ifndef B3_METAL_ABI_H\n")
        f.write("#define B3_METAL_ABI_H\n")
        f.write("#pragma once\n\n")
        f.write("#if defined( __METAL_VERSION__ )\n")
        f.write("// MSL struct layouts consumed by src/metal/*.metal. Order is\n")
        f.write("// declaration order (main source, then contact-only structs)\n")
        f.write("// because MSL requires declaration before use.\n")
        for name in union_structs:
            f.write(union_structs[name] + "\n")
        f.write("\n// Reserved function-constant indices for later phases\n")
        f.write("// (5b fast math, 2 compact solver, 4 hull-hull binning). No kernel\n")
        f.write("// reads them yet; they only reserve the contract.\n")
        f.write("constant bool B3_DOUBLE [[function_constant( 0 )]];\n")
        f.write("constant bool B3_HALF_MATERIALS [[function_constant( 1 )]];\n")
        f.write("constant uint B3_HULL_CLASS [[function_constant( 2 )]];\n")
        f.write("constant uint B3_SUBSTEP_COUNT [[function_constant( 3 )]];\n")
        f.write("#endif\n")
        f.write("#else\n")
        f.write("// C-side ABI assertions, moved verbatim from metal_backend.m.\n")
        f.write("// Metal's native float3 is 16-byte aligned, so the shader\n")
        f.write("// deliberately uses scalar fields matching this layout.\n")
        for i in assert_lines:
            f.write(lines[i] + "\n")
        f.write("#endif // B3_METAL_ABI_H\n")
    print("wrote %s %s %s" % (MAIN_OUT, CONTACT_OUT, ABI_OUT))

    # --- rewrite metal_backend.m: drop asserts, strings, and string pragmas ---
    m_first, m_last = find_literal(lines, "static const char* b3_metalSource =")
    c_first, c_last = find_literal(lines, "static const char* b3_contactSource =")
    drop = set(assert_lines)
    # overlength-strings pragma guards around the two literals (leave the
    # earlier vf64 guard at the top of the file alone).
    for i in (m_first - 3, m_first - 2):
        assert lines[i].strip() in (
            "#pragma clang diagnostic push",
            '#pragma clang diagnostic ignored "-Woverlength-strings"',
        ), "unexpected guard line %d: %r" % (i + 1, lines[i])
        drop.add(i)
    for i in range(m_first, m_last + 1):
        drop.add(i)
    # Declaration line and the pop closing the main-source guard.
    assert lines[m_first - 1].strip() == "static const char* b3_metalSource =", (
        "main marker mismatch: %r" % lines[m_first - 1]
    )
    drop.add(m_first - 1)
    assert lines[m_last + 1].strip() == "#pragma clang diagnostic pop", (
        "expected pop after main source: %r" % lines[m_last + 1]
    )
    drop.add(m_last + 1)
    for i in (c_first - 3, c_first - 2):
        assert lines[i].strip() in (
            "#pragma clang diagnostic push",
            '#pragma clang diagnostic ignored "-Woverlength-strings"',
        ), "unexpected guard line %d: %r" % (i + 1, lines[i])
        drop.add(i)
    for i in range(c_first, c_last + 1):
        drop.add(i)
    assert lines[c_first - 1].strip() == "static const char* b3_contactSource =", (
        "contact marker mismatch: %r" % lines[c_first - 1]
    )
    drop.add(c_first - 1)
    assert lines[c_last + 1].strip() == "#pragma clang diagnostic pop", (
        "expected pop after contact source: %r" % lines[c_last + 1]
    )
    drop.add(c_last + 1)
    # Stale intro comment describing the moved checks (content-addressed).
    comment_a = "// These checks protect the shared C/MSL ABI. Metal's native float3 is 16-byte"
    comment_b = "// aligned, so the shader deliberately uses scalar fields matching this layout."
    found = [i for i, l in enumerate(lines) if l.strip() == comment_a]
    assert len(found) == 1, "intro comment not found exactly once"
    assert lines[found[0] + 1].strip() == comment_b, (
        "intro comment continuation mismatch: %r" % lines[found[0] + 1]
    )
    drop.add(found[0])
    drop.add(found[0] + 1)

    new_lines = []
    for i, l in enumerate(lines):
        if i in drop:
            if i == m_first - 3:
                new_lines.append('#include "metal_abi.h"')
            continue
        new_lines.append(l)
    with open(SRC, "w", encoding="utf-8") as f:
        f.write("\n".join(new_lines) + "\n")
    print("rewrote %s (%d lines removed)" % (SRC, len(drop)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
