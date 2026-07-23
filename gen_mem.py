#!/usr/bin/env python3
"""
gen_mem.py
===================================================

#### DOCUMENTATION REDGARDING MEMORY LAYOUT AND STORAGE FORMATS

there are 4 storage formats possible for the weights in the memory image:
RM_RM: row-major inside each 4x4 tile, and row-major across the tile grid
RM_CM: row-major inside each 4x4 tile, and column-major across the tile grid
CM_RM: column-major inside each 4x4 tile, and row-major across the tile grid
CM_CM: column-major inside each 4x4 tile, and column-major across the tile grid

the output of a systolic array operation currently gives out 2 format RM_RM and RM_CM
hence it is recommended to use RM_CM storage format of weights and biases in memory
meanwhile inputs can be saved in both RM_RM or CM_RM format, and need to be changed according to it in assembly code

===================================================

Generic weight -> memory-image writer for custom RTL models.

Given a JSON spec listing one or more "sets", each describing:
    - file            : path to a text file containing the matrix values,
                         stored row-major (whitespace / comma / newline separated numbers)
    - start_address    : integer address (word address) at which this set's
                         data begins in the output .mem file
    - rows, cols        : logical shape of the matrix as stored in `file`
    - format            : "F1_F2" where
                             F1 (format1) = how the 16 elements *inside* each
                                            4x4 tile are serialized
                                            ("RM" = row-major, "CM" = column-major)
                             F2 (format2) = the order in which 4x4 tiles are
                                            visited across the tile grid
                                            ("RM" = row-major tile traversal,
                                             "CM" = column-major tile traversal)

All sets are written into the SAME output .mem file, each at its own
start_address. Matrices whose rows/cols aren't a multiple of 4 are
zero-padded up to the next multiple of 4 before tiling.

Example JSON spec (spec.json):
[
  {
    "file": "layer1_weights.txt",
    "start_address": 0,
    "rows": 8,
    "cols": 8,
    "format": "RM_CM"
  },
  {
    "file": "layer2_weights.txt",
    "start_address": 128,
    "rows": 16,
    "cols": 4,
    "format": "CM_RM"
  }
]

Consecutive placement:
    "start_address" is OPTIONAL per set. Omit it and
    that set is placed immediately after wherever the previous set finished
    writing (i.e. previous set's last used address + 1). This includes any
    zero-padding the previous set needed to reach a multiple of 4 in both
    dimensions, so you never have to hand-compute offsets. The very first
    set, if it omits start_address, defaults to address 0.

    Example:
    [
      {"file": "layer1_weights.txt", "rows": 8, "cols": 8,  "format": "RM_CM"},
      {"file": "layer1_biases.txt",  "rows": 4, "cols": 64, "format": "RM_CM"},
      {"file": "layer2_weights.txt", "rows": 8, "cols": 32, "format": "CM_CM"}
    ]
    -> layer1_weights.txt starts at 0, layer1_biases.txt starts right after
       it ends, layer2_weights.txt starts right after that.

    You can still mix explicit and auto: any set CAN give a start_address
    (e.g. to leave a gap, or jump to a fixed base for a new region), and
    later "auto" sets will resume counting from wherever that set ended.

Usage:
    python3 gen_mem.py spec.json output.mem --word-bits 8

Notes / assumptions (adjust as needed for your flow):
    - Values are treated as signed integers and written as zero-padded hex,
      one value per line, at word granularity (word-addressed .mem file,
      i.e. line N in the file == address N). --word-bits controls the hex
      width (default 8 -> 2 hex chars, matches int8 quantized weights).
    - Any address never written to (gaps between sets, or before the first
      set if it doesn't start at 0) is filled with 0x00 in the final file.
    - The parser for the weights file just pulls all integers/floats out
      via regex, in row-major order, and reshapes to (rows, cols). If your
      weights are floats you want to keep as floats instead of quantized
      ints, use --keep-float (values are then written as raw hex of their
      IEEE-754 bit pattern is NOT done automatically -- flag it and I'll
      extend this; by default this script assumes pre-quantized integers).
"""

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np


NUM_RE = re.compile(r"-?\d+\.?\d*(?:[eE][-+]?\d+)?")


def load_matrix(filepath, rows, cols):
    """Read row-major numbers out of a text file and reshape to (rows, cols)."""
    text = Path(filepath).read_text()
    nums = [float(x) if ("." in x or "e" in x.lower()) else int(x)
            for x in NUM_RE.findall(text)]
    expected = rows * cols
    if len(nums) < expected:
        raise ValueError(
            f"{filepath}: expected {expected} values ({rows}x{cols}), "
            f"found only {len(nums)}"
        )
    if len(nums) > expected:
        nums = nums[:expected]
    return np.array(nums).reshape(rows, cols)


def pad_to_multiple_of_4(mat):
    """Zero-pad rows/cols up to the next multiple of 4."""
    r, c = mat.shape
    pad_r = (-r) % 4
    pad_c = (-c) % 4
    if pad_r or pad_c:
        mat = np.pad(mat, ((0, pad_r), (0, pad_c)), mode="constant", constant_values=0)
    return mat


def split_tiles(mat):
    """Split an (R*4, C*4) matrix into a grid of 4x4 tiles -> dict[(i,j)] = tile."""
    R, C = mat.shape
    tiles = {}
    for i in range(R // 4):
        for j in range(C // 4):
            tiles[(i, j)] = mat[4 * i:4 * i + 4, 4 * j:4 * j + 4]
    return tiles, (R // 4, C // 4)


def tile_grid_traversal_order(grid_shape, format2):
    """Order in which to visit tiles across the grid."""
    R, C = grid_shape
    if format2 == "RM":
        return [(i, j) for i in range(R) for j in range(C)]
    elif format2 == "CM":
        return [(i, j) for j in range(C) for i in range(R)]
    else:
        raise ValueError(f"Unknown format2 '{format2}', expected RM or CM")


def tile_to_values(tile, format1):
    """Serialize a single 4x4 tile's 16 elements."""
    if format1 == "RM":
        return tile.flatten(order="C")  # row-major: across rows
    elif format1 == "CM":
        return tile.flatten(order="F")  # column-major: down columns
    else:
        raise ValueError(f"Unknown format1 '{format1}', expected RM or CM")


def matrix_to_stream(mat, fmt):
    """Full pipeline: pad -> tile -> traverse -> serialize -> flat value list."""
    format1, format2 = fmt.split("_")
    padded = pad_to_multiple_of_4(mat)
    tiles, grid_shape = split_tiles(padded)
    order = tile_grid_traversal_order(grid_shape, format2)
    values = []
    for key in order:
        values.extend(tile_to_values(tiles[key], format1).tolist())
    return values


def process_spec(spec_path, out_path, word_bits=8):
    spec = json.loads(Path(spec_path).read_text())

    memory = {}  # address -> value
    max_addr = -1
    next_addr = 0  # running pointer for "auto"/omitted start_address

    for entry in spec:
        fname = entry["file"]
        rows = entry["rows"]
        cols = entry["cols"]
        fmt = entry["format"]

        raw_start = entry.get("start_address", "auto")
        start_addr = next_addr if raw_start == "auto" else raw_start

        mat = load_matrix(fname, rows, cols)
        values = matrix_to_stream(mat, fmt)

        addr = start_addr
        for v in values:
            memory[addr] = v
            max_addr = max(max_addr, addr)
            addr += 1

        next_addr = addr  # next auto set resumes right here

        print(f"[gen_mem] {fname}: {rows}x{cols} ({fmt}) -> "
              f"{len(values)} words @ 0x{start_addr:X}..0x{addr - 1:X}")

    hex_width = (word_bits + 3) // 4
    mask = (1 << word_bits) - 1

    lines = []
    for addr in range(0, max_addr + 1):
        val = memory.get(addr, 0)
        val_int = int(val) & mask  # two's-complement wrap for negatives
        lines.append(f"{val_int:0{hex_width}X}")

    Path(out_path).write_text("\n".join(lines) + "\n")
    print(f"[gen_mem] wrote {len(lines)} words to {out_path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("spec", help="path to JSON spec file")
    ap.add_argument("output", help="output .mem file path")
    ap.add_argument("--word-bits", type=int, default=8,
                     help="bit width per memory word (default: 8)")
    args = ap.parse_args()
    process_spec(args.spec, args.output, args.word_bits)


if __name__ == "__main__":
    main()