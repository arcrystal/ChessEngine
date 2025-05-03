from numba import types
from numba.typed import List

PAWN = 1
KNIGHT = 2
BISHOP = 3
ROOK = 4
QUEEN = 5
KING = 6
EMPTY = 7

board_str = """
  +-------------------------+
8 | 56 57 58 59 60 61 62 63 |
  |                         |
7 | 48 49 50 51 52 53 54 55 |
  |                         |
6 | 40 41 42 43 44 45 46 47 |
  |                         |
5 | 32 33 34 35 36 37 38 39 |
  |                         |
4 | 24 25 26 27 28 29 30 31 |
  |                         |
3 | 16 17 18 19 20 21 22 23 |
  |                         |
2 | 08 09 10 11 12 13 14 15 |
  |                         |
1 | 00 01 02 03 04 05 06 07 |
  +-------------------------+
    a  b  c  d  e  f  g  h
"""
# Define move triples
f2f3 = (13, 21, 0)
f2f4 = (13, 29, 0)
g2g3 = (14, 22, 0)
g2g4 = (14, 30, 0)
e7e5 = (51, 35, 0)
e7e6 = (51, 43, 0)
d8h4 = (59, 31, 0)

# Helper to create Numba list from Python tuple of 3 moves
def make_sequence(a, b, c):
    l = List.empty_list(types.UniTuple(types.int64, 3))
    l.append(a)
    l.append(b)
    l.append(c)
    return l

# Compose test mates
move4_mates = List.empty_list(types.ListType(types.UniTuple(types.int64, 3)))
for seq in [
    (f2f3, e7e6, g2g4),
    (f2f3, e7e5, g2g4),
    (f2f4, e7e6, g2g4),
    (f2f4, e7e5, g2g4),
    (g2g4, e7e6, f2f3),
    (g2g4, e7e6, f2f4),
    (g2g4, e7e5, f2f3),
    (g2g4, e7e5, f2f4),
]:
    move4_mates.append(make_sequence(*seq))

