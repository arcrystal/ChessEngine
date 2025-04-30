from bitboards_magic import bishop_attacks_on_the_fly, rook_attacks_on_the_fly, print_bitboard
from bitboard_game import BitboardGameState

gs = BitboardGameState()

square = 3  # Queen on d1
gs.make_move((12, 20, 0)) # e2 e3
blockers = int(gs.occupied)
bb = bishop_attacks_on_the_fly(square, blockers)
gs.print_board()
print_bitboard(bb, "Queen attacks from d1")

gs.undo_move()
gs.make_move((15, 16, 0)) # h2 a3
square = 7  # Rook on h1
blockers = int(gs.occupied)
bb = rook_attacks_on_the_fly(square, blockers)

gs.print_board()
print_bitboard(bb, "Rook attacks from h1")
