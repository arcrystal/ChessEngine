# ChessEngine

File structure:
src/
 ├── bitboard.py              # Piece move generation (sliding, knight, king, etc.)
 ├── bitboard_utils.py         # Pure helpers (square calc, attacks, is_in_check)
 ├── bitboard_game.py         # BitboardGameState class + apply_move / undo_move
 ├── generate_moves.py         # generate_all_moves, generate_legal_moves
 ├── engine.py                 # Minimax/Quiescence