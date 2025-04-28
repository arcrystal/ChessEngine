import numpy as np

def bitboard_to_square(bb):
    """Convert a bitboard with a single bit set into its square index (0-63)."""
    return np.uint8(np.log2(bb))

def is_in_check(gs, white_to_move):
    """Check if the current player's king is attacked."""
    king_bb = gs.white_king if white_to_move else gs.black_king
    king_sq = bitboard_to_square(king_bb)

    enemy_moves = generate_all_moves(gs, only_attacks=True)  # later optimize
    for move in enemy_moves:
        _, to_sq, _ = move
        if to_sq == king_sq:
            return True
    return False

def apply_move(gs, move):
    """
    Apply a move to the BitboardGameState.
    Returns move_info needed to undo it.
    """
    from_sq, to_sq, promotion = move
    moving_piece_bb = 1 << from_sq
    captured_piece_bb = gs.occupancy[to_sq]

    # Save undo info
    move_info = (
        gs.white_pieces.copy(),
        gs.black_pieces.copy(),
        gs.white_king, 
        gs.black_king,
        gs.castling_rights.copy(), 
        gs.en_passant, 
        gs.halfmove_clock
    )

    # Apply move
    if gs.white_to_move:
        gs.white_pieces &= ~moving_piece_bb
        gs.white_pieces |= (1 << to_sq)
        if captured_piece_bb:
            gs.black_pieces &= ~(1 << to_sq)
        if from_sq == bitboard_to_square(gs.white_king):
            gs.white_king = 1 << to_sq
    else:
        gs.black_pieces &= ~moving_piece_bb
        gs.black_pieces |= (1 << to_sq)
        if captured_piece_bb:
            gs.white_pieces &= ~(1 << to_sq)
        if from_sq == bitboard_to_square(gs.black_king):
            gs.black_king = 1 << to_sq

    # Update occupancy, halfmove clock, etc.
    gs.update_bitboards()
    gs.white_to_move ^= 1

    return move_info

def undo_move(gs, move, move_info):
    """
    Undo a move by restoring saved game state.
    """
    (
        white_pieces,
        black_pieces,
        white_king, 
        black_king,
        castling_rights,
        en_passant, 
        halfmove_clock
    ) = move_info

    gs.white_pieces = white_pieces
    gs.black_pieces = black_pieces
    gs.white_king = white_king
    gs.black_king = black_king
    gs.castling_rights = castling_rights
    gs.en_passant = en_passant
    gs.halfmove_clock = halfmove_clock
    gs.white_to_move ^= 1

    gs.update_bitboards()