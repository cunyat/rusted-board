use crate::engine::{draw_layer, Board, Move, MoveDir, MoveKind, Piece, PieceType};

mod engine;

fn main() {
    // for i in [
    //     34359738368_u64,
    // ] {
    //     println!("{} = 1 << {}",i, i.trailing_zeros())
    // }

    let mut board = Board::initial();

    match board.make_move(Move::quiet(
        Piece::White(PieceType::Pawn),
        1 << 11,
        1 << 27,
        MoveDir::N,
    )) {
        Ok(_) => {}
        Err(e) => println!("[ERROR] {}", e),
    };

    match board.make_move(Move::quiet(
        Piece::Black(PieceType::Pawn),
        1 << 52,
        1 << 36,
        MoveDir::S,
    )) {
        Ok(_) => {}
        Err(e) => println!("[ERROR] {}", e),
    };

    match board.make_move(Move::new(
        Piece::White(PieceType::Pawn),
        1 << 27,
        1 << 36,
        MoveDir::NE,
        MoveKind::Capture,
    )) {
        Ok(_) => {}
        Err(e) => println!("[ERROR] {}", e),
    };

    board.debug();
    draw_layer(board.player_attacking_layer())

    // for mv in board.generate_moves() {
    //     engine::draw_move(&mv);
    // }
}
