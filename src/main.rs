use crate::engine::Color::{Black, White};
use crate::engine::Kind::Pawn;
use crate::engine::{Board, Move, MoveKind, Piece};

mod engine;

fn main() {
    let mut board = Board::initial();

    match board.make_move(&Move::new(
        Piece::new(Pawn, White),
        1 << 11,
        1 << 27,
        MoveKind::Quiet,
    )) {
        Ok(_) => {}
        Err(err) => panic!("[ERROR]: {}", err),
    }

    match board.make_move(&Move::new(
        Piece::new(Pawn, Black),
        1 << 52,
        1 << 36,
        MoveKind::Quiet,
    )) {
        Ok(_) => {}
        Err(err) => panic!("[ERROR]: {}", err),
    }

    match board.make_move(&Move::new(
        Piece::new(Pawn, White),
        1 << 27,
        1 << 36,
        MoveKind::Capture,
    )) {
        Ok(_) => {}
        Err(err) => panic!("[ERROR]: {}", err),
    }

    board.draw();

    for mv in board.generate_moves() {
        mv.draw();
    }
}
