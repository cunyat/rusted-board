use crate::engine::MoveDir;

#[derive(Debug, Copy, Clone, PartialEq)]
pub enum Kind {
    Pawn,
    Bishop,
    Knight,
    Rock,
    Queen,
    King,
}

pub trait Piece {
    fn kind(&self) -> Kind;

    fn moveset(&self) -> Vec<MoveDir>;
}
