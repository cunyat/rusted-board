use crate::engine::movement::Direction;
use crate::engine::Piece;

pub enum SpecialMoves {
    PawnAdvance,
    PawnCapture,
    CastleLong,
    CastleShort,
}

pub struct PotentialMove {
    pub(crate) piece: Piece,
    pub(crate) dir: Direction,
    pub(crate) slicing: bool,
    pub(crate) special: Option<SpecialMoves>,
}

impl PotentialMove {
    pub fn new(
        piece: Piece,
        dir: Direction,
        slicing: bool,
        special: Option<SpecialMoves>,
    ) -> PotentialMove {
        PotentialMove {
            piece,
            dir,
            slicing,
            special,
        }
    }
}
