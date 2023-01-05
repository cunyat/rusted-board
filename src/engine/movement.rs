use std::error::Error;
use std::fmt;
use std::fmt::Formatter;

use crate::engine::draw_table;
use crate::engine::piece::Piece;

/// Move represents a move.
/// It is used for applying a movement in the board
/// and keeping movements history.
#[derive(Debug, Clone, PartialEq)]
pub struct Move {
    pub(crate) piece: Piece,
    pub(crate) from: u64,
    pub(crate) to: u64,
    pub(crate) kind: Kind,
}

impl Move {
    pub fn new(piece: Piece, from: u64, to: u64, kind: Kind) -> Move {
        Move {
            piece,
            from,
            to,
            kind,
        }
    }

    pub fn draw(&self) {
        let mut out = [' '; 64];

        out[self.from.trailing_zeros() as usize] = self.piece.to_char();
        out[self.to.trailing_zeros() as usize] = match self.kind {
            Kind::Quiet => 'O',
            Kind::Capture => 'X',
            Kind::CastleShort => 'C',
            Kind::CastleLong => 'C',
        };

        draw_table(out);
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum Kind {
    Quiet,
    Capture,
    CastleShort,
    CastleLong,
}

#[derive(PartialEq, Debug)]
pub enum Direction {
    N,
    NE,
    E,
    SE,
    S,
    SW,
    W,
    NW,

    // Knight moves
    NNE,
    NEE,
    SEE,
    SSE,
    SSW,
    SWW,
    NWW,
    NNW,

    // King castle moves
    CastleShort,
    CastleLong,
}

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

impl Direction {
    pub fn is_north(&self) -> bool {
        match self {
            Direction::N
            | Direction::NE
            | Direction::NW
            | Direction::NNE
            | Direction::NEE
            | Direction::NWW
            | Direction::NNW => true,
            _ => false,
        }
    }

    pub fn is_south(&self) -> bool {
        match self {
            Direction::S
            | Direction::SE
            | Direction::SW
            | Direction::SEE
            | Direction::SSE
            | Direction::SSW
            | Direction::SWW => true,
            _ => false,
        }
    }

    pub fn is_east(&self) -> bool {
        match self {
            Direction::E
            | Direction::NE
            | Direction::SE
            | Direction::NNE
            | Direction::NEE
            | Direction::SSE
            | Direction::SEE => true,
            _ => false,
        }
    }

    pub fn is_west(&self) -> bool {
        match self {
            Direction::W
            | Direction::NW
            | Direction::SW
            | Direction::NNW
            | Direction::NWW
            | Direction::SSW
            | Direction::SWW => true,
            _ => false,
        }
    }

    pub fn is_knight(&self) -> bool {
        match self {
            Direction::NNE
            | Direction::NEE
            | Direction::SEE
            | Direction::SSE
            | Direction::SSW
            | Direction::SWW
            | Direction::NWW
            | Direction::NNW => true,
            _ => false,
        }
    }

    pub fn is_diagonal(&self) -> bool {
        match self {
            Direction::NE | Direction::SE | Direction::SW | Direction::NW => true,
            _ => false,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct MoveError {
    mv: Move,
    reason: String,
}

impl MoveError {
    pub fn new(mv: Move, reason: String) -> MoveError {
        MoveError { mv, reason }
    }
}

impl fmt::Display for MoveError {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "unable to move {:?}: {}", self.mv.piece, self.reason)
    }
}

impl Error for MoveError {}
