use crate::engine::movegen::PotentialMove;
use crate::engine::movegen::SpecialMoves::{CastleLong, CastleShort, PawnAdvance, PawnCapture};
use crate::engine::movement::Direction;
use crate::engine::player::Color;
use core::fmt;
use std::fmt::Formatter;

#[derive(Debug, Copy, Clone, PartialEq)]
pub enum Kind {
    Pawn,
    Bishop,
    Knight,
    Rook,
    Queen,
    King,
}

#[derive(Debug, Copy, Clone, PartialEq)]
pub struct Piece {
    pub(crate) kind: Kind,
    pub(crate) color: Color,
}

impl Piece {
    pub fn new(kind: Kind, color: Color) -> Piece {
        Piece { kind, color }
    }

    pub fn all() -> [Piece; 12] {
        [
            Piece::new(Kind::Pawn, Color::White),
            Piece::new(Kind::Bishop, Color::White),
            Piece::new(Kind::Knight, Color::White),
            Piece::new(Kind::Rook, Color::White),
            Piece::new(Kind::Queen, Color::White),
            Piece::new(Kind::King, Color::White),
            Piece::new(Kind::Pawn, Color::Black),
            Piece::new(Kind::Bishop, Color::Black),
            Piece::new(Kind::Knight, Color::Black),
            Piece::new(Kind::Rook, Color::Black),
            Piece::new(Kind::Queen, Color::Black),
            Piece::new(Kind::King, Color::Black),
        ]
    }

    pub fn white() -> [Piece; 6] {
        [
            Piece::new(Kind::Pawn, Color::White),
            Piece::new(Kind::Bishop, Color::White),
            Piece::new(Kind::Knight, Color::White),
            Piece::new(Kind::Rook, Color::White),
            Piece::new(Kind::Queen, Color::White),
            Piece::new(Kind::King, Color::White),
        ]
    }

    pub fn black() -> [Piece; 6] {
        [
            Piece::new(Kind::Pawn, Color::Black),
            Piece::new(Kind::Bishop, Color::Black),
            Piece::new(Kind::Knight, Color::Black),
            Piece::new(Kind::Rook, Color::Black),
            Piece::new(Kind::Queen, Color::Black),
            Piece::new(Kind::King, Color::Black),
        ]
    }

    pub fn for_color(color: Color) -> [Piece; 6] {
        match color {
            Color::Black => Piece::black(),
            Color::White => Piece::white(),
        }
    }

    pub fn potential_moves(&self) -> Vec<PotentialMove> {
        match self.kind {
            Kind::Pawn => {
                let (dir, east_capture, west_capture) = match self.color {
                    Color::White => (Direction::N, Direction::Ne, Direction::Nw),
                    Color::Black => (Direction::S, Direction::Se, Direction::Sw),
                };

                vec![
                    PotentialMove::new(*self, dir, false, Some(PawnAdvance)),
                    PotentialMove::new(*self, east_capture, false, Some(PawnCapture)),
                    PotentialMove::new(*self, west_capture, false, Some(PawnCapture)),
                ]
            }
            Kind::Bishop => vec![
                PotentialMove::new(*self, Direction::Ne, true, None),
                PotentialMove::new(*self, Direction::Se, true, None),
                PotentialMove::new(*self, Direction::Sw, true, None),
                PotentialMove::new(*self, Direction::Nw, true, None),
            ],
            Kind::Knight => vec![
                PotentialMove::new(*self, Direction::Nne, false, None),
                PotentialMove::new(*self, Direction::Nee, false, None),
                PotentialMove::new(*self, Direction::See, false, None),
                PotentialMove::new(*self, Direction::Sse, false, None),
                PotentialMove::new(*self, Direction::Ssw, false, None),
                PotentialMove::new(*self, Direction::Sww, false, None),
                PotentialMove::new(*self, Direction::Nww, false, None),
                PotentialMove::new(*self, Direction::Nnw, false, None),
            ],
            Kind::Rook => vec![
                PotentialMove::new(*self, Direction::N, true, None),
                PotentialMove::new(*self, Direction::E, true, None),
                PotentialMove::new(*self, Direction::S, true, None),
                PotentialMove::new(*self, Direction::W, true, None),
            ],
            Kind::Queen => vec![
                PotentialMove::new(*self, Direction::N, true, None),
                PotentialMove::new(*self, Direction::Ne, true, None),
                PotentialMove::new(*self, Direction::E, true, None),
                PotentialMove::new(*self, Direction::Se, true, None),
                PotentialMove::new(*self, Direction::S, true, None),
                PotentialMove::new(*self, Direction::Sw, true, None),
                PotentialMove::new(*self, Direction::W, true, None),
                PotentialMove::new(*self, Direction::Nw, true, None),
            ],
            Kind::King => vec![
                PotentialMove::new(*self, Direction::N, false, None),
                PotentialMove::new(*self, Direction::Ne, false, None),
                PotentialMove::new(*self, Direction::E, false, None),
                PotentialMove::new(*self, Direction::Se, false, None),
                PotentialMove::new(*self, Direction::S, false, None),
                PotentialMove::new(*self, Direction::Sw, false, None),
                PotentialMove::new(*self, Direction::W, false, None),
                PotentialMove::new(*self, Direction::Nw, false, None),
                PotentialMove::new(*self, Direction::CastleShort, false, Some(CastleShort)),
                PotentialMove::new(*self, Direction::CastleLong, false, Some(CastleLong)),
            ],
        }
    }

    pub fn to_char(self) -> char {
        match (self.kind, self.color) {
            (Kind::Pawn, Color::Black) => 'p',
            (Kind::Bishop, Color::Black) => 'b',
            (Kind::Knight, Color::Black) => 'n',
            (Kind::Rook, Color::Black) => 'r',
            (Kind::Queen, Color::Black) => 'q',
            (Kind::King, Color::Black) => 'k',

            (Kind::Pawn, Color::White) => 'P',
            (Kind::Bishop, Color::White) => 'B',
            (Kind::Knight, Color::White) => 'N',
            (Kind::Rook, Color::White) => 'R',
            (Kind::Queen, Color::White) => 'Q',
            (Kind::King, Color::White) => 'K',
        }
    }
}

impl fmt::Display for Piece {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} {}", self.color, self.kind)
    }
}

impl fmt::Display for Kind {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}",
            match self {
                Kind::Pawn => "pawn",
                Kind::Bishop => "bishop",
                Kind::Knight => "knight",
                Kind::Rook => "rook",
                Kind::Queen => "queen",
                Kind::King => "king",
            }
        )
    }
}
