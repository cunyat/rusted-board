use crate::engine::movegen::PotentialMove;
use crate::engine::movegen::SpecialMoves::{CastleLong, CastleShort, PawnAdvance, PawnCapture};
use crate::engine::movement::Direction;
use crate::engine::player::Color;

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

    pub fn potential_moves(&self) -> Vec<PotentialMove> {
        match self.kind {
            Kind::Pawn => {
                let (dir, east_capture, west_capture) = match self.color {
                    Color::White => (Direction::N, Direction::NE, Direction::NW),
                    Color::Black => (Direction::S, Direction::SE, Direction::SW),
                };

                vec![
                    PotentialMove::new(self.clone(), dir, false, Some(PawnAdvance)),
                    PotentialMove::new(self.clone(), east_capture, false, Some(PawnCapture)),
                    PotentialMove::new(self.clone(), west_capture, false, Some(PawnCapture)),
                ]
            }
            Kind::Bishop => vec![
                PotentialMove::new(self.clone(), Direction::NE, true, None),
                PotentialMove::new(self.clone(), Direction::SE, true, None),
                PotentialMove::new(self.clone(), Direction::SW, true, None),
                PotentialMove::new(self.clone(), Direction::NW, true, None),
            ],
            Kind::Knight => vec![
                PotentialMove::new(self.clone(), Direction::NNE, false, None),
                PotentialMove::new(self.clone(), Direction::NEE, false, None),
                PotentialMove::new(self.clone(), Direction::SEE, false, None),
                PotentialMove::new(self.clone(), Direction::SSE, false, None),
                PotentialMove::new(self.clone(), Direction::SSW, false, None),
                PotentialMove::new(self.clone(), Direction::SWW, false, None),
                PotentialMove::new(self.clone(), Direction::NWW, false, None),
                PotentialMove::new(self.clone(), Direction::NNW, false, None),
            ],
            Kind::Rook => vec![
                PotentialMove::new(self.clone(), Direction::N, true, None),
                PotentialMove::new(self.clone(), Direction::E, true, None),
                PotentialMove::new(self.clone(), Direction::S, true, None),
                PotentialMove::new(self.clone(), Direction::W, true, None),
            ],
            Kind::Queen => vec![
                PotentialMove::new(self.clone(), Direction::N, true, None),
                PotentialMove::new(self.clone(), Direction::NE, true, None),
                PotentialMove::new(self.clone(), Direction::E, true, None),
                PotentialMove::new(self.clone(), Direction::SE, true, None),
                PotentialMove::new(self.clone(), Direction::S, true, None),
                PotentialMove::new(self.clone(), Direction::SW, true, None),
                PotentialMove::new(self.clone(), Direction::W, true, None),
                PotentialMove::new(self.clone(), Direction::NW, true, None),
            ],
            Kind::King => vec![
                PotentialMove::new(self.clone(), Direction::N, false, None),
                PotentialMove::new(self.clone(), Direction::NE, false, None),
                PotentialMove::new(self.clone(), Direction::E, false, None),
                PotentialMove::new(self.clone(), Direction::SE, false, None),
                PotentialMove::new(self.clone(), Direction::S, false, None),
                PotentialMove::new(self.clone(), Direction::SW, false, None),
                PotentialMove::new(self.clone(), Direction::W, false, None),
                PotentialMove::new(self.clone(), Direction::NW, false, None),
                PotentialMove::new(
                    self.clone(),
                    Direction::CastleShort,
                    false,
                    Some(CastleShort),
                ),
                PotentialMove::new(self.clone(), Direction::CastleLong, false, Some(CastleLong)),
            ],
        }
    }

    pub fn to_char(&self) -> char {
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
