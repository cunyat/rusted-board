use std::fmt;
use std::fmt::Formatter;

#[derive(Debug, Copy, Clone, PartialEq)]
pub enum Color {
    Black,
    White,
}

impl fmt::Display for Color {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}",
            match self {
                Color::Black => "black",
                Color::White => "white",
            }
        )
    }
}

pub(crate) enum CastlingStatus {
    Available,
    Unavailable,
    Realized,
}

pub(crate) struct Player {
    castling_long: CastlingStatus,
    castling_short: CastlingStatus,
}

impl Player {
    pub(crate) fn new() -> Player {
        Player {
            castling_long: CastlingStatus::Available,
            castling_short: CastlingStatus::Available,
        }
    }

    pub(crate) fn castling_short_realized(&mut self) {
        self.castling_short = CastlingStatus::Realized;
        self.castling_long = CastlingStatus::Unavailable;
    }

    pub(crate) fn castling_long_realized(&mut self) {
        self.castling_short = CastlingStatus::Unavailable;
        self.castling_long = CastlingStatus::Realized;
    }

    pub(crate) fn castling_short_lost(&mut self) {
        self.castling_short = CastlingStatus::Unavailable;
    }

    pub(crate) fn castling_long_lost(&mut self) {
        self.castling_long = CastlingStatus::Unavailable;
    }

    pub(crate) fn can_castle_long(&self) -> bool {
        matches!(self.castling_long, CastlingStatus::Available)
    }

    pub(crate) fn can_castle_short(&self) -> bool {
        matches!(self.castling_short, CastlingStatus::Available)
    }
}
