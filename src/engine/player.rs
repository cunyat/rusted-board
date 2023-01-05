#[derive(Debug, Copy, Clone, PartialEq)]
pub enum Color {
    Black,
    White,
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
        match self.castling_long {
            CastlingStatus::Available => true,
            _ => false,
        }
    }

    pub(crate) fn can_castle_short(&self) -> bool {
        match self.castling_short {
            CastlingStatus::Available => true,
            _ => false,
        }
    }
}
