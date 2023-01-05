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
    color: Color,
    castling: CastlingStatus,
}

impl Player {
    pub(crate) fn black() -> Player {
        Player {
            color: Color::Black,
            castling: CastlingStatus::Available,
        }
    }

    pub(crate) fn white() -> Player {
        Player {
            color: Color::White,
            castling: CastlingStatus::Available,
        }
    }

    pub(crate) fn can_castle_long(&self) -> bool {
        return true;
    }

    pub(crate) fn can_castle_short(&self) -> bool {
        return true;
    }
}
