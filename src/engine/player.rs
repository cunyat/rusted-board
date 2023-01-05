use crate::engine::Move;

#[derive(Debug, PartialEq)]
pub(crate) enum Color {
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

    pub fn handle_move(&mut self, mv: &Move) {
        match self.castling {
            CastlingStatus::Available => {}
            CastlingStatus::Unavailable | CastlingStatus::Realized => return,
        }

        if mv.is_castle() {
            self.castling = CastlingStatus::Realized;
        }
    }

    pub(crate) fn castle_realized(&mut self) {
        self.castling = CastlingStatus::Realized
    }
    pub(crate) fn castle_lost(&mut self) {
        self.castling = CastlingStatus::Unavailable
    }
}
