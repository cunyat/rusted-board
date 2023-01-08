use std::error::Error;
use std::fmt;

use crate::engine::piece::Piece;
use crate::engine::position::Square;
use crate::engine::{draw_table, Color, Kind as PieceKind};

/// Move represents a move.
/// It is used for applying a movement in the board
/// and keeping movements history.
///
/// Internally it stores all info in a single u32 word:
/// - 0..6 bits containing move origin as offset (0..64)
/// - 6..12 bits containing move destination as offset (0..64)
/// - 12..16 bits containing the kind of move (see Kind enum)
/// - 16..20 bits containing piece and color
/// - 21st bit indicating check
/// - 22nd bit indicating checkmate
///
/// Also we can extract more features, for example:
///  - 16th bit indicates any kind of capture
///  - 20th bit indicates color (0 = white, 1 = black)
#[derive(Clone, PartialEq, Copy)]
pub struct Move {
    mv: u32,
}

impl Move {
    pub fn new_old(piece: Piece, from: u64, to: u64, kind: Kind) -> Move {
        Move {
            mv: (from.trailing_zeros() & 0x3f)
                | ((to.trailing_zeros() & 0x3f) << 6)
                | (kind as u32 & 0xf) << 12
                | (encode_piece(piece) & 0xf) << 16,
        }
    }

    pub fn new(
        piece: Piece,
        from: Square,
        to: Square,
        kind: Kind,
        check: bool,
        checkmate: bool,
    ) -> Move {
        Move {
            mv: ((from.offset() as u32) & 0x3f)
                | (((to.offset() as u32) & 0x3f) << 6)
                | (kind as u32 & 0xf) << 12
                | (encode_piece(piece) & 0xf) << 16
                | (check as u32 & 0x1) << 20
                | (checkmate as u32 & 0x1) << 21,
        }
    }

    pub fn piece(&self) -> Piece {
        decode_piece(self.mv)
    }

    pub fn origin(&self) -> Square {
        Square::must((self.mv & 0x3f) as u8)
    }

    pub fn dest(&self) -> Square {
        Square::must(((self.mv >> 6) & 0x3f) as u8)
    }

    pub fn kind(&self) -> Kind {
        match (self.mv >> 12) & 0xf {
            0 => Kind::Quiet,
            1 => Kind::PawnDouble,
            2 => Kind::CastleShort,
            3 => Kind::CastleLong,
            4 => Kind::KnightPromotion,
            5 => Kind::BishopPromotion,
            6 => Kind::RookPromotion,
            7 => Kind::QueenPromotion,
            9 => Kind::Capture,
            10 => Kind::CapturingKnightPromotion,
            11 => Kind::CapturingBishopPromotion,
            12 => Kind::CapturingRookPromotion,
            13 => Kind::CapturingQueenPromotion,
            14 => Kind::EnPassantCapture,
            unknown => panic!("unknown value decoding move flag: {}", unknown),
        }
    }

    pub fn draw(&self) {
        let mut out = [' '; 64];

        out[self.origin().offset() as usize] = self.piece().to_char();
        out[self.dest().offset() as usize] = match self.kind() {
            Kind::Quiet => 'O',
            Kind::Capture => 'X',
            Kind::CastleShort => 'C',
            Kind::CastleLong => 'C',
            _ => '?',
        };

        draw_table(out);
    }

    fn display(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Move {{ raw: {}, piece: {}, from: {}, to: {}, kind: {} }}",
            self.mv,
            self.piece().to_char(),
            self.origin(),
            self.dest(),
            self.kind() as u8
        )
    }
}

impl fmt::Display for Move {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.display(f)
    }
}

impl fmt::Debug for Move {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.display(f)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum Kind {
    Quiet = 0,
    PawnDouble = 1,
    CastleShort = 2,
    CastleLong = 3,
    KnightPromotion = 4,
    BishopPromotion = 5,
    RookPromotion = 6,
    QueenPromotion = 7,
    // free = 8,
    Capture = 9,
    CapturingKnightPromotion = 10,
    CapturingBishopPromotion = 11,
    CapturingRookPromotion = 12,
    CapturingQueenPromotion = 13,
    EnPassantCapture = 14,
    // free = 15
}

impl Kind {
    pub(crate) fn is_capture(&self) -> bool {
        matches!(
            self,
            Kind::Capture
                | Kind::CapturingKnightPromotion
                | Kind::CapturingBishopPromotion
                | Kind::CapturingRookPromotion
                | Kind::CapturingQueenPromotion
                | Kind::EnPassantCapture
        )
    }

    pub(crate) fn is_promotion(&self) -> bool {
        matches!(
            self,
            Kind::KnightPromotion
                | Kind::BishopPromotion
                | Kind::RookPromotion
                | Kind::QueenPromotion
                | Kind::CapturingKnightPromotion
                | Kind::CapturingBishopPromotion
                | Kind::CapturingRookPromotion
                | Kind::CapturingQueenPromotion
        )
    }

    pub(crate) fn promotion_piece_kind(&self) -> Option<PieceKind> {
        match self {
            Kind::BishopPromotion | Kind::CapturingBishopPromotion => Some(PieceKind::Bishop),
            Kind::KnightPromotion | Kind::CapturingKnightPromotion => Some(PieceKind::Knight),
            Kind::RookPromotion | Kind::CapturingRookPromotion => Some(PieceKind::Rook),
            Kind::QueenPromotion | Kind::CapturingQueenPromotion => Some(PieceKind::Queen),
            _ => None,
        }
    }
}

fn encode_piece(piece: Piece) -> u32 {
    match (piece.color, piece.kind) {
        (Color::White, PieceKind::Pawn) => 0,
        (Color::White, PieceKind::Bishop) => 1,
        (Color::White, PieceKind::Knight) => 2,
        (Color::White, PieceKind::Rook) => 3,
        (Color::White, PieceKind::Queen) => 4,
        (Color::White, PieceKind::King) => 5,
        (Color::Black, PieceKind::Pawn) => 8,
        (Color::Black, PieceKind::Bishop) => 9,
        (Color::Black, PieceKind::Knight) => 10,
        (Color::Black, PieceKind::Rook) => 11,
        (Color::Black, PieceKind::Queen) => 12,
        (Color::Black, PieceKind::King) => 13,
    }
}

fn decode_piece(mv: u32) -> Piece {
    match (mv >> 16) & 0xf {
        0 => Piece::new(PieceKind::Pawn, Color::White),
        1 => Piece::new(PieceKind::Bishop, Color::White),
        2 => Piece::new(PieceKind::Knight, Color::White),
        3 => Piece::new(PieceKind::Rook, Color::White),
        4 => Piece::new(PieceKind::Queen, Color::White),
        5 => Piece::new(PieceKind::King, Color::White),
        8 => Piece::new(PieceKind::Pawn, Color::Black),
        9 => Piece::new(PieceKind::Bishop, Color::Black),
        10 => Piece::new(PieceKind::Knight, Color::Black),
        11 => Piece::new(PieceKind::Rook, Color::Black),
        12 => Piece::new(PieceKind::Queen, Color::Black),
        13 => Piece::new(PieceKind::King, Color::Black),
        unknown => panic!("unexpected piece value from move encoding {}", unknown),
    }
}

#[derive(PartialEq, Debug)]
pub enum Direction {
    N,
    Ne,
    E,
    Se,
    S,
    Sw,
    W,
    Nw,

    // Knight moves
    Nne,
    Nee,
    See,
    Sse,
    Ssw,
    Sww,
    Nww,
    Nnw,

    // King castle moves
    CastleShort,
    CastleLong,
}

impl Direction {
    pub fn is_north(&self) -> bool {
        matches!(
            self,
            Direction::N
                | Direction::Ne
                | Direction::Nw
                | Direction::Nne
                | Direction::Nee
                | Direction::Nww
                | Direction::Nnw
        )
    }

    pub fn is_south(&self) -> bool {
        matches!(
            self,
            Direction::S
                | Direction::Se
                | Direction::Sw
                | Direction::See
                | Direction::Sse
                | Direction::Ssw
                | Direction::Sww
        )
    }

    pub fn is_east(&self) -> bool {
        matches!(
            self,
            Direction::E
                | Direction::Ne
                | Direction::Se
                | Direction::Nne
                | Direction::Nee
                | Direction::Sse
                | Direction::See
        )
    }

    pub fn is_west(&self) -> bool {
        matches!(
            self,
            Direction::W
                | Direction::Nw
                | Direction::Sw
                | Direction::Nnw
                | Direction::Nww
                | Direction::Ssw
                | Direction::Sww
        )
    }

    pub fn is_knight(&self) -> bool {
        matches!(
            self,
            Direction::Nne
                | Direction::Nee
                | Direction::See
                | Direction::Sse
                | Direction::Ssw
                | Direction::Sww
                | Direction::Nww
                | Direction::Nnw
        )
    }

    pub fn is_diagonal(&self) -> bool {
        matches!(
            self,
            Direction::Ne | Direction::Se | Direction::Sw | Direction::Nw
        )
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct MoveError {
    reason: String,
}

impl MoveError {
    pub fn new(reason: String) -> MoveError {
        MoveError { reason }
    }
}

impl fmt::Display for MoveError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "unable to move: {}", self.reason)
    }
}

impl Error for MoveError {}
