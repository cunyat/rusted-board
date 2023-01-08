use std::collections::HashSet;
use std::ops::Range;

use crate::engine::board::{Layers, INITIAL_LAYERS};
use crate::engine::movement::Kind as MoveKind;
use crate::engine::{draw_table, Color, Kind, Move, Piece};

pub struct Position {
    layers: Layers,

    cache: [u64; 2],
}

const CACHE_WHITE_PIECES: usize = 0;
const CACHE_BLACK_PIECES: usize = 1;

const CACHE_WHITE_MOVES: usize = 0;
const CACHE_BLACK_MOVES: usize = 1;

const PAWN_LAYER: usize = 0;
const BISHOP_LAYER: usize = 1;
const KNIGHT_LAYER: usize = 2;
const ROOK_LAYER: usize = 3;
const QUEEN_LAYER: usize = 4;
const KING_LAYER: usize = 5;

const BLACK_LAYER_OFFSET: usize = 6;

const WHITE_LAYERS_RANGE: Range<usize> = 0..6;
const BLACK_LAYERS_RANGE: Range<usize> = 6..12;

pub type InvalidPosition = &'static str;

impl Position {
    pub fn new(layers: Layers) -> Result<Position, InvalidPosition> {
        if !is_valid_position(layers) {
            return Err("layers do not contain a valid position");
        }

        let mut pos = Position {
            layers,
            cache: [0; 2],
        };

        pos.generate_cached_layers();
        Ok(pos)
    }

    pub fn must_new(layers: Layers) -> Position {
        match Position::new(layers) {
            Ok(pos) => pos,
            Err(e) => panic!("{}", e),
        }
    }

    pub fn initial() -> Position {
        Position::must_new(INITIAL_LAYERS)
    }

    pub fn white_pieces(&self) -> &u64 {
        &self.cache[CACHE_WHITE_PIECES]
    }

    pub fn black_pieces(&self) -> &u64 {
        &self.cache[CACHE_BLACK_PIECES]
    }

    pub fn get_layer(&self, piece: &Piece) -> &u64 {
        &self.layers[piece.layer_index()]
    }

    pub fn is_piece_at_position(&self, piece: &Piece, offset: &u8) -> bool {
        (self.get_layer(piece) & (1 << offset)) != 0
    }

    pub fn find_piece(&self, offset: u8, color: Color) -> Option<Piece> {
        Piece::for_color(color)
            .iter()
            .find(|piece| self.layers[piece.layer_index()] & (1 << offset) != 0)
            .map(|p| p.clone())
    }

    pub fn draw(&self) {
        let mut out: [char; 64] = [' '; 64];

        Piece::all().into_iter().for_each(|piece| {
            for pos in extract_layer_offsets(&self.layers[piece.layer_index()]) {
                out[pos as usize] = piece.to_char();
            }
        });

        draw_table(out);
    }

    fn generate_cached_layers(&mut self) {
        self.cache[CACHE_WHITE_PIECES] = self.layers[0..6]
            .iter()
            .fold(0u64, |acc, layer| acc | layer);

        self.cache[CACHE_BLACK_PIECES] = self.layers[6..12]
            .iter()
            .fold(0u64, |acc, layer| acc | layer);
    }

    /// Applies a movement in the position.
    /// It assumes that move is validated and legal,
    /// so it should only get called from the game engine.
    pub(crate) fn apply_move(&mut self, mv: &Move) {
        let idx = mv.piece().layer_index();

        // move the piece
        self.layers[idx] = self.layers[idx] ^ mv.bitmap_from() ^ mv.bitmap_to();

        self.apply_capture(mv);
        self.apply_promotion(mv);
        self.apply_castling(mv);
        self.generate_cached_layers();
    }

    fn apply_promotion(&mut self, mv: &Move) {
        if !mv.kind().is_promotion() {
            return;
        }

        let new_piece = match mv.kind().promotion_piece_kind() {
            Some(kind) => Piece::new(kind, mv.piece().color),
            None => return,
        };

        self.layers[mv.piece().layer_index()] ^= mv.bitmap_to();
        self.layers[new_piece.layer_index()] ^= mv.bitmap_to();
    }

    fn apply_castling(&mut self, mv: &Move) {
        if !matches!(mv.kind(), MoveKind::CastleLong | MoveKind::CastleShort) {
            return;
        }

        let idx = Piece::new(Kind::Rook, mv.piece().color).layer_index();

        match mv.kind() {
            MoveKind::CastleShort => {
                self.layers[idx] ^= (mv.bitmap_from() << 1) ^ (mv.bitmap_from() << 3)
            }
            MoveKind::CastleLong => {
                self.layers[idx] ^= (mv.bitmap_from() >> 1) ^ (mv.bitmap_from() >> 4)
            }
            _ => {}
        }
    }

    fn apply_capture(&mut self, mv: &Move) {
        if !mv.kind().is_capture() {
            return;
        }

        let piece_pos = match mv.kind() {
            MoveKind::Capture
            | MoveKind::CapturingKnightPromotion
            | MoveKind::CapturingBishopPromotion
            | MoveKind::CapturingRookPromotion
            | MoveKind::CapturingQueenPromotion => mv.bitmap_to(),
            MoveKind::EnPassantCapture => match mv.piece().color {
                Color::Black => mv.bitmap_to() << 8,
                Color::White => mv.bitmap_to() >> 8,
            },
            _ => return,
        };

        let range = match mv.piece().color {
            Color::Black => 0..6,
            Color::White => 6..12,
        };

        self.layers[range]
            .iter_mut()
            .filter(|layer| **layer & piece_pos != 0)
            .for_each(|layer| *layer ^= piece_pos)
    }
}

fn move_ray(origin: u8, dest: u8) -> u64 {
    match origin.abs_diff(dest) {
        // adjacent moves
        1 | 8 | 9 => 0,
        // diagonal adjacent move, excluding full rank movement
        7 if dest / 8 != origin / 8 => 0,
        dist if dist % 8 == 0 => expand_ray(origin, dest, 8),
        _ if dest / 8 == origin / 8 => expand_ray(origin, dest, 1),
        dist if dist % 9 == 0 => expand_ray(origin, dest, 9),
        dist if dist % 7 == 0 => expand_ray(origin, dest, 7),
        _ => 0,
    }
}

fn expand_ray(origin: u8, dest: u8, offset: u8) -> u64 {
    let mut ray = 0;
    for i in 1..(dest.abs_diff(origin) / offset) {
        ray |= if origin > dest {
            1 << (origin - (i * offset))
        } else {
            1 << (origin + (i * offset))
        }
    }

    ray
}

fn is_valid_position(layers: Layers) -> bool {
    let mut uniq = HashSet::new();
    layers
        .iter()
        .flat_map(|layer| extract_layer_offsets(layer))
        .all(move |x| uniq.insert(x))
}

pub fn extract_layer_offsets(layer: &u64) -> Vec<u8> {
    let mut last: u32 = layer.trailing_zeros();
    let mut offsets = vec![];
    while last < 64 {
        offsets.push(last as u8);
        last = (layer.overflowing_shr(last + 1).0).trailing_zeros() + last + 1;
    }

    offsets
}

impl Piece {
    fn layer_index(&self) -> usize {
        self.color.layer_index_offset() + self.kind.layer_base_index()
    }
}

impl Color {
    fn layer_index_offset(&self) -> usize {
        match self {
            Color::Black => 6,
            Color::White => 0,
        }
    }

    fn rival(&self) -> Color {
        match self {
            Color::Black => Color::White,
            Color::White => Color::Black,
        }
    }

    fn layer_range(&self) -> Range<usize> {
        match self {
            Color::Black => 0..6,
            Color::White => 6..12,
        }
    }
}

impl Kind {
    fn layer_base_index(&self) -> usize {
        match self {
            Kind::Pawn => PAWN_LAYER,
            Kind::Bishop => BISHOP_LAYER,
            Kind::Knight => KNIGHT_LAYER,
            Kind::Rook => ROOK_LAYER,
            Kind::Queen => QUEEN_LAYER,
            Kind::King => KING_LAYER,
        }
    }
}

#[cfg(test)]
mod test {
    use crate::engine::position::{extract_layer_offsets, move_ray, Position};
    use crate::engine::Color::{Black, White};
    use crate::engine::Kind::{King, Pawn};
    use crate::engine::MoveKind::{CastleLong, CastleShort, QueenPromotion, RookPromotion};
    use crate::engine::{Move, Piece};

    const CASTLING_POS: [u64; 12] = [
        0xff << 8,
        0,
        0,
        0x81,
        0,
        1 << 4,
        0xff << 48,
        0,
        0,
        0x81 << 56,
        0,
        1 << 60,
    ];

    #[test]
    fn it_extracts_layer_offsets() {
        assert_eq!(
            vec![0, 1, 6, 63],
            extract_layer_offsets(&(1 | (1 << 1) | (1 << 6) | 1 << 63))
        );

        assert_eq!(
            (0..64).collect::<Vec<u8>>(),
            extract_layer_offsets(&u64::MAX),
        );

        assert_eq!(Vec::<u8>::new(), extract_layer_offsets(&0));
    }

    #[test]
    fn it_generates_move_ray() {
        assert_eq!((1 << 28) | (1 << 36), move_ray(20, 44));
        assert_eq!(0x8080808080800, move_ray(3, 59));
        assert_eq!(0x7e0000, move_ray(16, 23));
        assert_eq!(0x7e0000, move_ray(23, 16));
        assert_eq!(0x8080808080800, move_ray(59, 3));
        assert_eq!(0x201008040000, move_ray(9, 54));
        assert_eq!(0x2040810204000, move_ray(56, 07));
        assert_eq!(0x2010080000, move_ray(46, 10));
        assert_eq!(0x2040000, move_ray(11, 32));
    }

    #[test]
    fn it_should_apply_short_castling() {
        let mut pos = Position::must_new(CASTLING_POS);
        let w_king = Piece::new(King, White);
        let b_king = Piece::new(King, Black);

        pos.apply_move(&Move::new(w_king, 4, 6, CastleShort, false, false));

        assert_eq!(1 << 6, pos.layers[5]);
        assert_eq!(1 | 1 << 5, pos.layers[3]);

        pos.apply_move(&Move::new(b_king, 60, 62, CastleShort, false, false));

        assert_eq!(1 << 62, pos.layers[11]);
        assert_eq!(1 << 56 | 1 << 61, pos.layers[9]);
    }

    #[test]
    fn it_should_apply_long_castling() {
        let mut pos = Position::must_new(CASTLING_POS);
        let w_king = Piece::new(King, White);
        let b_king = Piece::new(King, Black);

        pos.apply_move(&Move::new(w_king, 4, 2, CastleLong, false, false));

        assert_eq!(1 << 2, pos.layers[5]);
        assert_eq!(1 << 3 | 1 << 7, pos.layers[3]);

        pos.apply_move(&Move::new(b_king, 60, 58, CastleLong, false, false));

        assert_eq!(1 << 58, pos.layers[11]);
        assert_eq!(1 << 59 | 1 << 63, pos.layers[9]);
    }

    #[test]
    fn it_should_apply_pawn_promotion() {
        let mut pos = Position::must_new([1 << 52, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1 << 10, 0]);

        let pawn = Piece::new(Pawn, White);
        pos.apply_move(&Move::new(pawn, 52, 60, QueenPromotion, false, false));
        assert_eq!(0, pos.layers[0]);
        assert_eq!(1 << 60, pos.layers[4]);

        let pawn = Piece::new(Pawn, Black);
        pos.apply_move(&Move::new(pawn, 10, 2, RookPromotion, false, false));
        assert_eq!(0, pos.layers[6]);
        assert_eq!(1 << 2, pos.layers[9])
    }

    // Bitmap cheatsheet :D
    //
    //     8    56 57 58 59 60 61 62 63
    //     7    48 49 50 51 52 53 54 55
    //     6    40 41 42 43 44 45 46 47
    //     5    32 33 34 35 36 37 38 39
    //     4    24 25 26 27 28 29 30 31
    //     3    16 17 18 19 20 21 22 23
    //     2    08 09 10 11 12 13 14 15
    //     1    00 01 02 03 04 05 06 07
    //           A  B  C  D  E  F  G  H
}
