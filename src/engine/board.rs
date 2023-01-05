use std::ops::{BitXorAssign, Range};

use crate::engine::draw_table;
use crate::engine::movement::{
    Direction, Kind as MoveKind, Move, MoveError, PotentialMove, SpecialMoves,
};
use crate::engine::piece::{Kind, Piece};
use crate::engine::player::{Color, Player};

type Layers = [u64; 12];

pub struct Turn {
    color: Color,
    number: u8,
}

/// Board represents game current status and handles all changes
pub struct Board {
    white: Player,
    black: Player,
    turn: Turn,

    layers: Layers,

    history: Vec<Move>,
}

impl Board {
    /// Generates the initial board for a normal game start.
    pub fn initial() -> Board {
        Board {
            white: Player::white(),
            black: Player::black(),
            turn: Turn {
                color: Color::White,
                number: 1,
            },
            layers: INITIAL_LAYERS,
            history: vec![],
        }
    }

    pub fn draw(&self) {
        let mut out: [char; 64] = [' '; 64];

        self.layers
            .into_iter()
            .enumerate()
            .for_each(|(idx, layer)| {
                let piece = self.get_piece_by_layer_index(idx);
                let mut offset = layer.trailing_zeros();

                while offset < 64 {
                    out[offset as usize] = piece.to_char();

                    offset = match layer_next_offset(&layer, offset) {
                        Some(value) => value,
                        None => break,
                    };
                }
            });

        draw_table(out);
    }

    fn player(&self) -> &Player {
        match self.turn.color {
            Color::White => &self.white,
            Color::Black => &self.black,
        }
    }

    fn player_pieces(&self) -> u64 {
        match self.turn.color {
            Color::White => self.white_pieces(),
            Color::Black => self.black_pieces(),
        }
    }

    fn rival_pieces(&self) -> u64 {
        match self.turn.color {
            Color::White => self.black_pieces(),
            Color::Black => self.white_pieces(),
        }
    }

    fn rival_king(&self) -> &u64 {
        &self.layers[match self.turn.color {
            Color::White => 5,
            Color::Black => 11,
        }]
    }

    fn white_pieces(&self) -> u64 {
        self.layers[0]
            | self.layers[1]
            | self.layers[2]
            | self.layers[3]
            | self.layers[4]
            | self.layers[5]
    }

    fn black_pieces(&self) -> u64 {
        self.layers[6]
            | self.layers[7]
            | self.layers[8]
            | self.layers[9]
            | self.layers[10]
            | self.layers[11]
    }

    fn all_pieces(&self) -> u64 {
        self.black_pieces() | self.white_pieces()
    }

    /// Returns a tuple containing the index and the layer copy
    /// for the given piece
    fn get_layer(&self, piece: &Piece) -> u64 {
        self.layers[self.get_layer_index(piece)]
    }

    /// Returns layer index containing given piece
    fn get_layer_index(&self, piece: &Piece) -> usize {
        match (piece.color, piece.kind) {
            (Color::White, Kind::Pawn) => 0,
            (Color::White, Kind::Bishop) => 1,
            (Color::White, Kind::Knight) => 2,
            (Color::White, Kind::Rock) => 3,
            (Color::White, Kind::Queen) => 4,
            (Color::White, Kind::King) => 5,
            (Color::Black, Kind::Pawn) => 6,
            (Color::Black, Kind::Bishop) => 7,
            (Color::Black, Kind::Knight) => 8,
            (Color::Black, Kind::Rock) => 9,
            (Color::Black, Kind::Queen) => 10,
            (Color::Black, Kind::King) => 11,
        }
    }

    fn get_piece_by_layer_index(&self, idx: usize) -> Piece {
        match idx {
            0 => Piece::new(Kind::Pawn, Color::White),
            1 => Piece::new(Kind::Bishop, Color::White),
            2 => Piece::new(Kind::Knight, Color::White),
            3 => Piece::new(Kind::Rock, Color::White),
            4 => Piece::new(Kind::Queen, Color::White),
            5 => Piece::new(Kind::King, Color::White),
            6 => Piece::new(Kind::Pawn, Color::Black),
            7 => Piece::new(Kind::Bishop, Color::Black),
            8 => Piece::new(Kind::Knight, Color::Black),
            9 => Piece::new(Kind::Rock, Color::Black),
            10 => Piece::new(Kind::Queen, Color::Black),
            11 => Piece::new(Kind::King, Color::Black),
            _ => panic!(
                "layers index must be [0, 12), trying to obtain piece for index {}",
                idx
            ),
        }
    }

    fn turn_layers_range(&self) -> Range<usize> {
        match self.turn.color {
            Color::White => 0..6,
            Color::Black => 6..12,
        }
    }

    pub fn generate_moves(&self) -> Vec<Move> {
        let mut moves: Vec<Move> = vec![];

        for idx in self.turn_layers_range() {
            let layer = self.layers[idx];
            let mut offset = layer.trailing_zeros();

            while offset < 64 {
                moves.append(
                    self.get_piece_by_layer_index(idx)
                        .potential_moves()
                        .iter()
                        .filter_map(|pm| self.generate_legal_moves(pm, 1 << offset))
                        .flatten()
                        .collect::<Vec<Move>>()
                        .as_mut()
                );

                offset = match layer_next_offset(&layer, offset) {
                    Some(value) => value,
                    None => break,
                };
            }
        }

        moves
    }

    fn generate_legal_moves(&self, pmove: &PotentialMove, pos: u64) -> Option<Vec<Move>> {
        if !self.special_is_possible(pmove) {
            return None;
        }

        if pmove.slicing {
            return self.generate_sliding_moves(pmove, &pos);
        }

        let new_pos = match apply_move(&pmove.dir, &pos) {
            None => return None,
            Some(np) => np,
        };

        if new_pos & self.player_pieces() != 0
            || (pmove.piece.kind == Kind::Pawn
            && pmove.dir.is_diagonal()
            && new_pos & self.rival_pieces() == 0)
            || (pmove.piece.kind == Kind::Pawn
            && !pmove.dir.is_diagonal()
            && new_pos & self.rival_pieces() != 0)
        {
            return None;
        }

        let mut moves = vec![Move::new(
            pmove.piece,
            pos,
            new_pos,
            self.calculate_move_kind(pmove, &new_pos),
        )];

        if pmove.piece.kind == Kind::Pawn && pos & pawn_initial_layer(pmove.piece.color) != 0 {
            if let Some(large_pos) = apply_move(&pmove.dir, &new_pos) {
                if large_pos & self.rival_pieces() == 0 {
                    moves.push(Move::new(pmove.piece, pos, large_pos, MoveKind::Quiet));
                }
            }
        }

        Some(moves)
    }

    fn generate_sliding_moves(&self, pmove: &PotentialMove, pos: &u64) -> Option<Vec<Move>> {
        let mut moves = vec![];
        let mut last_pos = *pos;
        while let Some(new_pos) = apply_move(&pmove.dir, &last_pos) {
            if new_pos & self.player_pieces() != 0 {
                break;
            }

            if new_pos & self.rival_pieces() != 0 {
                moves.push(Move::new(pmove.piece, *pos, new_pos, MoveKind::Capture));
                break;
            } else {
                moves.push(Move::new(pmove.piece, *pos, new_pos, MoveKind::Quiet));
            }

            last_pos = new_pos
        }

        match moves.is_empty() {
            true => None,
            false => Some(moves),
        }
    }

    fn special_is_possible(&self, pmove: &PotentialMove) -> bool {
        match pmove.special {
            Some(SpecialMoves::CastleLong) => self.player().can_castle_long(),
            Some(SpecialMoves::CastleShort) => self.player().can_castle_short(),
            _ => true,
        }
    }

    fn calculate_move_kind(&self, pmove: &PotentialMove, pos: &u64) -> MoveKind {
        if self.rival_pieces() & pos != 0 {
            return MoveKind::Capture;
        }

        if pmove.dir.is_castle() {
            return MoveKind::Castle;
        }

        return MoveKind::Quiet;
    }

    fn next_turn(&mut self) {
        self.turn = match self.turn.color {
            Color::White => Turn {
                color: Color::Black,
                number: self.turn.number,
            },
            Color::Black => Turn {
                color: Color::White,
                number: self.turn.number + 1,
            },
        }
    }

    pub fn make_move(&mut self, mv: &Move) -> Result<(), MoveError> {
        if let Err(e) = self.is_valid_move(mv) {
            return Err(e);
        }

        // Applying capture before applying the move
        // to ensure we dont remove the moved piece.
        if mv.kind == MoveKind::Capture {
            self.apply_capture(&mv.to);
        }

        let idx = self.get_layer_index(&mv.piece);
        self.layers[idx] = self.layers[idx] ^ mv.from ^ mv.to;
        self.next_turn();
        self.history.push(mv.clone());

        Ok(())
    }

    fn apply_capture(&mut self, pos: &u64) {
        self.layers
            .iter_mut()
            .filter(|layer| **layer & *pos != 0)
            .for_each(|layer| layer.bitxor_assign(pos));
    }

    fn is_valid_move(&self, mv: &Move) -> Result<(), MoveError> {
        if self.get_layer(&mv.piece) & mv.from == 0 {
            return Err(MoveError::new(
                mv.clone(),
                "indicated piece is not at origin position".to_string(),
            ));
        }

        if self.player_pieces() & mv.to != 0 {
            return Err(MoveError::new(
                mv.clone(),
                "destination already occupied by player piece".to_string(),
            ));
        }

        match mv.kind {
            MoveKind::Quiet => {
                if self.all_pieces() & mv.to != 0 {
                    return Err(MoveError::new(
                        mv.clone(),
                        "quiet move on occupied cell".to_string(),
                    ));
                }
            }
            MoveKind::Capture => {
                if self.rival_pieces() & mv.to == 0 {
                    return Err(MoveError::new(
                        mv.clone(),
                        "capturing move without rival piece in dest".to_string(),
                    ));
                }
            }
            MoveKind::Castle => {}
        }

        return Ok(());
    }
}

fn layer_next_offset(layer: &u64, last: u32) -> Option<u32> {
    if *layer == 0 || last >= 63 {
        return None;
    }

    match layer & u64::MAX << last + 1 {
        0 => None,
        a => Some(a.trailing_zeros()),
    }
}

fn pawn_initial_layer(color: Color) -> u64 {
    match color {
        Color::White => 0xff << 8,
        Color::Black => 0xff << 48,
    }
}

/// Tries to apply move from a base position and
/// return the new position.
/// In case the move ends outside the board, returns None.
fn apply_move(dir: &Direction, pos: &u64) -> Option<u64> {
    if moves_outside_board(dir, pos) {
        return None;
    }

    let offset = move_direction_offset(dir);

    if offset > 0 {
        pos.checked_shl(offset as u32)
    } else {
        pos.checked_shr(offset.abs() as u32)
    }
}

/// Indicates if piece would go out of the board
/// moving in the indicated direction.
fn moves_outside_board(dir: &Direction, position: &u64) -> bool {
    if position & T_BORDER != 0 && dir.is_north() {
        return true;
    }

    if position & R_BORDER != 0 && dir.is_east() {
        return true;
    }

    if position & B_BORDER != 0 && dir.is_south() {
        return true;
    }

    if position & L_BORDER != 0 && dir.is_west() {
        return true;
    }

    if dir.is_knight() {
        if position & (T_BORDER >> 8) != 0 && matches!(dir, Direction::NNE | Direction::NNW) {
            return true;
        }
        if position & (R_BORDER >> 1) != 0 && matches!(dir, Direction::NEE | Direction::SEE) {
            return true;
        }
        if position & (B_BORDER << 8) != 0 && matches!(dir, Direction::SSE | Direction::SSW) {
            return true;
        }
        if position & (L_BORDER << 1) != 0 && matches!(dir, Direction::NWW | Direction::SWW) {
            return true;
        }
    }

    return false;
}

/// Return the offset to transform a bitmap
/// for a given move direction
fn move_direction_offset(dir: &Direction) -> i32 {
    match dir {
        Direction::N => 8,
        Direction::NE => 9,
        Direction::E => 1,
        Direction::SE => -7,
        Direction::S => -8,
        Direction::SW => -9,
        Direction::W => -1,
        Direction::NW => 7,
        Direction::NNE => 17,
        Direction::NEE => 10,
        Direction::SEE => -6,
        Direction::SSE => -15,
        Direction::SSW => -17,
        Direction::SWW => -10,
        Direction::NWW => 6,
        Direction::NNW => 15,
        Direction::CastleShort => 2,
        Direction::CastleLong => -2,
    }
}

pub fn draw_layer(p: u64) {
    let mut out = [' '; 64];
    for i in 0..64 {
        out[i] = if (p >> i) & 1 == 1 { '1' } else { ' ' };
    }

    draw_table(out);
}

const INITIAL_LAYERS: Layers = [
    // 0,
    0b11111111 << 8,
    0b00100100,
    0b01000010,
    0b10000001,
    0b00001000,
    0b00010000,
    0b11111111 << 48,
    0b00100100 << 56,
    0b01000010 << 56,
    0b10000001 << 56,
    0b00001000 << 56,
    0b00010000 << 56,
];

const T_BORDER: u64 = 0b11111111 << 56;
const B_BORDER: u64 = 0b11111111;
const R_BORDER: u64 = 9259542123273814144;
const L_BORDER: u64 = 72340172838076673;

#[cfg(test)]
mod test {
    use crate::engine::board::{moves_outside_board, Turn};
    use crate::engine::Board;
    use crate::engine::Color::White;
    use crate::engine::movement::{Direction, Kind as MoveKind, Move};
    use crate::engine::piece::{Kind, Piece};
    use crate::engine::player::{Color, Player};

    #[test]
    fn it_moves_inside_board() {
        for (dir, pos) in [
            (Direction::N, 1 << 4),
            (Direction::N, 1 << 52),
            (Direction::NNE, 1 << 35),
            (Direction::NNE, 1 << 22),
            (Direction::NNE, 1 << 40),
            (Direction::NE, 1 << 48),
            (Direction::NE, 1 << 54),
            (Direction::NE, 1 << 9),
            (Direction::NEE, 1 << 53),
            (Direction::NEE, 1 << 13),
            (Direction::E, 1 << 30),
            (Direction::E, 1 << 25),
            (Direction::SEE, 1 << 13),
            (Direction::SE, 1 << 14),
            (Direction::SE, 1 << 44),
            (Direction::SSE, 1 << 22),
            (Direction::SSE, 1 << 34),
            (Direction::S, 1 << 11),
            (Direction::S, 1 << 36),
            (Direction::SSW, 1 << 17),
            (Direction::SSW, 1 << 29),
            (Direction::SW, 1 << 13),
            (Direction::SW, 1 << 9),
            (Direction::SWW, 1 << 10),
            (Direction::SWW, 1 << 52),
            (Direction::W, 1 << 33),
            (Direction::W, 1 << 33),
            (Direction::NWW, 1 << 50),
            (Direction::NWW, 1 << 07),
            (Direction::NW, 1 << 49),
            (Direction::NW, 1 << 39),
            (Direction::NNW, 1 << 41),
            (Direction::NNW, 1 << 4),
        ] {
            assert_eq!(
                false,
                moves_outside_board(&dir, &pos),
                "moves inside board from (1 << {}) to {:?}",
                pos.trailing_zeros(),
                dir,
            );
        }
    }

    #[test]
    fn it_should_generate_initial_moves() {
        let board = Board::initial();
        let pawn = Piece::new(Kind::Pawn, Color::White);
        let knight = Piece::new(Kind::Knight, Color::White);

        assert_eq!(
            vec![
                Move::new(pawn, 1 << 8, 1 << 16, MoveKind::Quiet),
                Move::new(pawn, 1 << 8, 1 << 24, MoveKind::Quiet),
                Move::new(pawn, 1 << 9, 1 << 17, MoveKind::Quiet),
                Move::new(pawn, 1 << 9, 1 << 25, MoveKind::Quiet),
                Move::new(pawn, 1 << 10, 1 << 18, MoveKind::Quiet),
                Move::new(pawn, 1 << 10, 1 << 26, MoveKind::Quiet),
                Move::new(pawn, 1 << 11, 1 << 19, MoveKind::Quiet),
                Move::new(pawn, 1 << 11, 1 << 27, MoveKind::Quiet),
                Move::new(pawn, 1 << 12, 1 << 20, MoveKind::Quiet),
                Move::new(pawn, 1 << 12, 1 << 28, MoveKind::Quiet),
                Move::new(pawn, 1 << 13, 1 << 21, MoveKind::Quiet),
                Move::new(pawn, 1 << 13, 1 << 29, MoveKind::Quiet),
                Move::new(pawn, 1 << 14, 1 << 22, MoveKind::Quiet),
                Move::new(pawn, 1 << 14, 1 << 30, MoveKind::Quiet),
                Move::new(pawn, 1 << 15, 1 << 23, MoveKind::Quiet),
                Move::new(pawn, 1 << 15, 1 << 31, MoveKind::Quiet),
                Move::new(knight, 1 << 1, 1 << 18, MoveKind::Quiet),
                Move::new(knight, 1 << 1, 1 << 16, MoveKind::Quiet),
                Move::new(knight, 1 << 6, 1 << 23, MoveKind::Quiet),
                Move::new(knight, 1 << 6, 1 << 21, MoveKind::Quiet),
            ],
            board.generate_moves(),
            "board generates initial moves for white player"
        );
    }

    #[test]
    fn piece_queen_generate_moves() {
        let origin = 0b00001000 << 24;
        let piece = Piece::new(Kind::Queen, Color::White);

        let board = Board {
            white: Player::white(),
            black: Player::black(),
            layers: [
                0,
                0,
                0,
                0,
                0b00001000 << 24,
                0,
                0b00001000 << 56,
                0,
                0,
                0,
                0b00001000 << 24,
                0,
            ],
            turn: Turn {
                color: White,
                number: 1,
            },
            history: vec![],
        };

        assert_eq!(
            board.generate_moves(),
            vec![
                // North
                Move::new(piece, origin, 1 << 35, MoveKind::Quiet),
                Move::new(piece, origin, 1 << 43, MoveKind::Quiet),
                Move::new(piece, origin, 1 << 51, MoveKind::Quiet),
                Move::new(piece, origin, 1 << 59, MoveKind::Capture),
                // North East
                Move::new(piece, origin, 1 << 36, MoveKind::Quiet),
                Move::new(piece, origin, 1 << 45, MoveKind::Quiet),
                Move::new(piece, origin, 1 << 54, MoveKind::Quiet),
                Move::new(piece, origin, 1 << 63, MoveKind::Quiet),
                // East
                Move::new(piece, origin, 1 << 28, MoveKind::Quiet),
                Move::new(piece, origin, 1 << 29, MoveKind::Quiet),
                Move::new(piece, origin, 1 << 30, MoveKind::Quiet),
                Move::new(piece, origin, 1 << 31, MoveKind::Quiet),
                // South East
                Move::new(piece, origin, 1 << 20, MoveKind::Quiet),
                Move::new(piece, origin, 1 << 13, MoveKind::Quiet),
                Move::new(piece, origin, 1 << 6, MoveKind::Quiet),
                // South
                Move::new(piece, origin, 1 << 19, MoveKind::Quiet),
                Move::new(piece, origin, 1 << 11, MoveKind::Quiet),
                Move::new(piece, origin, 1 << 3, MoveKind::Quiet),
                // South West
                Move::new(piece, origin, 1 << 18, MoveKind::Quiet),
                Move::new(piece, origin, 1 << 9, MoveKind::Quiet),
                Move::new(piece, origin, 1, MoveKind::Quiet),
                // West
                Move::new(piece, origin, 1 << 26, MoveKind::Quiet),
                Move::new(piece, origin, 1 << 25, MoveKind::Quiet),
                Move::new(piece, origin, 1 << 24, MoveKind::Quiet),
                // North West
                Move::new(piece, origin, 1 << 34, MoveKind::Quiet),
                Move::new(piece, origin, 1 << 41, MoveKind::Quiet),
                Move::new(piece, origin, 1 << 48, MoveKind::Quiet),
            ]
        )
    }
}

// Bitmap cheatsheet :D
//
// 56 57 58 59 60 61 62 63
// 48 49 50 51 52 53 54 55
// 40 41 42 43 44 45 46 47
// 32 33 34 35 36 37 38 39
// 24 25 26 27 28 29 30 31
// 16 17 18 19 20 21 22 23
// 08 09 10 11 12 13 14 15
// 00 01 02 03 04 05 06 07
