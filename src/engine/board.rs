use crate::engine::draw_table;
use crate::engine::movegen::{PotentialMove, SpecialMoves};
use crate::engine::movement::{Direction, Kind as MoveKind, Move, MoveError};
use crate::engine::piece::{Kind, Piece};
use crate::engine::player::{Color, Player};
use crate::engine::position::{extract_layer_offsets, Position};

pub(crate) type Layers = [u64; 12];

pub struct Turn {
    color: Color,
    number: u8,
}

/// Board represents game current status and handles all changes
pub struct Board {
    white: Player,
    black: Player,
    turn: Turn,

    position: Position,

    history: Vec<Move>,
}

impl Board {
    /// Generates the initial board for a normal game start.
    pub fn initial() -> Board {
        Board {
            white: Player::new(),
            black: Player::new(),
            turn: Turn {
                color: Color::White,
                number: 1,
            },
            position: Position::initial(),
            history: vec![],
        }
    }

    pub fn draw(&self) {
        self.position.draw();
    }

    fn player(&self) -> &Player {
        match self.turn.color {
            Color::White => &self.white,
            Color::Black => &self.black,
        }
    }

    fn player_last_file_bitmap(&self) -> u64 {
        match self.turn.color {
            Color::White => 0xff << 56,
            Color::Black => 0xff,
        }
    }

    fn player_pawn_start_file_bitmap(&self) -> u64 {
        match self.turn.color {
            Color::White => 0xff << 8,
            Color::Black => 0xff << 48,
        }
    }

    fn check_played_lost_castling(&mut self, mv: &Move) {
        let player = match self.turn.color {
            Color::White => &mut self.white,
            Color::Black => &mut self.black,
        };

        match mv.piece().kind {
            Kind::Rook => {
                if player.can_castle_short() && mv.offset_from() % 8 == 7 {
                    player.castling_short_lost();
                }

                if player.can_castle_long() && mv.offset_from() % 8 == 0 {
                    player.castling_long_lost();
                }
            }
            Kind::King if mv.kind() == MoveKind::CastleShort => player.castling_short_realized(),
            Kind::King if mv.kind() == MoveKind::CastleLong => player.castling_long_realized(),
            Kind::King => {
                player.castling_short_lost();
                player.castling_long_lost();
            }
            _ => {}
        }
    }

    fn player_pieces(&self) -> &u64 {
        match self.turn.color {
            Color::White => self.white_pieces(),
            Color::Black => self.black_pieces(),
        }
    }

    fn rival_pieces(&self) -> &u64 {
        match self.turn.color {
            Color::White => self.black_pieces(),
            Color::Black => self.white_pieces(),
        }
    }

    fn white_pieces(&self) -> &u64 {
        self.position.white_pieces()
    }

    fn black_pieces(&self) -> &u64 {
        self.position.black_pieces()
    }

    #[cfg(test)]
    fn get_layer(&self, piece: &Piece) -> &u64 {
        self.position.get_layer(piece)
    }

    pub fn generate_moves(&self) -> Vec<Move> {
        let mut moves: Vec<Move> = vec![];

        for piece in Piece::all() {
            if piece.color != self.turn.color {
                continue;
            }

            for pos in extract_layer_offsets(self.position.get_layer(&piece)) {
                for pmove in piece.potential_moves() {
                    match self.generate_legal_moves(&pmove, 1 << pos) {
                        None => {}
                        Some(mut mv) => moves.append(mv.as_mut()),
                    }
                }
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
            // Pawns need to capture to move in diagonal
            || (pmove.piece.kind == Kind::Pawn
                && pmove.dir.is_diagonal()
                && new_pos & self.rival_pieces() == 0
                && (
                    self.history.is_empty()
                    || !is_capture_en_passant(new_pos, self.history.last().unwrap()))
            )
            // Pawns can't capture in forward move
            || (pmove.piece.kind == Kind::Pawn
                && !pmove.dir.is_diagonal()
                && new_pos & self.rival_pieces() != 0)
        {
            return None;
        }

        let mut moves = vec![Move::new_old(
            pmove.piece,
            pos,
            new_pos,
            self.calculate_move_kind(pmove, &new_pos),
        )];

        // This is a bit weird, but the `if let Some(_)..` is unstable
        // when has boolean operators like && or || :(
        // So we first check if piece its pawn to reduce unneeded operations for other pieces.
        if pmove.piece.kind == Kind::Pawn && pos & pawn_initial_layer(pmove.piece.color) != 0 {
            if let Some(large_pos) = apply_move(&pmove.dir, &new_pos) {
                if large_pos & self.rival_pieces() == 0 {
                    moves.push(Move::new_old(pmove.piece, pos, large_pos, MoveKind::Quiet));
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
                moves.push(Move::new_old(pmove.piece, *pos, new_pos, MoveKind::Capture));
                break;
            } else {
                moves.push(Move::new_old(pmove.piece, *pos, new_pos, MoveKind::Quiet));
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
        } else if pmove.piece.kind == Kind::Pawn && pmove.dir.is_diagonal() {
            // Already validated
            return MoveKind::EnPassantCapture;
        }

        match pmove.dir {
            Direction::CastleShort => MoveKind::CastleShort,
            Direction::CastleLong => MoveKind::CastleLong,
            _ => MoveKind::Quiet,
        }
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

    pub fn make_move(
        &mut self,
        origin: u32,
        dest: u32,
        promotion: Option<Kind>,
    ) -> Result<Move, MoveError> {
        let mv = match self.validate_move(origin, dest, promotion) {
            Ok(mv) => mv,
            Err(err) => return Err(err),
        };

        self.position.apply_move(&mv);
        self.check_played_lost_castling(&mv);
        self.next_turn();
        self.history.push(mv.clone());

        Ok(mv)
    }

    fn validate_move(
        &self,
        origin: u32,
        dest: u32,
        promotion: Option<Kind>,
    ) -> Result<Move, MoveError> {
        if origin >= 64 || dest >= 64 {
            return move_error("origin and destination must be in range [0, 64)");
        }

        let (borigin, bdest) = (1 << origin, 1 << dest);

        let piece = match self.position.find_piece(origin as u8, self.turn.color) {
            None => return move_error("no piece found at origin for current player"),
            Some(p) => p,
        };

        if self.player_pieces() & bdest != 0 {
            return move_error("destination already occupied by player piece");
        }

        let mut kind = MoveKind::Quiet;
        let mut check = false;
        let mut checkmate = false;

        if self.rival_pieces() & bdest != 0 {
            if piece.kind == Kind::Pawn && (origin.abs_diff(dest) % 8) == 0 {
                return move_error("pawns cannot capture while advancing forward");
            }
            // todo: check if king is capturing a defended piece

            kind = MoveKind::Capture;
        }

        match self.validate_king_castling(&piece, &origin, &dest, &kind) {
            Err(e) => return Err(e),
            Ok(mv_kind) => kind = mv_kind,
        };

        match self.validate_promotion(&piece, &bdest, promotion, &kind) {
            Err(e) => return Err(e),
            Ok(prom_kind) => kind = prom_kind,
        };

        if piece.kind == Kind::Pawn {
            match origin.abs_diff(dest) {
                8 => {}
                7 | 9 => {
                    if let Some(last_mv) = self.history.last() {
                        if is_capture_en_passant(bdest, last_mv) {
                            kind = MoveKind::EnPassantCapture;
                        }
                    }
                }
                16 => {
                    if self.player_pawn_start_file_bitmap() & borigin == 0 {
                        return move_error("pawns can only move double from start square");
                    }

                    kind = MoveKind::PawnDouble;
                }
                _ => {
                    return move_error(
                        "pawns can only move one or two squares forward or capture in diagonal.",
                    )
                }
            }

            if (piece.color == Color::Black && dest > origin)
                || (piece.color == Color::White && dest < origin)
            {
                return move_error("pawns can only go forward");
            }
        }

        // todo: validate king does not move to an attacked square
        // todo: validate that any move does not reveal a check to king
        // todo: promotions can also generate checks :')

        Ok(Move::new(piece, origin, dest, kind, check, checkmate))
    }

    fn validate_king_castling(
        &self,
        piece: &Piece,
        origin: &u32,
        dest: &u32,
        kind: &MoveKind,
    ) -> Result<MoveKind, MoveError> {
        if piece.kind != Kind::King || origin.abs_diff(*dest) != 2 {
            return Ok(kind.clone());
        }

        // todo: validate that squares origin..=dest are not attacked.

        match (origin, dest) {
            (4, 2) | (60, 58) => {
                if !self.player().can_castle_long() {
                    move_error("player has lost long castling")
                } else {
                    Ok(MoveKind::CastleLong)
                }
            }
            (4, 6) | (60, 62) => {
                if !self.player().can_castle_long() {
                    move_error("player has lost short castling")
                } else {
                    Ok(MoveKind::CastleShort)
                }
            }

            _ => move_error("invalid move positions for castling"),
        }
    }

    fn validate_promotion(
        &self,
        piece: &Piece,
        bdest: &u64,
        prom: Option<Kind>,
        mv_kind: &MoveKind,
    ) -> Result<MoveKind, MoveError> {
        if prom.is_some() && piece.kind != Kind::Pawn {
            return move_error("only pawns can promote");
        }

        if prom.is_some() && bdest & self.player_last_file_bitmap() == 0 {
            return move_error("promoting when not in last file");
        }

        if piece.kind != Kind::Pawn || self.player_last_file_bitmap() & bdest == 0 {
            return Ok(mv_kind.clone());
        }

        match prom {
            None => return move_error("missing promotion for pawn"),
            Some(Kind::Knight) if *mv_kind == MoveKind::Capture => {
                Ok(MoveKind::CapturingKnightPromotion)
            }
            Some(Kind::Bishop) if *mv_kind == MoveKind::Capture => {
                Ok(MoveKind::CapturingBishopPromotion)
            }
            Some(Kind::Rook) if *mv_kind == MoveKind::Capture => {
                Ok(MoveKind::CapturingRookPromotion)
            }
            Some(Kind::Queen) if *mv_kind == MoveKind::Capture => {
                Ok(MoveKind::CapturingQueenPromotion)
            }
            Some(Kind::Knight) => Ok(MoveKind::KnightPromotion),
            Some(Kind::Bishop) => Ok(MoveKind::BishopPromotion),
            Some(Kind::Rook) => Ok(MoveKind::RookPromotion),
            Some(Kind::Queen) => Ok(MoveKind::QueenPromotion),
            Some(_) => return move_error("move kind must be a promotion type"),
        }
    }
}

fn is_capture_en_passant(dest: u64, last_mv: &Move) -> bool {
    return last_mv.kind() == MoveKind::PawnDouble
        && ((last_mv.bitmap_to() == (dest << 8) && last_mv.piece().color == Color::White)
            || (last_mv.bitmap_to() == (dest >> 8) && last_mv.piece().color == Color::Black));
}

fn move_error<T>(msg: &str) -> Result<T, MoveError> {
    Err(MoveError::new(msg.to_string()))
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
    if (position & T_BORDER != 0 && dir.is_north())
        || (position & R_BORDER != 0 && dir.is_east())
        || (position & B_BORDER != 0 && dir.is_south())
        || (position & L_BORDER != 0 && dir.is_west())
    {
        true
    } else if dir.is_knight()
        && ((position & (T_BORDER >> 8) != 0 && matches!(dir, Direction::NNE | Direction::NNW))
            || (position & (R_BORDER >> 1) != 0 && matches!(dir, Direction::NEE | Direction::SEE))
            || (position & (B_BORDER << 8) != 0 && matches!(dir, Direction::SSE | Direction::SSW))
            || (position & (L_BORDER << 1) != 0 && matches!(dir, Direction::NWW | Direction::SWW)))
    {
        true
    } else {
        false
    }
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

#[allow(dead_code)]
pub fn draw_layer(p: u64) {
    let mut out = [' '; 64];
    for i in 0..64 {
        out[i] = if (p >> i) & 1 == 1 { '1' } else { ' ' };
    }

    draw_table(out);
}

pub(crate) const INITIAL_LAYERS: Layers = [
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
    use crate::engine::board::{is_capture_en_passant, moves_outside_board, Turn};
    use crate::engine::movement::{Direction, Move};
    use crate::engine::piece::{Kind, Piece};
    use crate::engine::player::Player;
    use crate::engine::position::Position;
    use crate::engine::Board;
    use crate::engine::Color::{Black, White};
    use crate::engine::Kind::{King, Pawn, Queen, Rook};
    use crate::engine::MoveKind::{
        Capture, CastleLong, CastleShort, EnPassantCapture, PawnDouble, QueenPromotion, Quiet,
    };

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
        let pawn = Piece::new(Pawn, White);
        let knight = Piece::new(Kind::Knight, White);

        assert_eq!(
            vec![
                Move::new_old(pawn, 1 << 8, 1 << 16, Quiet),
                Move::new_old(pawn, 1 << 8, 1 << 24, Quiet),
                Move::new_old(pawn, 1 << 9, 1 << 17, Quiet),
                Move::new_old(pawn, 1 << 9, 1 << 25, Quiet),
                Move::new_old(pawn, 1 << 10, 1 << 18, Quiet),
                Move::new_old(pawn, 1 << 10, 1 << 26, Quiet),
                Move::new_old(pawn, 1 << 11, 1 << 19, Quiet),
                Move::new_old(pawn, 1 << 11, 1 << 27, Quiet),
                Move::new_old(pawn, 1 << 12, 1 << 20, Quiet),
                Move::new_old(pawn, 1 << 12, 1 << 28, Quiet),
                Move::new_old(pawn, 1 << 13, 1 << 21, Quiet),
                Move::new_old(pawn, 1 << 13, 1 << 29, Quiet),
                Move::new_old(pawn, 1 << 14, 1 << 22, Quiet),
                Move::new_old(pawn, 1 << 14, 1 << 30, Quiet),
                Move::new_old(pawn, 1 << 15, 1 << 23, Quiet),
                Move::new_old(pawn, 1 << 15, 1 << 31, Quiet),
                Move::new_old(knight, 1 << 1, 1 << 18, Quiet),
                Move::new_old(knight, 1 << 1, 1 << 16, Quiet),
                Move::new_old(knight, 1 << 6, 1 << 23, Quiet),
                Move::new_old(knight, 1 << 6, 1 << 21, Quiet),
            ],
            board.generate_moves(),
            "board generates initial moves for white player"
        );
    }

    #[test]
    fn it_generates_moves_for_queen() {
        let origin = 0b00001000 << 24;
        let piece = Piece::new(Kind::Queen, White);
        let board = Board {
            white: Player::new(),
            black: Player::new(),
            position: Position::must_new([0, 0, 0, 0, 1 << 27, 0, 1 << 59, 0, 0, 0, 0, 0]),
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
                Move::new_old(piece, origin, 1 << 35, Quiet),
                Move::new_old(piece, origin, 1 << 43, Quiet),
                Move::new_old(piece, origin, 1 << 51, Quiet),
                Move::new_old(piece, origin, 1 << 59, Capture),
                // North East
                Move::new_old(piece, origin, 1 << 36, Quiet),
                Move::new_old(piece, origin, 1 << 45, Quiet),
                Move::new_old(piece, origin, 1 << 54, Quiet),
                Move::new_old(piece, origin, 1 << 63, Quiet),
                // East
                Move::new_old(piece, origin, 1 << 28, Quiet),
                Move::new_old(piece, origin, 1 << 29, Quiet),
                Move::new_old(piece, origin, 1 << 30, Quiet),
                Move::new_old(piece, origin, 1 << 31, Quiet),
                // South East
                Move::new_old(piece, origin, 1 << 20, Quiet),
                Move::new_old(piece, origin, 1 << 13, Quiet),
                Move::new_old(piece, origin, 1 << 6, Quiet),
                // South
                Move::new_old(piece, origin, 1 << 19, Quiet),
                Move::new_old(piece, origin, 1 << 11, Quiet),
                Move::new_old(piece, origin, 1 << 3, Quiet),
                // South West
                Move::new_old(piece, origin, 1 << 18, Quiet),
                Move::new_old(piece, origin, 1 << 9, Quiet),
                Move::new_old(piece, origin, 1, Quiet),
                // West
                Move::new_old(piece, origin, 1 << 26, Quiet),
                Move::new_old(piece, origin, 1 << 25, Quiet),
                Move::new_old(piece, origin, 1 << 24, Quiet),
                // North West
                Move::new_old(piece, origin, 1 << 34, Quiet),
                Move::new_old(piece, origin, 1 << 41, Quiet),
                Move::new_old(piece, origin, 1 << 48, Quiet),
            ]
        )
    }

    /// Generates a board without bishops, horses and queen
    /// for testing castling mechanics
    fn castling_board() -> Board {
        Board {
            white: Player::new(),
            black: Player::new(),
            position: Position::must_new([
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
            ]),
            turn: Turn {
                color: White,
                number: 1,
            },
            history: vec![],
        }
    }

    #[test]
    fn it_can_castle_short() {
        let mut board = castling_board();

        let w_castle = Move::new_old(Piece::new(King, White), 1 << 4, 1 << 6, CastleShort);

        debug_assert_eq!(
            Some(&w_castle),
            board
                .generate_moves()
                .iter()
                .find(|mv| mv.offset_to() == w_castle.offset_to() && mv.kind() == CastleShort),
            "generates castle move"
        );

        assert_eq!(
            Ok(w_castle),
            board.make_move(
                w_castle.offset_from() as u32,
                w_castle.offset_to() as u32,
                None
            )
        );

        assert_eq!(false, board.white.can_castle_short());
        assert_eq!(false, board.white.can_castle_long());
        assert_eq!(w_castle.bitmap_to(), *board.get_layer(&w_castle.piece()));
        assert_eq!(
            0b00100001,
            *board.get_layer(&Piece::new(Rook, White)),
            "rock is also moved when castled"
        );

        let b_castle = Move::new_old(Piece::new(King, Black), 1 << 60, 1 << 62, CastleShort);

        debug_assert_eq!(
            Some(&b_castle),
            board
                .generate_moves()
                .iter()
                .find(|mv| b_castle.offset_to() == mv.offset_to() && mv.kind() == CastleShort),
            "generates castle move"
        );

        assert_eq!(
            Ok(b_castle),
            board.make_move(
                b_castle.offset_from() as u32,
                b_castle.offset_to() as u32,
                None
            )
        );

        assert_eq!(false, board.black.can_castle_short());
        assert_eq!(false, board.black.can_castle_long());
        assert_eq!(b_castle.bitmap_to(), *board.get_layer(&b_castle.piece()));
        assert_eq!(
            0b00100001 << 56,
            *board.get_layer(&Piece::new(Rook, Black)),
            "rock is also moved when castled"
        );
    }

    #[test]
    fn it_can_castle_long() {
        let mut board = castling_board();

        let w_castle = Move::new_old(Piece::new(King, White), 1 << 4, 1 << 2, CastleLong);

        debug_assert_eq!(
            Some(&w_castle),
            board
                .generate_moves()
                .iter()
                .find(|mv| mv.offset_to() == w_castle.offset_to() && mv.kind() == CastleLong),
            "generates white's castle move"
        );

        assert_eq!(
            Ok(w_castle),
            board.make_move(
                w_castle.offset_from() as u32,
                w_castle.offset_to() as u32,
                None
            )
        );

        assert_eq!(false, board.white.can_castle_long());
        assert_eq!(false, board.white.can_castle_short());
        assert_eq!(w_castle.bitmap_to(), *board.get_layer(&w_castle.piece()));
        assert_eq!(
            0b10001000,
            *board.get_layer(&Piece::new(Rook, White)),
            "rock is also moved when castled"
        );

        let b_castle = Move::new_old(Piece::new(King, Black), 1 << 60, 1 << 58, CastleLong);

        debug_assert_eq!(
            Some(&b_castle),
            board
                .generate_moves()
                .iter()
                .find(|mv| b_castle.offset_to() == mv.offset_to() && mv.kind() == CastleLong),
            "generates black's castle move"
        );

        assert_eq!(
            Ok(b_castle),
            board.make_move(
                b_castle.offset_from() as u32,
                b_castle.offset_to() as u32,
                None
            )
        );

        assert_eq!(false, board.black.can_castle_long());
        assert_eq!(false, board.black.can_castle_short());
        assert_eq!(b_castle.bitmap_to(), *board.get_layer(&b_castle.piece()));
        assert_eq!(
            0b10001000 << 56,
            *board.get_layer(&Piece::new(Rook, Black)),
            "rock is also moved when castled"
        );
    }

    #[test]
    fn it_tracks_when_player_loses_castling() {
        let mut board = castling_board();

        board.make_move(7, 6, None).expect("move 1 should be valid");
        assert_eq!(false, board.white.can_castle_short());
        assert_eq!(true, board.white.can_castle_long());

        board
            .make_move(63, 61, None)
            .expect("move 2 should be valid");
        assert_eq!(false, board.black.can_castle_short());
        assert_eq!(true, board.black.can_castle_long());

        board.make_move(0, 2, None).expect("move 3 should be valid");
        assert_eq!(false, board.white.can_castle_short());
        assert_eq!(false, board.white.can_castle_long());

        board
            .make_move(56, 58, None)
            .expect("move 4 should be valid");
        assert_eq!(false, board.black.can_castle_short());
        assert_eq!(false, board.black.can_castle_long());

        board = castling_board();

        board
            .make_move(4, 5, None)
            .expect("king's move must be valid");
        assert_eq!(false, board.white.can_castle_short());
        assert_eq!(false, board.white.can_castle_long());

        board
            .make_move(60, 61, None)
            .expect("king's move must be valid");
        assert_eq!(false, board.black.can_castle_short());
        assert_eq!(false, board.black.can_castle_long());
    }

    #[test]
    fn it_can_capture_en_passant() {
        let w_pawn = Piece::new(Pawn, White);
        let b_pawn = Piece::new(Pawn, Black);
        for tc in [
            (19, Move::new(w_pawn, 11, 27, PawnDouble, false, false)),
            (23, Move::new(w_pawn, 15, 31, PawnDouble, false, false)),
            (43, Move::new(b_pawn, 51, 35, PawnDouble, false, false)),
            (40, Move::new(b_pawn, 48, 32, PawnDouble, false, false)),
        ] {
            assert_eq!(true, is_capture_en_passant(1 << tc.0, &tc.1))
        }
    }

    #[test]
    fn it_can_not_capture_en_passant() {
        let w_pawn = Piece::new(Pawn, White);
        let b_pawn = Piece::new(Pawn, Black);
        for tc in [
            (19, Move::new(w_pawn, 19, 27, Quiet, false, false)),
            (31, Move::new(w_pawn, 23, 31, PawnDouble, false, false)),
            (43, Move::new(b_pawn, 43, 35, Quiet, false, false)),
            (31, Move::new(b_pawn, 40, 32, PawnDouble, false, false)),
        ] {
            assert_eq!(false, is_capture_en_passant(1 << tc.0, &tc.1))
        }
    }

    #[test]
    fn it_handles_en_passant_pawn_capture() {
        let mut board = Board::initial();
        let w_pawn = Piece::new(Pawn, White);
        let b_pawn = Piece::new(Pawn, Black);

        assert_eq!(
            Ok(Move::new(w_pawn, 11, 27, PawnDouble, false, false)),
            board.make_move(11, 27, None)
        );

        assert_eq!(
            Ok(Move::new(b_pawn, 55, 47, Quiet, false, false)),
            board.make_move(55, 47, None)
        );

        assert_eq!(
            Ok(Move::new(w_pawn, 27, 35, Quiet, false, false)),
            board.make_move(27, 35, None)
        );

        assert_eq!(
            Ok(Move::new(b_pawn, 50, 34, PawnDouble, false, false)),
            board.make_move(50, 34, None)
        );

        let mv = Move::new(w_pawn, 35, 42, EnPassantCapture, false, false);
        match board
            .generate_moves()
            .iter()
            .find(|mv| mv.kind() == EnPassantCapture)
        {
            None => panic!("did not generate en passant capture move."),
            Some(gmv) => assert_eq!(mv, *gmv),
        }

        assert_eq!(Ok(mv), board.make_move(35, 42, None));
        assert_eq!(
            0,
            board.position.get_layer(&w_pawn) & (1 << 34),
            "pawn has been removed after en passant capture"
        )
    }

    #[test]
    fn it_should_promote_pawn() {
        let mut board = Board {
            white: Player::new(),
            black: Player::new(),
            turn: Turn {
                color: White,
                number: 1,
            },
            position: Position::must_new([1 << 52, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
            history: vec![],
        };

        let mv = Move::new(
            Piece::new(Pawn, White),
            52,
            60,
            QueenPromotion,
            false,
            false,
        );

        assert_eq!(Ok(mv), board.make_move(52, 60, Some(Queen)));
        assert_eq!(0, *board.position.get_layer(&mv.piece()));
        assert_eq!(
            1 << 60,
            *board.position.get_layer(&Piece::new(Queen, White))
        );
    }
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
