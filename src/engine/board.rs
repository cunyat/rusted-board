use std::error::Error;
use std::fmt;
use std::fmt::{Display, Formatter};

pub struct Board {
    turn: Turn,
    layers: [Layer; 12],
    extra: [u64; 2],
}

#[derive(Debug)]
pub struct Layer {
    piece: Piece,
    bitmap: u64,
}

impl<'a> IntoIterator for &'a Layer {
    type Item = usize;
    type IntoIter = LayerIter<'a>;

    fn into_iter(self) -> Self::IntoIter {
        LayerIter {
            last: None,
            ended: false,
            data: self,
        }
    }
}

pub struct LayerIter<'a> {
    last: Option<usize>,
    ended: bool,
    data: &'a Layer,
}

impl<'a> Iterator for LayerIter<'a> {
    type Item = usize;

    fn next(&mut self) -> Option<Self::Item> {
        if self.ended {
            return None;
        }
        let result: usize;

        match self.last {
            None => {
                result = self.data.first_index();
                if result >= 64 {
                    self.ended = true;
                    return None;
                }
            }
            Some(index) => {
                result = match self.data.next_index(index) {
                    Some(index) => index,
                    None => {
                        self.ended = true;
                        return None;
                    }
                };
            }
        }

        self.last = Some(result);
        self.last
    }
}

impl Layer {
    const MAX_INDEX: usize = 63;

    fn first_index(&self) -> usize {
        self.bitmap.trailing_zeros() as usize
    }

    fn next_index(&self, index: usize) -> Option<usize> {
        if self.bitmap == 0 || index >= Layer::MAX_INDEX {
            None
        } else {
            match self.bitmap & u64::MAX << index + 1 {
                0 => None,
                a => Some(a.trailing_zeros() as usize),
            }
        }
    }

    fn draw(&self) -> [char; 64] {
        let mut output = [' '; 64];
        self.into_iter()
            .for_each(|idx| output[idx] = self.piece.to_char());

        output
    }

    pub fn fancy_draw(&self) {
        let mut output = [' '; 64];
        self.into_iter()
            .for_each(|idx| output[idx] = self.piece.to_char());

        draw_table(output);
    }

    fn split_pieces_layers(&self) -> Vec<u64> {
        self.into_iter().map(|index| 1 << index).collect()
    }

    pub(crate) fn apply_move(&mut self, mv: Move) {
        self.bitmap = self.bitmap ^ mv.origin ^ mv.dest
    }

    pub(crate) fn remove_piece(&mut self, position: u64) {
        self.bitmap = self.bitmap ^ position;
    }
}

const T_BORDER: u64 = 0b11111111 << 56;
const B_BORDER: u64 = 0b11111111;
const R_BORDER: u64 = 9259542123273814144;
const L_BORDER: u64 = 72340172838076673;

const INITIAL_BOARD: [Layer; 12] = [
    Layer {
        piece: Piece::White(PieceType::Pawn),
        bitmap: 0b11111111 << 8,
    },
    // Layer { piece: Piece::White(PieceType::Pawn), bitmap: 0 },
    Layer {
        piece: Piece::White(PieceType::Bishop),
        bitmap: 0b00100100,
    },
    Layer {
        piece: Piece::White(PieceType::Knight),
        bitmap: 0b01000010,
    },
    Layer {
        piece: Piece::White(PieceType::Rock),
        bitmap: 0b10000001,
    },
    Layer {
        piece: Piece::White(PieceType::Queen),
        bitmap: 0b00001000,
    },
    Layer {
        piece: Piece::White(PieceType::King),
        bitmap: 0b00010000,
    },
    Layer {
        piece: Piece::Black(PieceType::Pawn),
        bitmap: 0b11111111 << 48,
    },
    Layer {
        piece: Piece::Black(PieceType::Bishop),
        bitmap: 0b00100100 << 56,
    },
    Layer {
        piece: Piece::Black(PieceType::Knight),
        bitmap: 0b01000010 << 56,
    },
    Layer {
        piece: Piece::Black(PieceType::Rock),
        bitmap: 0b10000001 << 56,
    },
    Layer {
        piece: Piece::Black(PieceType::Queen),
        bitmap: 0b00001000 << 56,
    },
    Layer {
        piece: Piece::Black(PieceType::King),
        bitmap: 0b00010000 << 56,
    },
];

impl Board {
    pub fn initial() -> Board {
        let mut board = Board {
            turn: Turn::White(1),
            layers: INITIAL_BOARD,
            extra: [0, 0],
        };
        board.calculate_extra_layers();
        board
    }

    pub fn debug(&self) {
        println!("turn: {}", self.turn);
        let pieces = self
            .layers
            .iter()
            .map(|layer| layer.draw())
            .fold([' '; 64], |acc, layer| {
                let mut macc = acc;
                for i in 0..64 {
                    if layer[i] != ' ' {
                        macc[i] = layer[i];
                    }
                }
                macc
            });

        draw_table(pieces);
        println!("White pieces layer:");
        draw_layer(self.extra[0]);
        println!("Black pieces layer:");
        draw_layer(self.extra[1]);
    }

    fn calculate_extra_layers(&mut self) {
        self.extra = [0, 0];

        for i in 0..6 {
            self.extra[0] |= self.layers[i].bitmap;
        }

        for i in 6..12 {
            self.extra[1] |= self.layers[i].bitmap;
        }
    }

    pub fn black_pieces(&self) -> u64 {
        self.extra[1]
    }

    pub fn white_pieces(&self) -> u64 {
        self.extra[0]
    }

    pub fn player_pieces(&self, p: Piece) -> u64 {
        match p {
            Piece::Black(_) => self.black_pieces(),
            Piece::White(_) => self.white_pieces(),
        }
    }

    pub fn rival_pieces(&self, p: Piece) -> u64 {
        match p {
            Piece::Black(_) => self.white_pieces(),
            Piece::White(_) => self.black_pieces(),
        }
    }

    pub fn all_pieces(&self) -> u64 {
        self.white_pieces() | self.black_pieces()
    }

    pub fn generate_moves(&self) -> Vec<Move> {
        match self.turn {
            Turn::White(_) => 0..6,
            Turn::Black(_) => 6..12,
        }
        .fold(vec![], |mut acc, idx| {
            let layer = &self.layers[idx];

            layer
                .split_pieces_layers()
                .iter()
                .fold(vec![], |mut acc, origin| {
                    acc.append(
                        layer
                            .piece
                            .generate_moves(
                                *origin,
                                &self.player_pieces(layer.piece),
                                &self.rival_pieces(layer.piece),
                            )
                            .as_mut(),
                    );
                    acc
                })
                .into_iter()
                .for_each(|mv| acc.push(mv));

            return acc;
        })
    }

    pub fn make_move(&mut self, mv: Move) -> Result<(), MoveError> {
        if let Err(e) = self.is_valid_move(mv) {
            return Err(e);
        }

        if mv.kind == MoveKind::Capture {
            match self.layer_index_by_position(mv.dest) {
                Some(idx) => self.layers[idx].remove_piece(mv.dest),
                None => {
                    return Err(MoveError {
                        mv,
                        reason: "could not find piece to capture :(".to_string(),
                    })
                }
            }
        }

        self.layers[mv.piece.index()].apply_move(mv);
        self.next_turn();
        self.calculate_extra_layers();
        self.validate_layers();

        return Ok(());
    }

    fn layer_index_by_position(&self, position: u64) -> Option<usize> {
        self.layers
            .iter()
            .enumerate()
            .find(|(_, layer)| layer.bitmap & position != 0)
            .map_or(None, |(idx, _)| Some(idx))
    }

    fn next_turn(&mut self) {
        self.turn = match self.turn {
            Turn::White(num) => Turn::Black(num),
            Turn::Black(num) => Turn::White(num + 1),
        }
    }

    fn is_valid_move(&self, mv: Move) -> Result<(), MoveError> {
        if self.layers[mv.piece.index()].bitmap & mv.origin == 0 {
            return Err(MoveError {
                mv,
                reason: "indicated piece is not at origin position".to_string(),
            });
        }

        if self.player_pieces(mv.piece) & mv.dest != 0 {
            return Err(MoveError {
                mv,
                reason: "destination already occupied by player piece".to_string(),
            });
        }

        match mv.kind {
            MoveKind::Quiet => {
                if self.all_pieces() & mv.dest != 0 {
                    return Err(MoveError {
                        mv,
                        reason: "quiet move on occupied cell".to_string(),
                    });
                }
            }
            MoveKind::Capture => {
                if self.rival_pieces(mv.piece) & mv.dest == 0 {
                    return Err(MoveError {
                        mv,
                        reason: "capturing move without rival piece in dest".to_string(),
                    });
                }
            }
            MoveKind::Castle => {}
            MoveKind::Check => {}
            MoveKind::Checkmate => {}
        }

        return Ok(());
    }

    pub fn player_attacking_layer(&self) -> u64 {
        self.layers
            .iter()
            .filter(|layer| match layer.piece {
                Piece::Black(_) => matches!(self.turn, Turn::Black(_)),
                Piece::White(_) => matches!(self.turn, Turn::White(_)),
            })
            .fold(0u64, move |acc, layer| {
                acc | layer
                    .split_pieces_layers()
                    .iter()
                    .flat_map(|origin| {
                        layer.piece.generate_moves(
                            *origin,
                            &self.player_pieces(layer.piece),
                            &self.rival_pieces(layer.piece),
                        )
                    })
                    .filter(|mv| {
                        mv.piece.piece_type() != PieceType::Pawn || mv.direction.is_diagonal()
                    })
                    .fold(0u64, |acc, mv| acc | mv.dest)
            })
    }

    fn validate_layers(&self) {
        let mut acc: u64 = 0;
        for layer in &self.layers {
            acc &= layer.bitmap;
            if acc != 0 {
                panic!("found layer collisions in {:?} layer", layer.piece)
            }
        }
    }
}

#[derive(Debug, Clone)]
pub struct MoveError {
    mv: Move,
    reason: String,
}

impl Display for MoveError {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "unable to move {:?}: {}", self.mv.piece, self.reason)
    }
}

impl Error for MoveError {}

#[derive(Debug, Copy, Clone, PartialEq)]
pub enum PieceType {
    Pawn,
    Bishop,
    Knight,
    Rock,
    Queen,
    King,
}

#[derive(Debug, Copy, Clone, PartialEq)]
pub enum Piece {
    Black(PieceType),
    White(PieceType),
}

#[derive(Debug, PartialEq)]
pub enum Turn {
    White(usize),
    Black(usize),
}

impl Display for Turn {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} {}",
            match self {
                Turn::White(t) | Turn::Black(t) => t,
            },
            match self {
                Turn::White(_) => "White",
                Turn::Black(_) => "Black",
            }
        )
    }
}

impl PieceType {
    fn index(&self) -> usize {
        match self {
            PieceType::Pawn => 0,
            PieceType::Bishop => 1,
            PieceType::Knight => 2,
            PieceType::Rock => 3,
            PieceType::Queen => 4,
            PieceType::King => 5,
        }
    }
}

impl TryFrom<usize> for Piece {
    type Error = String;

    fn try_from(value: usize) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(Piece::White(PieceType::Pawn)),
            1 => Ok(Piece::White(PieceType::Bishop)),
            2 => Ok(Piece::White(PieceType::Knight)),
            3 => Ok(Piece::White(PieceType::Rock)),
            4 => Ok(Piece::White(PieceType::Queen)),
            5 => Ok(Piece::White(PieceType::King)),
            6 => Ok(Piece::Black(PieceType::Pawn)),
            7 => Ok(Piece::Black(PieceType::Bishop)),
            8 => Ok(Piece::Black(PieceType::Knight)),
            9 => Ok(Piece::Black(PieceType::Rock)),
            10 => Ok(Piece::Black(PieceType::Queen)),
            11 => Ok(Piece::Black(PieceType::King)),
            _ => Err(String::from("Piece is represented by integer [0, 11]")),
        }
    }
}

impl Piece {
    fn index(&self) -> usize {
        match self {
            Piece::White(t) => t.index(),
            Piece::Black(t) => t.index() + 6,
        }
    }

    fn can_slice(&self) -> bool {
        match self.piece_type() {
            PieceType::Bishop | PieceType::Rock | PieceType::Queen => true,
            _ => false,
        }
    }

    fn generate_moves(&self, origin: u64, player_pieces: &u64, rival_pieces: &u64) -> Vec<Move> {
        match self.piece_type() {
            PieceType::Pawn => {
                let mut moves = vec![];

                for dir in MoveDir::piece_moves(self) {
                    if let Some(mv) = dir.apply(&origin) {
                        if dir.is_diagonal() && (mv & rival_pieces != 0) {
                            moves.push(Move::new(*self, origin, mv, dir, MoveKind::Capture));
                        }

                        if !dir.is_diagonal() && (mv & (rival_pieces | player_pieces) == 0) {
                            moves.push(Move::quiet(*self, origin, mv, dir));

                            let double_move = dir.apply(&mv);
                            if ((dir == MoveDir::N && origin & (0b11111111 << 8) != 0)
                                || (dir == MoveDir::S && origin & (0b11111111 << 48) != 0))
                                && double_move.is_some()
                            {
                                moves.push(Move::quiet(*self, origin, double_move.unwrap(), dir));
                            }
                        }
                    }
                }

                moves
            }
            PieceType::King | PieceType::Knight => MoveDir::piece_moves(self)
                .iter()
                .map(|dir| (dir, dir.apply(&origin)))
                .filter(|(_, mv)| mv.is_some() && (mv.unwrap() & player_pieces == 0))
                .map(|(dir, mv)| Move::quiet(*self, origin, mv.unwrap(), *dir))
                .collect(),
            PieceType::Bishop | PieceType::Rock | PieceType::Queen => MoveDir::piece_moves(self)
                .iter()
                .fold(vec![], |mut moves, dir| {
                    let mut bitmap = origin.clone();
                    while let Some(mv) = dir.apply(&bitmap) {
                        if mv & player_pieces != 0 {
                            break;
                        }
                        if mv & rival_pieces != 0 {
                            moves.push(Move::new(*self, origin, mv, *dir, MoveKind::Capture));
                            break;
                        }

                        moves.push(Move::quiet(*self, origin, mv, *dir));
                        bitmap = mv;
                    }
                    moves
                }),
        }
    }
}

impl Into<char> for PieceType {
    fn into(self) -> char {
        match self {
            PieceType::Pawn => 'p',
            PieceType::Bishop => 'b',
            PieceType::Knight => 'n',
            PieceType::Rock => 'r',
            PieceType::Queen => 'q',
            PieceType::King => 'k',
        }
    }
}

impl Piece {
    fn to_char(&self) -> char {
        match self {
            Piece::Black(t) => match t {
                PieceType::Pawn => 'p',
                PieceType::Bishop => 'b',
                PieceType::Knight => 'n',
                PieceType::Rock => 'r',
                PieceType::Queen => 'q',
                PieceType::King => 'k',
            },
            Piece::White(t) => match t {
                PieceType::Pawn => 'P',
                PieceType::Bishop => 'B',
                PieceType::Knight => 'N',
                PieceType::Rock => 'R',
                PieceType::Queen => 'Q',
                PieceType::King => 'K',
            },
        }
    }

    fn piece_type(&self) -> PieceType {
        match self {
            Piece::Black(pt) | Piece::White(pt) => pt.clone(),
        }
    }
}

impl Display for Piece {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.to_char())
    }
}

pub fn draw_layer(p: u64) {
    let mut out = [' '; 64];
    for i in 0..64 {
        out[i] = if (p >> i) & 1 == 1 { '1' } else { ' ' };
    }

    draw_table(out);
}

pub fn draw_move(mv: &Move) {
    let mut out = [' '; 64];

    out[mv.origin.trailing_zeros() as usize] = mv.piece.to_char();
    out[mv.dest.trailing_zeros() as usize] = 'X';

    draw_table(out);
}

fn draw_table(p: [char; 64]) {
    println!("┌───┬───┬───┬───┬───┬───┬───┬───┐");
    println!(
        "│ {} │ {} │ {} │ {} │ {} │ {} │ {} │ {} │",
        p[0 + 56],
        p[1 + 56],
        p[2 + 56],
        p[3 + 56],
        p[4 + 56],
        p[5 + 56],
        p[6 + 56],
        p[7 + 56]
    );
    println!("├───┼───┼───┼───┼───┼───┼───┼───┤");
    println!(
        "│ {} │ {} │ {} │ {} │ {} │ {} │ {} │ {} │",
        p[0 + 48],
        p[1 + 48],
        p[2 + 48],
        p[3 + 48],
        p[4 + 48],
        p[5 + 48],
        p[6 + 48],
        p[7 + 48]
    );
    println!("├───┼───┼───┼───┼───┼───┼───┼───┤");
    println!(
        "│ {} │ {} │ {} │ {} │ {} │ {} │ {} │ {} │",
        p[0 + 40],
        p[1 + 40],
        p[2 + 40],
        p[3 + 40],
        p[4 + 40],
        p[5 + 40],
        p[6 + 40],
        p[7 + 40]
    );
    println!("├───┼───┼───┼───┼───┼───┼───┼───┤");
    println!(
        "│ {} │ {} │ {} │ {} │ {} │ {} │ {} │ {} │",
        p[0 + 32],
        p[1 + 32],
        p[2 + 32],
        p[3 + 32],
        p[4 + 32],
        p[5 + 32],
        p[6 + 32],
        p[7 + 32]
    );
    println!("├───┼───┼───┼───┼───┼───┼───┼───┤");
    println!(
        "│ {} │ {} │ {} │ {} │ {} │ {} │ {} │ {} │",
        p[0 + 24],
        p[1 + 24],
        p[2 + 24],
        p[3 + 24],
        p[4 + 24],
        p[5 + 24],
        p[6 + 24],
        p[7 + 24]
    );
    println!("├───┼───┼───┼───┼───┼───┼───┼───┤");
    println!(
        "│ {} │ {} │ {} │ {} │ {} │ {} │ {} │ {} │",
        p[0 + 16],
        p[1 + 16],
        p[2 + 16],
        p[3 + 16],
        p[4 + 16],
        p[5 + 16],
        p[6 + 16],
        p[7 + 16]
    );
    println!("├───┼───┼───┼───┼───┼───┼───┼───┤");
    println!(
        "│ {} │ {} │ {} │ {} │ {} │ {} │ {} │ {} │",
        p[0 + 8],
        p[1 + 8],
        p[2 + 8],
        p[3 + 8],
        p[4 + 8],
        p[5 + 8],
        p[6 + 8],
        p[7 + 8]
    );
    println!("├───┼───┼───┼───┼───┼───┼───┼───┤");
    println!(
        "│ {} │ {} │ {} │ {} │ {} │ {} │ {} │ {} │",
        p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7]
    );
    println!("└───┴───┴───┴───┴───┴───┴───┴───┘");
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub enum MoveDir {
    N = 8,
    NE = 9,
    E = 1,
    SE = -7,
    S = -8,
    SW = -9,
    W = -1,
    NW = 7,

    // Knight moves
    NNE = 17,
    NEE = 10,
    SEE = -6,
    SSE = -15,
    SSW = -17,
    SWW = -10,
    NWW = 6,
    NNW = 15,

    // King castle moves
    CastleShort = 2,
    CastleLong = -2,
}

impl MoveDir {
    fn piece_moves(piece: &Piece) -> Vec<MoveDir> {
        match piece {
            Piece::Black(PieceType::Pawn) => vec![MoveDir::S, MoveDir::SE, MoveDir::SW],
            Piece::White(PieceType::Pawn) => vec![MoveDir::N, MoveDir::NE, MoveDir::NW],
            Piece::Black(pt) | Piece::White(pt) => match pt {
                PieceType::Pawn => panic!("pawn moves without color, it should be already handled"),
                PieceType::Bishop => vec![MoveDir::NE, MoveDir::SE, MoveDir::SW, MoveDir::NW],
                PieceType::Knight => vec![
                    MoveDir::NNE,
                    MoveDir::NEE,
                    MoveDir::SEE,
                    MoveDir::SSE,
                    MoveDir::SSW,
                    MoveDir::SWW,
                    MoveDir::NWW,
                    MoveDir::NNW,
                ],
                PieceType::Rock => vec![MoveDir::N, MoveDir::E, MoveDir::S, MoveDir::W],
                PieceType::Queen => vec![
                    MoveDir::N,
                    MoveDir::E,
                    MoveDir::S,
                    MoveDir::W,
                    MoveDir::NE,
                    MoveDir::SE,
                    MoveDir::SW,
                    MoveDir::NW,
                ],
                PieceType::King => vec![
                    MoveDir::N,
                    MoveDir::E,
                    MoveDir::S,
                    MoveDir::W,
                    MoveDir::NE,
                    MoveDir::SE,
                    MoveDir::SW,
                    MoveDir::NW,
                    MoveDir::CastleShort,
                    MoveDir::CastleLong,
                ],
            },
        }
    }

    fn value(self) -> i32 {
        self as i32
    }

    fn apply(self, position: &u64) -> Option<u64> {
        if self.collisions(position) {
            return None;
        }

        if self.value() > 0 {
            position.checked_shl(self.value() as u32)
        } else {
            position.checked_shr(self.value().abs() as u32)
        }
    }

    fn collisions(self, position: &u64) -> bool {
        if position & T_BORDER != 0 && self.is_north() {
            return true;
        }
        if position & R_BORDER != 0 && self.is_east() {
            return true;
        }
        if position & B_BORDER != 0 && self.is_south() {
            return true;
        }
        if position & L_BORDER != 0 && self.is_west() {
            return true;
        }

        if self.is_knight() {
            if position & (T_BORDER >> 8) != 0 && (self == MoveDir::NNE || self == MoveDir::NNW) {
                return true;
            }
            if position & (R_BORDER >> 1) != 0 && (self == MoveDir::NEE || self == MoveDir::SEE) {
                return true;
            }
            if position & (B_BORDER << 8) != 0 && (self == MoveDir::SSE || self == MoveDir::SSW) {
                return true;
            }
            if position & (L_BORDER << 1) != 0 && (self == MoveDir::NWW || self == MoveDir::SWW) {
                return true;
            }
        }

        return false;
    }

    fn is_north(self) -> bool {
        self.value() >= 6
    }

    fn is_south(self) -> bool {
        self.value() <= 6
    }

    fn is_east(self) -> bool {
        match self {
            MoveDir::E
            | MoveDir::NE
            | MoveDir::SE
            | MoveDir::NNE
            | MoveDir::NEE
            | MoveDir::SSE
            | MoveDir::SEE => true,
            _ => false,
        }
    }

    fn is_west(self) -> bool {
        match self {
            MoveDir::W
            | MoveDir::NW
            | MoveDir::SW
            | MoveDir::NNW
            | MoveDir::NWW
            | MoveDir::SSW
            | MoveDir::SWW => true,
            _ => false,
        }
    }

    fn is_knight(self) -> bool {
        match self {
            MoveDir::NNE
            | MoveDir::NEE
            | MoveDir::SEE
            | MoveDir::SSE
            | MoveDir::SSW
            | MoveDir::SWW
            | MoveDir::NWW
            | MoveDir::NNW => true,
            _ => false,
        }
    }

    pub fn is_diagonal(self) -> bool {
        match self {
            MoveDir::NE | MoveDir::SE | MoveDir::SW | MoveDir::NW => true,
            _ => false,
        }
    }

    pub fn is_castle(self) -> bool {
        self == MoveDir::CastleShort || self == MoveDir::CastleLong
    }
}

#[derive(PartialEq, Copy, Clone, Debug)]
pub enum MoveKind {
    Quiet,
    Capture,
    Castle,
    Check,
    Checkmate,
}

#[derive(PartialEq, Copy, Clone, Debug)]
pub struct Move {
    piece: Piece,
    origin: u64,
    dest: u64,
    direction: MoveDir,
    kind: MoveKind,
}

impl Move {
    pub fn new(piece: Piece, origin: u64, dest: u64, direction: MoveDir, kind: MoveKind) -> Move {
        Move {
            piece,
            origin,
            dest,
            direction,
            kind,
        }
    }

    pub fn quiet(piece: Piece, origin: u64, dest: u64, direction: MoveDir) -> Move {
        Move {
            piece,
            origin,
            dest,
            direction,
            kind: MoveKind::Quiet,
        }
    }

    pub fn is_castle(&self) -> bool {
        match self.piece {
            Piece::White(PieceType::King) => {
                self.origin == 1 << 4 && (self.dest == 1 << 6 || self.dest == 1 << 2)
            }
            Piece::Black(PieceType::King) => {
                self.origin == 1 << 60 && (self.dest == 1 << 62 || self.dest == 1 << 58)
            }
            _ => false,
        }
    }

    pub fn is_piece(&self, piece: PieceType) -> bool {
        self.piece.piece_type() == piece
    }
}

#[cfg(test)]
mod test {
    use super::Piece::{Black, White};
    use super::PieceType::*;
    use super::*;

    #[test]
    fn layer_first_index() {
        assert_eq!(
            8,
            Layer {
                piece: White(Pawn),
                bitmap: 1 << 8
            }
            .first_index()
        );
        assert_eq!(
            0,
            Layer {
                piece: White(Pawn),
                bitmap: 1
            }
            .first_index()
        );
        assert_eq!(
            64,
            Layer {
                piece: White(Pawn),
                bitmap: 0
            }
            .first_index()
        );
        assert_eq!(
            2,
            Layer {
                piece: White(Pawn),
                bitmap: 0b100100
            }
            .first_index()
        );
        assert_eq!(
            36,
            Layer {
                piece: White(Pawn),
                bitmap: 0b10010000 << 32
            }
            .first_index()
        );
    }

    #[test]
    fn board_initial_moves() {
        let mut board = Board::initial();
        assert_eq!(
            vec![
                Move::quiet(White(Pawn), 1 << 8, 1 << 16, MoveDir::N),
                Move::quiet(White(Pawn), 1 << 8, 1 << 24, MoveDir::N),
                Move::quiet(White(Pawn), 1 << 9, 1 << 17, MoveDir::N),
                Move::quiet(White(Pawn), 1 << 9, 1 << 25, MoveDir::N),
                Move::quiet(White(Pawn), 1 << 10, 1 << 18, MoveDir::N),
                Move::quiet(White(Pawn), 1 << 10, 1 << 26, MoveDir::N),
                Move::quiet(White(Pawn), 1 << 11, 1 << 19, MoveDir::N),
                Move::quiet(White(Pawn), 1 << 11, 1 << 27, MoveDir::N),
                Move::quiet(White(Pawn), 1 << 12, 1 << 20, MoveDir::N),
                Move::quiet(White(Pawn), 1 << 12, 1 << 28, MoveDir::N),
                Move::quiet(White(Pawn), 1 << 13, 1 << 21, MoveDir::N),
                Move::quiet(White(Pawn), 1 << 13, 1 << 29, MoveDir::N),
                Move::quiet(White(Pawn), 1 << 14, 1 << 22, MoveDir::N),
                Move::quiet(White(Pawn), 1 << 14, 1 << 30, MoveDir::N),
                Move::quiet(White(Pawn), 1 << 15, 1 << 23, MoveDir::N),
                Move::quiet(White(Pawn), 1 << 15, 1 << 31, MoveDir::N),
                Move::quiet(White(Knight), 1 << 1, 1 << 18, MoveDir::NNE),
                Move::quiet(White(Knight), 1 << 1, 1 << 16, MoveDir::NNW),
                Move::quiet(White(Knight), 1 << 6, 1 << 23, MoveDir::NNE),
                Move::quiet(White(Knight), 1 << 6, 1 << 21, MoveDir::NNW),
            ],
            board.generate_moves(),
            "board generates initial moves for white player"
        );

        board.turn = Turn::Black(1);

        assert_eq!(
            vec![
                Move::quiet(Black(Pawn), 1 << 48, 1 << 40, MoveDir::S),
                Move::quiet(Black(Pawn), 1 << 48, 1 << 32, MoveDir::S),
                Move::quiet(Black(Pawn), 1 << 49, 1 << 41, MoveDir::S),
                Move::quiet(Black(Pawn), 1 << 49, 1 << 33, MoveDir::S),
                Move::quiet(Black(Pawn), 1 << 50, 1 << 42, MoveDir::S),
                Move::quiet(Black(Pawn), 1 << 50, 1 << 34, MoveDir::S),
                Move::quiet(Black(Pawn), 1 << 51, 1 << 43, MoveDir::S),
                Move::quiet(Black(Pawn), 1 << 51, 1 << 35, MoveDir::S),
                Move::quiet(Black(Pawn), 1 << 52, 1 << 44, MoveDir::S),
                Move::quiet(Black(Pawn), 1 << 52, 1 << 36, MoveDir::S),
                Move::quiet(Black(Pawn), 1 << 53, 1 << 45, MoveDir::S),
                Move::quiet(Black(Pawn), 1 << 53, 1 << 37, MoveDir::S),
                Move::quiet(Black(Pawn), 1 << 54, 1 << 46, MoveDir::S),
                Move::quiet(Black(Pawn), 1 << 54, 1 << 38, MoveDir::S),
                Move::quiet(Black(Pawn), 1 << 55, 1 << 47, MoveDir::S),
                Move::quiet(Black(Pawn), 1 << 55, 1 << 39, MoveDir::S),
                Move::quiet(Black(Knight), 1 << 57, 1 << 42, MoveDir::SSE),
                Move::quiet(Black(Knight), 1 << 57, 1 << 40, MoveDir::SSW),
                Move::quiet(Black(Knight), 1 << 62, 1 << 47, MoveDir::SSE),
                Move::quiet(Black(Knight), 1 << 62, 1 << 45, MoveDir::SSW),
            ],
            board.generate_moves(),
            "board generates initial moves for black player"
        )
    }

    #[test]
    fn piece_queen_generate_moves() {
        let origin = 0b00001000 << 24;
        let piece = White(Queen);
        assert_eq!(
            White(Queen).generate_moves(origin, &(0b00001000 << 8), &(0b00001000 << 56)),
            vec![
                Move::quiet(piece, origin, 1 << 35, MoveDir::N),
                Move::quiet(piece, origin, 1 << 43, MoveDir::N),
                Move::quiet(piece, origin, 1 << 51, MoveDir::N),
                Move::new(piece, origin, 1 << 59, MoveDir::N, MoveKind::Capture),
                Move::quiet(piece, origin, 1 << 28, MoveDir::E),
                Move::quiet(piece, origin, 1 << 29, MoveDir::E),
                Move::quiet(piece, origin, 1 << 30, MoveDir::E),
                Move::quiet(piece, origin, 1 << 31, MoveDir::E),
                Move::quiet(piece, origin, 1 << 19, MoveDir::S),
                Move::quiet(piece, origin, 1 << 26, MoveDir::W),
                Move::quiet(piece, origin, 1 << 25, MoveDir::W),
                Move::quiet(piece, origin, 1 << 24, MoveDir::W),
                Move::quiet(piece, origin, 1 << 36, MoveDir::NE),
                Move::quiet(piece, origin, 1 << 45, MoveDir::NE),
                Move::quiet(piece, origin, 1 << 54, MoveDir::NE),
                Move::quiet(piece, origin, 1 << 63, MoveDir::NE),
                Move::quiet(piece, origin, 1 << 20, MoveDir::SE),
                Move::quiet(piece, origin, 1 << 13, MoveDir::SE),
                Move::quiet(piece, origin, 1 << 6, MoveDir::SE),
                Move::quiet(piece, origin, 1 << 18, MoveDir::SW),
                Move::quiet(piece, origin, 1 << 9, MoveDir::SW),
                Move::quiet(piece, origin, 1, MoveDir::SW),
                Move::quiet(piece, origin, 1 << 34, MoveDir::NW),
                Move::quiet(piece, origin, 1 << 41, MoveDir::NW),
                Move::quiet(piece, origin, 1 << 48, MoveDir::NW),
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
