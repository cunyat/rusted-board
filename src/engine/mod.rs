pub use self::board::{draw_layer, Board};
pub use self::movement::{Kind as MoveKind, Move};
pub use self::piece::{Kind, Piece};
pub use self::player::Color;
pub use self::position::{Position, Square};

mod board;
// mod calculated; ignore file while not being used
mod movegen;
mod movement;
mod piece;
mod player;
mod position;

#[macro_export]
macro_rules! sq {
    ($val:expr) => {
        match Square::try_from($val) {
            Err(e) => panic!("invalid square value: {}", e),
            Ok(sq) => sq,
        }
    };
}

#[macro_export]
macro_rules! mv {
    ($kind:ident, $color:ident, $from:expr, $to:expr) => {
        Move::new(
            Piece::new($kind, $color),
            sq!($from),
            sq!($to),
            $crate::engine::movement::Kind::Quiet,
            false,
            false,
        )
    };
    ($p_kind:ident, $color:ident, $from:expr, $to:expr, $m_kind:ident) => {
        Move::new(
            Piece::new($p_kind, $color),
            sq!($from),
            sq!($to),
            $m_kind,
            false,
            false,
        )
    };
}

pub fn draw_table(p: [char; 64]) {
    println!("┌───┬───┬───┬───┬───┬───┬───┬───┐");
    println!(
        "│ {} │ {} │ {} │ {} │ {} │ {} │ {} │ {} │",
        p[56],
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
        p[48],
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
        p[40],
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
        p[32],
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
        p[24],
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
        p[16],
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
        p[8],
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
