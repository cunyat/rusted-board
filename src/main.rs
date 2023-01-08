extern crate core;

use crate::engine::Board;

mod engine;

fn main() {
    let mut board = Board::initial();
    let moves = [
        (11, 27, None),
        (52, 36, None),
        (27, 35, None),
        (50, 34, None),
        (35, 42, None),
        (49, 42, None),
        (6, 21, None),
        (51, 43, None),
        (12, 28, None),
        (62, 45, None),
        (1, 18, None),
        (61, 52, None),
        (5, 26, None),
        (60, 62, None),
        (4, 6, None),
    ];

    for mv in moves {
        match board.make_move(mv.0, mv.1, mv.2) {
            Ok(_) => {}
            Err(err) => panic!("[ERROR]: {} at move {:?}", err, mv),
        }
    }

    board.draw();

    for mv in board.generate_moves() {
        mv.draw();
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

#[allow(dead_code)]
fn generate_bishop_moveset() {
    let mut rank_attacks = [0u64; 64];

    for sq in 0..64 {
        let mut northeast = 9;
        loop {
            if sq + northeast >= 64 || (sq + northeast) % 8 == 0 {
                break;
            }
            rank_attacks[sq] |= 1 << (sq + northeast);
            northeast += 9;
        }

        let mut southeast = 7;
        loop {
            if southeast > sq || (sq - southeast) % 8 == 0 {
                break;
            }
            rank_attacks[sq] |= 1 << (sq - southeast);
            southeast += 7
        }

        let mut southwest = 9;
        loop {
            if southwest > sq || (sq - southwest) % 8 == 7 {
                break;
            }
            rank_attacks[sq] |= 1 << (sq - southwest);
            southwest += 9;
        }
        //
        let mut northwest = 7;
        loop {
            if sq + northwest >= 64 || (sq + northwest) % 8 == 7 {
                break;
            }
            rank_attacks[sq] |= 1 << (sq + northwest);
            northwest += 7;
        }
    }

    println!("const DIAGONAL_ATTACKS: [u64; 64] = [");
    for sq in 0..64 {
        println!("    {:#x},", rank_attacks[sq]);
    }
    println!("];")
}
