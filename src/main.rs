use crate::engine::{draw_layer, Board, Color, Kind, Piece};

mod engine;

fn main() {
    let mut board = Board::initial();

    match board.make_move(11, 27, None) {
        Ok(_) => {}
        Err(err) => panic!("[ERROR]: {}", err),
    }

    match board.make_move(52, 36, None) {
        Ok(_) => {}
        Err(err) => panic!("[ERROR]: {}", err),
    }

    // match board.make_move(6, 21, None) {
    //     Ok(_) => {}
    //     Err(err) => panic!("[ERROR]: {}", err),
    // }

    match board.make_move(27, 35, None) {
        Ok(_) => {}
        Err(err) => panic!("[ERROR]: {}", err),
    }

    match board.make_move(50, 34, None) {
        Ok(_) => {}
        Err(err) => panic!("[ERROR]: {}", err),
    }

    board.draw();

    for mv in board.generate_moves() {
        mv.draw();
    }
}

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

        println!("index: {}", sq);
        draw_layer(rank_attacks[sq]);
    }
    println!("];")
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
