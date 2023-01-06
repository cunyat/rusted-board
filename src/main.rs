extern crate core;

use crate::engine::Board;

mod engine;

fn main() {
    let mut board = Board::initial();

    match board.make_move(11, 27) {
        Ok(_) => {}
        Err(err) => panic!("[ERROR]: {}", err),
    }

    match board.make_move(52, 36) {
        Ok(_) => {}
        Err(err) => panic!("[ERROR]: {}", err),
    }

    match board.make_move(6, 21) {
        Ok(_) => {}
        Err(err) => panic!("[ERROR]: {}", err),
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
