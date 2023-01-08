use crate::engine::{Color, Square};

fn bishop_attacking(sq: Square) -> u64 {
    BISHOP_ATTACKS[*sq.offset() as usize]
}

fn rock_attacking(sq: Square) -> u64 {
    ROCK_ATTACKS[*sq.offset() as usize]
}

fn queen_attacking(sq: Square) -> u64 {
    bishop_attacking(sq) | rock_attacking(sq)
}

fn knight_attacking(sq: Square) -> u64 {
    let bb = sq.bitmap();
    let h1 = ((bb >> 1) & 0x7f7f7f7f7f7f7f7f) | ((bb << 1) & 0xfefefefefefefefe);
    let h2 = ((bb >> 2) & 0x3f3f3f3f3f3f3f3f) | ((bb << 2) & 0xfcfcfcfcfcfcfcfc);
    (h1 << 16) | (h1 >> 16) | (h2 << 8) | (h2 >> 8)
}

pub fn king_moveset(sq: Square) -> u64 {
    let bb = sq.bitmap();
    let row = ((bb >> 1) & 0x7f7f7f7f7f7f7f7f) | ((bb << 1) & 0xfefefefefefefefe);
    row | (row << 8) | (row >> 8) | (bb << 8) | (bb >> 8)
}

fn pawn_attacking(sq: Square, color: Color) -> u64 {
    let bb = sq.bitmap();
    match color {
        Color::Black => {
            ((bb >> 1) & 0x7f7f7f7f7f7f7f7f) >> 8 | ((bb << 1) & 0xfefefefefefefefe) >> 8
        }
        Color::White => {
            ((bb >> 1) & 0x7f7f7f7f7f7f7f7f) << 8 | ((bb << 1) & 0xfefefefefefefefe) << 8
        }
    }
}

#[cfg(test)]
mod test {
    use crate::engine::calculated::{king_moveset, knight_attacking, pawn_attacking};
    use crate::engine::Color::{Black, White};
    use crate::engine::Square;
    use crate::sq;

    #[test]
    fn it_generates_knight_moves() {
        assert_eq!(0x284400442800, knight_attacking(sq!(28)));
        assert_eq!(0x402000, knight_attacking(sq!(7)));
        assert_eq!(0x4020000000000, knight_attacking(sq!(56)))
    }

    #[test]
    fn it_generates_pawn_moves() {
        assert_eq!(0x280000, pawn_attacking(sq!(12), White));
        assert_eq!(1 << 22, pawn_attacking(sq!(15), White));
        assert_eq!(1 << 33, pawn_attacking(sq!(24), White));

        assert_eq!(0x28 << 40, pawn_attacking(sq!(52), Black));
        assert_eq!(1 << 38, pawn_attacking(sq!(47), Black));
        assert_eq!(1 << 25, pawn_attacking(sq!(32), Black));
    }

    #[test]
    fn it_generates_king_attacks() {
        assert_eq!(0x382838 << 16, king_moveset(sq!(28)));
        assert_eq!(0xc040, king_moveset(sq!(7)));
        assert_eq!(0x203 << 48, king_moveset(sq!(56)));
    }

    // todo: tests for bishop, rook and queen attacks
}

const ROCK_ATTACKS: [u64; 64] = [
    0x1010101010101fe,
    0x2020202020202fd,
    0x4040404040404fb,
    0x8080808080808f7,
    0x10101010101010ef,
    0x20202020202020df,
    0x40404040404040bf,
    0x808080808080807f,
    0x10101010101fe01,
    0x20202020202fd02,
    0x40404040404fb04,
    0x80808080808f708,
    0x101010101010ef10,
    0x202020202020df20,
    0x404040404040bf40,
    0x8080808080807f80,
    0x101010101fe0101,
    0x202020202fd0202,
    0x404040404fb0404,
    0x808080808f70808,
    0x1010101010ef1010,
    0x2020202020df2020,
    0x4040404040bf4040,
    0x80808080807f8080,
    0x1010101fe010101,
    0x2020202fd020202,
    0x4040404fb040404,
    0x8080808f7080808,
    0x10101010ef101010,
    0x20202020df202020,
    0x40404040bf404040,
    0x808080807f808080,
    0x10101fe01010101,
    0x20202fd02020202,
    0x40404fb04040404,
    0x80808f708080808,
    0x101010ef10101010,
    0x202020df20202020,
    0x404040bf40404040,
    0x8080807f80808080,
    0x101fe0101010101,
    0x202fd0202020202,
    0x404fb0404040404,
    0x808f70808080808,
    0x1010ef1010101010,
    0x2020df2020202020,
    0x4040bf4040404040,
    0x80807f8080808080,
    0x1fe010101010101,
    0x2fd020202020202,
    0x4fb040404040404,
    0x8f7080808080808,
    0x10ef101010101010,
    0x20df202020202020,
    0x40bf404040404040,
    0x807f808080808080,
    0xfe01010101010101,
    0xfd02020202020202,
    0xfb04040404040404,
    0xf708080808080808,
    0xef10101010101010,
    0xdf20202020202020,
    0xbf40404040404040,
    0x7f80808080808080,
];

const BISHOP_ATTACKS: [u64; 64] = [
    0x8040201008040200,
    0x80402010080500,
    0x804020110a00,
    0x8041221400,
    0x182442800,
    0x10204885000,
    0x102040810a000,
    0x102040810204000,
    0x4020100804020002,
    0x8040201008050005,
    0x804020110a000a,
    0x804122140014,
    0x18244280028,
    0x1020488500050,
    0x102040810a000a0,
    0x204081020400040,
    0x2010080402000204,
    0x4020100805000508,
    0x804020110a000a11,
    0x80412214001422,
    0x1824428002844,
    0x102048850005088,
    0x2040810a000a010,
    0x408102040004020,
    0x1008040200020408,
    0x2010080500050810,
    0x4020110a000a1120,
    0x8041221400142241,
    0x182442800284482,
    0x204885000508804,
    0x40810a000a01008,
    0x810204000402010,
    0x804020002040810,
    0x1008050005081020,
    0x20110a000a112040,
    0x4122140014224180,
    0x8244280028448201,
    0x488500050880402,
    0x810a000a0100804,
    0x1020400040201008,
    0x402000204081020,
    0x805000508102040,
    0x110a000a11204080,
    0x2214001422418000,
    0x4428002844820100,
    0x8850005088040201,
    0x10a000a010080402,
    0x2040004020100804,
    0x200020408102040,
    0x500050810204080,
    0xa000a1120408000,
    0x1400142241800000,
    0x2800284482010000,
    0x5000508804020100,
    0xa000a01008040201,
    0x4000402010080402,
    0x2040810204080,
    0x5081020408000,
    0xa112040800000,
    0x14224180000000,
    0x28448201000000,
    0x50880402010000,
    0xa0100804020100,
    0x40201008040201,
];
