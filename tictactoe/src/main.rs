use std::fmt;

type Position = (usize, usize);

#[derive(PartialEq, Copy, Clone, Debug)]
enum Player {
    Player1,
    Player2,
}

impl Player {
    fn get_next(&self) -> Self {
        match self {
            Player::Player1 => Player::Player2,
            Player::Player2 => Player::Player1,
        }
    }
}
#[derive(PartialEq, Debug)]
enum GameResult {
    PlayerWon(Player),
    Draw,
}
#[derive(Debug)]
enum GameError {
    NotLegalPosition,
    PositionOccupied,
}

trait Game<const M: usize, const N: usize, const K: usize>: Clone + fmt::Display {
    fn get_legal_moves(&self) -> Vec<Position>;
    fn get_winner(&self) -> Option<GameResult>;
    fn execute_move(&mut self, player: Player, pos: Position) -> Result<(), GameError>;
    fn undo_move(&mut self, pos: Position) -> Result<(), GameError>;
}

#[derive(PartialEq, Copy, Clone, Debug)]
struct Board<const M: usize, const N: usize, const K: usize>([[Option<Player>; N]; M]);

impl<const M: usize, const N: usize, const K: usize> Board<M, N, K> {
    fn new() -> Self {
        Board([[None; N]; M])
    }
}

impl<const M: usize, const N: usize, const K: usize> fmt::Display for Board<M, N, K> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        unimplemented!();
    }
}

impl<const M: usize, const N: usize, const K: usize> Game<M, N, K> for Board<M, N, K> {
    fn get_legal_moves(&self) -> Vec<Position> {
        unimplemented!();
    }

    fn get_winner(&self) -> Option<GameResult> {
        unimplemented!();
    }

    fn execute_move(&mut self, player: Player, pos: Position) -> Result<(), GameError> {
        unimplemented!();
    }

    fn undo_move(&mut self, pos: Position) -> Result<(), GameError> {
        unimplemented!();
    }
}

trait Agent {
    fn get_next_move<G: Game<M, N, K>, const M: usize, const N: usize, const K: usize>(
        &mut self,
        game: &G,
        player: Player,
    ) -> Position;
}

struct Minimax;

impl Agent for Minimax {
    fn get_next_move<G: Game<M, N, K>, const M: usize, const N: usize, const K: usize>(
        &mut self,
        game: &G,
        player: Player,
    ) -> Position {
        unimplemented!();
    }
}

fn parse_input(s: &str) -> Result<Position, Box<dyn std::error::Error>> {
    let mut ms = s.matches(char::is_numeric);
    Ok((
        ms.next().ok_or("no item1")?.parse::<usize>()?,
        ms.next().ok_or("no item2")?.parse::<usize>()?,
    ))
}

fn main() {
    let mut b: Board<4, 4, 4> = Board::new();
    let mut ai = Minimax;
    let mut player = Player::Player1;
    println!("{}", b);
    while b.get_winner().is_none() {
        match player {
            Player::Player1 => {
                let mut line = String::new();
                std::io::stdin().read_line(&mut line).unwrap();
                if let Ok(pos) = parse_input(line.as_str()) {
                    if let Err(msg) = b.execute_move(player, pos) {
                        println!("Error: {:?}", msg);
                        continue;
                    }
                    println!("{}", b);
                } else {
                    println!("Not valid input");
                    continue;
                }
            }
            Player::Player2 => {
                b.execute_move(player, ai.get_next_move(&b, player))
                    .unwrap();
                println!("{}", b);
            }
        }
        player = player.get_next();
    }
}

mod test {
    use crate::{GameResult::*, Player::*, *};
    #[test]
    fn three_by_three_tests() {
        let board = Board::<3, 3, 3>([
            [Some(Player1), None, None],
            [Some(Player1), None, None],
            [Some(Player1), None, None],
        ]);

        assert_eq!(board.get_winner(), Some(PlayerWon(Player1)));
        assert_eq!(
            board.get_legal_moves(),
            vec![(0, 1), (0, 2), (1, 1), (1, 2), (2, 1), (2, 2)]
        );

        let board = Board::<3, 3, 3>([
            [None, Some(Player2), None],
            [None, Some(Player2), None],
            [None, Some(Player2), None],
        ]);

        assert_eq!(board.get_winner(), Some(PlayerWon(Player2)));

        let board = Board::<3, 3, 3>([
            [None, Some(Player2), None],
            [None, Some(Player2), None],
            [None, Some(Player1), None],
        ]);

        assert_eq!(board.get_winner(), None);

        let board = Board::<3, 3, 2>([
            [None, Some(Player2), None],
            [None, Some(Player2), None],
            [None, Some(Player1), None],
        ]);

        assert_eq!(board.get_winner(), Some(PlayerWon(Player2)));

        let board = Board::<3, 3, 3>([
            [None, Some(Player2), None],
            [None, Some(Player1), None],
            [Some(Player2), Some(Player2), Some(Player2)],
        ]);

        assert_eq!(board.get_winner(), Some(PlayerWon(Player2)));

        let board = Board::<3, 3, 2>([
            [None, Some(Player2), None],
            [None, Some(Player1), None],
            [Some(Player2), None, Some(Player1)],
        ]);

        assert_eq!(board.get_winner(), Some(PlayerWon(Player1)));

        let board = Board::<3, 3, 2>([
            [None, Some(Player2), None],
            [Some(Player1), None, None],
            [Some(Player2), Some(Player1), None],
        ]);

        assert_eq!(board.get_winner(), Some(PlayerWon(Player1)));
        assert_eq!(
            board.get_legal_moves(),
            vec![(0, 0), (0, 2), (1, 1), (1, 2), (2, 2)]
        );

        let board = Board::<3, 3, 2>([
            [None, Some(Player2), None],
            [Some(Player1), None, Some(Player2)],
            [Some(Player2), None, None],
        ]);

        assert_eq!(board.get_winner(), Some(PlayerWon(Player2)));

        let board = Board::<2, 3, 2>([
            [None, Some(Player2), None],
            [Some(Player1), None, Some(Player2)],
        ]);

        assert_eq!(board.get_winner(), Some(PlayerWon(Player2)));

        let board = Board::<3, 2, 2>([
            [None, Some(Player2)],
            [Some(Player1), None],
            [Some(Player2), Some(Player1)],
        ]);

        assert_eq!(board.get_winner(), Some(PlayerWon(Player1)));

        let board = Board::<3, 2, 2>([
            [None, Some(Player1)],
            [None, Some(Player2)],
            [Some(Player2), Some(Player1)],
        ]);

        assert_eq!(board.get_winner(), Some(PlayerWon(Player2)));

        let board = Board::<2, 3, 2>([
            [None, Some(Player2), None],
            [Some(Player2), None, Some(Player1)],
        ]);

        assert_eq!(board.get_winner(), Some(PlayerWon(Player2)));

        let board = Board::<3, 3, 2>([
            [None, None, None],
            [None, None, Some(Player1)],
            [None, Some(Player1), None],
        ]);

        assert_eq!(board.get_winner(), Some(PlayerWon(Player1)));
    }
}
