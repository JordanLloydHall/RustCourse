use std::fmt;

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

    fn get_last(&self) -> Self {
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

trait Game {
    type Move;
    fn moves(&self) -> Vec<Self::Move>;
    fn winner(&self) -> Option<GameResult>;
    fn execute_move(&mut self, player_move: Self::Move) -> Result<(), GameError>;
    fn undo_move(&mut self, player_move: Self::Move) -> Result<(), GameError>;
}

#[derive(PartialEq, Copy, Clone, Debug)]
struct MNKBoard<const M: usize, const N: usize, const K: usize> {
    board: [[Option<Player>; N]; M], 
    current_player: Player
}

impl<const M: usize, const N: usize, const K: usize> MNKBoard<M, N, K> {
    fn new() -> Self {
        MNKBoard {
            board: [[None; N]; M], 
            current_player: Player::Player1
        }
    }
}

impl<const M: usize, const N: usize, const K: usize> fmt::Display for MNKBoard<M, N, K> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        unimplemented!();
    }
}

impl<const M: usize, const N: usize, const K: usize> Game for MNKBoard<M, N, K> {
    type Move = (usize, usize);
    fn moves(&self) -> Vec<Self::Move> {
        unimplemented!();
    }

    fn winner(&self) -> Option<GameResult> {
        unimplemented!();
    }

    fn execute_move(&mut self, player_move: Self::Move) -> Result<(), GameError> {
        unimplemented!();
    }

    fn undo_move(&mut self, player_move: Self::Move) -> Result<(), GameError> {
        unimplemented!();
    }
}

struct Minimax;

impl Minimax {
    fn next_move<G: Game>(
        &mut self,
        game: &G
    ) -> G::Move {
        unimplemented!();
    }
}

fn parse_input(s: &str) -> Result<(usize, usize), Box<dyn std::error::Error>> {
    let mut ms = s.matches(char::is_numeric);
    Ok((
        ms.next().ok_or("no item1")?.parse::<usize>()?,
        ms.next().ok_or("no item2")?.parse::<usize>()?,
    ))
}

fn main() {
    let mut b: MNKBoard<4, 4, 4> = MNKBoard::new();
    let mut ai = Minimax;
    let mut player = Player::Player1;
    println!("{}", b);
    while b.winner().is_none() {
        match player {
            Player::Player1 => {
                let mut line = String::new();
                std::io::stdin().read_line(&mut line).unwrap();
                if let Ok(pos) = parse_input(line.as_str()) {
                    if let Err(msg) = b.execute_move(pos) {
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
                b.execute_move(ai.next_move(&b))
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
        let board = MNKBoard::<3, 3, 3>{
            board: [[Some(Player1), None, None],
            [Some(Player1), None, None],
            [Some(Player1), None, None]],
            current_player: Player1
        };

        assert_eq!(board.winner(), Some(PlayerWon(Player1)));
        assert_eq!(
            board.moves(),
            vec![(0, 1), (0, 2), (1, 1), (1, 2), (2, 1), (2, 2)]
        );

        let board = MNKBoard::<3, 3, 3> {
            board: [[None, Some(Player2), None],
            [None, Some(Player2), None],
            [None, Some(Player2), None]],
            current_player: Player1
        };

        assert_eq!(board.winner(), Some(PlayerWon(Player2)));

        let board = MNKBoard::<3, 3, 3> {
            board: [[None, Some(Player2), None],
            [None, Some(Player2), None],
            [None, Some(Player1), None]],
            current_player: Player1
        };

        assert_eq!(board.winner(), None);

        let board = MNKBoard::<3, 3, 2> {
            board: [[None, Some(Player2), None],
            [None, Some(Player2), None],
            [None, Some(Player1), None]],
            current_player: Player1
        };

        assert_eq!(board.winner(), Some(PlayerWon(Player2)));

        let board = MNKBoard::<3, 3, 3> {
            board: [[None, Some(Player2), None],
            [None, Some(Player1), None],
            [Some(Player2), Some(Player2), Some(Player2)]],
            current_player: Player1
        };

        assert_eq!(board.winner(), Some(PlayerWon(Player2)));

        let board = MNKBoard::<3, 3, 2> {
            board: [[None, Some(Player2), None],
            [None, Some(Player1), None],
            [Some(Player2), None, Some(Player1)]],
            current_player: Player1
        };

        assert_eq!(board.winner(), Some(PlayerWon(Player1)));

        let board = MNKBoard::<3, 3, 2> {
            board: [[None, Some(Player2), None],
            [Some(Player1), None, None],
            [Some(Player2), Some(Player1), None]],
            current_player: Player1
        };

        assert_eq!(board.winner(), Some(PlayerWon(Player1)));
        assert_eq!(
            board.moves(),
            vec![(0, 0), (0, 2), (1, 1), (1, 2), (2, 2)]
        );

        let board = MNKBoard::<3, 3, 2> {
            board: [[None, Some(Player2), None],
            [Some(Player1), None, Some(Player2)],
            [Some(Player2), None, None]],
            current_player: Player1
        };

        assert_eq!(board.winner(), Some(PlayerWon(Player2)));

        let board = MNKBoard::<2, 3, 2> {
            board: [[None, Some(Player2), None],
            [Some(Player1), None, Some(Player2)]],
            current_player: Player1
        };

        assert_eq!(board.winner(), Some(PlayerWon(Player2)));

        let board = MNKBoard::<3, 2, 2> {
            board: [[None, Some(Player2)],
            [Some(Player1), None],
            [Some(Player2), Some(Player1)]],
            current_player: Player1
        };

        assert_eq!(board.winner(), Some(PlayerWon(Player1)));

        let board = MNKBoard::<3, 2, 2> {
            board: [[None, Some(Player1)],
            [None, Some(Player2)],
            [Some(Player2), Some(Player1)]],
            current_player: Player1
        };

        assert_eq!(board.winner(), Some(PlayerWon(Player2)));

        let board = MNKBoard::<2, 3, 2> {
            board: [[None, Some(Player2), None],
            [Some(Player2), None, Some(Player1)]],
            current_player: Player1
        };

        assert_eq!(board.winner(), Some(PlayerWon(Player2)));

        let board = MNKBoard::<3, 3, 2> {
            board: [[None, None, None],
            [None, None, Some(Player1)],
            [None, Some(Player1), None]],
            current_player: Player1
        };

        assert_eq!(board.winner(), Some(PlayerWon(Player1)));
    }
}
