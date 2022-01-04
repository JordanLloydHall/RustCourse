use std::{fmt, collections::HashMap};
use strum::{EnumIter, IntoEnumIterator};

#[derive(PartialEq, Copy, Clone, Debug, EnumIter, Eq, Hash)]
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
    MoveNotFound
}

trait Game {
    type Move;
    fn moves(&self) -> Vec<Self::Move>;
    fn winner(&self) -> Option<GameResult>;
    fn execute_move(&mut self, player_move: Self::Move) -> Result<(), GameError>;
    fn undo_move(&mut self, player_move: Self::Move) -> Result<(), GameError>;
}
 
#[derive(PartialEq, Clone, Debug)]
struct MNKBoard<const M: usize, const N: usize, const K: usize> {
    ply: usize,
    board: [[Option<Player>; N]; M], 
    move_sequence: Vec<<MNKBoard<M,N,K> as Game>::Move>,
    current_player: Player
}

impl<const M: usize, const N: usize, const K: usize> MNKBoard<M, N, K> {
    fn new() -> Self {
        MNKBoard {
            ply: 0,
            board: [[None; N]; M],
            move_sequence: vec![],
            current_player: Player::Player1
        }
    }

    fn iter2d(&self) -> impl Iterator<Item = ((usize, usize), &Option<Player>)> {
        self.board.iter()
        .enumerate()
        .flat_map(|(y, row)| 
            row.iter().enumerate().map(move |(x, cell)| ((x, y), cell)))
    }

    // Check if moves supplied contain at least one K-in-a-row combination for the specified player 
    // Time complexity: O(L), where L = moves.len()
    fn has_win_combination(mut moves: Vec<&Option<Player>>, player: Player) -> bool {
        if moves.len() < K {
            return false; 
        }

        let mut count: HashMap<Option<Player>, usize> = HashMap::new();
        let mut to_remove: usize = 0; 
        let remainder = moves.split_off(K);

        for &&m in &moves {
            *count.entry(m).or_insert(0) += 1;
        }

        if *count.entry(Some(player)).or_insert(0) == K {
            return true; 
        }

        for &&m in &remainder {
            *count.entry(*moves[to_remove]).or_insert(0) -= 1;
            *count.entry(m).or_insert(0) += 1;
            to_remove += 1;
            if  *count.entry(Some(player)).or_insert(0) == K {
                return true;
            }
        }

        false
        // moves.windows(K)
        //      .map(|ms| ms.iter().filter(|&&&m| m == Some(player)))
        //      .any(|ms| ms.count() == K)
    }

    // Check if current move results in the win of the current player
    // Time complexity: O(K)
    fn check_current_move(&self) -> Option<GameResult> {
        let current_move = self.move_sequence.last();

        if current_move.is_none() {
            return None;
        }

        let &(r, c) = current_move.unwrap();
        let player = self.board[r][c].unwrap();

        // Initialise horizontal and vertical ranges
        let horizontal = ((c - K + 1)..(c + K)).filter(|col| col + 1 >= K && col + K <= N);
        let vertical = ((r - K + 1)..(r + K)).filter(|row| row + 1 >= K && row + K <= M);

        // Check horizontal moves
        let horizontal_moves: Vec<_> = horizontal.clone().map(|col| &self.board[r][col]).collect();
        if Self::has_win_combination(horizontal_moves, player) { return Some(GameResult::PlayerWon(player)); }

        // Check vertical moves
        let vertical_moves: Vec<_> = vertical.clone().map(|row| &self.board[row][c]).collect();
        if Self::has_win_combination(vertical_moves, player) { return Some(GameResult::PlayerWon(player)); }

        // Check main diagonal moves 
        let main_diag_moves: Vec<_> = horizontal.clone().zip(vertical.clone()).map(|(row, col)| &self.board[row][col]).collect();
        if Self::has_win_combination(main_diag_moves, player) { return Some(GameResult::PlayerWon(player)); }

        // Check anti diagonal moves 
        let anti_diag_moves: Vec<_> = horizontal.zip(vertical.rev()).map(|(row, col)| &self.board[row][col]).collect(); 
        if Self::has_win_combination(anti_diag_moves, player) { return Some(GameResult::PlayerWon(player)); }

        None
    }

    // Inspect the entire board to find a winner
    // Usage: when a non-trivial board state is preloaded with no move sequence provided
    // Time complexity: O(M * N)
    fn check_board(&self) -> Option<GameResult> {
        // Check horizontal moves
        for horizontal_moves in self.board {
            for player in Player::iter() {
                if Self::has_win_combination(horizontal_moves.iter().collect(), player) { 
                    return Some(GameResult::PlayerWon(player)); 
                }
            }
        }

        // Check vertical moves 
        for col in 0..N {
            let vertical_moves: Vec<_> = (0..M).map(|row| &self.board[row][col]).collect();
            for player in Player::iter() {
                if Self::has_win_combination(vertical_moves.clone(), player) { 
                    return Some(GameResult::PlayerWon(player)); 
                }
            }
        }

        // Initialise diagonal moves 
        let mut main_diag_moves: Vec<Vec<&Option<Player>>> = vec![vec![]; M + N - 1];
        let mut anti_diag_moves: Vec<Vec<&Option<Player>>> = vec![vec![]; main_diag_moves.len()];

        for row in 0..M {
            for col in 0..N {
                main_diag_moves[row + col].push(&self.board[row][col]);
            }
        }

        // Main diagonals 
        for diagonal_moves in main_diag_moves {
            for player in Player::iter() {
                if Self::has_win_combination(diagonal_moves.clone(), player) { 
                    return Some(GameResult::PlayerWon(player)); 
                }
            }
        }

        let min_anti_diag = - (M as i32) + 1; 

        for row in 0..M {
            for col in 0..N {
                let index = col as i32 - row as i32 - min_anti_diag; 
                anti_diag_moves[index as usize].push(&self.board[row][col]); 
            }
        }

        // Anti diagonals 
        for diagonal_moves in anti_diag_moves {
            for player in Player::iter() {
                if Self::has_win_combination(diagonal_moves.clone(), player) { 
                    return Some(GameResult::PlayerWon(player)); 
                }
            }
        }

        None 
    }
}

impl<const M: usize, const N: usize, const K: usize> fmt::Display for MNKBoard<M, N, K> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        unimplemented!();
    }
}

impl<const M: usize, const N: usize, const K: usize> Game for MNKBoard<M, N, K> {
    type Move = (usize, usize);

    // Generate list of legal moves that can be made by a player 
    // Time complexity: O(M * N)
    fn moves(&self) -> Vec<Self::Move> {
        self.iter2d()
        .filter(|(_, c)| c.is_none())
        .map(|((r, c), _)| (c, r))
        .collect()
    }

    // Evaluate the game state to determine if there is a winner 
    // An analysis for this function can be conducted to find amortized time complexity 
    fn winner(&self) -> Option<GameResult> {
        if self.ply == M * N {
            return Some(GameResult::Draw);
        }

        match self.move_sequence.last() {
            // Board preloaded with no move sequences given; inspect whole board to determine game result 
            // Time complexity: O(M * N)
            None => { self.check_board() }
            // Use last move to determine game result 
            // Time complexity: O(K)
            _    => { self.check_current_move() }
        }
    }

    fn execute_move(&mut self, (r, c): Self::Move) -> Result<(), GameError> {
        if !(0..self.board.len()).contains(&r) || !(0..self.board[0].len()).contains(&c) {
            return Err(GameError::NotLegalPosition);
        }
        if self.board[r][c].is_some() {
            return Err(GameError::PositionOccupied);
        }

        self.ply += 1; 
        self.board[r][c] = Some(self.current_player);
        Ok(())
    }

    // Undo all moves starting from player_move if it is a valid move, otherwise throw a MoveNotFound error
    fn undo_move(&mut self, player_move: Self::Move) -> Result<(), GameError> {
        let index = self.move_sequence.iter().position(|&m| m == player_move);

        match index {
            None => { Err(GameError::MoveNotFound) }, 
            Some(pos) => { // ply = pos + 1
                let removed_moves = self.move_sequence.drain(pos..);
                removed_moves.into_iter().for_each(|(r, c)| self.board[r][c] = None);

                let removed_ply_count = self.ply - pos;
                if removed_ply_count % 2 != 0 { 
                    self.current_player = self.current_player.get_last();
                }

                self.ply = pos - 1;
                Ok(())
            }
        }
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
            ply: 3,
            board: [[Some(Player1), None, None],
                    [Some(Player1), None, None],
                    [Some(Player1), None, None]],
            move_sequence: vec![],
            current_player: Player1
        };

        assert_eq!(board.winner(), Some(PlayerWon(Player1)));
        assert_eq!(
            board.moves(),
            vec![(0, 1), (0, 2), (1, 1), (1, 2), (2, 1), (2, 2)]
        );

        let board = MNKBoard::<3, 3, 3> {
            ply: 3,
            board: [[None, Some(Player2), None],
                    [None, Some(Player2), None],
                    [None, Some(Player2), None]],
            move_sequence: vec![],
            current_player: Player1
        };

        assert_eq!(board.winner(), Some(PlayerWon(Player2)));

        let board = MNKBoard::<3, 3, 3> {
            ply: 3,
            board: [[None, Some(Player2), None],
                    [None, Some(Player2), None],
                    [None, Some(Player1), None]],
            move_sequence: vec![],
            current_player: Player1
        };

        assert_eq!(board.winner(), None);

        let board = MNKBoard::<3, 3, 2> {
            ply: 3,
            board: [[None, Some(Player2), None],
                    [None, Some(Player2), None],
                    [None, Some(Player1), None]],
            move_sequence: vec![],
            current_player: Player1
        };

        assert_eq!(board.winner(), Some(PlayerWon(Player2)));

        let board = MNKBoard::<3, 3, 3> {
            ply: 5, 
            board: [[None, Some(Player2), None],
                    [None, Some(Player1), None],
                    [Some(Player2), Some(Player2), Some(Player2)]],
            move_sequence: vec![],
            current_player: Player1
        };

        assert_eq!(board.winner(), Some(PlayerWon(Player2)));

        let board = MNKBoard::<3, 3, 2> {
            ply: 4,
            board: [[None, Some(Player2), None],
                    [None, Some(Player1), None],
                    [Some(Player2), None, Some(Player1)]],
            move_sequence: vec![], 
            current_player: Player1
        };

        assert_eq!(board.winner(), Some(PlayerWon(Player1)));

        let board = MNKBoard::<3, 3, 2> {
            ply: 4,
            board: [[None, Some(Player2), None],
                    [Some(Player1), None, None],
                    [Some(Player2), Some(Player1), None]],
            move_sequence: vec![],
            current_player: Player1
        };

        assert_eq!(board.winner(), Some(PlayerWon(Player1)));
        assert_eq!(
            board.moves(),
            vec![(0, 0), (0, 2), (1, 1), (1, 2), (2, 2)]
        );

        let board = MNKBoard::<3, 3, 2> {
            ply: 4,
            board: [[None, Some(Player2), None],
                    [Some(Player1), None, Some(Player2)],
                    [Some(Player2), None, None]],
            move_sequence: vec![],
            current_player: Player1
        };

        assert_eq!(board.winner(), Some(PlayerWon(Player2)));

        let board = MNKBoard::<2, 3, 2> {
            ply: 3,
            board: [[None, Some(Player2), None],
            [Some(Player1), None, Some(Player2)]],
            move_sequence: vec![],
            current_player: Player1
        };

        assert_eq!(board.winner(), Some(PlayerWon(Player2)));

        let board = MNKBoard::<3, 2, 2> {
            ply: 4,
            board: [[None, Some(Player2)],
            [Some(Player1), None],
            [Some(Player2), Some(Player1)]],
            move_sequence: vec![],
            current_player: Player1
        };

        assert_eq!(board.winner(), Some(PlayerWon(Player1)));

        let board = MNKBoard::<3, 2, 2> {
            ply: 4, 
            board: [[None, Some(Player1)],
            [None, Some(Player2)],
            [Some(Player2), Some(Player1)]],
            move_sequence: vec![],
            current_player: Player1
        };

        assert_eq!(board.winner(), Some(PlayerWon(Player2)));

        let board = MNKBoard::<2, 3, 2> {
            ply: 3,
            board: [[None, Some(Player2), None],
            [Some(Player2), None, Some(Player1)]],
            move_sequence: vec![],
            current_player: Player1
        };

        assert_eq!(board.winner(), Some(PlayerWon(Player2)));

        let board = MNKBoard::<3, 3, 2> {
            ply: 2, 
            board: [[None, None, None],
            [None, None, Some(Player1)],
            [None, Some(Player1), None]],
            move_sequence: vec![],
            current_player: Player1
        };

        assert_eq!(board.winner(), Some(PlayerWon(Player1)));
    }
}
