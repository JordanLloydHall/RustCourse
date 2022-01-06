use colored::*;
use itertools::Itertools;
use std::{cmp, collections::HashMap, fmt};
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
    MoveNotFound,
}

trait Game {
    type Move: Clone;
    fn moves(&self) -> Vec<Self::Move>;
    fn winner(&self) -> Option<GameResult>;
    fn execute_move(&mut self, player_move: Self::Move) -> Result<(), GameError>;
    fn undo_move(&mut self, player_move: Self::Move) -> Result<(), GameError>;
}

#[derive(PartialEq, Clone, Debug)]
struct MNKBoard<const M: usize, const N: usize, const K: usize> {
    ply: usize,
    board: [[Option<Player>; N]; M],
    move_sequence: Vec<<MNKBoard<M, N, K> as Game>::Move>,
    current_player: Player,
}

impl<const M: usize, const N: usize, const K: usize> MNKBoard<M, N, K> {
    fn new() -> Self {
        MNKBoard {
            ply: 0,
            board: [[None; N]; M],
            move_sequence: vec![],
            current_player: Player::Player1,
        }
    }

    fn iter2d(&self) -> impl Iterator<Item = ((usize, usize), &Option<Player>)> {
        self.board.iter().enumerate().flat_map(|(r, row_cells)| {
            row_cells
                .iter()
                .enumerate()
                .map(move |(c, cell)| ((r, c), cell))
        })
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

        // TODO: replace with for each
        for &&m in &moves {
            *count.entry(m).or_insert(0) += 1;
        }

        if *count.entry(Some(player)).or_insert(0) == K {
            return true;
        }

        // TODO: replace with windows
        for &&m in &remainder {
            *count.entry(*moves[to_remove]).or_insert(0) -= 1;
            *count.entry(m).or_insert(0) += 1;
            to_remove += 1;
            if *count.entry(Some(player)).or_insert(0) == K {
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
        current_move?;

        let &(r, c) = current_move.unwrap();
        let (r32, c32) = (r as i32, c as i32);
        let (m32, n32, k32) = (M as i32, N as i32, K as i32);
        let player = self.board[r][c].unwrap();

        // Initialise horizontal and vertical ranges
        let horizontal = ((c32 - k32 + 1)..(c32 + k32))
            .filter(|&col| col >= 0 && col < n32)
            .map(|v| v as usize);

        let vertical = ((r32 - k32 + 1)..(r32 + k32))
            .filter(|&row| row >= 0 && row < m32)
            .map(|v| v as usize);

        // Check horizontal moves
        let horizontal_moves: Vec<_> = horizontal.clone().map(|col| &self.board[r][col]).collect();
        if Self::has_win_combination(horizontal_moves, player) {
            return Some(GameResult::PlayerWon(player));
        }

        // Check vertical moves
        let vertical_moves: Vec<_> = vertical.clone().map(|row| &self.board[row][c]).collect();
        if Self::has_win_combination(vertical_moves, player) {
            return Some(GameResult::PlayerWon(player));
        }

        // Check main diagonal moves
        let main_diag_moves: Vec<_> = horizontal
            .clone()
            .zip(vertical.clone())
            .map(|(row, col)| &self.board[row][col])
            .collect();
        if Self::has_win_combination(main_diag_moves, player) {
            return Some(GameResult::PlayerWon(player));
        }

        // Check anti diagonal moves
        let anti_diag_moves: Vec<_> = horizontal
            .zip(vertical.rev())
            .map(|(row, col)| &self.board[row][col])
            .collect();
        if Self::has_win_combination(anti_diag_moves, player) {
            return Some(GameResult::PlayerWon(player));
        }

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

        let min_anti_diag = -(M as i32) + 1;

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
        let sep_row = format!("{}\n", "-".repeat(4 * N - 1));
        let board = self
            .board
            .iter()
            .map(|row| {
                let pretty_row = row.iter().map(|c| match c {
                    Some(Player::Player1) => " O ".cyan().bold(),
                    Some(Player::Player2) => " X ".red().bold(),
                    None => "   ".normal(),
                });
                format!("{}\n", pretty_row.format("|"))
            })
            .format(&sep_row);
        write!(f, "{}", board)
    }
}

impl<const M: usize, const N: usize, const K: usize> Game for MNKBoard<M, N, K> {
    type Move = (usize, usize);

    // Generate list of legal moves that can be made by a player
    // Time complexity: O(M * N)
    // TODO: use HashSet to improve time
    fn moves(&self) -> Vec<Self::Move> {
        self.iter2d()
            .filter(|(_, c)| c.is_none())
            .map(|((r, c), _)| (r, c))
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
            None => self.check_board(),
            // Use last move to determine game result
            // Time complexity: O(K)
            _ => self.check_current_move(),
        }
    }

    fn execute_move(&mut self, player_move @ (r, c): Self::Move) -> Result<(), GameError> {
        if !(0..M).contains(&r) || !(0..N).contains(&c) {
            return Err(GameError::NotLegalPosition);
        }
        if self.board[r][c].is_some() {
            return Err(GameError::PositionOccupied);
        }

        self.ply += 1;
        self.board[r][c] = Some(self.current_player);
        self.current_player = self.current_player.get_next();
        self.move_sequence.push(player_move);
        Ok(())
    }

    // Undo all moves starting from player_move if it is a valid move, otherwise throw a MoveNotFound error
    fn undo_move(&mut self, player_move: Self::Move) -> Result<(), GameError> {
        let index = self.move_sequence.iter().position(|&m| m == player_move);

        match index {
            None => Err(GameError::MoveNotFound),
            Some(pos) => {
                // ply = pos + 1
                let removed_moves = self.move_sequence.drain(pos..);
                removed_moves
                    .into_iter()
                    .for_each(|(r, c)| self.board[r][c] = None);

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
    const WIN_VAL: i32 = i32::MAX;
    const LOSE_VAL: i32 = i32::MIN;

    fn evaluate<G: Game>(&self, game: &G, depth: i32) -> i32 {
        match game.winner() {
            Some(GameResult::PlayerWon(Player::Player1)) => Minimax::WIN_VAL - depth,
            Some(GameResult::PlayerWon(Player::Player2)) => Minimax::LOSE_VAL + depth,
            _ => 0,
        }
    }

    fn alphabeta<G: Game>(
        &self,
        game: &mut G,
        depth: i32,
        mut alpha: i32,
        mut beta: i32,
        player: Player,
    ) -> i32 {
        if game.winner().is_some() {
            return self.evaluate(game, depth);
        }

        match player {
            Player::Player1 => {
                let mut eval = i32::MIN;
                for m in game.moves() {
                    game.execute_move(m.clone()).unwrap();
                    eval = cmp::max(
                        eval,
                        self.alphabeta(game, depth + 1, alpha, beta, Player::Player2),
                    );
                    game.undo_move(m).unwrap();
                    if eval >= beta {
                        break;
                    }
                    alpha = cmp::max(alpha, eval);
                }
                eval
            }
            Player::Player2 => {
                let mut eval = i32::MAX;
                for m in game.moves() {
                    game.execute_move(m.clone()).unwrap();
                    eval = cmp::min(
                        eval,
                        self.alphabeta(game, depth + 1, alpha, beta, Player::Player1),
                    );
                    game.undo_move(m.clone()).unwrap();
                    if eval <= alpha {
                        break;
                    }
                    beta = cmp::min(beta, eval);
                }
                eval
            }
        }
    }

    fn next_move<G: Game>(&mut self, game: &mut G) -> G::Move {
        let move_evals: Vec<_> = game
            .moves()
            .iter()
            .map(|m| {
                game.execute_move(m.clone()).unwrap();
                let eval = self.alphabeta(
                    game,
                    0,
                    Minimax::LOSE_VAL,
                    Minimax::WIN_VAL,
                    Player::Player2,
                );
                game.undo_move(m.clone()).unwrap();
                (m.clone(), eval)
            })
            .collect();

        let (best_move, _) = move_evals
            .iter()
            .max_by(|(_, v1), (_, v2)| v1.cmp(v2))
            .unwrap();
        best_move.clone()
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
    let mut b: MNKBoard<3, 3, 3> = MNKBoard::new();
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
                let ai_move = ai.next_move(&mut b);
                b.execute_move(ai_move).unwrap();
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
        let board = MNKBoard::<3, 3, 3> {
            ply: 3,
            board: [
                [Some(Player1), None, None],
                [Some(Player1), None, None],
                [Some(Player1), None, None],
            ],
            move_sequence: vec![],
            current_player: Player1,
        };

        assert_eq!(board.winner(), Some(PlayerWon(Player1)));
        assert_eq!(
            board.moves(),
            vec![(0, 1), (0, 2), (1, 1), (1, 2), (2, 1), (2, 2)]
        );

        let board = MNKBoard::<3, 3, 3> {
            ply: 3,
            board: [
                [None, Some(Player2), None],
                [None, Some(Player2), None],
                [None, Some(Player2), None],
            ],
            move_sequence: vec![],
            current_player: Player1,
        };

        assert_eq!(board.winner(), Some(PlayerWon(Player2)));

        let board = MNKBoard::<3, 3, 3> {
            ply: 3,
            board: [
                [None, Some(Player2), None],
                [None, Some(Player2), None],
                [None, Some(Player1), None],
            ],
            move_sequence: vec![],
            current_player: Player1,
        };

        assert_eq!(board.winner(), None);

        let board = MNKBoard::<3, 3, 2> {
            ply: 3,
            board: [
                [None, Some(Player2), None],
                [None, Some(Player2), None],
                [None, Some(Player1), None],
            ],
            move_sequence: vec![],
            current_player: Player1,
        };

        assert_eq!(board.winner(), Some(PlayerWon(Player2)));

        let board = MNKBoard::<3, 3, 3> {
            ply: 5,
            board: [
                [None, Some(Player2), None],
                [None, Some(Player1), None],
                [Some(Player2), Some(Player2), Some(Player2)],
            ],
            move_sequence: vec![],
            current_player: Player1,
        };

        assert_eq!(board.winner(), Some(PlayerWon(Player2)));

        let board = MNKBoard::<3, 3, 2> {
            ply: 4,
            board: [
                [None, Some(Player2), None],
                [None, Some(Player1), None],
                [Some(Player2), None, Some(Player1)],
            ],
            move_sequence: vec![],
            current_player: Player1,
        };

        assert_eq!(board.winner(), Some(PlayerWon(Player1)));

        let board = MNKBoard::<3, 3, 2> {
            ply: 4,
            board: [
                [None, Some(Player2), None],
                [Some(Player1), None, None],
                [Some(Player2), Some(Player1), None],
            ],
            move_sequence: vec![],
            current_player: Player1,
        };

        assert_eq!(board.winner(), Some(PlayerWon(Player1)));
        assert_eq!(board.moves(), vec![(0, 0), (0, 2), (1, 1), (1, 2), (2, 2)]);

        let board = MNKBoard::<3, 3, 2> {
            ply: 4,
            board: [
                [None, Some(Player2), None],
                [Some(Player1), None, Some(Player2)],
                [Some(Player2), None, None],
            ],
            move_sequence: vec![],
            current_player: Player1,
        };

        assert_eq!(board.winner(), Some(PlayerWon(Player2)));

        let board = MNKBoard::<2, 3, 2> {
            ply: 3,
            board: [
                [None, Some(Player2), None],
                [Some(Player1), None, Some(Player2)],
            ],
            move_sequence: vec![],
            current_player: Player1,
        };

        assert_eq!(board.winner(), Some(PlayerWon(Player2)));

        let board = MNKBoard::<3, 2, 2> {
            ply: 4,
            board: [
                [None, Some(Player2)],
                [Some(Player1), None],
                [Some(Player2), Some(Player1)],
            ],
            move_sequence: vec![],
            current_player: Player1,
        };

        assert_eq!(board.winner(), Some(PlayerWon(Player1)));

        let board = MNKBoard::<3, 2, 2> {
            ply: 4,
            board: [
                [None, Some(Player1)],
                [None, Some(Player2)],
                [Some(Player2), Some(Player1)],
            ],
            move_sequence: vec![],
            current_player: Player1,
        };

        assert_eq!(board.winner(), Some(PlayerWon(Player2)));

        let board = MNKBoard::<2, 3, 2> {
            ply: 3,
            board: [
                [None, Some(Player2), None],
                [Some(Player2), None, Some(Player1)],
            ],
            move_sequence: vec![],
            current_player: Player1,
        };

        assert_eq!(board.winner(), Some(PlayerWon(Player2)));

        let board = MNKBoard::<3, 3, 2> {
            ply: 2,
            board: [
                [None, None, None],
                [None, None, Some(Player1)],
                [None, Some(Player1), None],
            ],
            move_sequence: vec![],
            current_player: Player1,
        };

        assert_eq!(board.winner(), Some(PlayerWon(Player1)));
    }

    #[test]
    fn check_result_by_last_move() {
        let board = MNKBoard::<3, 3, 3> {
            ply: 5,
            board: [
                [Some(Player1), Some(Player1), Some(Player1)],
                [None, Some(Player2), None],
                [None, Some(Player2), None],
            ],
            move_sequence: vec![(0, 0)],
            current_player: Player2,
        };

        assert_eq!(board.winner(), Some(PlayerWon(Player1)));

        let board = MNKBoard::<3, 3, 3> {
            ply: 5,
            board: [
                [Some(Player1), Some(Player1), Some(Player1)],
                [None, Some(Player2), None],
                [None, Some(Player2), None],
            ],
            move_sequence: vec![(0, 1)],
            current_player: Player2,
        };

        assert_eq!(board.winner(), Some(PlayerWon(Player1)));

        let board = MNKBoard::<3, 3, 3> {
            ply: 5,
            board: [
                [Some(Player1), Some(Player1), Some(Player1)],
                [None, Some(Player2), None],
                [None, Some(Player2), None],
            ],
            move_sequence: vec![(0, 2)],
            current_player: Player2,
        };

        assert_eq!(board.winner(), Some(PlayerWon(Player1)));
    }
}
