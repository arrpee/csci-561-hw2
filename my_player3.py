import numpy as np
from skimage.measure import label

import numpy.typing as npt
from typing import Iterator, List, Tuple, Set, Optional, Union


class GO:
    def __init__(
        self,
        piece: int,
        move_num: int,
        previous_board: npt.NDArray[np.int_],
        board: npt.NDArray[np.int_],
    ) -> None:
        self.size: int = 5
        self.komi: float = 2.5
        self.max_move_num: int = 25

        self.piece: int = piece
        self.opp_piece: int = 3 - self.piece
        self.move_num: int = move_num

        self.previous_board: npt.NDArray[np.int_] = previous_board
        self.board: npt.NDArray[np.int_] = board

        self.all_positions = [(i, j) for i in range(5) for j in range(5)]
        self.center_positions = [(i, j) for i in range(1, 4) for j in range(1, 4)]

        self.adjacent_positions = {
            (i, j): [pos for pos in self.get_adjacent_positions(i, j)]
            for i, j in self.all_positions
        }

        self.adjacent_diagonal_positions = {
            (i, j): [pos for pos in self.get_adjacent_diagonal_positions(i, j)]
            for i, j in self.all_positions
        }

        self.move_history: List[Optional[Tuple[int, int]]] = []
        self.board_history: List[npt.NDArray[np.int_]] = []

        self.allies: npt.NDArray[np.int_] = label(self.board, connectivity=1)
        self.allies_history: List[npt.NDArray[np.int_]] = []

    def make_move(self, i: int, j: int) -> None:
        self.board_history.append(self.previous_board)
        self.previous_board = np.copy(self.board)  # type: ignore

        self.allies_history.append(self.allies)
        self.move_history.append((i, j))
        self.board[i, j] = self.piece
        self.remove_dead_pieces()
        self.set_allies()

        self.move_num += 1
        self.piece, self.opp_piece = self.opp_piece, self.piece

    def make_pass_move(
        self,
    ) -> None:
        self.move_history.append(None)

        self.move_num += 1
        self.piece, self.opp_piece = self.opp_piece, self.piece

    def unmake_move(self) -> None:
        if self.move_history.pop():
            self.board = self.previous_board
            self.previous_board = self.board_history.pop()
            self.allies = self.allies_history.pop()
        self.move_num -= 1
        self.piece, self.opp_piece = self.opp_piece, self.piece

    def set_allies(self) -> None:
        self.allies = label(self.board, connectivity=1)

    def check_if_same_boards(self) -> bool:
        return (self.previous_board == self.board).all()  # type: ignore

    def check_valid_move(self, i: int, j: int) -> bool:
        if self.board[i, j]:
            # Invalid if position already has a piece
            return False

        temp = self.allies
        self.board[i, j] = self.piece
        self.set_allies()
        if self.check_liberty(i, j):
            # Valid if the position has liberty
            self.board[i, j] = 0
            self.allies = temp
            return True

        pieces_removed = self.remove_dead_pieces()
        self.set_allies()
        if not self.check_liberty(i, j) or (
            pieces_removed and self.check_if_same_boards()
        ):
            # Invalid if the position has no liberty after removing dead pieces or if KO rule will be violated
            for x, y in pieces_removed:
                self.board[x, y] = self.opp_piece
            self.board[i, j] = 0
            self.allies = temp
            return False

        for x, y in pieces_removed:
            self.board[x, y] = self.opp_piece
        self.board[i, j] = 0
        self.allies = temp
        return True

    def check_move_capture(self, i: int, j: int) -> bool:
        self.board[i, j] = self.piece
        capture = self.check_dead_pieces()
        self.board[i, j] = 0

        return capture

    def check_liberty(self, i: int, j: int) -> bool:
        for x, y in np.argwhere(self.allies == self.allies[i, j]):
            for a, b in self.adjacent_positions[(x, y)]:
                if not self.board[a, b]:
                    return True

        return False

    def check_game_end(self) -> bool:
        return self.move_num >= self.max_move_num or bool(
            self.move_history
            and not self.move_history[-1]
            and self.check_if_same_boards()
        )

    def check_dead_pieces(self) -> bool:
        for i, j in np.argwhere(self.board == self.opp_piece):
            if not self.check_liberty(i, j):
                return True

        return False

    def remove_dead_pieces(self) -> Set[Tuple[int, int]]:
        dead_pieces: Set[Tuple[int, int]] = set()

        for i, j in np.argwhere(self.board == self.opp_piece):
            if (i, j) not in dead_pieces and not self.check_liberty(i, j):
                dead_pieces |= set(
                    (x, y) for x, y in np.argwhere(self.allies == self.allies[i, j])
                )

        for i, j in dead_pieces:
            self.board[i, j] = 0

        return dead_pieces

    def get_valid_moves(self) -> Iterator[Optional[Tuple[int, int]]]:
        valid_moves = []
        discovered = set()
        last_move = self.get_last_move()

        if self.move_num <= 8:
            for i, j in self.center_positions:
                if self.check_valid_move(i, j):
                    discovered.add((i, j))
                    yield (i, j)
            if self.move_num <= 6:
                return

        if last_move:
            for i, j in self.adjacent_positions[last_move]:
                if self.check_valid_move(i, j):
                    if self.check_move_capture(i, j):
                        discovered.add((i, j))
                        yield (i, j)
                    else:
                        valid_moves.append((i, j))

            for i, j in self.adjacent_diagonal_positions[last_move]:
                if self.check_valid_move(i, j):
                    if self.check_move_capture(i, j):
                        discovered.add((i, j))
                        yield (i, j)
                    else:
                        valid_moves.append((i, j))

        for i, j in self.all_positions:
            if (i, j) not in discovered and self.check_valid_move(i, j):
                if self.check_move_capture(i, j):
                    yield (i, j)
                else:
                    valid_moves.append((i, j))

        for move in valid_moves:
            yield move

        if self.move_num > 15:
            yield None

    def get_last_move(self) -> Optional[Tuple[int, int]]:

        if self.move_history:
            return self.move_history[-1]

        for i, j in np.argwhere(self.board != self.previous_board):
            if self.board[i, j] == self.opp_piece:
                return (i, j)

        return None

    def get_pattern_counts(self, piece: int) -> int:
        _, ct = label(self.board == piece, connectivity=2, return_num=True)
        return ct  # type: ignore

    def get_liberty_counts(self, piece: int) -> int:
        discovered: Set[Tuple[int, int]] = set()
        ct = 0
        for i, j in np.argwhere(self.board == piece):
            if (i, j) not in discovered:
                for a, b in np.argwhere(self.allies == self.allies[i, j]):
                    discovered.add((a, b))
                    for x, y in self.adjacent_positions[(a, b)]:
                        if not self.board[x, y]:
                            ct += 1
        return ct

    def get_piece_counts(self) -> Tuple[int, int]:
        return (self.board == 1).sum(), (self.board == 2).sum()

    def get_winner(self) -> int:
        black_count, white_count = self.get_piece_counts()

        if black_count > white_count + self.komi:
            return 1
        elif black_count < white_count + self.komi:
            return 2
        else:
            return 0

    def get_adjacent_positions(self, i: int, j: int) -> List[Tuple[int, int]]:
        positions = []
        if i > 0:
            positions.append((i - 1, j))
        if i < self.size - 1:
            positions.append((i + 1, j))
        if j > 0:
            positions.append((i, j - 1))
        if j < self.size - 1:
            positions.append((i, j + 1))
        return positions

    def get_adjacent_diagonal_positions(self, i: int, j: int) -> List[Tuple[int, int]]:
        positions = []
        if i > 0 and j > 0:
            positions.append((i - 1, j - 1))
        if i < self.size - 1 and j < self.size - 1:
            positions.append((i + 1, j + 1))
        if i < self.size - 1 and j > 0:
            positions.append((i + 1, j - 1))
        if i > 0 and j < self.size - 1:
            positions.append((i - 1, j + 1))
        return positions


class Player:
    WIN_UTILITY = 1000

    def __init__(self) -> None:
        self.piece: int
        self.opp_piece: int

        self.board: GO
        self.search_depth = 4

        self.filepath_input: str = "input.txt"
        self.filepath_output: str = "output.txt"
        self.filepath_move_num: str = "move_num.txt"

    def read_board(self) -> None:
        with open(self.filepath_input, "r") as f:
            lines = f.read().splitlines()

        self.piece = self.num_move = int(lines[0])
        self.opp_piece = 3 - self.piece

        previous_board = np.array([[int(x) for x in line] for line in lines[1:6]])
        board = np.array([[int(x) for x in line] for line in lines[6:]])

        if board.sum() > 1:
            # If there is <= one piece on the board find the actual move no
            self.read_move_num()

        self.board = GO(self.piece, self.num_move, previous_board, board)

    def write_move(self, move: Optional[Tuple[int, int]]) -> None:
        self.write_move_num()

        with open(self.filepath_output, "w") as f:
            f.write(f"{move[0]},{move[1]}" if move else "PASS")

    def read_move_num(self) -> None:
        with open(self.filepath_move_num, "r") as f:
            m = int(f.read())
            if m < 25:
                self.num_move = m

    def write_move_num(self) -> None:
        with open(self.filepath_move_num, "w") as f:
            f.write(f"{self.board.move_num + 2}")

    def compute_move(self) -> Optional[Tuple[int, int]]:
        if self.board.move_num <= 4:
            for (i, j) in [
                (2, 2),
                (2, 3),
                (3, 2),
                (2, 1),
                (1, 2),
            ]:
                if self.board.check_valid_move(i, j):
                    return (i, j)
        return self.alpha_beta_search(self.search_depth)

    def evaluate_position(self) -> Union[int, float]:
        utility = 0
        multiplier = 1 if self.piece == 1 else -1

        black_counts, white_counts = self.board.get_piece_counts()
        utility += (black_counts - white_counts) * multiplier * 10

        utility += self.board.get_liberty_counts(self.piece) * 6
        utility -= self.board.get_liberty_counts(self.opp_piece) * 6

        utility -= self.board.get_pattern_counts(self.piece) * 4
        utility += self.board.get_pattern_counts(self.opp_piece) * 4

        for i, j in np.argwhere(self.board.board == self.piece):
            if i == 0 or i == 4:
                utility -= 2
            if j == 0 or j == 4:
                utility -= 2
        return utility

    def alpha_beta_search(self, depth: int) -> Optional[Tuple[int, int]]:
        best_move, _ = self.alpha_beta_max(float("-inf"), float("inf"), depth)
        return best_move

    def alpha_beta_max(
        self, alpha: Union[float, int], beta: Union[float, int], depth: int
    ) -> Tuple[Optional[Tuple[int, int]], Union[float, int]]:

        if self.board.check_game_end():
            return (
                None,
                Player.WIN_UTILITY
                if self.board.get_winner() == self.piece
                else -Player.WIN_UTILITY,
            )
        if not depth:
            return None, self.evaluate_position()

        best_move: Optional[Tuple[int, int]] = None
        best_value = float("-inf")
        for move in self.board.get_valid_moves():
            if move:
                self.board.make_move(move[0], move[1])
            else:
                self.board.make_pass_move()
            _, value = self.alpha_beta_min(alpha, beta, depth - 1)
            self.board.unmake_move()
            if value > best_value:
                best_value = value
                best_move = move

            if value >= beta:
                return best_move, value

            alpha = max(alpha, value)

        return best_move, best_value

    def alpha_beta_min(
        self, alpha: Union[float, int], beta: Union[float, int], depth: int
    ) -> Tuple[Optional[Tuple[int, int]], Union[float, int]]:
        if self.board.check_game_end():
            return (
                None,
                Player.WIN_UTILITY
                if self.board.get_winner() == self.piece
                else -Player.WIN_UTILITY,
            )

        if not depth:
            return None, self.evaluate_position()

        best_move: Optional[Tuple[int, int]] = None
        best_value = float("inf")
        for move in self.board.get_valid_moves():
            if move:
                self.board.make_move(move[0], move[1])
            else:
                self.board.make_pass_move()
            _, value = self.alpha_beta_max(alpha, beta, depth - 1)
            self.board.unmake_move()

            if value < best_value:
                best_value = value
                best_move = move

            if value <= alpha:
                return best_move, value

            beta = min(beta, value)

        return best_move, best_value


if __name__ == "__main__":
    player = Player()
    player.read_board()
    move = player.compute_move()
    player.write_move(move)
