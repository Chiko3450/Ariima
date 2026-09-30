import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random

# Define the neural network architecture for the AI
class MovePredictor(nn.Module):
    def __init__(self, input_size, output_size):
        super(MovePredictor, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Define the Piece class (same as before)
class Piece:
    def __init__(self, player, rank):
        self.player = player
        self.rank = rank
        self.frozen = False

    def __str__(self):
        return f"{self.player}{self.rank[0].upper()}"

# Board class to manage the game board (same as before)
class Board:
    def __init__(self):
        self.board = [[None for _ in range(8)] for _ in range(8)]
        self.trap_squares = [(2, 2), (2, 5), (5, 2), (5, 5)]  # c3, f3, c6, f6
        self.setup_board()

    def setup_board(self):
        self.board[6] = [Piece('A', rank) for rank in ['elephant', 'camel', 'horse', 'dog', 'dog', 'horse', 'camel', 'elephant']]
        self.board[7] = [Piece('A', rank) for rank in ['rabbit', 'rabbit', 'rabbit', 'rabbit', 'rabbit', 'rabbit', 'rabbit', 'rabbit']]  

        self.board[0] = [Piece('B', rank) for rank in ['elephant', 'camel', 'horse', 'dog', 'dog', 'horse', 'camel', 'elephant']]
        self.board[1] = [Piece('B', rank) for rank in ['rabbit', 'rabbit', 'rabbit', 'rabbit', 'rabbit', 'rabbit', 'rabbit', 'rabbit']]  

    def display(self):
        for x, row in enumerate(self.board):
            print(f"{8 - x} " + " | ".join(str(piece) if piece else " . " for piece in row))
        print("   A   B   C   D   E   F   G   H\n")

    def get_valid_moves(self, player):
        moves = []
        for x in range(8):
            for y in range(8):
                piece = self.board[x][y]
                if piece and piece.player == player and not piece.frozen:
                    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        new_x, new_y = x + dx, y + dy
                        if self.is_valid_move(piece, (x, y), (new_x, new_y)):
                            moves.append(((x, y), (new_x, new_y)))
        return moves

    def move_piece(self, start, end):
        x1, y1 = start
        x2, y2 = end
        piece = self.board[x1][y1]
        if piece and self.is_valid_move(piece, start, end):
            self.board[x2][y2] = piece
            self.board[x1][y1] = None
            return True
        return False

    def is_valid_move(self, piece, start, end):
        x1, y1 = start
        x2, y2 = end
        if not (0 <= x2 < 8 and 0 <= y2 < 8):  # Bounds check
            return False
        if self.board[x2][y2] is not None:  # Can't move to occupied square
            return False
        if piece.rank == 'rabbit' and x2 < x1:  # Rabbits can't move backward
            return False
        return True

    def check_victory(self):
        if any(self.board[0][col] and self.board[0][col].player == 'A' and self.board[0][col].rank == 'rabbit' for col in range(8)):
            return 'A'
        if any(self.board[7][col] and self.board[7][col].player == 'B' and self.board[7][col].rank == 'rabbit' for col in range(8)):
            return 'B'
        return None

# Define the AI Agent using a neural network
class Agent:
    def __init__(self, input_size, output_size):
        self.model = MovePredictor(input_size, output_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()

    def act(self, state, valid_moves):
        # Flatten the board state and convert to tensor
        state_tensor = torch.tensor(state.flatten(), dtype=torch.float32)

        # Get move predictions from the model
        move_scores = self.model(state_tensor)

        # Find the indices of valid moves
        valid_move_indices = [valid_moves.index(move) for move in valid_moves]
        valid_move_scores = [move_scores[idx] for idx in valid_move_indices]

        # Select the best move (with the highest score)
        best_move_index = valid_move_scores.index(max(valid_move_scores))
        return valid_moves[best_move_index]

    def train(self, state, valid_moves, target_move, target_value):
        self.optimizer.zero_grad()

        # Flatten the state
        state_tensor = torch.tensor(state.flatten(), dtype=torch.float32)

        # Get model predictions
        move_scores = self.model(state_tensor)

        # Convert the target move to the corresponding index in the valid moves
        target_move_index = valid_moves.index(target_move)

        # Calculate loss
        target_tensor = torch.tensor([target_value], dtype=torch.float32)
        loss = self.criterion(move_scores[target_move_index], target_tensor)

        # Backpropagation
        loss.backward()
        self.optimizer.step()

# Board class and other game mechanics remain unchanged (same as before)

class Game:
    def __init__(self):
        self.board = Board()
        self.current_player = 'A'
        self.agent = Agent(input_size=64, output_size=64)  # Input size = 64 (8x8 board), output size = 64 (possible moves)

    def get_state(self):
        state = np.zeros((8, 8), dtype=int)
        for x in range(8):
            for y in range(8):
                piece = self.board.board[x][y]
                if piece:
                    state[x, y] = 1 if piece.player == 'A' else -1
        return state

    def play(self):
        for _ in range(100):  # Limit the number of moves per game
            self.board.display()  # Display the board after each move
            valid_moves = self.board.get_valid_moves(self.current_player)
            if not valid_moves:
                print(f"Player {self.current_player} has no valid moves. Game Over!")
                break

            if self.current_player == 'A':  # Human player's turn
                print("Player A's Turn (Human)")
                user_input = input("Enter your move (e.g., A7 to B6): ")
                try:
                    move = self.parse_move(user_input)
                except ValueError:
                    print("Invalid move format! Try again.")
                    continue
                if move in valid_moves:
                    self.board.move_piece(*move)
                else:
                    print("Invalid move! Please try again.")
                    continue
            else:  # AI's turn as Player B
                print("Player B's Turn (AI)")
                state = self.get_state()
                move = self.agent.act(state, valid_moves)
                self.board.move_piece(*move)

            self.current_player = 'B' if self.current_player == 'A' else 'A'
            winner = self.board.check_victory()
            if winner:
                print(f"Player {winner} wins!")
                break

    def parse_move(self, move_str):
        start, end = move_str.split(" to ")
        row_map = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7}
        start_col, start_row = start[0], int(start[1])  # Column as letter, Row as number
        end_col, end_row = end[0], int(end[1])

        start_x = row_map[start_col]
        end_x = row_map[end_col]

        start_y = 8 - start_row
        end_y = 8 - end_row

        return ((start_y, start_x), (end_y, end_x))

# Run the game
if __name__ == "__main__":
    game = Game()
    game.play()
