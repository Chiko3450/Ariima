import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd
import time
import gym
import os
from collections import deque


#env = gym.make('Arimaa')
# Piece class to represent each piece on the board
class Piece:
    def __init__(self, player, rank):
        self.player = player
        self.rank = rank
        self.frozen = False

    def __str__(self):
        return f"{self.player}{self.rank}"

# Board class to manage the game board
class Board:
    def __init__(self):
        self.board = [[None for _ in range(8)] for _ in range(8)]  # Empty board
        self.trap_squares = [(2, 2), (2, 5), (5, 2), (5, 5)]  # c3, f3, c6, f6
        self.setup_board()

    def setup_board(self):
        # Prompt Player A and AI Player B to place their pieces
        self.place_pieces('A', is_ai=True)
        self.place_pieces('B', is_ai=True)

    def place_pieces(self, player, is_ai=False, model=None):
        print(f"\n{player}'s turn to place pieces!")
        pieces = ['Elephant', 'Camel', 'Horse', 'Dog', 'Dog', 'Cat', 'Cat']
        piece_symbols = {
            'Elephant': 'E',
            'Camel': 'M',
            'Horse': 'H',
            'Dog': 'D',
            'Cat': 'C'
        }
        allowed_rows = [6, 7] if player == 'A' else [0, 1]  # A -> rows 6,7 | B -> rows 0,1

        if is_ai:
            if model and player == 'B':  # Use saved model for Player B's piece placement
                print(f"\n{player} (AI) is using the starting positions from the saved model!")
                for row in range(8):
                    for col in range(8):
                        piece = model[row][col]  # Assuming model stores player piece info
                        if piece:
                            self.board[row][col] = Piece(player, piece_symbols[piece])

            else:
                print(f"\n{player} (AI) is placing its pieces automatically!")
                available_positions = [(r, c) for r in allowed_rows for c in range(8)]
                random.shuffle(available_positions)  # Shuffle to avoid conflicts

                for piece in pieces:
                    if available_positions:
                        row, col = available_positions.pop()
                        self.board[row][col] = Piece(player, piece[0])  # Place the piece

        else:  # Human Player
            for piece in pieces:
                placed = False
                while not placed:
                    self.display()
                    print(f"\nWhere do you want to place your {piece} (row 1-2, col A-H)?")
                    
                    try:
                        row, col = input(f"Enter row (1-2) and column (A-H) for {piece}: ").split()
                        row = int(row)
                        col = ord(col.upper()) - ord('A')  
                        row = 8 - row  

                        if row in allowed_rows and 0 <= col < 8 and self.board[row][col] is None:
                            self.board[row][col] = Piece(player, piece[0])  # Place the piece
                            placed = True
                        else:
                            print("Invalid position! Pick a valid spot in your allowed area.")
                    except (ValueError, IndexError):
                        print("Invalid input! Please enter a valid row and column.")

        # Only place rabbits **after other pieces**
        for row in allowed_rows:
            for col in range(8):
                if self.board[row][col] is None:
                    self.board[row][col] = Piece(player, 'r')  # Place Rabbit

    def display(self):
        for x, row in enumerate(self.board):
            line = f"{8 - x} "
            for y, piece in enumerate(row):
                if (x, y) in self.trap_squares and piece is None:
                    line += " x "  
                else:
                    line += f"{str(piece) if piece else ' . '}"
                line += " | "
            print(line.rstrip(" | "))
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
        start_x, start_y = start
        end_x, end_y = end

        piece = self.board[start_x][start_y]
        target_piece = self.board[end_x][end_y]

        if not piece:
            return False, 0  # No piece at the start position

        # Check if the target position is occupied by an opponent's piece
        if target_piece and target_piece.player != piece.player:
            # Check if the current piece can remove the opponent's piece (i.e., it's stronger)
            if self.can_remove(piece, target_piece):
                # Remove the opponent's piece from the board
                self.board[end_x][end_y] = None  # Remove opponent's piece

                # Move the current piece to the target position
                self.board[end_x][end_y] = piece
                self.board[start_x][start_y] = None

                return True, 1  # Successful move and reward for capturing

            else:
                return False, 0  # Can't move over a stronger opponent's piece

        # If the target position is empty or occupied by a friendly piece, perform a normal move
        self.board[end_x][end_y] = piece
        self.board[start_x][start_y] = None
        return True, 1  # Successful move


    def can_remove(self, piece, target_piece):
        # Compare the ranks of the two pieces
        return target_piece and target_piece.player != piece.player and self.rank_value(target_piece.rank) < self.rank_value(piece.rank)


    def rank_value(self, rank):
        # Return a value representing the rank strength (higher value = stronger piece)
        rank_order = {'R': 1, 'C': 2, 'H': 3, 'D': 4, 'M': 5, 'E': 6}
        return rank_order.get(rank, 0)  # Default to 0 if rank is not found


    def is_valid_move(self, piece, start, end):
        x1, y1 = start
        x2, y2 = end
        if not (0 <= x2 < 8 and 0 <= y2 < 8):  # Bounds check
            return False
        if self.board[x2][y2] is not None:  # Can't move to occupied square
            return False
        if piece.rank == 'r' and x2 < x1:  # Rabbits can't move backward
        # Rabbits can't move backward (y-coordinate check)
            return False
        return True

    def check_trap_squares(self):
        for x, y in self.trap_squares:
            piece = self.board[x][y]
            if piece:
                neighbors = [(x + dx, y + dy) for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]]
                has_ally = any(
                    0 <= nx < 8 and 0 <= ny < 8 and self.board[nx][ny] and self.board[nx][ny].player == piece.player
                    for nx, ny in neighbors
                )
                if not has_ally:  # Piece is removed if no ally is adjacent
                    self.board[x][y] = None


    def can_push_or_pull(self, piece, target, direction):
        x, y = target
        dx, dy = direction
        nx, ny = x + dx, y + dy

        # Check bounds and if the target square is empty
        if not (0 <= nx < 8 and 0 <= ny < 8) or self.board[nx][ny] is not None:
            return False

        target_piece = self.board[x][y]
        return target_piece and target_piece.player != piece.player and target_piece.rank > piece.rank

    def freeze_pieces(self):
        for x in range(8):
            for y in range(8):
                piece = self.board[x][y]
                if piece and not piece.frozen:
                    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < 8 and 0 <= ny < 8:
                            neighbor = self.board[nx][ny]
                            if neighbor and neighbor.player != piece.player and self.rank_value(neighbor.rank) > self.rank_value(piece.rank):
                                piece.frozen = True
                                break

    def check_victory(self):
        # Check if Player A rabbit reaches Player B's back row
        if any(self.board[0][col] and self.board[0][col].player == 'A' and self.board[0][col].rank == 'r' for col in range(8)):
            return 'A'
        # Check if Player B rabbit reaches Player A's back row
        if any(self.board[7][col] and self.board[7][col].player == 'B' and self.board[7][col].rank == 'r' for col in range(8)):
            return 'B'
        # Check if all rabbits of one player are eliminated
        a_rabbits = sum(1 for row in self.board for piece in row if piece and piece.player == 'A' and piece.rank == 'r')
        b_rabbits = sum(1 for row in self.board for piece in row if piece and piece.player == 'B' and piece.rank == 'r')
        if a_rabbits == 0:
            return 'B'
        if b_rabbits == 0:
            return 'A'
        return None
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95  # Discount factor
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001

        self.model = self.build_model()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()

    def build_model(self):
        model = nn.Sequential(
            nn.Linear(self.state_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, self.action_size)
        )
        return model

    def act(self, state, valid_moves):
        if np.random.rand() <= self.epsilon:  # Explore
            return random.choice(valid_moves)

        state = torch.tensor(state.flatten(), dtype=torch.float32)
        q_values = self.model(state)
        best_move_index = torch.argmax(q_values[:len(valid_moves)]).item()
        return valid_moves[best_move_index]
  # Choose best move from valid moves

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train(self, batch_size=32):
        if len(self.memory) < batch_size:
            return

        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                next_state_tensor = torch.tensor(next_state.flatten(), dtype=torch.float32)
                target = reward + self.gamma * torch.max(self.model(next_state_tensor)).item()

            state_tensor = torch.tensor(state.flatten(), dtype=torch.float32)
            q_values = self.model(state_tensor)
            q_values[action] = target

            self.optimizer.zero_grad()
            loss = self.criterion(q_values, torch.tensor(q_values.detach(), dtype=torch.float32))
            loss.backward()
            self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay



# Game class with updated rules
class Game:
    def __init__(self, model_path='arimaa_ai_dqn.pth'):
        self.board = Board()
        self.current_player = 'A'
        
        # Initialize the agent
        self.agent = DQNAgent(state_size=64, action_size=64)  # 8x8 board flattened
        
        # Load saved model if it exists
        if os.path.exists(model_path):
            print("Loading saved model...")
            self.agent.model.load_state_dict(torch.load(model_path))
            self.agent.model.eval()  # Set the model to evaluation mode
        else:
            print("No saved model found. Training from scratch.")
        
        self.total_rewards = 0
        self.min_turns = float('inf')  # Initialize to infinity for comparison

    def get_state(self):
        state = np.zeros((8, 8), dtype=int)
        for x in range(8):
            for y in range(8):
                piece = self.board.board[x][y]
                if piece:
                    state[x, y] = 1 if piece.player == 'A' else -1
        return state

    def play(self, episodes=1000, batch_size=32, delay=1):
        for episode in range(episodes):
            if episode == 0:  # Only create the board once
                self.board = Board()  # Reset board
            self.current_player = 'A'  # Player A starts
            turn_counter = 0  # Initialize the turn counter

            while turn_counter < 1000:  # Limit the number of turns per game
                self.board.display()  # Display the board after each move
                valid_moves = self.board.get_valid_moves(self.current_player)
                if not valid_moves:  # Game ends if no moves are available
                    print(f"Player {self.current_player} has no valid moves. Game Over!")
                    break

                if self.current_player == 'A':  # Player A's turn (AI)
                    print("Player A's Turn (AI)")
                    
                    # Get the game state representation
                    state = self.get_state()

                    # AI logic to make a move
                    move = self.agent.act(state, valid_moves)

                    # Execute the move
                    self.board.move_piece(*move)
                    #time.sleep(1)

                else:  # Player B's turn (AI)
                    print("Player B's Turn (AI)")

                    # Get the game state representation
                    state = self.get_state()

                    # AI logic to make a move
                    move = self.agent.act(state, valid_moves)

                    # Execute the move
                    self.board.move_piece(*move)

                # Update turn and check for victory
                self.current_player = 'B' if self.current_player == 'A' else 'A'
                winner = self.board.check_victory()

                if winner:
                    print(f"Player {winner} wins!")
                    if winner == 'A':  # AI wins
                        reward = -5  # Positive reward for AI victory
                        self.total_rewards += reward  # Accumulate reward
                    else:  # AI loses
                        reward = 10  # Negative reward for AI loss
                        self.total_rewards += reward  # Accumulate reward

                    # Update the AI's memory with the reward
                    self.agent.remember(state, move, reward, self.get_state(), done=True)

                    # Optionally, you can train the agent after each game (depends on your training setup)
                    self.agent.train(batch_size=batch_size)
                    
                    # Display the AI's total reward after the game ends
                    print(f"AI Total Reward after this game: {self.total_rewards}")

                    # Save model after AI win, if this game took fewer turns
                    if winner == 'A' and turn_counter < self.min_turns:
                        print(f"Saving model after AI victory (took {turn_counter} turns, new best)!")
                        torch.save(self.agent.model.state_dict(), 'arimaa_ai_dqn.pth')
                        self.min_turns = turn_counter  # Update the minimum turns

                    return

                # Increment the turn counter
                turn_counter += 1

            # If turn_counter reaches 100, it's a draw
            if turn_counter == 1000:
                print("Game ended in a draw!")
                reward = -5  # Assign a -5 reward for the AI in case of a draw
                self.total_rewards += reward  # Accumulate the reward

                # Update the AI's memory with the reward
                self.agent.remember(state, move, reward, self.get_state(), done=True)

                # Optionally, you can train the agent after each game
                self.agent.train(batch_size=batch_size)
                
                # Display the AI's total reward
                print(f"AI Total Reward after this game: {self.total_rewards}")
                return





    def parse_move(self, move_str):
        start, end = move_str.split(" to ")
        row_map = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7}
        
        # Parse the starting and ending squares
        start_col, start_row = start[0], int(start[1])  # Column as letter, Row as number
        end_col, end_row = end[0], int(end[1])
        
        # Convert column (A-H) to 0-7
        start_x = row_map[start_col]
        end_x = row_map[end_col]
        
        # Convert rows from 1-8 to 7-0 (reverse the row order)
        start_y = 8 - start_row
        end_y = 8 - end_row
        
        print(f"Parsed Move: ({start_x}, {start_y}) to ({end_x}, {end_y})")  # Debug print
        
        return ((start_y, start_x), (end_y, end_x))  # Correct the order of coordinates

if __name__ == "__main__":
    game = Game()
    game.play()  # Start the game