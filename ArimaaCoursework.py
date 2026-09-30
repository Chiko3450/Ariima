import random

class Piece:
    def __init__(self, player, rank):
        self.player = player  # 'A' for player A, 'B' for player B
        self.rank = rank      # Rank: 1-7 (1 being the strongest, 7 the weakest)

    def __str__(self):
        return f"{self.player}{self.rank}"

class Board:
    def __init__(self):
        self.board = [[None for _ in range(8)] for _ in range(8)]
        self.setup_board()

    def setup_board(self):
        # Set up player A's pieces
        self.board[0] = [Piece('A', i) for i in range(1, 9)]
        # Set up player B's pieces
        self.board[7] = [Piece('B', i) for i in range(1, 9)]

    def display(self):
        for row in self.board:
            print(" | ".join(str(piece) if piece else " . " for piece in row))
        print()

    def get_valid_moves(self, player):
        moves = []
        for x in range(8):
            for y in range(8):
                piece = self.board[x][y]
                if piece and piece.player == player:
                    # Generate possible moves (up, down, left, right)
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
        else:
            print("Invalid move")

    def is_valid_move(self, piece, start, end):
        x1, y1 = start
        x2, y2 = end
        if 0 <= x2 < 8 and 0 <= y2 < 8:  # Must stay within bounds
            return True  # For now, we allow any move
        return False

class Game:
    def __init__(self):
        self.board = Board()
        self.current_player = 'A'

    def switch_player(self):
        self.current_player = 'B' if self.current_player == 'A' else 'A'

    def player_move(self):
        while True:
            try:
                move = input("Enter your move (start_x start_y end_x end_y): ")
                start_x, start_y, end_x, end_y = map(int, move.split())
                if (start_x, start_y) in self.board.get_valid_moves('A'):
                    self.board.move_piece((start_x, start_y), (end_x, end_y))
                    break
                else:
                    print("Invalid move. Please try again.")
            except ValueError:
                print("Invalid input. Please enter four integers.")

    def ai_move(self):
        valid_moves = self.board.get_valid_moves('B')
        if valid_moves:
            move = random.choice(valid_moves)
            print(f"AI moves from {move[0]} to {move[1]}")
            self.board.move_piece(move[0], move[1])

    def play(self):
        while True:
            self.board.display()
            print(f"Player {self.current_player}'s turn.")
            if self.current_player == 'A':
                self.player_move()
            else:
                self.ai_move()
            self.switch_player()

if __name__ == "__main__":
    game = Game()
    game.play()
