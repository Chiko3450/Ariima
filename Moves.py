# Import necessary modules
import cgi
import cgitb
import random
from Arimaa_Improved_AI import Game  # Import your Game class where the DQN agent is defined

# Enable CGI error handling
cgitb.enable()

# Create an instance of the Game
game = Game()

def print_header():
    print("Content-Type: text/html\n")  # Required to specify the content type
    print("<html><body><h1>Arimaa Game</h1>")

def print_board():
    print("<h3>Game Board:</h3>")
    game.board.display()  # Call the display method to print the current state of the board

def print_valid_moves():
    valid_moves = game.board.get_valid_moves(game.current_player)
    print(f"<h3>Valid moves for {game.current_player}: </h3>")
    print("<ul>")
    for move in valid_moves:
        print(f"<li>{move}</li>")
    print("</ul>")

def play_turn():
    print("<h3>Player's Turn</h3>")
    # Let the DQN agent select a move based on the current state
    valid_moves = game.board.get_valid_moves(game.current_player)
    state = game.get_state()  # Get the current game state

    move = game.agent.act(state, valid_moves)  # DQN AI chooses a move
    print(f"<p>AI selected move: {move}</p>")
    # Perform the move and continue the game logic
    game.board.move_piece(move[0], move[1])
    game.add_move_to_history(game.current_player, move)

def handle_request():
    form = cgi.FieldStorage()

    # Handle user inputs
    action = form.getvalue("action")
    if action == "play_turn":
        play_turn()
    else:
        print_board()
        print_valid_moves()

    print("<hr><br><a href='/cgi-bin/game.py?action=play_turn'>Play AI's Turn</a>")
    print("<br><a href='/cgi-bin/game.py?action=show_board'>Show Board</a>")

def main():
    print_header()
    handle_request()
    print("</body></html>")

if __name__ == "__main__":
    main()
