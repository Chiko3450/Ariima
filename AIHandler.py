import numpy as np
import random
import math
import torch
import torch.nn as nn
import torch.optim as optim

from collections import deque

class MCTSNode:
    def __init__(self, state, player, parent=None):  # ✅ Set parent=None
        self.state = state  # Board state
        self.player = player
        self.parent = parent  # ✅ Store parent node (fix NameError)
        self.untried_moves = state.get_valid_moves(player)  # List of moves
        self.children = []
        self.visits = 0
        self.wins = 0
        self.value = 0

    def is_fully_expanded(self):
        return len(self.untried_moves) == 0

    def best_child(self, exploration_weight=1.41):
        return max(self.children, key=lambda child: child.value / (child.visits + 1) + exploration_weight * math.sqrt(math.log(self.visits + 1) / (child.visits + 1)))

    def select_child(self):
        return self.best_child()

    def expand(self):
        move = self.untried_moves.pop()
        new_state = self.state.clone()  # Ensure `clone()` is implemented
        new_state.move_piece(*move)
        
        child_node = MCTSNode(new_state, self.player, parent=self)  # ✅ Set parent=self
        self.children.append(child_node)
        return child_node

    def update(self, reward):
        self.visits += 1
        self.wins += reward  # Keep track of wins instead of undefined `value`

# Monte Carlo Tree Search Algorithm
class MCTS:
    def __init__(self, board, player, simulations=100):
        self.root = MCTSNode(board, player)
        self.simulations = simulations

    def search(self):
        for _ in range(self.simulations):
            node = self.root
            while node.is_fully_expanded() and node.children:
                node = node.select_child()

            if not node.is_fully_expanded():
                node = node.expand()

            # Ensure there are children after expanding, or handle no valid moves
            if node.children:
                result = self.rollout(node.state)
                self.backpropagate(node, result)
            else:
                print(f"Warning: No valid children found for node with state: {node.state}")
                return None  # If no valid children, return None or handle it as a draw/loss

        # Ensure there are children before trying to select the best
        if self.root.children:
            return self.root.best_child(exploration_weight=0).state.last_move
        else:
            print("No valid moves found. Returning None.")
            return None  # If no valid move, return None


    def best_move(self):
        """ Returns the best move found by MCTS """
        return self.search()  # Calls `search()` to get the best move

    def rollout(self, board):
        for _ in range(50):  # Limit random moves to prevent infinite loops
            moves = board.get_valid_moves(board.current_player)
            if not moves:
                return -1  # Lose condition
            move = random.choice(moves)
            board.move_piece(*move)
            if board.check_victory():
                return 1
        return 0  # Draw

    def backpropagate(self, node, result):
        while node:
            node.update(result)
            node = node.parent
