import queue
import numpy as np
from .cube import Cube
from .constants import MOVES
from typing import List, Tuple

# The improved heuristic
def heuristic(cube: Cube, state: np.ndarray) -> int:
    return np.sum(state != cube.goal_state)

class Node:
    def __init__(self, state: np.ndarray, g_cost: int = 0, h_cost: int = 0, parent=None, action=None):
        self.state = state
        self.g_cost = g_cost
        self.h_cost = h_cost
        self.parent = parent
        self.action = action
        self.child_nodes = {}  # Dictionary to store child nodes

    def __lt__(self, other):
        return (self.g_cost + self.h_cost) < (other.g_cost + other.h_cost)

def mcts(cube: Cube, budget: int = 20000) -> Tuple[List[int], int]:
    def ucb_value(node: Node) -> float:
        if node.parent is None or node.parent.g_cost == 0:
            return float('inf')
        exploitation = node.g_cost / node.parent.g_cost
        exploration = np.sqrt(np.log(node.parent.g_cost) / node.g_cost)
        return exploitation + exploration

    def select(node: Node) -> Node:
        while node.child_nodes:
            node = max(node.child_nodes.values(), key=ucb_value)
        return node

    def expand(node: Node) -> Node:
        unexplored_moves = [move for move in range(len(MOVES)) if move not in node.child_nodes]
        if unexplored_moves:
            random_move = np.random.choice(unexplored_moves)
            child_state = Cube.move_state(node.state, random_move)
            child_node = Node(
                state=child_state,
                g_cost=node.g_cost + 1,
                h_cost=heuristic(cube, child_state),
                parent=node,
                action=random_move
            )
            node.child_nodes[random_move] = child_node
            return child_node
        else:
            return node  # No unexplored moves, return the current node

    def playout(node: Node) -> float:
        # Limit the number of explored states in playout to 14
        max_explored_states = 14
        for _ in range(max_explored_states):
            random_move = np.random.choice(len(MOVES))
            child_state = Cube.move_state(node.state, random_move)
            node = Node(
                state=child_state,
                g_cost=node.g_cost + 1,
                h_cost=heuristic(cube, child_state),
                parent=node,
                action=random_move
            )
            if np.array_equal(node.state, cube.goal_state):
                return float('inf')  # Maximum reward for reaching the goal state
        return heuristic(cube, node.state)  # Return estimated cost if goal state not reached

    start_node = Node(state=cube.clone_state(), g_cost=0, h_cost=heuristic(cube, cube.clone_state()))
    for _ in range(budget):
        selected_node = select(start_node)
        expanded_node = expand(selected_node)
        reward = playout(expanded_node)

    # Reconstruct the path
    path = []
    while expanded_node.parent is not None:
        path.append(expanded_node.action)
        expanded_node = expanded_node.parent
    path.reverse()
    return path, budget, budget
