import queue
import numpy as np
from .cube import Cube
from .constants import MOVES
from typing import List, Tuple

# Define a heuristic function (for simplicity, we use the number of misplaced stickers)
def heuristic(cube: Cube, state: np.ndarray) -> int:
    return np.sum(state != cube.goal_state)

class Node:
    def __init__(self, state: np.ndarray, g_cost: int = 0, h_cost: int = 0, parent=None, action=None):
        self.state = state
        self.g_cost = g_cost
        self.h_cost = h_cost
        self.parent = parent
        self.action = action

    def __lt__(self, other):
        return (self.g_cost + self.h_cost) < (other.g_cost + other.h_cost)

def astar(cube: Cube, max_iterations: int = 9999999) -> Tuple[List[int], int]:
    start_node = Node(state=cube.clone_state(), g_cost=0, h_cost=heuristic(cube, cube.clone_state()))
    frontier = queue.PriorityQueue()
    frontier.put(start_node)
    explored = set()

    iteration = 0
    while not frontier.empty() and iteration < max_iterations:
        current_node = frontier.get()

        if np.array_equal(current_node.state, cube.goal_state):
            # Reconstruct the path
            path = []
            while current_node.parent is not None:
                path.append(current_node.action)
                current_node = current_node.parent
            path.reverse()
            return path, iteration

        explored.add(tuple(current_node.state))

        for move in range(len(MOVES)):
            child_state = Cube.move_state(current_node.state, move)

            if tuple(child_state) not in explored:
                child_node = Node(
                    state=child_state,
                    g_cost=current_node.g_cost + 1,
                    h_cost=heuristic(cube, child_state),
                    parent=current_node,
                    action=move
                )
                frontier.put(child_node)

        iteration += 1

    return [], iteration
