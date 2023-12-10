import queue
import numpy as np
from .cube import Cube
from .constants import MOVES
from typing import List, Tuple

def heuristic(cube: Cube, state: np.ndarray) -> int:
    goal_state = cube.goal_state.reshape((6, 4))  # Reshape to a 2D array for easier indexing
    state = state.reshape((6, 4))

    total_distance = 0

    for face in range(6):
        for sticker in range(4):
            current_position = np.where(state == goal_state[face, sticker])
            goal_position = np.where(goal_state == goal_state[face, sticker])

            # Extract indices from tuples
            current_position = np.array(current_position).T[0]
            goal_position = np.array(goal_position).T[0]

            distance = np.sum(np.abs(current_position - goal_position))
            total_distance += distance

    return total_distance

class Node:
    def __init__(self, state: np.ndarray, g_cost: int = 0, h_cost: int = 0, parent=None, action=None):
        self.state = state
        self.g_cost = g_cost
        self.h_cost = h_cost
        self.parent = parent
        self.action = action

    def __lt__(self, other):
        return (self.g_cost + self.h_cost) < (other.g_cost + other.h_cost)
    
def bidirectional_bfs(cube: Cube, max_iterations: int = 9999999) -> Tuple[List[int], int]:
    start_node = Node(state=cube.clone_state(), g_cost=0, h_cost=heuristic(cube, cube.clone_state()))
    goal_node = Node(state=cube.goal_state, g_cost=0, h_cost=heuristic(cube, cube.goal_state))

    start_frontier = queue.Queue()
    goal_frontier = queue.Queue()
    start_frontier.put(start_node)
    goal_frontier.put(goal_node)

    start_explored = set()
    goal_explored = set()

    iteration = 0
    while not start_frontier.empty() and not goal_frontier.empty() and iteration < max_iterations:
        # Explore from the start state
        start_current_node = start_frontier.get()
        start_explored.add(start_current_node)

        for move in range(len(MOVES)):
            child_state = Cube.move_state(start_current_node.state, move)

            if tuple(child_state) not in start_explored:
                child_node = Node(
                    state=child_state,
                    g_cost=start_current_node.g_cost + 1,
                    h_cost=heuristic(cube, child_state),
                    parent=start_current_node,
                    action=move
                )
                start_frontier.put(child_node)

                # Check for intersection with the goal state
                if any(np.array_equal(node.state, child_node.state) for node in goal_explored):
                    goal_node = next(node for node in goal_explored if np.array_equal(node.state, child_node.state))
                    path_start = construct_path(child_node)
                    path_goal = construct_path(goal_node, reverse=True)
                    path_goal.pop()  # Remove the duplicate node in the middle
                    path_goal.reverse()
                    return path_start + path_goal, iteration

        # Explore from the goal state
        goal_current_node = goal_frontier.get()
        goal_explored.add(goal_current_node)

        for move in range(len(MOVES)):
            child_state = Cube.move_state(goal_current_node.state, move)

            if tuple(child_state) not in goal_explored:
                child_node = Node(
                    state=child_state,
                    g_cost=goal_current_node.g_cost + 1,
                    h_cost=heuristic(cube, child_state),
                    parent=goal_current_node,
                    action=move
                )
                goal_frontier.put(child_node)

        iteration += 1

    return [], iteration

def construct_path(node, reverse=False):
    path = []
    while node is not None:
        path.append(node.action)
        node = node.parent
    if reverse:
        path.reverse()
    return path