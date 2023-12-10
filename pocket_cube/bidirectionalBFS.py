import queue
import numpy as np
from .cube import Cube
from .constants import MOVES
from typing import List, Tuple

# Define a heuristic function
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

    frontier_start = queue.PriorityQueue()
    frontier_goal = queue.PriorityQueue()
    frontier_start.put(start_node)
    frontier_goal.put(goal_node)

    explored_start = {tuple(start_node.state): start_node}
    explored_goal = {tuple(goal_node.state): goal_node}

    iteration = 0
    while not frontier_start.empty() and not frontier_goal.empty() and iteration < max_iterations:
        # Forward search
        current_node_start = frontier_start.get()

        for state, node in explored_goal.items():
            if np.array_equal(current_node_start.state, state):
                print("explored_start")
                print(explored_start)
                print("explored_goal")
                print(explored_goal)

                # Reconstruct the path
                path_start = reconstruct_path(current_node_start)
                path_goal = reconstruct_path(node)
                path_goal.reverse()
                return path_start + path_goal, iteration

        explored_start[tuple(current_node_start.state)] = current_node_start
        expand_and_enqueue(frontier_start, explored_start, current_node_start, cube)

        # Backward search
        current_node_goal = frontier_goal.get()

        for state, node in explored_start.items():
            if np.array_equal(current_node_goal.state, state):
                print("explored_start")
                print(explored_start)
                print("explored_goal")
                print(explored_goal)

                # Reconstruct the path
                path_start = reconstruct_path(node)
                path_goal = reconstruct_path(current_node_goal)
                path_goal.reverse()
                return path_start + path_goal, iteration

        explored_goal[tuple(current_node_goal.state)] = current_node_goal
        expand_and_enqueue(frontier_goal, explored_goal, current_node_goal, cube)

        iteration += 1

    return [], iteration

def expand_and_enqueue(frontier, explored, current_node, cube):
    for move in range(len(MOVES)):
        child_state = Cube.move_state(current_node.state, move)
        child_node = Node(
            state=child_state,
            g_cost=current_node.g_cost + 1,
            h_cost=heuristic(cube, child_state),
            parent=current_node,
            action=move
        )

        if tuple(child_state) not in explored:
            explored[tuple(child_state)] = child_node
            frontier.put(child_node)

def reconstruct_path(node):
    path = []
    while node.parent is not None:
        path.append(node.action)
        node = node.parent
    path.reverse()
    return path
