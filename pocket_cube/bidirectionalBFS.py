from typing import List, Tuple
from collections import deque
from .cube import Cube
from .astar import Node
from .constants import MOVES
from .moves import Move
import numpy as np

def bidirectional_bfs(cube: Cube) -> Tuple[List[int], int]:
    start_node = Node(state=cube.clone_state(), g_cost=0, h_cost=0)
    end_node = Node(state=cube.goal_state, g_cost=0, h_cost=0)

    start_queue = deque([start_node])
    end_queue = deque([end_node])

    start_explored = set()
    end_explored = set()

    while start_queue and end_queue:
        # Forward BFS
        current_node = start_queue.popleft()
        start_explored.add(tuple(current_node.state))

        if np.array_equal(current_node.state, end_node.state):
            # Reconstruct the path
            path = reconstruct_bidirectional_path(current_node, end_node)
            return path, len(start_explored) + len(end_explored)

        for move in range(len(MOVES)):
            child_state = Cube.move_state(current_node.state, move)

            if tuple(child_state) not in start_explored:
                child_node = Node(
                    state=child_state,
                    g_cost=current_node.g_cost + 1,
                    h_cost=0,  # Since we're using BFS, the heuristic is always 0
                    parent=current_node,
                    action=move
                )
                start_queue.append(child_node)

        # Backward BFS
        current_node = end_queue.popleft()
        end_explored.add(tuple(current_node.state))

        if tuple(current_node.state) in start_explored:
            # Reconstruct the path
            path = reconstruct_bidirectional_path(current_node, end_node)
            return path, len(start_explored) + len(end_explored)

        for move in range(len(MOVES)):
            child_state = Cube.move_state(current_node.state, move)

            if tuple(child_state) not in end_explored:
                child_node = Node(
                    state=child_state,
                    g_cost=current_node.g_cost + 1,
                    h_cost=0,
                    parent=current_node,
                    action=move
                )
                end_queue.append(child_node)

    return [], len(start_explored) + len(end_explored)

def reconstruct_bidirectional_path(node1: Node, node2: Node) -> List[int]:
    path1 = []
    while node1 is not None:
        if node1.action is not None:
            path1.append(node1.action)
        node1 = node1.parent

    path2 = []
    while node2 is not None:
        if node2.action is not None:
            path2.append(node2.action)
        node2 = node2.parent

    path2.reverse()
    return path1 + path2[1:]
