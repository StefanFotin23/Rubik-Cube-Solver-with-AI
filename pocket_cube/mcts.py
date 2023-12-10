import numpy as np
from .cube import Cube  # Importă clasa Cube din modulul tău
from .constants import MOVES  # Importă constanta MOVES din modulul tău
from typing import List
import random
import math

class MCTSNode:
    def __init__(self, state: np.ndarray, parent=None, action=None):
        self.state = state
        self.parent = parent
        self.action = action
        self.children = []
        self.visits = 0
        self.reward = 0

def ucb_score(node: MCTSNode, exploration_constant: float):
    if node.visits == 0:
        return float('inf')
    return (node.reward / node.visits) + exploration_constant * math.sqrt(math.log(node.parent.visits) / node.visits)

def select_node(node: MCTSNode, exploration_constant: float):
    while node.children:
        node = max(node.children, key=lambda x: ucb_score(x, exploration_constant))
    return node

def expand_node(node: MCTSNode):
    untried_moves = [move for move in range(len(MOVES)) if move not in [child.action for child in node.children]]
    if untried_moves:
        move = random.choice(untried_moves)
        child_state = Cube.move_state(node.state, move)
        child_node = MCTSNode(state=child_state, parent=node, action=move)
        node.children.append(child_node)
        return child_node
    else:
        return random.choice(node.children)

def simulate(node: MCTSNode, goal_state: np.ndarray):
    cube_copy = Cube()
    cube_copy.set_state(node.state)
    return cube_copy.solve()

def backpropagate(node: MCTSNode, reward: int):
    while node is not None:
        node.visits += 1
        node.reward += reward
        node = node.parent

def mcts(cube: Cube, exploration_constant: float = 1.0, budget: int = 1000) -> List[int]:
    root = MCTSNode(state=cube.clone_state())

    for _ in range(budget):
        selected_node = select_node(root, exploration_constant)
        expanded_node = expand_node(selected_node)
        reward = simulate(expanded_node, cube.goal_state)
        backpropagate(expanded_node, reward)

    best_child = max(root.children, key=lambda x: x.visits)
    solution_path = []
    while best_child.parent is not None:
        solution_path.append(best_child.action)
        best_child = best_child.parent
    solution_path.reverse()

    return solution_path
