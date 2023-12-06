from __future__ import annotations

from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib import animation

from .constants import MOVES, CORNERS, COLORS, LETTERS
from .moves import Move, MoveInput, MoveSequence

import matplotlib.pyplot as plt
import numpy as np


class Cube:

    def __init__(self, moves: Moves | None = None, scrambled: bool = True):
        self.goal_state = np.repeat(np.arange(6), 4)
        self.state = np.repeat(np.arange(6), 4)

        if moves or scrambled:
            self.scramble(moves)

    def scramble(self, moves: Moves | None = None):

        if moves is None:
            num_of_moves = np.random.randint(5, 11)
            moves = list(np.random.randint(len(MOVES), size=num_of_moves))

        self.state = Cube.move_state(self.state, moves)

    def move(self, move: Moves) -> Cube:
        cube = Cube()
        cube.state = Cube.move_state(self.clone_state(), move)
        return cube

    @staticmethod
    def move_state(state: np.ndarray, move: Moves) -> np.ndarray:
        move = Move.parse(move)

        if isinstance(move, list):
            for m in move:
                state = state[MOVES[m.value]]
        else:
            state = state[MOVES[move.value]]

        return state

    def clone_state(self) -> np.ndarray:
        return np.copy(self.state)

    def clone(self) -> Cube:
        cube = Cube()
        cube.state = self.clone_state()
        return cube

    def hash(self) -> str:
        return Cube.hash_state(self.state)

    @staticmethod
    def hash_state(state: np.ndarray) -> str:
        return ''.join(map(str, state))

    @staticmethod
    def _draw_corner(ax, position, colors):

        vertices = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
                         [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]]) + position

        indices = [(0, 1, 2, 3), (4, 5, 6, 7), (0, 1, 5, 4),
                   (2, 3, 7, 6), (0, 3, 7, 4), (1, 2, 6, 5)]

        faces = [[vertices[idx] for idx in face] for face in indices]

        ax.add_collection3d(Poly3DCollection(faces, facecolors=colors, linewidths=1, edgecolors='black'))

    @staticmethod
    def _draw_cube(state: np.ndarray, ax):

        for corner, (state_idxs, color_idxs) in CORNERS.items():
            colors = ["gray"] * 6

            for sticker_idx, color_idx in zip(state_idxs, color_idxs):
                colors[color_idx] = COLORS[state[sticker_idx]]

            Cube._draw_corner(ax, corner, colors)

    @staticmethod
    def render_state(state):
        fig, ax = plt.subplots(figsize=(7, 5))
        base_coords = np.array([(0, 1), (1, 1), (0, 0), (1, 0)])
        offsets = np.array([[0, 0], [1, 0], [2, 0], [-1, 0], [0, 1], [0, -1]]) * 2

        idx = 0

        for offset in offsets:
            for coords in base_coords:
                rect = plt.Rectangle(coords + offset, 1, 1, edgecolor='black', linewidth=1)
                rect.set_facecolor(COLORS[state[idx]])
                ax.add_patch(rect)

                idx += 1

        ax.set_xlim(-2.1, 6.1)
        ax.set_ylim(-2.1, 4.1)
        ax.axis('off')
        plt.show()

    def render(self):
        Cube.render_state(self.state)

    def render3D(self):

        fig = plt.figure(figsize=(4, 4))
        ax = fig.add_subplot(111, projection='3d')

        Cube._draw_cube(self.state, ax)

        ax.axis('off')
        ax.set_xlim([0, 2])
        ax.set_ylim([0, 2])
        ax.set_zlim([0, 2])
        plt.show()

    @staticmethod
    def render3D_moves(initial_state: np.ndarray, moves: MoveSequence, save: bool = False):
        moves = Move.parse(moves)

        original_state = np.copy(initial_state)
        state = initial_state

        fig = plt.figure(figsize=(4, 4), frameon=False)
        ax = fig.add_subplot(111, projection='3d')

        Cube._draw_cube(state, ax)

        ax.axis('off')
        ax.set_xlim([0, 2])
        ax.set_ylim([0, 2])
        ax.set_zlim([0, 2])

        move_index = 0

        def init():
            Cube._draw_cube(state, ax)
            return ax

        def animate(i):
            nonlocal move_index

            if i == 0:  # For the initial frame, show the original state
                state[:] = np.copy(original_state)
                Cube._draw_cube(state, ax)

            else:
                if move_index < len(moves):  # Check if there are more moves to perform
                    state[:] = Cube.move_state(state, moves[move_index])
                    ax.clear()

                    Cube._draw_cube(state, ax)
                    move_index += 1
                    ax.axis('off')
                    ax.set_xlim([0, 2])
                    ax.set_ylim([0, 2])
                    ax.set_zlim([0, 2])
                else:

                    move_index = 0
                    state[:] = np.copy(original_state)
                    Cube._draw_cube(state, ax)

        ani = animation.FuncAnimation(fig, animate, frames=len(moves) + 2, init_func=init,
                                      interval=1000, blit=False)

        if save:
            ani.save('rubiks_cube_animation.gif', writer='pillow', fps=1)

        plt.show()
        return ani

    def render_text(self):
        lines = [
            [None, None, 16, 17],
            [None, None, 18, 19],
            [12, 13, 0, 1, 4, 5, 8, 9],
            [14, 15, 2, 3, 6, 7, 10, 11],
            [None, None, 20, 21],
            [None, None, 22, 23]
        ]

        for line in lines:
            print("".join(LETTERS[self.state[idx]] if idx is not None else " " for idx in line))