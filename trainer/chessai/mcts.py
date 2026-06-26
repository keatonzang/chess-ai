"""AlphaZero-style PUCT MCTS guided by the policy/value network."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import chess
import numpy as np
import torch

from .encoding import board_to_planes, legal_move_indices


@dataclass
class Node:
    prior: float
    to_play: bool
    move: chess.Move | None = None
    children: dict = field(default_factory=dict)
    visits: int = 0
    value_sum: float = 0.0

    @property
    def q(self) -> float:
        return self.value_sum / self.visits if self.visits else 0.0

    def expanded(self) -> bool:
        return len(self.children) > 0


class MCTS:
    def __init__(self, model, device="cuda:0", c_puct=1.5, batch_eval=True):
        self.model = model
        self.device = device
        self.c_puct = c_puct

    @torch.no_grad()
    def _evaluate(self, board: chess.Board):
        """Return (policy_dict {move: prob}, value) for the side to move."""
        planes = torch.from_numpy(board_to_planes(board))[None].to(self.device)
        with torch.amp.autocast(self.device.split(":")[0],
                                enabled=self.device.startswith("cuda")):
            logits, value = self.model(planes)
        logits = logits[0].float().cpu().numpy()
        value = float(value[0])

        legal = legal_move_indices(board)
        if not legal:
            return {}, value
        idxs = np.array(list(legal.keys()))
        masked = logits[idxs]
        masked -= masked.max()
        exp = np.exp(masked)
        exp /= exp.sum()
        policy = {legal[idx]: float(p) for idx, p in zip(legal.keys(), exp)}
        return policy, value

    def run(self, board: chess.Board, n_sims: int,
            add_noise: bool = False, dirichlet_alpha: float = 0.3,
            noise_frac: float = 0.25):
        root = Node(prior=0.0, to_play=board.turn)
        policy, _ = self._evaluate(board)
        self._expand(root, board, policy)

        if add_noise and root.children:
            moves = list(root.children.keys())
            noise = np.random.dirichlet([dirichlet_alpha] * len(moves))
            for m, nz in zip(moves, noise):
                c = root.children[m]
                c.prior = c.prior * (1 - noise_frac) + nz * noise_frac

        for _ in range(n_sims):
            node = root
            sim_board = board.copy()
            path = [node]
            # selection
            while node.expanded():
                move, node = self._select_child(node)
                sim_board.push(move)
                path.append(node)
            # expansion + evaluation
            value = self._evaluate_and_expand(node, sim_board)
            # backup
            self._backup(path, value)

        return root

    def _expand(self, node: Node, board: chess.Board, policy: dict):
        for move, prob in policy.items():
            node.children[move] = Node(prior=prob, to_play=not board.turn,
                                       move=move)

    def _evaluate_and_expand(self, node: Node, board: chess.Board) -> float:
        if board.is_game_over():
            result = board.result(claim_draw=True)
            if result == "1-0":
                outcome = 1.0
            elif result == "0-1":
                outcome = -1.0
            else:
                outcome = 0.0
            # value from perspective of side to move at this node
            return outcome if board.turn == chess.WHITE else -outcome
        policy, value = self._evaluate(board)
        self._expand(node, board, policy)
        return value

    def _select_child(self, node: Node):
        best_score = -float("inf")
        best_move = None
        best_child = None
        sqrt_total = math.sqrt(node.visits + 1)
        for move, child in node.children.items():
            u = self.c_puct * child.prior * sqrt_total / (1 + child.visits)
            # child.q is from the child's to_play perspective; negate for parent
            q = -child.q
            score = q + u
            if score > best_score:
                best_score = score
                best_move = move
                best_child = child
        return best_move, best_child

    def _backup(self, path, value: float):
        # value is from the perspective of the side to move at the leaf
        for node in reversed(path):
            node.visits += 1
            node.value_sum += value
            value = -value


def best_move(model, board: chess.Board, n_sims: int = 200, device="cuda:0",
              temperature: float = 0.0, c_puct: float = 1.5):
    """Pick a move via MCTS. temperature=0 -> most-visited (strongest)."""
    mcts = MCTS(model, device=device, c_puct=c_puct)
    root = mcts.run(board, n_sims)
    if not root.children:
        return None, {}
    moves = list(root.children.keys())
    visits = np.array([root.children[m].visits for m in moves], dtype=np.float64)
    if temperature <= 1e-6:
        choice = moves[int(visits.argmax())]
    else:
        probs = visits ** (1.0 / temperature)
        probs /= probs.sum()
        choice = moves[int(np.random.choice(len(moves), p=probs))]
    visit_dist = {m: int(v) for m, v in zip(moves, visits)}
    return choice, visit_dist
