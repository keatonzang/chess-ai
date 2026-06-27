"""Pure self-play reinforcement learning (AlphaZero-style, no Stockfish).

The network learns ONLY from the outcomes of its own games (win/draw/loss) and
from MCTS policy improvement. No external evaluation is used as a training
target — this is trial-and-error self-play from random weights.

Efficiency: MCTS uses push/undo on a single board per game (no per-simulation
board copies) and batches all neural-net leaf evaluations across the parallel
games into one forward pass per simulation step.
"""

from __future__ import annotations

import math

import chess
import numpy as np
import torch

from .encoding import board_to_planes, legal_move_indices, move_to_index
from .mcts import Node


def cp_to_winprob(cp: float) -> float:
    """Lichess-style centipawn -> win probability in [0, 1]."""
    return 1.0 / (1.0 + math.exp(-0.00368208 * cp))


class SelfPlay:
    def __init__(self, model, device="cuda:0", c_puct=1.5, sims=100,
                 dirichlet_alpha=0.3, noise_frac=0.25, max_moves=160,
                 temp_moves=24):
        self.model = model
        self.device = device
        self.c_puct = c_puct
        self.sims = sims
        self.dir_alpha = dirichlet_alpha
        self.noise_frac = noise_frac
        self.max_moves = max_moves
        self.temp_moves = temp_moves

    @torch.no_grad()
    def _batch_eval(self, boards):
        if not boards:
            return []
        planes = np.empty((len(boards), 18, 8, 8), dtype=np.float32)
        for i, b in enumerate(boards):
            planes[i] = board_to_planes(b)
        x = torch.from_numpy(planes).to(self.device, non_blocking=True)
        with torch.amp.autocast(self.device.split(":")[0],
                                enabled=self.device.startswith("cuda")):
            logits, values = self.model(x)
        logits = logits.float().cpu().numpy()
        values = values.float().cpu().numpy()
        out = []
        for i, b in enumerate(boards):
            legal = legal_move_indices(b)
            if not legal:
                out.append(({}, float(values[i])))
                continue
            keys = list(legal.keys())
            row = logits[i][keys]
            row -= row.max()
            ex = np.exp(row)
            ex /= ex.sum()
            out.append(({legal[k]: float(p) for k, p in zip(keys, ex)},
                        float(values[i])))
        return out

    def _select_child(self, node: Node):
        best = -1e30
        best_move = None
        best_child = None
        sqrt_total = math.sqrt(node.visits + 1)
        cp = self.c_puct
        for move, child in node.children.items():
            u = cp * child.prior * sqrt_total / (1 + child.visits)
            score = -child.q + u
            if score > best:
                best = score
                best_move = move
                best_child = child
        return best_move, best_child

    @staticmethod
    def _backup(path, value):
        for node in reversed(path):
            node.visits += 1
            node.value_sum += value
            value = -value

    def _add_noise(self, root: Node):
        if not root.children:
            return
        moves = list(root.children.keys())
        noise = np.random.dirichlet([self.dir_alpha] * len(moves))
        for m, nz in zip(moves, noise):
            c = root.children[m]
            c.prior = c.prior * (1 - self.noise_frac) + nz * self.noise_frac

    def _run_sims(self, roots, boards, active):
        for _ in range(self.sims):
            eval_pairs = []   # (gi, path)
            eval_boards = []
            npushed = {}
            for gi in active:
                node = roots[gi]
                b = boards[gi]
                path = [node]
                pushed = 0
                while node.expanded():
                    move, child = self._select_child(node)
                    b.push(move)
                    node = child
                    path.append(node)
                    pushed += 1
                npushed[gi] = pushed
                if b.is_game_over(claim_draw=False):
                    res = b.result(claim_draw=False)
                    outcome = 1.0 if res == "1-0" else -1.0 if res == "0-1" else 0.0
                    val = outcome if b.turn == chess.WHITE else -outcome
                    self._backup(path, val)
                else:
                    eval_pairs.append((gi, path))
                    eval_boards.append(b)

            results = self._batch_eval(eval_boards)
            for (gi, path), (policy, value) in zip(eval_pairs, results):
                leaf = path[-1]
                turn = boards[gi].turn
                for move, prob in policy.items():
                    leaf.children[move] = Node(prior=prob, to_play=not turn,
                                               move=move)
                self._backup(path, value)

            # restore every board to its root position
            for gi in active:
                for _ in range(npushed[gi]):
                    boards[gi].pop()

    def search_one(self, board, add_noise=True, temperature=0.0):
        """Single-position MCTS (push/undo). Returns (chosen_move, policy_idx).

        policy_idx is the MCTS visit distribution as [(move_index, prob), ...]
        in ``board``'s own frame. Used by the bot-vs-Stockfish actor.
        """
        b = board.copy()
        policy, _ = self._batch_eval([b])[0]
        if not policy:
            return None, []
        root = Node(prior=0.0, to_play=b.turn)
        for move, prob in policy.items():
            root.children[move] = Node(prior=prob, to_play=not b.turn, move=move)
        if add_noise:
            self._add_noise(root)

        for _ in range(self.sims):
            node = root
            path = [node]
            pushed = 0
            while node.expanded():
                move, child = self._select_child(node)
                b.push(move)
                node = child
                path.append(node)
                pushed += 1
            if b.is_game_over(claim_draw=False):
                res = b.result(claim_draw=False)
                outcome = 1.0 if res == "1-0" else -1.0 if res == "0-1" else 0.0
                val = outcome if b.turn == chess.WHITE else -outcome
                self._backup(path, val)
            else:
                pol, v = self._batch_eval([b])[0]
                leaf = path[-1]
                for move, prob in pol.items():
                    leaf.children[move] = Node(prior=prob, to_play=not b.turn,
                                               move=move)
                self._backup(path, v)
            for _ in range(pushed):
                b.pop()

        moves = list(root.children.keys())
        visits = np.array([root.children[m].visits for m in moves],
                          dtype=np.float64)
        total = visits.sum()
        policy_idx = [(move_to_index(m, board), v / total)
                      for m, v in zip(moves, visits)]
        if temperature <= 1e-6:
            choice = moves[int(visits.argmax())]
        else:
            p = visits ** (1.0 / temperature)
            p /= p.sum()
            choice = moves[int(np.random.choice(len(moves), p=p))]
        return choice, policy_idx

    def play(self, n_games=64):
        """Play n_games in parallel; return a list of (fen, policy, value) samples."""
        boards = [chess.Board() for _ in range(n_games)]
        histories = [[] for _ in range(n_games)]   # (fen, policy_idx, turn)
        results = [None] * n_games
        move_no = 0

        while True:
            active = [i for i in range(n_games)
                      if results[i] is None
                      and not boards[i].is_game_over(claim_draw=True)
                      and len(histories[i]) < self.max_moves]
            if not active:
                break

            init = self._batch_eval([boards[i] for i in active])
            roots = {}
            for (policy, _), gi in zip(init, active):
                root = Node(prior=0.0, to_play=boards[gi].turn)
                for move, prob in policy.items():
                    root.children[move] = Node(prior=prob,
                                               to_play=not boards[gi].turn,
                                               move=move)
                self._add_noise(root)
                roots[gi] = root

            self._run_sims(roots, boards, active)

            for gi in active:
                root = roots[gi]
                if not root.children:
                    results[gi] = boards[gi].result(claim_draw=True)
                    continue
                moves = list(root.children.keys())
                visits = np.array([root.children[m].visits for m in moves],
                                  dtype=np.float64)
                board = boards[gi]
                policy_idx = [(move_to_index(m, board), v / visits.sum())
                              for m, v in zip(moves, visits)]
                histories[gi].append((board.fen(), policy_idx, board.turn))

                if move_no < self.temp_moves:
                    p = visits / visits.sum()
                    choice = moves[int(np.random.choice(len(moves), p=p))]
                else:
                    choice = moves[int(visits.argmax())]
                board.push(choice)
                if board.is_game_over(claim_draw=True):
                    results[gi] = board.result(claim_draw=True)
            move_no += 1

        for gi in range(n_games):
            if results[gi] is None:
                results[gi] = boards[gi].result(claim_draw=True)

        samples = []
        for gi in range(n_games):
            res = results[gi]
            outcome = 1.0 if res == "1-0" else -1.0 if res == "0-1" else 0.0
            for fen, policy_idx, turn in histories[gi]:
                v = outcome if turn == chess.WHITE else -outcome
                samples.append({
                    "fen": fen,
                    "value": round(v, 5),
                    "policy": [[int(i), round(float(p), 5)] for i, p in policy_idx],
                })
        return samples, results
