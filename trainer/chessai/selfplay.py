"""AlphaZero-style self-play to generate RL training data.

Runs many games in parallel and batches all neural-net evaluations across games
each simulation step (one forward pass instead of one-per-node), which makes
MCTS self-play tractable on a single GPU.

Output records match datagen's JSONL schema so the same train.py can fine-tune
on them:
  {"fen": ..., "value": <game outcome from side-to-move>, "policy": [[idx,p],...]}
where the policy target is the normalized MCTS visit distribution.
"""

from __future__ import annotations

import json
import math
import os

import chess
import numpy as np
import torch

from .encoding import board_to_planes, legal_move_indices, move_to_index
from .mcts import Node


class BatchedSelfPlay:
    def __init__(self, model, device="cuda:0", c_puct=1.5, sims=100,
                 dirichlet_alpha=0.3, noise_frac=0.25):
        self.model = model
        self.device = device
        self.c_puct = c_puct
        self.sims = sims
        self.dir_alpha = dirichlet_alpha
        self.noise_frac = noise_frac

    @torch.no_grad()
    def _batch_eval(self, boards: list[chess.Board]):
        if not boards:
            return []
        planes = np.stack([board_to_planes(b) for b in boards])
        x = torch.from_numpy(planes).to(self.device)
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
            policy = {legal[k]: float(p) for k, p in zip(keys, ex)}
            out.append((policy, float(values[i])))
        return out

    def _add_noise(self, root: Node):
        if not root.children:
            return
        moves = list(root.children.keys())
        noise = np.random.dirichlet([self.dir_alpha] * len(moves))
        for m, nz in zip(moves, noise):
            c = root.children[m]
            c.prior = c.prior * (1 - self.noise_frac) + nz * self.noise_frac

    def _select_leaf(self, root: Node, board: chess.Board):
        """Descend from root to a leaf, pushing moves onto a copy of board."""
        node = root
        sim_board = board.copy()
        path = [node]
        while node.expanded():
            best_score = -1e9
            best_move = None
            best_child = None
            sqrt_total = math.sqrt(node.visits + 1)
            for move, child in node.children.items():
                u = self.c_puct * child.prior * sqrt_total / (1 + child.visits)
                score = -child.q + u
                if score > best_score:
                    best_score = score
                    best_move = move
                    best_child = child
            sim_board.push(best_move)
            node = best_child
            path.append(node)
        return path, sim_board

    @staticmethod
    def _backup(path, value):
        for node in reversed(path):
            node.visits += 1
            node.value_sum += value
            value = -value

    def _run_mcts_batch(self, roots, boards, active):
        """Run self.sims simulations for all active games, batching leaf evals."""
        for _ in range(self.sims):
            pending = []  # (game_idx, path, leaf_board)
            for gi in active:
                path, leaf_board = self._select_leaf(roots[gi], boards[gi])
                pending.append((gi, path, leaf_board))

            # terminal vs needs-eval
            eval_boards = []
            eval_refs = []
            for gi, path, leaf in pending:
                if leaf.is_game_over(claim_draw=True):
                    res = leaf.result(claim_draw=True)
                    outcome = 1.0 if res == "1-0" else -1.0 if res == "0-1" else 0.0
                    val = outcome if leaf.turn == chess.WHITE else -outcome
                    self._backup(path, val)
                else:
                    eval_boards.append(leaf)
                    eval_refs.append((path, leaf))

            results = self._batch_eval(eval_boards)
            for (path, leaf), (policy, value) in zip(eval_refs, results):
                node = path[-1]
                for move, prob in policy.items():
                    node.children[move] = Node(prior=prob, to_play=not leaf.turn,
                                               move=move)
                self._backup(path, value)

    def play(self, n_games=64, max_moves=200, temp_moves=20, resign_value=-0.92,
             out_path=None):
        """Play n_games in parallel; return list of JSONL-ready records."""
        boards = [chess.Board() for _ in range(n_games)]
        histories = [[] for _ in range(n_games)]  # (fen, visit_policy, turn)
        results = [None] * n_games
        move_no = 0

        while True:
            active = [i for i in range(n_games) if results[i] is None
                      and not boards[i].is_game_over(claim_draw=True)
                      and len(histories[i]) < max_moves]
            if not active:
                break

            # fresh root + priors for each active game
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

            self._run_mcts_batch(roots, boards, active)

            for gi in active:
                root = roots[gi]
                if not root.children:
                    results[gi] = boards[gi].result(claim_draw=True)
                    continue
                moves = list(root.children.keys())
                visits = np.array([root.children[m].visits for m in moves],
                                  dtype=np.float64)
                # record visit policy target
                board = boards[gi]
                policy_idx = [(move_to_index(m, board), v / visits.sum())
                              for m, v in zip(moves, visits)]
                histories[gi].append((board.fen(), policy_idx, board.turn))

                # choose move (temperature early, greedy later)
                if move_no < temp_moves:
                    p = visits / visits.sum()
                    choice = moves[int(np.random.choice(len(moves), p=p))]
                else:
                    choice = moves[int(visits.argmax())]
                board.push(choice)

                if board.is_game_over(claim_draw=True):
                    results[gi] = board.result(claim_draw=True)
            move_no += 1

        # finalize outcomes
        for gi in range(n_games):
            if results[gi] is None:
                results[gi] = boards[gi].result(claim_draw=True)

        records = []
        for gi in range(n_games):
            res = results[gi]
            outcome = 1.0 if res == "1-0" else -1.0 if res == "0-1" else 0.0
            for fen, policy_idx, turn in histories[gi]:
                v = outcome if turn == chess.WHITE else -outcome
                records.append({
                    "fen": fen,
                    "value": round(v, 5),
                    "policy": [[int(i), round(float(p), 5)] for i, p in policy_idx],
                })

        if out_path:
            os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
            with open(out_path, "a") as f:
                for r in records:
                    f.write(json.dumps(r) + "\n")
        return records, results
