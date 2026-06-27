// PUCT MCTS in TypeScript — mirrors trainer/chessai/mcts.py.
// Uses chess.js for rules and an injected NN evaluator (ONNX in a worker).
import { Chess } from "chess.js";
import { boardToPlanes, moveToIndex } from "./encoding";

export type Evaluator = (
  planes: Float32Array
) => Promise<{ logits: Float32Array; value: number }>;

export interface Analysis {
  value: number; // [-1,1] from side-to-move POV
  whiteValue: number; // [-1,1] from White's POV (for the eval bar)
  winPct: number; // side-to-move win probability 0..100
  turn: "w" | "b";
  moves: { uci: string; san: string; prob: number }[]; // top policy moves
}

/** One network forward pass: the bot's own evaluation of a position. */
export async function analyzePosition(
  evaluator: Evaluator,
  fen: string,
  topN = 4
): Promise<Analysis> {
  const game = new Chess(fen);
  const { logits, value } = await evaluator(boardToPlanes(fen));
  const turn = game.turn() as "w" | "b";
  const legal = game.moves({ verbose: true }) as any[];
  let moves: Analysis["moves"] = [];
  if (legal.length) {
    const idxs = legal.map((m) =>
      moveToIndex(m.from, m.to, m.promotion ?? null, turn)
    );
    let max = -Infinity;
    for (const i of idxs) if (logits[i] > max) max = logits[i];
    let sum = 0;
    const exps = idxs.map((i) => {
      const e = Math.exp(logits[i] - max);
      sum += e;
      return e;
    });
    moves = legal
      .map((m, k) => ({ uci: m.from + m.to + (m.promotion ?? ""), san: m.san, prob: exps[k] / sum }))
      .sort((a, b) => b.prob - a.prob)
      .slice(0, topN);
  }
  const whiteValue = turn === "w" ? value : -value;
  const winPct = Math.round((value + 1) * 50);
  return { value, whiteValue, winPct, turn, moves };
}

class Node {
  prior: number;
  visits = 0;
  valueSum = 0;
  children = new Map<string, Node>(); // key: uci move
  constructor(prior: number) {
    this.prior = prior;
  }
  get q(): number {
    return this.visits ? this.valueSum / this.visits : 0;
  }
  get expanded(): boolean {
    return this.children.size > 0;
  }
}

export interface SearchOptions {
  sims: number;
  cPuct?: number;
  temperature?: number; // 0 = strongest (most visits)
}

function moveUci(m: { from: string; to: string; promotion?: string }): string {
  return m.from + m.to + (m.promotion ?? "");
}

export class MCTS {
  constructor(private evaluator: Evaluator, private cPuct = 1.5) {}

  private async evaluate(
    game: Chess
  ): Promise<{ policy: Map<string, number>; value: number }> {
    const planes = boardToPlanes(game.fen());
    const { logits, value } = await this.evaluator(planes);
    const turn = game.turn() as "w" | "b";
    const moves = game.moves({ verbose: true }) as any[];
    const policy = new Map<string, number>();
    if (moves.length === 0) return { policy, value };

    const idxs = moves.map((m) =>
      moveToIndex(m.from, m.to, m.promotion ?? null, turn)
    );
    let max = -Infinity;
    for (const i of idxs) if (logits[i] > max) max = logits[i];
    let sum = 0;
    const exps = idxs.map((i) => {
      const e = Math.exp(logits[i] - max);
      sum += e;
      return e;
    });
    moves.forEach((m, k) => policy.set(moveUci(m), exps[k] / sum));
    return { policy, value };
  }

  private expand(node: Node, policy: Map<string, number>) {
    for (const [uci, prob] of policy) node.children.set(uci, new Node(prob));
  }

  private selectChild(node: Node): [string, Node] {
    let best = -Infinity;
    let bestMove = "";
    let bestChild: Node | null = null;
    const sqrtTotal = Math.sqrt(node.visits + 1);
    for (const [uci, child] of node.children) {
      const u = (this.cPuct * child.prior * sqrtTotal) / (1 + child.visits);
      const q = -child.q; // child q is from child's perspective
      const score = q + u;
      if (score > best) {
        best = score;
        bestMove = uci;
        bestChild = child;
      }
    }
    return [bestMove, bestChild as Node];
  }

  private terminalValue(game: Chess): number {
    // value from perspective of side to move
    if (game.isCheckmate()) return -1; // side to move is mated
    return 0; // stalemate / draw
  }

  async search(
    rootFen: string,
    opts: SearchOptions
  ): Promise<{ bestMove: string | null; visits: Record<string, number> }> {
    const game = new Chess(rootFen);
    const root = new Node(0);
    const rootEval = await this.evaluate(game);
    this.expand(root, rootEval.policy);
    if (root.children.size === 0) return { bestMove: null, visits: {} };

    for (let s = 0; s < opts.sims; s++) {
      let node = root;
      const path: Node[] = [root];
      let depth = 0;
      while (node.expanded) {
        const [mv, child] = this.selectChild(node);
        game.move(mv);
        node = child;
        path.push(node);
        depth++;
      }
      let value: number;
      if (game.isGameOver()) {
        value = this.terminalValue(game);
      } else {
        const ev = await this.evaluate(game);
        this.expand(node, ev.policy);
        value = ev.value;
      }
      // backup
      for (let i = path.length - 1; i >= 0; i--) {
        path[i].visits += 1;
        path[i].valueSum += value;
        value = -value;
      }
      for (let i = 0; i < depth; i++) game.undo();
    }

    const visits: Record<string, number> = {};
    for (const [uci, child] of root.children) visits[uci] = child.visits;

    const temp = opts.temperature ?? 0;
    let bestMove: string;
    if (temp <= 1e-6) {
      bestMove = Object.entries(visits).sort((a, b) => b[1] - a[1])[0][0];
    } else {
      const entries = Object.entries(visits);
      const weights = entries.map(([, v]) => Math.pow(v, 1 / temp));
      const total = weights.reduce((a, b) => a + b, 0);
      let r = Math.random() * total;
      bestMove = entries[0][0];
      for (let i = 0; i < entries.length; i++) {
        r -= weights[i];
        if (r <= 0) {
          bestMove = entries[i][0];
          break;
        }
      }
    }
    return { bestMove, visits };
  }
}
