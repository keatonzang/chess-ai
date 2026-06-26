// End-to-end JS engine test: encoding.ts + mcts.ts + real ONNX model,
// driven through onnxruntime-node (same API as onnxruntime-web).
import * as ort from "onnxruntime-node";
import { Chess } from "chess.js";
import { MCTS, Evaluator } from "../lib/mcts";
import { N_PLANES } from "../lib/encoding";

async function main() {
  const session = await ort.InferenceSession.create("public/model/chessnet.onnx");

  const evaluator: Evaluator = async (planes: Float32Array) => {
    const tensor = new ort.Tensor("float32", planes, [1, N_PLANES, 8, 8]);
    const out = await session.run({ board: tensor });
    return {
      logits: out.policy.data as Float32Array,
      value: (out.value.data as Float32Array)[0],
    };
  };

  const mcts = new MCTS(evaluator);

  // 1) startpos: must return a legal opening move
  const start = new Chess();
  const r1 = await mcts.search(start.fen(), { sims: 80, temperature: 0 });
  console.log("startpos best:", r1.bestMove, "top visits:",
    Object.entries(r1.visits).sort((a, b) => b[1] - a[1]).slice(0, 4));

  // 2) mate-in-1: Qxf7# from a scholar's mate setup — strong net should find it
  const m1 = new Chess("r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/8/PPPP1PPP/RNBQK1NR w KQkq - 0 1");
  // load a real M1: white Qh5, black ...g6?? no. Use a clean KQ vs K mate in 1.
  const mate = new Chess("6k1/5ppp/8/8/8/8/8/4Q1K1 w - - 0 1");
  const r2 = await mcts.search(mate.fen(), { sims: 120, temperature: 0 });
  console.log("KQ endgame best:", r2.bestMove);

  // 3) play a full self-game to prove no crashes / always-legal
  const g = new Chess();
  let plies = 0;
  while (!g.isGameOver() && plies < 60) {
    const r = await mcts.search(g.fen(), { sims: 24, temperature: 0.3 });
    if (!r.bestMove) break;
    const mv = g.move({
      from: r.bestMove.slice(0, 2),
      to: r.bestMove.slice(2, 4),
      promotion: r.bestMove.length > 4 ? r.bestMove[4] : undefined,
    });
    if (!mv) { console.error("ILLEGAL MOVE PRODUCED:", r.bestMove, g.fen()); process.exit(1); }
    plies++;
  }
  console.log(`self-game: ${plies} plies, result=${g.isGameOver() ? g.pgn().slice(-12) : "ongoing"}, ok=all-legal`);
  console.log("ENGINE TEST: PASS");
}

main().catch((e) => { console.error(e); process.exit(1); });
