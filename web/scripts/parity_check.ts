// Verify TS encoding matches the Python fixture exactly.
import fixture from "./parity_fixture.json";
import { boardToPlanes, moveToIndex } from "../lib/encoding";

type Entry = {
  fen: string;
  turn: "w" | "b";
  planes: number[];
  moves: { from: string; to: string; promo: string | null; index: number }[];
};

let planeErrors = 0;
let moveErrors = 0;
let moveCount = 0;

for (const e of fixture as Entry[]) {
  const planes = boardToPlanes(e.fen);
  if (planes.length !== e.planes.length) {
    console.error(`plane length mismatch for ${e.fen}`);
    planeErrors++;
    continue;
  }
  for (let i = 0; i < planes.length; i++) {
    if (Math.abs(planes[i] - e.planes[i]) > 1e-6) {
      planeErrors++;
      if (planeErrors <= 5) {
        console.error(`plane mismatch ${e.fen} idx ${i}: ts=${planes[i]} py=${e.planes[i]}`);
      }
      break;
    }
  }
  for (const m of e.moves) {
    moveCount++;
    const idx = moveToIndex(m.from, m.to, m.promo, e.turn);
    if (idx !== m.index) {
      moveErrors++;
      if (moveErrors <= 10) {
        console.error(`move mismatch ${e.fen} ${m.from}${m.to}${m.promo ?? ""}: ts=${idx} py=${m.index}`);
      }
    }
  }
}

console.log(`positions=${(fixture as Entry[]).length} moves=${moveCount}`);
console.log(`planeErrors=${planeErrors} moveErrors=${moveErrors}`);
console.log(planeErrors === 0 && moveErrors === 0 ? "PARITY: PASS" : "PARITY: FAIL");
process.exit(planeErrors === 0 && moveErrors === 0 ? 0 : 1);
