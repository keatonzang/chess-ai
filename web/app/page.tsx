import ChessGame from "../components/ChessGame";

export default function Home() {
  return (
    <main className="wrap">
      <header className="hd">
        <h1>
          Chess<span className="accent">RL</span>
        </h1>
        <p className="tag">
          A neural net that learned chess by distilling Stockfish across puzzles,
          weird openings, endgames &amp; rare positions — then sharpened with
          self-play. It thinks in your browser with MCTS search.
        </p>
      </header>
      <ChessGame />
      <footer className="ft">
        <span>Runs 100% client-side · WebGPU / WASM</span>
        <a href="https://github.com/keatonzang/chess-ai" target="_blank" rel="noreferrer">
          source on GitHub
        </a>
      </footer>
    </main>
  );
}
