import ChessGame from "../components/ChessGame";

export default function Home() {
  return (
    <main className="wrap">
      <header className="hd">
        <h1>
          Chess<span className="accent">RL</span>
        </h1>
        <p className="tag">
          A neural net learning chess <strong>purely by trial and error</strong> —
          self-play reinforcement learning from random weights, no human games and
          no engine evaluations as training signal. It thinks in your browser with
          MCTS search.
        </p>
        <p className="note">
          ⚠️ Early-stage agent: it&apos;s actively training and still a beginner —
          it will hang pieces and miss wins. That&apos;s the real, unpolished RL
          model, warts and all. It gets stronger with every training session.
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
