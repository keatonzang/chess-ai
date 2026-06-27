"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { Chess, Square } from "chess.js";
import { Chessboard } from "react-chessboard";
import { getEngine, Engine } from "../lib/engine";
import { MCTS, analyzePosition, Analysis } from "../lib/mcts";

type Status = "loading" | "ready" | "thinking" | "gameover" | "error";

const LEVELS = [
  { name: "Casual", sims: 24, temp: 0.6 },
  { name: "Club", sims: 80, temp: 0.25 },
  { name: "Strong", sims: 200, temp: 0.0 },
  { name: "Max", sims: 480, temp: 0.0 },
];

export default function ChessGame() {
  const gameRef = useRef(new Chess());
  const engineRef = useRef<Engine | null>(null);
  const reqId = useRef(0);

  const [fen, setFen] = useState(gameRef.current.fen());
  const [status, setStatus] = useState<Status>("loading");
  const [backend, setBackend] = useState<string>("");
  const [level, setLevel] = useState(1);
  const [playerColor, setPlayerColor] = useState<"w" | "b">("w");
  const [message, setMessage] = useState("Loading neural engine…");
  const [lastMove, setLastMove] = useState<{ from: string; to: string } | null>(null);
  const [thinkMs, setThinkMs] = useState<number | null>(null);
  const [history, setHistory] = useState<string[]>([]);
  const [analysis, setAnalysis] = useState<Analysis | null>(null);
  const [showEval, setShowEval] = useState(true);

  // ---- engine setup (loads ONNX on the client) ----
  useEffect(() => {
    let cancelled = false;
    getEngine("/model/chessnet.onnx")
      .then((eng) => {
        if (cancelled) return;
        engineRef.current = eng;
        setBackend(eng.backend);
        setStatus("ready");
        setMessage("Your move.");
      })
      .catch((err) => {
        if (cancelled) return;
        setStatus("error");
        setMessage("Engine failed to load: " + String(err?.message ?? err));
      });
    return () => {
      cancelled = true;
    };
  }, []);

  const refresh = useCallback(() => {
    const g = gameRef.current;
    setFen(g.fen());
    setHistory(g.history());
    if (g.isGameOver()) {
      setStatus("gameover");
      if (g.isCheckmate())
        setMessage(g.turn() === playerColor ? "Checkmate — the bot wins." : "Checkmate — you win! 🎉");
      else if (g.isDraw()) setMessage("Draw.");
      else setMessage("Game over.");
    }
  }, [playerColor]);

  const requestEngineMove = useCallback(() => {
    const g = gameRef.current;
    const eng = engineRef.current;
    if (g.isGameOver() || !eng) return;
    setStatus("thinking");
    setMessage("Bot is thinking…");
    const lvl = LEVELS[level];
    const id = ++reqId.current;
    const fen = g.fen();
    const t0 = performance.now();
    const mcts = new MCTS(eng.evaluator);
    mcts
      .search(fen, { sims: lvl.sims, temperature: lvl.temp })
      .then(({ bestMove }) => {
        if (id !== reqId.current) return; // stale (new game / undo)
        applyEngineMove(bestMove, Math.round(performance.now() - t0));
      })
      .catch((err) => {
        setStatus("error");
        setMessage("Engine error: " + String(err?.message ?? err));
      });
  }, [level]);

  const applyEngineMove = useCallback(
    (uci: string | null, ms: number) => {
      if (!uci) {
        refresh();
        return;
      }
      const g = gameRef.current;
      const move = g.move({
        from: uci.slice(0, 2),
        to: uci.slice(2, 4),
        promotion: uci.length > 4 ? uci[4] : undefined,
      });
      if (move) {
        setLastMove({ from: move.from, to: move.to });
        setThinkMs(ms);
      }
      refresh();
      if (!g.isGameOver()) {
        setStatus("ready");
        setMessage("Your move.");
      }
    },
    [refresh]
  );

  // trigger engine when it's the bot's turn
  useEffect(() => {
    if (status !== "ready") return;
    const g = gameRef.current;
    if (!g.isGameOver() && g.turn() !== playerColor) {
      requestEngineMove();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [fen, status, playerColor]);

  // live analysis (the bot's own evaluation of the current position)
  useEffect(() => {
    const eng = engineRef.current;
    if (!eng || !showEval) return;
    let cancelled = false;
    analyzePosition(eng.evaluator, fen)
      .then((a) => {
        if (!cancelled) setAnalysis(a);
      })
      .catch(() => {});
    return () => {
      cancelled = true;
    };
  }, [fen, backend, showEval]);

  const onDrop = useCallback(
    (source: Square, target: Square): boolean => {
      const g = gameRef.current;
      if (status === "thinking" || g.turn() !== playerColor) return false;
      let move;
      try {
        move = g.move({ from: source, to: target, promotion: "q" });
      } catch {
        return false;
      }
      if (!move) return false;
      setLastMove({ from: move.from, to: move.to });
      setThinkMs(null);
      refresh();
      if (!g.isGameOver()) {
        setStatus("ready"); // effect will fire engine move
      }
      return true;
    },
    [status, playerColor, refresh]
  );

  const newGame = useCallback(
    (color: "w" | "b") => {
      reqId.current++; // invalidate any in-flight search
      gameRef.current = new Chess();
      setPlayerColor(color);
      setLastMove(null);
      setThinkMs(null);
      setFen(gameRef.current.fen());
      setHistory([]);
      if (status !== "loading" && status !== "error") {
        setStatus("ready");
        setMessage(color === "w" ? "Your move." : "Bot opens…");
      }
    },
    [status]
  );

  const undo = useCallback(() => {
    const g = gameRef.current;
    if (status === "thinking") return;
    reqId.current++; // invalidate any in-flight search
    g.undo(); // bot move
    g.undo(); // your move
    setLastMove(null);
    refresh();
    if (!g.isGameOver()) {
      setStatus("ready");
      setMessage("Your move.");
    }
  }, [status, refresh]);

  const highlight: Record<string, React.CSSProperties> = {};
  if (lastMove) {
    highlight[lastMove.from] = { background: "rgba(255, 213, 79, 0.45)" };
    highlight[lastMove.to] = { background: "rgba(255, 213, 79, 0.45)" };
  }

  const disabled = status === "loading";

  return (
    <div className="game">
      <div className="boardwrap">
        {showEval && (
          <div className="evalbar" title="Bot's evaluation (White's perspective)">
            <div
              className="evalfill"
              style={{
                height: `${analysis ? (analysis.whiteValue + 1) * 50 : 50}%`,
              }}
            />
            <span className="evalnum">
              {analysis
                ? (analysis.whiteValue >= 0 ? "+" : "") +
                  analysis.whiteValue.toFixed(2)
                : "—"}
            </span>
          </div>
        )}
        <div className="board">
        <Chessboard
          position={fen}
          onPieceDrop={onDrop}
          boardOrientation={playerColor === "w" ? "white" : "black"}
          customSquareStyles={highlight}
          customBoardStyle={{ borderRadius: "8px", boxShadow: "0 10px 40px rgba(0,0,0,.45)" }}
          customDarkSquareStyle={{ backgroundColor: "#4a6b8a" }}
          customLightSquareStyle={{ backgroundColor: "#d6e0ea" }}
          arePiecesDraggable={status === "ready" || status === "thinking"}
        />
        </div>
      </div>

      <aside className="panel">
        <div className={`status ${status}`}>
          <span className="dot" />
          {message}
        </div>

        <div className="control">
          <label>Difficulty</label>
          <div className="levels">
            {LEVELS.map((l, i) => (
              <button
                key={l.name}
                className={i === level ? "lvl active" : "lvl"}
                onClick={() => setLevel(i)}
                disabled={disabled}
              >
                {l.name}
              </button>
            ))}
          </div>
          <small>{LEVELS[level].sims} MCTS simulations / move</small>
        </div>

        {showEval && analysis && (
          <div className="control">
            <label>
              Bot's view —{" "}
              {analysis.turn === playerColor ? "your" : "bot's"} side to move,{" "}
              {analysis.winPct}% win
            </label>
            <div className="topmoves">
              {analysis.moves.map((m) => (
                <div key={m.uci} className="tm">
                  <span className="tmsan">{m.san}</span>
                  <span className="tmbar">
                    <span
                      className="tmfill"
                      style={{ width: `${Math.round(m.prob * 100)}%` }}
                    />
                  </span>
                  <span className="tmpct">{Math.round(m.prob * 100)}%</span>
                </div>
              ))}
            </div>
          </div>
        )}

        <div className="control">
          <label>
            <input
              type="checkbox"
              checked={showEval}
              onChange={(e) => setShowEval(e.target.checked)}
              style={{ marginRight: 6 }}
            />
            Show bot evaluation
          </label>
        </div>

        <div className="control">
          <label>New game as</label>
          <div className="levels">
            <button className="lvl" onClick={() => newGame("w")} disabled={disabled}>
              ♔ White
            </button>
            <button className="lvl" onClick={() => newGame("b")} disabled={disabled}>
              ♚ Black
            </button>
          </div>
        </div>

        <div className="control">
          <button className="wide" onClick={undo} disabled={disabled || history.length < 2}>
            ↶ Undo
          </button>
        </div>

        <div className="meta">
          {backend && <div>engine: <b>{backend}</b></div>}
          {thinkMs !== null && <div>last think: <b>{thinkMs} ms</b></div>}
          <div>moves: <b>{history.length}</b></div>
        </div>

        {history.length > 0 && (
          <div className="moves">
            {history.map((m, i) =>
              i % 2 === 0 ? (
                <span key={i} className="mv">
                  <i>{i / 2 + 1}.</i> {m} {history[i + 1] ?? ""}
                </span>
              ) : null
            )}
          </div>
        )}
      </aside>
    </div>
  );
}
