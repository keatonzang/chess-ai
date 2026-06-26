import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Chess RL — play the bot",
  description:
    "A neural-network chess engine trained by distilling Stockfish over puzzles, openings, endgames and rare positions, then refined with self-play. Runs fully in your browser.",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
