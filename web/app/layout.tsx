import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Chess RL — play the bot",
  description:
    "A neural-network chess engine learning purely by self-play reinforcement learning from random weights — no human games, no engine evaluations. Runs fully in your browser.",
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
