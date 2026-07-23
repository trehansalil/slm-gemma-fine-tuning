import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "Inference Benchmarks",
  description: "KV Cache & Speculative Decoding benchmarks",
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
