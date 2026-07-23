import { NextRequest, NextResponse } from "next/server";

const MODAL_URL = process.env.KV_CACHE_URL || process.env.NEXT_PUBLIC_KV_CACHE_URL || "";

export async function POST(req: NextRequest) {
  if (!MODAL_URL) {
    return NextResponse.json({ error: "KV cache endpoint not configured" }, { status: 503 });
  }
  const body = await req.json();
  const res = await fetch(MODAL_URL, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  const data = await res.json();
  return NextResponse.json(data, { status: res.status });
}
