# Deploy to Vercel

## When to use
When the user says `/deploy-vercel`, wants to deploy a demo/playground for their fine-tuned model, or asks to create a web interface for the model.

## Arguments
- `--backend` (optional): Inference backend to use. One of `hf-inference`, `modal`, or `static`. Default: ask the user.
- `REPO_NAME` (optional): HuggingFace repo ID for inference endpoint (if using `hf-inference` backend).

## Instructions

Follow these steps exactly:

### 1. Ask the user which deployment mode they want

Since the model files are too large for Vercel serverless functions, present these options:

1. **HuggingFace Inference API** — Frontend on Vercel calls HF Inference Endpoints. Requires the model to be pushed to HF first (use `/push-to-hf`).
2. **Modal backend** — Frontend on Vercel calls a Modal serverless GPU endpoint. Uses existing Modal setup in the project.
3. **Static demo** — No inference backend. Displays model info, training metrics, and pre-computed example outputs.

### 2. Create the web app

Create a Next.js app in a `web/` directory at the project root.

```bash
ls web/ 2>/dev/null || echo "No web/ directory yet"
```

#### Directory structure:
```
web/
├── package.json
├── vercel.json
├── next.config.js
├── public/
├── app/
│   ├── layout.tsx
│   ├── page.tsx
│   ├── globals.css
│   └── api/
│       └── generate/
│           └── route.ts    (only for hf-inference or modal backends)
└── components/
    ├── ModelCard.tsx
    ├── MetricsChart.tsx
    ├── ChatInterface.tsx   (only for hf-inference or modal backends)
    └── ExampleOutputs.tsx  (only for static mode)
```

### 3. Set up package.json

```json
{
  "name": "slm-demo",
  "version": "0.1.0",
  "private": true,
  "scripts": {
    "dev": "next dev",
    "build": "next build",
    "start": "next start"
  },
  "dependencies": {
    "next": "^14",
    "react": "^18",
    "react-dom": "^18"
  },
  "devDependencies": {
    "@types/node": "^20",
    "@types/react": "^18",
    "typescript": "^5",
    "tailwindcss": "^3",
    "autoprefixer": "^10",
    "postcss": "^8"
  }
}
```

### 4. Create the UI components

#### ModelCard.tsx
Display:
- Model name and architecture (from `config.json`)
- Training pipeline stages (SFT → Instruction Tuning → DPO → RLAIF)
- Parameter count
- Base model info

#### MetricsChart.tsx
- Parse `metrics.jsonl` at build time (via `getStaticProps` or read at build)
- Show training loss curve and perplexity over steps
- Use a lightweight chart library or SVG

#### ChatInterface.tsx (for inference backends)
- Text input for prompts
- Streaming response display
- Loading states
- Temperature/max tokens controls

#### ExampleOutputs.tsx (for static mode)
- Pre-computed prompt/response pairs
- Side-by-side comparison of base vs fine-tuned outputs

### 5. Create the API route (if using inference backend)

#### For HuggingFace Inference:
```typescript
// app/api/generate/route.ts
import { NextRequest, NextResponse } from 'next/server';

export async function POST(req: NextRequest) {
  const { prompt, max_tokens = 256, temperature = 0.7 } = await req.json();

  const response = await fetch(
    `https://api-inference.huggingface.co/models/${process.env.HF_MODEL_ID}`,
    {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${process.env.HF_TOKEN}`,
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        inputs: prompt,
        parameters: { max_new_tokens: max_tokens, temperature },
      }),
    }
  );

  const result = await response.json();
  return NextResponse.json(result);
}
```

#### For Modal:
```typescript
// app/api/generate/route.ts
export async function POST(req: NextRequest) {
  const { prompt, max_tokens = 256, temperature = 0.7 } = await req.json();

  const response = await fetch(process.env.MODAL_ENDPOINT_URL!, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ prompt, max_tokens, temperature }),
  });

  const result = await response.json();
  return NextResponse.json(result);
}
```

### 6. Configure vercel.json

```json
{
  "framework": "nextjs",
  "buildCommand": "npm run build",
  "outputDirectory": ".next",
  "env": {
    "HF_MODEL_ID": "@hf-model-id",
    "HF_TOKEN": "@hf-token"
  }
}
```

### 7. Copy metrics data for build-time access

```bash
cp models/slm125m/instruction/metrics.jsonl web/public/metrics-instruction.jsonl 2>/dev/null
cp models/slm125m/dpo/metrics.jsonl web/public/metrics-dpo.jsonl 2>/dev/null
cp models/slm125m/rlaif/metrics.jsonl web/public/metrics-rlaif.jsonl 2>/dev/null
```

### 8. Deploy

```bash
cd web
npm install
npx vercel --prod
```

If the user hasn't linked a Vercel project yet:
```bash
npx vercel link
```

Set environment variables if using an inference backend:
```bash
npx vercel env add HF_MODEL_ID
npx vercel env add HF_TOKEN
```

### 9. Report the deployment URL

Print the Vercel deployment URL and confirm which features are live.
