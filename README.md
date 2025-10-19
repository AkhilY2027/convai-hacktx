## Conversational Accent Practice AI

An end-to-end demo to help users practice accents and intonation:
- Generates a target accent sample from your prompt using ElevenLabs
- Records your voice in the browser
- Compares your recording locally with Python and returns a score + tips
- Optional: use Google Gemini for qualitative feedback

---

## Quick Start

1) Create API keys file

Create a file named `.env.local` in the project root with your keys:

```bash
# Required: ElevenLabs (for target accent audio)
XI_API_KEY=your_elevenlabs_api_key
```

2) Install Python dependencies (for local analysis)

The app calls `audioComparison.py` to compare your audio with the target sample.

Option A — pip (recommended, minimal):
```bash
pip install librosa numpy scipy vosk soundfile
```

Option B — conda (uses provided environment.yml):
```bash
conda env create -f environment.yml
conda activate mfa-dev
# Add minimal analysis libs if not present in the env
pip install librosa vosk
```

Vosk model (offline ASR):
- This repo includes `models/vosk-model-small-en-us-0.15/`.
- If missing, download and extract: https://alphacephei.com/vosk/models
- Optional env var:
	```bash
	export VOSK_MODEL=./models/vosk-model-small-en-us-0.15
	```

3) Install Node.js dependencies and run

```bash
npm install
npm run dev
```

Then open http://localhost:3000

Notes:
- Node.js 18.18+ or 20+ is recommended.
- You can also use yarn, pnpm, or bun if preferred.

---

## Usage

1. Select an accent (e.g., British, Australian, Indian)
2. Wait for the target sample to generate (ElevenLabs)
3. Click Record and read the on-screen prompt
4. Stop the recording
5. Click Analyze to get a score, metrics, and tips

---

## Optional: Gemini-Based Feedback

If you prefer AI-generated, qualitative feedback:
1. Add `GOOGLE_API_KEY` (and optionally `GEMINI_MODEL`) in `.env.local`
2. In `components/ConvAI.tsx`, change the analysis request to use the Gemini route:
	 - Replace the fetch URL from `/api/accent-feedback` to `/api/gemini-feedback`
3. Restart the dev server

---

## Project Structure (high level)

```
convai-hacktx/
├── app/
│   └── api/
│       ├── elevenlabs-accent/route.ts   # Generates target accent audio
│       ├── accent-feedback/route.ts     # Local Python comparison
│       └── gemini-feedback/route.ts     # Optional Gemini comparison
├── components/
│   └── ConvAI.tsx                       # Main UI
├── audioComparison.py                   # Local analysis script
├── models/
│   └── vosk-model-small-en-us-0.15/     # Vosk ASR model (included)
├── environment.yml                      # Optional conda env
└── SETUP.md                             # Deeper setup & troubleshooting
```

---

## Troubleshooting

- python3 not found
	- Install Python 3.8+ from https://www.python.org/
	- On Windows, use `python` instead of `python3`

- No Vosk model directory found
	- Ensure `models/vosk-model-small-en-us-0.15/` exists
	- Or set `VOSK_MODEL` to your extracted model path

- Analysis failed (500)
	- Check terminal output for Python errors
	- Verify `librosa`, `numpy`, `scipy`, `vosk`, `soundfile` are installed
	- Try a shorter recording first (5–10 seconds)

- Dev server issues
	- Run `npm install` first
	- Ensure Node 18.18+ or 20+
	- If using pnpm, run `pnpm install` then `pnpm dev` (enable Corepack if needed)

---

## Credits

- ElevenLabs for TTS and accent generation
- Vosk for offline speech recognition
- Librosa/NumPy/SciPy for audio analysis
- Google Gemini (optional) for qualitative feedback
