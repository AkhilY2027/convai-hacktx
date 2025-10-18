# Accent Practice AI - Setup Guide

## Overview
This app helps users practice and improve their accents using:
- **ElevenLabs** for target accent audio generation
- **Local Python analysis** (audioComparison.py) for detailed pronunciation feedback
- **(Optional) Google Gemini** for AI-based qualitative feedback

---

## Environment Variables

Create a `.env.local` file in the project root:

```bash
# Required for ElevenLabs accent generation
XI_API_KEY=your_elevenlabs_api_key

# Optional: for Gemini-based feedback (if you want to use /api/gemini-feedback)
GOOGLE_API_KEY=your_google_api_key
GEMINI_MODEL=gemini-1.5-pro-002
```

---

## Python Setup (for Local Audio Analysis)

The `/api/accent-feedback` route calls `audioComparison.py` for local pronunciation scoring.

### 1. Install Python Dependencies

```bash
pip install librosa numpy scipy vosk soundfile
```

### 2. Download Vosk Model

The script uses **Vosk** for offline speech recognition.

1. Download a small English model:
   - [vosk-model-small-en-us-0.15.zip](https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip)
2. Extract to `./models/vosk-model-small-en-us-0.15`
3. Set env var (optional):
   ```bash
   export VOSK_MODEL=./models/vosk-model-small-en-us-0.15
   ```

### 3. Test Python Script

```bash
python3 audioComparison.py <user_audio.webm> <target_audio.mp3>
```

Expected output (JSON):
```json
{
  "score": 0.85,
  "weak_phonemes": [["word", 0.6], ["another", 0.7]],
  "ref_text": "The quick brown fox..."
}
```

---

## Running the App

```bash
npm install
npm run dev
```

Open [http://localhost:3000](http://localhost:3000)

### Usage Flow
1. Select an accent (British, Australian, Indian, etc.)
2. Wait for the target accent sample to load (ElevenLabs)
3. Click "Record Your Voice" and read the prompt
4. Click "Stop Recording"
5. Click "Analyze" to get detailed feedback from the Python script

---

## Optional: Use Gemini Instead

If you prefer AI-based qualitative feedback over local analysis:

1. Set `GOOGLE_API_KEY` in `.env.local`
2. In `components/ConvAI.tsx`, change the fetch URL:
   ```typescript
   const res = await fetch("/api/gemini-feedback", {
   ```
3. Restart dev server

---

## Troubleshooting

### "python3: command not found"
- Install Python 3.8+ from [python.org](https://www.python.org/)
- On Windows, use `python` instead of `python3` in the route if needed

### "No Vosk model directory found"
- Download and extract a Vosk model to `./models/`
- Set `VOSK_MODEL` env var to the extracted folder path

### "Analysis failed (500)"
- Check the terminal for Python stderr output
- Verify all Python dependencies are installed
- Try a shorter recording (5-10 seconds) first

### Audio format issues
- The script expects 16kHz mono audio internally (librosa handles conversion)
- User recordings are `audio/webm`, target samples are `audio/mpeg` (both work)

---

## Project Structure

```
convai-hacktx/
├── app/
│   ├── api/
│   │   ├── accent-feedback/route.ts    # Local Python analysis
│   │   ├── elevenlabs-accent/route.ts  # Target accent generation
│   │   └── gemini-feedback/route.ts    # Optional AI feedback
│   └── page.tsx
├── components/
│   └── ConvAI.tsx                      # Main UI component
├── audioComparison.py                  # Python audio analysis script
├── models/
│   └── vosk-model-small-en-us-0.15/   # Vosk ASR model
├── .env.local                          # Environment variables
└── SETUP.md                            # This file
```

---

## Credits

- **ElevenLabs** for high-quality accent audio synthesis
- **Vosk** for offline speech recognition
- **Librosa** for audio feature extraction
- **Google Gemini** (optional) for AI-based feedback
