import librosa
import numpy as np
from scipy.spatial.distance import cosine
try:
    from phonemizer import phonemize  # requires eSpeak or other backend installed
    _HAS_PHONEMIZER = True
except Exception:
    phonemize = None  # type: ignore
    _HAS_PHONEMIZER = False
try:
    from vosk import Model, KaldiRecognizer  # type: ignore
    _HAS_VOSK = True
except Exception:
    Model = None  # type: ignore
    KaldiRecognizer = None  # type: ignore
    _HAS_VOSK = False
import os
import json
import difflib
from typing import Optional, Dict, Any, List, Tuple
from scipy.spatial.distance import cdist
import shutil
import tempfile
import subprocess
from pathlib import Path
try:
    import soundfile as sf  # for writing wavs
    _HAS_SF = True
except Exception:
    sf = None  # type: ignore
    _HAS_SF = False
try:
    from scipy.io import wavfile as scipy_wavfile
    _HAS_SCIPY_WAV = True
except Exception:
    scipy_wavfile = None  # type: ignore
    _HAS_SCIPY_WAV = False
try:
    from librosa.sequence import dtw as librosa_dtw
    _HAS_DTW = True
except Exception:
    librosa_dtw = None  # type: ignore
    _HAS_DTW = False


# -----------------------------
# STEP 1: AUDIO LOADING
# -----------------------------
def load_audio(path, target_sr=16000):
    y, sr = librosa.load(path, sr=target_sr)
    y = librosa.util.normalize(y)
    return y, sr


# -----------------------------
# STEP 2: TRANSCRIPTION
# -----------------------------
def _resolve_vosk_model_path(model_path=None):
    """Resolve a usable Vosk model directory.
    Order: explicit arg -> $VOSK_MODEL -> scan ./models and ./ for 'vosk-model*' (prefer en-us).
    """
    # 1) Explicit argument
    if model_path and os.path.isdir(model_path):
        return model_path

    # 2) Environment variable
    env_path = os.getenv("VOSK_MODEL")
    if env_path and os.path.isdir(env_path):
        return env_path

    # 3) Scan common locations
    candidates = []
    for root in ("models", "."):
        if os.path.isdir(root):
            for name in os.listdir(root):
                full = os.path.join(root, name)
                if name.lower().startswith("vosk-model") and os.path.isdir(full):
                    candidates.append(full)

    # Prefer en-us models
    for c in candidates:
        if "en-us" in os.path.basename(c).lower():
            return c
    if candidates:
        return candidates[0]

    # 4) None found
    raise RuntimeError(
        "No Vosk model directory found. Set VOSK_MODEL env var or place a model under ./models.\n"
        "Download a model (e.g., small EN):\n"
        "https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip\n"
        "Then extract to .\\models\\vosk-model-small-en-us-0.15 (Windows)."
    )


def transcribe_whisper(audio_path, model_path: str | None = None):
    """
    Offline transcription using Vosk. The function name is kept to minimize code churn.
    Expects a Vosk ASR model extracted at `model_path`.
    """
    if not _HAS_VOSK:
        raise RuntimeError(
            "Vosk is not installed. Install with `pip install vosk` and download a model."
        )
    model_path = _resolve_vosk_model_path(model_path)

    # Load audio as 16kHz mono float, then convert to 16-bit PCM bytes
    y, sr = load_audio(audio_path, target_sr=16000)
    if y is None or len(y) == 0:
        return ""
    pcm16 = (np.clip(y, -1.0, 1.0) * 32767.0).astype(np.int16).tobytes()

    model = Model(model_path)
    rec = KaldiRecognizer(model, 16000)
    rec.SetWords(True)

    # Feed in chunks of PCM bytes
    chunk_bytes = 8000
    for i in range(0, len(pcm16), chunk_bytes):
        rec.AcceptWaveform(pcm16[i:i + chunk_bytes])

    try:
        result = json.loads(rec.FinalResult())
    except Exception:
        result = {}
    text = result.get("text", "").strip() if isinstance(result, dict) else ""
    words = result.get("result", []) if isinstance(result, dict) else []
    # Normalize words to a list of {word,start,end}
    norm_words = []
    for w in words or []:
        try:
            norm_words.append({
                "word": str(w.get("word", "")).lower(),
                "start": float(w.get("start", 0.0)),
                "end": float(w.get("end", 0.0)),
            })
        except Exception:
            continue
    return {"text": text, "words": norm_words}


def _align_word_sequences(ref_words, user_words):
    """Align reference and user word sequences by string using difflib.
    Returns list of (ref_idx, user_idx) pairs where words are equal.
    """
    ref_tokens = [w["word"] for w in ref_words]
    user_tokens = [w["word"] for w in user_words]
    matcher = difflib.SequenceMatcher(a=ref_tokens, b=user_tokens)
    pairs = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            # pair words in this equal block
            length = min(i2 - i1, j2 - j1)
            for k in range(length):
                pairs.append((i1 + k, j1 + k))
    return pairs


# -----------------------------
# STEP 3: FEATURE EXTRACTION
# -----------------------------
def extract_features(y: np.ndarray, sr: int) -> Optional[Dict[str, Any]]:
    """Extract robust features for similarity.
    Returns None if segment is too short to analyze reliably.
    """
    if y is None or len(y) == 0:
        return None

    duration = len(y) / float(sr)
    # Require at least ~180ms for stable features
    if duration < 0.18:
        return None

    # Choose FFT params based on segment length
    seg_len = len(y)
    # Pick n_fft as power of two <= min(2048, seg_len)
    max_fft = min(2048, seg_len)
    if max_fft < 64:
        return None
    n_fft = 2 ** int(np.floor(np.log2(max_fft)))
    n_fft = max(256, int(n_fft))
    hop_length = max(64, int(n_fft // 4))

    try:
        mfcc_seq = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, n_fft=n_fft, hop_length=hop_length)
        if mfcc_seq is None or mfcc_seq.shape[1] < 3:
            return None
        mfcc_mean = np.mean(mfcc_seq, axis=1)
        mfcc_std = np.std(mfcc_seq, axis=1) + 1e-8
        try:
            pitch, _ = librosa.pyin(y, fmin=80, fmax=350)
            pitch_mean = float(np.nanmean(pitch)) if pitch is not None else np.nan
        except Exception:
            pitch_mean = np.nan
        try:
            energy = librosa.feature.rms(y=y, frame_length=n_fft, hop_length=hop_length)
            energy_mean = float(np.mean(energy)) if energy is not None else 0.0
        except Exception:
            energy_mean = 0.0

        return {
            "mfcc_mean": mfcc_mean,
            "mfcc_std": mfcc_std,
            "mfcc_seq": mfcc_seq,
            "pitch_mean": pitch_mean,
            "energy": energy_mean,
            "duration": duration,
        }
    except Exception:
        return None


# -----------------------------
# STEP 4: PHONEME COMPARISON
# -----------------------------
def phoneme_similarity(f_user: Dict[str, Any], f_ref: Dict[str, Any]) -> Optional[float]:
    """Compute similarity between two feature dicts.
    Combines cosine similarity of MFCC stats, optional DTW over MFCC sequences,
    and pitch/energy proximity. Returns None if not computable.
    """
    if f_user is None or f_ref is None:
        return None

    # Cosine similarity on concatenated normalized stats
    v_user = np.concatenate([f_user.get("mfcc_mean", np.zeros(13)), f_user.get("mfcc_std", np.zeros(13))])
    v_ref = np.concatenate([f_ref.get("mfcc_mean", np.zeros(13)), f_ref.get("mfcc_std", np.zeros(13))])
    # L2 normalize to avoid scale issues
    def _l2(x):
        n = np.linalg.norm(x) + 1e-8
        return x / n
    v_user_n = _l2(v_user)
    v_ref_n = _l2(v_ref)
    stat_sim = 1.0 - cosine(v_user_n, v_ref_n)
    if np.isnan(stat_sim) or np.isinf(stat_sim):
        stat_sim = 0.0

    # Optional DTW similarity between MFCC sequences (frame x time)
    dtw_sim = None
    if _HAS_DTW and f_user.get("mfcc_seq") is not None and f_ref.get("mfcc_seq") is not None:
        A = f_user["mfcc_seq"].T  # time x features
        B = f_ref["mfcc_seq"].T
        # Use cosine distance per frame
        try:
            D = cdist(A, B, metric="cosine")
            # DTW expects a cost matrix; get warping path cost
            cost, _ = librosa_dtw(D=D)
            # Normalize by path length (approx diagonal length)
            norm_cost = cost[-1, -1] / (A.shape[0] + B.shape[0])
            # Map to [0,1]: lower cost -> higher similarity
            dtw_sim = float(np.exp(-3.0 * norm_cost))
        except Exception:
            dtw_sim = None

    # Pitch and energy proximity terms
    pitch_user = f_user.get("pitch_mean", np.nan)
    pitch_ref = f_ref.get("pitch_mean", np.nan)
    if np.isnan(pitch_user) or np.isnan(pitch_ref) or pitch_ref == 0:
        pitch_term = 0.5  # neutral if unreliable
    else:
        pitch_diff = abs(pitch_user - pitch_ref) / (abs(pitch_ref) + 1e-6)
        pitch_term = max(0.0, 1.0 - pitch_diff)

    energy_user = f_user.get("energy", 0.0)
    energy_ref = f_ref.get("energy", 0.0)
    if energy_ref == 0:
        energy_term = 0.5
    else:
        energy_diff = abs(energy_user - energy_ref) / (abs(energy_ref) + 1e-6)
        energy_term = max(0.0, 1.0 - energy_diff)

    # Combine weights
    weights = {
        "stat": 0.5,
        "dtw": 0.3 if dtw_sim is not None else 0.0,
        "pitch": 0.1,
        "energy": 0.1,
    }
    weight_sum = sum(weights.values()) if sum(weights.values()) > 0 else 1.0
    combined = (weights["stat"]*stat_sim + weights["dtw"]*(dtw_sim or 0.0) +
                weights["pitch"]*pitch_term + weights["energy"]*energy_term) / weight_sum

    return float(max(0.0, min(1.0, combined)))


# -----------------------------
# STEP 5: ACCENT COMPARISON PIPELINE
# -----------------------------
def compare_accent(user_audio, ref_audio):
    # Load both
    y_user, sr = load_audio(user_audio)
    y_ref, _ = load_audio(ref_audio, target_sr=sr)

    # Transcribe both
    print("Transcribing both audios...")
    t_user_res = transcribe_whisper(user_audio)
    t_ref_res = transcribe_whisper(ref_audio)
    t_user = t_user_res.get("text", "") if isinstance(t_user_res, dict) else str(t_user_res)
    t_ref = t_ref_res.get("text", "") if isinstance(t_ref_res, dict) else str(t_ref_res)

    # Quick text validation
    if t_user.lower().strip() != t_ref.lower().strip():
        print("Warning: Texts differ between user and reference.")
        print(f"User said: {t_user}")
        print(f"Reference: {t_ref}")

    # Preferred path: align by word timestamps from Vosk
    ref_words = t_ref_res.get("words", []) if isinstance(t_ref_res, dict) else []
    user_words = t_user_res.get("words", []) if isinstance(t_user_res, dict) else []
    scores: List[Tuple[str, float]] = []
    if ref_words and user_words:
        pairs = _align_word_sequences(ref_words, user_words)
        # Compute basic text match ratio
        match_ratio = (len(pairs) / max(1, len(ref_words))) * 100.0
        print(f"Text match rate: {match_ratio:.1f}% ({len(pairs)} of {len(ref_words)} words aligned)")
        for (ri, ui) in pairs:
            rw = ref_words[ri]
            uw = user_words[ui]
            # Convert seconds to sample indices, clamp bounds
            rs = max(0, min(len(y_ref), int(round(rw["start"] * sr))))
            re = max(0, min(len(y_ref), int(round(rw["end"] * sr))))
            us = max(0, min(len(y_user), int(round(uw["start"] * sr))))
            ue = max(0, min(len(y_user), int(round(uw["end"] * sr))))
            if re - rs <= 0 or ue - us <= 0:
                continue
            seg_ref = y_ref[rs:re]
            seg_user = y_user[us:ue]
            f_user = extract_features(seg_user, sr)
            f_ref = extract_features(seg_ref, sr)
            if f_user is None or f_ref is None:
                continue
            sim = phoneme_similarity(f_user, f_ref)
            if sim is None or np.isnan(sim) or np.isinf(sim):
                continue
            scores.append((rw["word"], sim))
    # Fallback path: previous even segmentation by phonemes/words
    if not scores:
        # Split reference into phonemes (graceful fallback if phonemizer not available)
        if _HAS_PHONEMIZER and phonemize:
            try:
                phonemes = phonemize(t_ref, language="en-us").split()
            except Exception:
                # Fallback to word-level segmentation
                phonemes = t_ref.split()
        else:
            phonemes = t_ref.split()
        if len(phonemes) == 0:
            phonemes = [""] * 4  # avoid div-by-zero; create dummy segments

        total_dur = len(y_user)/sr
        seg_len = total_dur / len(phonemes)
        for i, ph in enumerate(phonemes):
            start, end = int(i*seg_len*sr), int((i+1)*seg_len*sr)
            if end <= start:
                continue
            seg_user = y_user[start:end]
            seg_ref = y_ref[start:end]

            f_user = extract_features(seg_user, sr)
            f_ref = extract_features(seg_ref, sr)
            if f_user is None or f_ref is None:
                continue
            sim = phoneme_similarity(f_user, f_ref)
            if sim is None or np.isnan(sim) or np.isinf(sim):
                continue
            scores.append((ph, sim))

    if scores:
        avg_score = float(np.mean([s for _, s in scores]))
        weakest = sorted(scores, key=lambda x: x[1])[:3]
    else:
        avg_score = 0.0
        weakest = []

    print("\nAccent Analysis Complete 🎤")
    print(f"Overall Accent Similarity: {avg_score*100:.2f}%")
    if len(scores) >= 5:
        show = 5
    else:
        show = len(scores)
    print("Segments needing most improvement:")
    for ph, s in weakest[:show]:
        print(f"  • {ph:<12} similarity {s:.2f}")

    return avg_score, weakest, t_ref


# -----------------------------
# STEP 6b: MFA COMPARISON AUDIO (optional)
# -----------------------------
def _mfa_available() -> bool:
    return shutil.which("mfa") is not None


def _resolve_mfa_assets() -> Tuple[str, str]:
    """Try to find dictionary (.dict or .zip) and acoustic model (.zip) under ./models/mfa.
    Returns (dict_path, acoustic_model_path). Raises if not found.
    """
    base = Path("models") / "mfa"
    if not base.exists():
        raise RuntimeError("MFA assets folder not found: models/mfa. Place dictionary and model there.")

    dict_candidates = list(base.glob("*.dict")) + list(base.glob("*dictionary*.zip")) + list(base.glob("*ipa*.zip"))
    model_candidates = list(base.glob("*.zip"))
    # Filter model candidates to likely acoustic models (exclude dictionary zips if present)
    model_candidates = [p for p in model_candidates if "dictionary" not in p.name.lower() and p.suffix == ".zip"]

    if not dict_candidates:
        raise RuntimeError("No MFA dictionary found in models/mfa (expected .dict or dictionary .zip)")
    if not model_candidates:
        raise RuntimeError("No MFA acoustic model .zip found in models/mfa")

    # Prefer ARPA English dictionary and US acoustic model if available
    def _score_dict(p: Path) -> int:
        n = p.name.lower()
        score = 0
        if "english" in n or "en_us" in n or "us" in n:
            score += 2
        if "arpa" in n:
            score += 1
        return score

    def _score_model(p: Path) -> int:
        n = p.name.lower()
        score = 0
        if "english" in n or "us" in n:
            score += 2
        if "arpa" in n:
            score += 1
        return score

    dict_path = max(dict_candidates, key=_score_dict)
    model_path = max(model_candidates, key=_score_model)
    return str(dict_path), str(model_path)


def _write_wav_mono16k(src_path: str, out_path: str):
    y, sr = load_audio(src_path, target_sr=16000)
    # y is float32 in [-1,1]
    if _HAS_SF:
        sf.write(out_path, y, 16000, subtype="PCM_16")
    elif _HAS_SCIPY_WAV:
        pcm16 = (np.clip(y, -1.0, 1.0) * 32767.0).astype(np.int16)
        scipy_wavfile.write(out_path, 16000, pcm16)
    else:
        raise RuntimeError("No WAV writer available (need soundfile or scipy)")


def _prepare_mfa_corpus(tmp_dir: Path, audio_path: str, transcript: str) -> Path:
    corpus = tmp_dir / "corpus"
    corpus.mkdir(parents=True, exist_ok=True)
    wav_path = corpus / (Path(audio_path).stem + ".wav")
    lab_path = corpus / (Path(audio_path).stem + ".lab")
    _write_wav_mono16k(audio_path, str(wav_path))
    lab_path.write_text(transcript, encoding="utf-8")
    return corpus


def _run_mfa_align(corpus_dir: Path, dict_path: str, model_path: str, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "mfa", "align",
        str(corpus_dir),
        dict_path,
        model_path,
        str(out_dir),
        "--clean",
        "--overwrite"
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"MFA align failed: {proc.stderr}\nCommand: {' '.join(cmd)}")


def _parse_textgrid_words(textgrid_path: Path) -> List[Dict[str, Any]]:
    """Very small TextGrid parser for word intervals. Assumes a tier named 'words' or the first IntervalTier.
    Returns list of {word, start, end} in seconds.
    """
    txt = textgrid_path.read_text(encoding="utf-8", errors="ignore")
    # Naive parse: find intervals sections. This won't cover all corner cases but works for MFA defaults.
    words: List[Dict[str, Any]] = []
    # Try to detect interval lines: xmin = ..., xmax = ..., text = "..."
    lines = txt.splitlines()
    xmin = xmax = label = None
    for line in lines:
        l = line.strip()
        if l.startswith("xmin ="):
            try:
                xmin = float(l.split("=", 1)[1].strip())
            except Exception:
                xmin = None
        elif l.startswith("xmax ="):
            try:
                xmax = float(l.split("=", 1)[1].strip())
            except Exception:
                xmax = None
        elif l.startswith("text ="):
            # text = "word"
            label = l.split("=", 1)[1].strip()
            if label.startswith('"') and label.endswith('"'):
                label = label[1:-1]
            if xmin is not None and xmax is not None:
                w = (label or "").strip().lower()
                if w != "":
                    words.append({"word": w, "start": xmin, "end": xmax})
            xmin = xmax = label = None
    return words


def generate_mfa_comparison(user_audio: str, ref_audio: str, transcript: str, output_wav: str = "mfa_comparison.wav") -> Optional[str]:
    """Run MFA forced alignment for user and reference to the given transcript, then
    export an alternating comparison WAV of [ref word] + [user word] for aligned words.
    Returns the output wav path on success, or None on failure.
    """
    if not _mfa_available():
        print("MFA not found on PATH. Install Montreal Forced Aligner and ensure 'mfa' CLI is available.")
        return None
    try:
        dict_path, model_path = _resolve_mfa_assets()
    except Exception as e:
        print(f"MFA assets not found: {e}")
        return None

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        # Prepare two corpora: one for ref, one for user (both use same transcript)
        ref_corpus = _prepare_mfa_corpus(tmp / "ref", ref_audio, transcript)
        user_corpus = _prepare_mfa_corpus(tmp / "user", user_audio, transcript)
        ref_out = tmp / "ref_out"
        user_out = tmp / "user_out"
        _run_mfa_align(ref_corpus, dict_path, model_path, ref_out)
        _run_mfa_align(user_corpus, dict_path, model_path, user_out)

        ref_tg = list(ref_out.glob("*.TextGrid"))
        user_tg = list(user_out.glob("*.TextGrid"))
        if not ref_tg or not user_tg:
            print("MFA did not produce TextGrid outputs as expected.")
            return None
        ref_words = _parse_textgrid_words(ref_tg[0])
        user_words = _parse_textgrid_words(user_tg[0])

        # Align words by label using difflib (string match)
        pairs = _align_word_sequences(ref_words, user_words)
        if not pairs:
            print("No matching words found between MFA alignments.")
            return None

        # Load audios for slicing
        y_ref, sr = load_audio(ref_audio, target_sr=16000)
        y_user, _ = load_audio(user_audio, target_sr=16000)

        # Build alternating sequence: ref word, short silence, user word, longer silence
        def silence(seconds: float) -> np.ndarray:
            return np.zeros(int(round(seconds * sr)), dtype=np.float32)

        out = []
        for (ri, ui) in pairs:
            rw, uw = ref_words[ri], user_words[ui]
            rs = max(0, min(len(y_ref), int(round(rw["start"] * sr))))
            re = max(0, min(len(y_ref), int(round(rw["end"] * sr))))
            us = max(0, min(len(y_user), int(round(uw["start"] * sr))))
            ue = max(0, min(len(y_user), int(round(uw["end"] * sr))))
            if re - rs <= 0 or ue - us <= 0:
                continue
            out.append(y_ref[rs:re])
            out.append(silence(0.08))
            out.append(y_user[us:ue])
            out.append(silence(0.18))

        if not out:
            print("No valid segments to export for MFA comparison.")
            return None

        y_out = np.concatenate(out).astype(np.float32)
        # Normalize softly to avoid clipping
        peak = float(np.max(np.abs(y_out)) + 1e-8)
        if peak > 0.98:
            y_out = y_out * (0.98 / peak)

        # Write WAV
        if _HAS_SF:
            sf.write(output_wav, y_out, sr, subtype="PCM_16")
        elif _HAS_SCIPY_WAV:
            pcm16 = (np.clip(y_out, -1.0, 1.0) * 32767.0).astype(np.int16)
            scipy_wavfile.write(output_wav, sr, pcm16)
        else:
            print("Could not write WAV: need soundfile or scipy installed.")
            return None

    return output_wav


# -----------------------------
# STEP 6: EXAMPLE USAGE
# -----------------------------
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print(json.dumps({"error": "Usage: python audioComparison.py <user_audio> <ref_audio>"}))
        sys.exit(1)
    
    user_audio = sys.argv[1]
    reference_audio = sys.argv[2]

    try:
        score, weak_phonemes, ref_text = compare_accent(user_audio, reference_audio)
        # Output JSON for easy parsing by Node.js
        result = {
            "score": float(score),
            "weak_phonemes": [(w, float(s)) for w, s in weak_phonemes],
            "ref_text": ref_text
        }
        print(json.dumps(result))
    except RuntimeError as e:
        print(json.dumps({"error": str(e)}))
        sys.exit(1)

    # Optionally feed this to LLM for natural feedback
    # feedback_prompt = f"""
    # The user practiced saying "{ref_text}" in a Southern accent.
    # Accent score: {score*100:.2f}%
    # Weak phonemes: {weak_phonemes}
    # Give coaching advice on how to improve those sounds.
    # """

    # feedback = openai.Chat.completions.create(
    #     model="gpt-4o-mini",
    #     messages=[{"role": "system", "content": "You are an accent coach."},
    #               {"role": "user", "content": feedback_prompt}]
    # )

    # print("\nAI Coach Feedback:")
    # print(feedback.choices[0].message.content)
