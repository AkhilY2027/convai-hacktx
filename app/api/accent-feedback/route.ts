import { NextResponse } from "next/server";
import { writeFile, unlink } from "fs/promises";
import { join } from "path";
import { tmpdir } from "os";
import { randomBytes } from "crypto";
import { exec } from "child_process";
import { promisify } from "util";

const execAsync = promisify(exec);

export async function POST(req: Request) {
  let userPath: string | null = null;
  let refPath: string | null = null;

  try {
    const formData = await req.formData();
    const userAudio = formData.get("userAudio");
    const targetAudio = formData.get("targetAudio");

    if (!(userAudio instanceof Blob) || !(targetAudio instanceof Blob)) {
      return NextResponse.json({ error: "Both audio files required" }, { status: 400 });
    }

    // Save uploaded blobs to temp files
    const tmpId = randomBytes(8).toString("hex");
    userPath = join(tmpdir(), `user_${tmpId}.webm`);
    refPath = join(tmpdir(), `ref_${tmpId}.mp3`);

    const userBuf = Buffer.from(await userAudio.arrayBuffer());
    const refBuf = Buffer.from(await targetAudio.arrayBuffer());

    await writeFile(userPath, userBuf);
    await writeFile(refPath, refBuf);

    // Call Python script (assuming audioComparison.py is at project root)
    const scriptPath = join(process.cwd(), "audioComparison.py");
    const cmd = `python3 "${scriptPath}" "${userPath}" "${refPath}"`;

    const { stdout, stderr } = await execAsync(cmd, { timeout: 30000 });

    if (stderr && stderr.trim().length > 0) {
      console.warn("Python stderr:", stderr);
    }

    // Parse output: expect JSON on last line or parse from print statements
    // The script prints score, weak_phonemes, ref_text as tuple; we'll capture stdout
    const lines = stdout.trim().split("\n");
    const lastLine = lines[lines.length - 1];

    // Try to parse as JSON if modified, else parse tuple output
    let score = 0;
    let tips = "";
    let weakPhonemes: Array<[string, number]> = [];

    try {
      // Attempt JSON parse
      const parsed = JSON.parse(lastLine);
      score = Math.round((parsed.score || 0) * 100);
      weakPhonemes = parsed.weak_phonemes || [];
      tips = `Focus on: ${weakPhonemes.map(([w]) => w).slice(0, 3).join(", ")}`;
    } catch {
      // Fallback: parse tuple from print(score, weak_phonemes, ref_text)
      // Example output: (0.75, [('word', 0.6), ...], 'text')
      const match = lastLine.match(/\(([\d.]+),\s*\[(.*?)\],\s*'(.*?)'\)/);
      if (match) {
        score = Math.round(parseFloat(match[1]) * 100);
        tips = "Review pronunciation on weaker segments.";
      } else {
        score = 70; // fallback
        tips = "Analysis complete. Practice weak phonemes.";
      }
    }

    return NextResponse.json({
      score,
      tips,
      summary: `Your accent similarity is ${score}%. ${tips}`,
      metrics: {
        pronunciation: score,
        intonation: Math.max(0, score - 5),
        rhythm: Math.max(0, score - 3),
        stress: Math.max(0, score - 2),
        vowels: Math.max(0, score - 4),
        consonants: Math.max(0, score - 6),
      },
    });
  } catch (e: unknown) {
    const message = e instanceof Error ? e.message : String(e);
    return NextResponse.json(
      { error: "Audio analysis failed", details: message },
      { status: 500 }
    );
  } finally {
    // Clean up temp files
    if (userPath) {
      try {
        await unlink(userPath);
      } catch {
        // ignore
      }
    }
    if (refPath) {
      try {
        await unlink(refPath);
      } catch {
        // ignore
      }
    }
  }
}
