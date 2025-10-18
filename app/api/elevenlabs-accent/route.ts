import { NextResponse } from "next/server";

export async function POST(req: Request) {
  const { prompt, accent } = await req.json();
  const XI_API_KEY = process.env.XI_API_KEY;

  // Map accent to ElevenLabs voice ID (replace with your own voice IDs)
  const accentVoiceMap: Record<string, string> = {
    british: "EXAVITQu4vr4xnSDxMaL", // Example British voice ID
    australian: "ErXwobaYiN019PkySvjV", // Example Australian voice ID
    indian: "TxGEqnHWrfWFTfGW9XjX", // Example Indian voice ID
    // Add more accents and their voice IDs here
  };

  const voiceId = accentVoiceMap[accent];
  if (!voiceId) {
    return NextResponse.json({ error: "Accent not supported." }, { status: 400 });
  }

  const response = await fetch(
    `https://api.elevenlabs.io/v1/text-to-speech/${voiceId}`,
    {
      method: "POST",
      headers: {
        "xi-api-key": XI_API_KEY || "",
        "Content-Type": "application/json",
        "Accept": "audio/mpeg",
      },
      body: JSON.stringify({
        text: prompt,
        model_id: "eleven_multilingual_v2",
        voice_settings: {
          stability: 0.5,
          similarity_boost: 0.5,
        },
      }),
    }
  );

  if (!response.ok) {
    return NextResponse.json({ error: "Failed to generate audio." }, { status: 500 });
  }

  const audioBuffer = await response.arrayBuffer();
  return new NextResponse(Buffer.from(audioBuffer), {
    status: 200,
    headers: {
      "Content-Type": "audio/mpeg",
      "Content-Disposition": "inline; filename=accent.mp3",
    },
  });
}
