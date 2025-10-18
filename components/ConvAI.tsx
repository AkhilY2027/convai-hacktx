"use client";
import React, { useState, useRef, useEffect } from "react";

const ACCENTS = [
  { label: "British", value: "british" },
  { label: "Australian", value: "australian" },
  { label: "Indian", value: "indian" },
  // Add more accents as needed
];

export function ConvAI() {
  const [accent, setAccent] = useState(ACCENTS[0].value);
  const [prompt] = useState("The quick brown fox jumps over the lazy dog.");
  const [isRecording, setIsRecording] = useState(false);
  const [userAudio, setUserAudio] = useState<Blob | null>(null);
  const [targetAudioUrl, setTargetAudioUrl] = useState<string | null>(null);
  const [feedback, setFeedback] = useState<{ score: number; tips: string } | null>(null);
  // Fetch target accent audio when accent or prompt changes
  useEffect(() => {
    async function fetchAccentAudio() {
      setTargetAudioUrl(null);
      try {
        const res = await fetch("/api/elevenlabs-accent", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ prompt, accent }),
        });
        if (!res.ok) throw new Error("Failed to fetch accent audio");
        const audioBlob = await res.blob();
        setTargetAudioUrl(URL.createObjectURL(audioBlob));
      } catch {
        setTargetAudioUrl(null);
      }
    }
    fetchAccentAudio();
  }, [accent, prompt]);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioChunksRef = useRef<Blob[]>([]);

  const handleRecordClick = async () => {
    if (isRecording) {
      // Stop recording
      mediaRecorderRef.current?.stop();
      setIsRecording(false);
    } else {
      // Start recording
      setUserAudio(null);
      setFeedback(null);
      audioChunksRef.current = [];
      try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        const mediaRecorder = new MediaRecorder(stream);
        mediaRecorderRef.current = mediaRecorder;
        mediaRecorder.ondataavailable = (event) => {
          if (event.data.size > 0) {
            audioChunksRef.current.push(event.data);
          }
        };
        mediaRecorder.onstop = async () => {
          const audioBlob = new Blob(audioChunksRef.current, { type: "audio/webm" });
          setUserAudio(audioBlob);
          stream.getTracks().forEach(track => track.stop());
          // Send user and target audio to accent-feedback API
          if (targetAudioUrl) {
            try {
              const formData = new FormData();
              formData.append("userAudio", audioBlob, "user.webm");
              // Fetch the target accent audio as a blob
              const targetRes = await fetch(targetAudioUrl);
              const targetBlob = await targetRes.blob();
              formData.append("targetAudio", targetBlob, "target.mp3");
              const res = await fetch("/api/accent-feedback", {
                method: "POST",
                body: formData,
              });
              if (res.ok) {
                const data = await res.json();
                setFeedback(data);
              }
            } catch {
              setFeedback(null);
            }
          }
        };
        mediaRecorder.start();
        setIsRecording(true);
      } catch {
        alert("Could not access microphone. Please check permissions.");
      }
    }
  };
  // MediaRecorder logic will go here

  return (
    <div className="bg-white/80 backdrop-blur-md rounded-xl shadow-lg p-2">
      <div className="max-w-xl mx-auto p-4 space-y-6">
        <h2 className="text-2xl font-bold">Accent Practice AI</h2>
        <label>
          Choose Accent:
          <select
            value={accent}
            onChange={e => setAccent(e.target.value)}
            className="ml-2 p-1 border rounded"
          >
            {ACCENTS.map(a => (
              <option key={a.value} value={a.value}>{a.label}</option>
            ))}
          </select>
        </label>
        <div>
          <strong>Prompt:</strong> {prompt}
        </div>
        <button
          className={`px-4 py-2 rounded ${isRecording ? "bg-red-500" : "bg-blue-500"} text-white`}
          onClick={handleRecordClick}
        >
          {isRecording ? "Stop Recording" : "Record Your Voice"}
        </button>
        {userAudio && (
          <div>
            <audio controls src={URL.createObjectURL(userAudio)} />
            <button className="ml-2 px-2 py-1 border rounded" onClick={() => setUserAudio(null)}>
              Retry
            </button>
          </div>
        )}
        {targetAudioUrl && (
          <div>
            <strong>Target Accent Example:</strong>
            <audio controls src={targetAudioUrl} />
          </div>
        )}
        {feedback && (
          <div className="mt-4 p-2 border rounded bg-gray-50">
            <strong>Score:</strong> {feedback.score}/100<br />
            <strong>Tips:</strong> {feedback.tips}
          </div>
        )}
      </div>
    </div>
  );
}