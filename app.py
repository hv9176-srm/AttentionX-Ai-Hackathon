import os
import json
import math
import tempfile
import subprocess
from typing import List, Dict, Any

import numpy as np
import torch
import whisper
import librosa
from google import genai
from moviepy import VideoFileClip, concatenate_videoclips
from moviepy import (
    VideoFileClip,
    TextClip,
    CompositeVideoClip,
    concatenate_videoclips
)


# ----------------------------
# Setup
# ----------------------------
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("Set GEMINI_API_KEY as an environment variable.")

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"--- Engine: {device} ---")

whisper_model = whisper.load_model("turbo").to(device)
client = genai.Client(api_key=GEMINI_API_KEY)


# ----------------------------
# Helpers
# ----------------------------
def extract_audio_wav(video_path: str, wav_path: str, sample_rate: int = 16000):
    """
    Extract mono WAV audio using ffmpeg.
    """
    cmd = [
        "ffmpeg",
        "-y",
        "-i", video_path,
        "-ac", "1",
        "-ar", str(sample_rate),
        wav_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg audio extraction failed:\n{result.stderr}")


def zscore_normalize(values: List[float]) -> List[float]:
    arr = np.array(values, dtype=np.float32)
    if len(arr) == 0:
        return []
    mean = float(arr.mean())
    std = float(arr.std())
    if std < 1e-8:
        return [0.0 for _ in arr]
    return [float((x - mean) / std) for x in arr]


def minmax_normalize(values: List[float]) -> List[float]:
    arr = np.array(values, dtype=np.float32)
    if len(arr) == 0:
        return []
    mn = float(arr.min())
    mx = float(arr.max())
    if abs(mx - mn) < 1e-8:
        return [0.0 for _ in arr]
    return [float((x - mn) / (mx - mn)) for x in arr]


def parse_json_response(text: str):
    import json
    import re

    text = text.strip()

    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text).strip()
        text = re.sub(r"```$", "", text).strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        candidate = match.group(0)
        return json.loads(candidate)

    raise ValueError(f"Could not parse JSON from model response:\n{text}")


def overlap(a_start, a_end, b_start, b_end):
    return max(a_start, b_start) < min(a_end, b_end)


# ----------------------------
# Transcription
# ----------------------------
def transcribe_video(video_path: str) -> List[Dict[str, Any]]:
    result = whisper_model.transcribe(video_path, verbose=False)
    segments = result.get("segments", [])
    if not segments:
        raise ValueError("No transcript segments found.")
    return segments
def get_captions_for_selected_segments(selected_segments, whisper_segments, padding=1.0):
    """
    Build caption entries relative to the final stitched reel timeline.
    Each caption has:
    - start_time (relative to final reel)
    - end_time (relative to final reel)
    - text
    """
    captions = []
    reel_cursor = 0.0

    for chosen in selected_segments:
        clip_start = max(0.0, float(chosen["start_time"]) - padding)
        clip_end = float(chosen["end_time"]) + padding

        for ws in whisper_segments:
            ws_start = float(ws["start"])
            ws_end = float(ws["end"])
            ws_text = ws.get("text", "").strip()

            if not ws_text:
                continue

            # overlap with chosen clip
            overlap_start = max(ws_start, clip_start)
            overlap_end = min(ws_end, clip_end)

            if overlap_end <= overlap_start:
                continue

            relative_start = reel_cursor + (overlap_start - clip_start)
            relative_end = reel_cursor + (overlap_end - clip_start)

            captions.append({
                "start_time": relative_start,
                "end_time": relative_end,
                "text": ws_text
            })

        reel_cursor += (clip_end - clip_start)

    return captions
def create_headline_clip(headline, video_width, duration):
    return TextClip(
        text=headline,
        font_size=70,
        color="white",
        stroke_color="black",
        stroke_width=4,
        method="caption",
        size=(int(video_width * 0.9), None),
        text_align="center"
    ).with_position(("center", 120)).with_duration(duration)

# 🔴 ADD STEP 3 HERE
def make_vertical_916(clip, target_width=1080, target_height=1920):
    clip_aspect = clip.w / clip.h
    target_aspect = target_width / target_height

    if clip_aspect > target_aspect:
        clip = clip.resized(height=target_height)
        x_center = clip.w / 2
        clip = clip.cropped(
            x_center=x_center,
            width=target_width,
            y_center=clip.h / 2,
            height=target_height
        )
    else:
        clip = clip.resized(width=target_width)
        y_center = clip.h / 2
        clip = clip.cropped(
            x_center=clip.w / 2,
            width=target_width,
            y_center=y_center,
            height=target_height
        )

    return clip


#  ADD THIS NOW
def create_headline_clip(headline, video_width, duration):
    return TextClip(
        text=headline,
        font_size=70,
        color="white",
        stroke_color="black",
        stroke_width=4,
        method="caption",
        size=(int(video_width * 0.9), None),
        text_align="center"
    ).with_position(("center", 120)).with_duration(duration)

# ----------------------------
# Audio analysis
# ----------------------------
def compute_audio_features_for_segments(
    wav_path: str,
    segments: List[Dict[str, Any]],
    sample_rate: int = 16000
) -> List[Dict[str, float]]:
    """
    For each transcript segment, compute:
    - mean RMS
    - peak RMS
    - dynamic jump (max frame-to-frame increase)
    """
    y, sr = librosa.load(wav_path, sr=sample_rate, mono=True)

    frame_length = 2048
    hop_length = 512

    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]

    # frame times
    times = librosa.frames_to_time(
        np.arange(len(rms)),
        sr=sr,
        hop_length=hop_length
    )

    features = []

    for seg in segments:
        start = float(seg["start"])
        end = float(seg["end"])

        idx = np.where((times >= start) & (times <= end))[0]
        if len(idx) == 0:
            features.append({
                "mean_rms": 0.0,
                "peak_rms": 0.0,
                "jump_rms": 0.0,
            })
            continue

        seg_rms = rms[idx]
        diffs = np.diff(seg_rms) if len(seg_rms) > 1 else np.array([0.0])

        features.append({
            "mean_rms": float(np.mean(seg_rms)),
            "peak_rms": float(np.max(seg_rms)),
            "jump_rms": float(np.max(diffs)) if len(diffs) else 0.0,
        })

    return features


# ----------------------------
# Text emotion scoring
# ----------------------------
def build_segment_batch_text(segments: List[Dict[str, Any]]) -> str:
    lines = []
    for i, s in enumerate(segments):
        text = s.get("text", "").strip().replace("\n", " ")
        lines.append(f'{i}: [{int(s["start"])}s-{int(s["end"])}s] {text}')
    return "\n".join(lines)

def score_text_emotion_with_gemini(segments: List[Dict[str, Any]]) -> List[float]:
    """
    Ask Gemini to score each transcript segment for emotional / viral intensity.
    Returns one score per segment in range 0-10.
    """

    # Keep full alignment with the rest of the pipeline
    batch_text = build_segment_batch_text(segments)

    prompt = f"""
You are analyzing transcript segments for short-form viral editing.

Task:
Score each segment for emotional / viral intensity from 0 to 10.

High score if the segment has:
- tension
- urgency
- fear
- shock
- conflict
- surprise
- emotional confession
- motivational punch
- deep insight
- intense or dramatic energy

Low score if:
- filler
- setup with no payoff
- repetitive or low-energy content
- generic explanation without impact

Return ONLY valid JSON in this format:
{{
  "scores": [
    {{"index": 0, "score": 4.5}},
    {{"index": 1, "score": 8.0}}
  ]
}}

SEGMENTS:
{batch_text}
""".strip()

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
    )

    print("RAW GEMINI RESPONSE:")
    print(response.text)

    parsed = parse_json_response(response.text)

    if "scores" not in parsed:
        raise ValueError(f"Gemini emotion scoring missing 'scores': {parsed}")

    scores = [0.0] * len(segments)

    for item in parsed["scores"]:
        idx = item.get("index")
        score = item.get("score")

        if isinstance(idx, int) and 0 <= idx < len(segments):
            try:
                scores[idx] = float(score)
            except Exception:
                scores[idx] = 0.0

    return scores


# ----------------------------
# Optional keyword bonus
# ----------------------------
def keyword_bonus(text: str) -> float:
    text = text.lower()

    strong_words = [
        "fire", "strange", "danger", "afraid", "panic", "run", "screamed",
        "shocked", "crazy", "unbelievable", "terrifying", "fight", "struggle",
        "problem", "warning", "truth", "secret", "never", "worst", "best"
    ]

    bonus = 0.0
    for word in strong_words:
        if word in text:
            bonus += 0.15

    return min(bonus, 1.0)


# ----------------------------
# Fuse scores
# ----------------------------
def fuse_multimodal_scores(
    segments: List[Dict[str, Any]],
    audio_features: List[Dict[str, float]],
    text_scores: List[float]
) -> List[Dict[str, Any]]:
    mean_rms_list = [x["mean_rms"] for x in audio_features]
    peak_rms_list = [x["peak_rms"] for x in audio_features]
    jump_rms_list = [x["jump_rms"] for x in audio_features]

    mean_norm = minmax_normalize(mean_rms_list)
    peak_norm = minmax_normalize(peak_rms_list)
    jump_norm = minmax_normalize(jump_rms_list)
    text_norm = [min(max(t / 10.0, 0.0), 1.0) for t in text_scores]

    fused = []

    for i, seg in enumerate(segments):
        text = seg.get("text", "").strip()
        duration = float(seg["end"] - seg["start"])

        # softer filter for educational content
        if text_scores[i] < 3:
            continue

        if duration <= 3:
            duration_bonus = 0.05
        elif duration <= 8:
            duration_bonus = 0.10
        elif duration <= 15:
            duration_bonus = 0.06
        else:
            duration_bonus = 0.0

        kw_bonus = keyword_bonus(text)

        score = (
            0.55 * text_norm[i] +
            0.15 * mean_norm[i] +
            0.15 * peak_norm[i] +
            0.10 * jump_norm[i] +
            0.05 * kw_bonus
        ) + duration_bonus

        fused.append({
            "index": i,
            "start_time": int(seg["start"]),
            "end_time": int(seg["end"]),
            "text": text,
            "duration": duration,
            "text_score": round(text_scores[i], 3),
            "mean_rms": round(audio_features[i]["mean_rms"], 6),
            "peak_rms": round(audio_features[i]["peak_rms"], 6),
            "jump_rms": round(audio_features[i]["jump_rms"], 6),
            "final_score": round(float(score), 6),
        })

    fused.sort(key=lambda x: x["final_score"], reverse=True)
    return fused

# ----------------------------
# Segment selection
# ----------------------------
def select_top_segments(
    ranked_segments: List[Dict[str, Any]],
    target_min_total: int = 15,
    target_max_total: int = 45,
    min_clip_len: int = 3,
    max_clip_len: int = 20,
) -> List[Dict[str, Any]]:
    selected = []
    total = 0

    for seg in ranked_segments:
        start = int(seg["start_time"])
        end = int(seg["end_time"])

        if end - start < min_clip_len:
            end = start + min_clip_len
        if end - start > max_clip_len:
            end = start + max_clip_len

        bad = False
        for chosen in selected:
            if overlap(start, end, chosen["start_time"], chosen["end_time"]):
                bad = True
                break
        if bad:
            continue

        if total + (end - start) > target_max_total:
            continue

        selected.append({
            "start_time": start,
            "end_time": end,
            "score": seg["final_score"],
            "source_index": seg["index"],
            "text": seg["text"]
        })
        total += (end - start)

        if total >= target_min_total:
            break

    selected.sort(key=lambda x: x["start_time"])
    selected = merge_nearby_segments(selected)
    return selected
def create_caption_overlays(captions, video_width):
    caption_clips = []

    for cap in captions:
        duration = float(cap["end_time"] - cap["start_time"])
        if duration <= 0:
            continue

        txt = TextClip(
            cap["text"],
            fontsize=58,
            color="white",
            stroke_color="black",
            stroke_width=3,
            font="Arial",
            size=(int(video_width * 0.85), None),
            method="caption",
            align="center"
        )

        txt = txt.set_start(cap["start_time"]).set_duration(duration)
        txt = txt.set_position(("center", 1500))

        caption_clips.append(txt)

    return caption_clips
def export_highlight_reel(
    video_path: str,
    selected_segments: list,
    whisper_segments: list,
    headline: str,
    output_path: str = "viral_reel_styled.mp4",
    padding: float = 1.0,
    target_width: int = 1080,
    target_height: int = 1920
):
    if not selected_segments:
        raise ValueError("No selected segments to export.")

    base_clips = []
    video = VideoFileClip(video_path)

    try:
        video_duration = float(video.duration)

        for seg in selected_segments:
            start = max(0, float(seg["start_time"]) - padding)
            end = min(video_duration, float(seg["end_time"]) + padding)

            if end <= start:
                continue

            subclip = video.subclipped(start, end)
            subclip = make_vertical_916(subclip, target_width, target_height)
            base_clips.append(subclip)

        if not base_clips:
            raise ValueError("No valid clips were created from selected segments.")

        final_video = concatenate_videoclips(base_clips, method="compose")

        captions = get_captions_for_selected_segments(
            selected_segments=selected_segments,
            whisper_segments=whisper_segments,
            padding=padding
        )

        headline_clip = create_headline_clip(
            headline=headline,
            video_width=target_width,
            duration=final_video.duration
        )

        caption_clips = create_caption_overlays(
            captions=captions,
            video_width=target_width
        )

        print("Caption count:", len(caption_clips))

        final_composite = CompositeVideoClip(
            [final_video, headline_clip] + caption_clips,
            size=(target_width, target_height)
        )

        print("Final video size:", final_video.w, "x", final_video.h)
        print("Final composite size:", final_composite.w, "x", final_composite.h)

        final_composite.write_videofile(
            output_path,
            codec="libx264",
            audio=True,
            audio_codec="aac",
            temp_audiofile="temp-audio.m4a",
            remove_temp=True,
            fps=24
        )

        final_composite.close()
        return output_path

    finally:
        video.close()
        for clip in base_clips:
            clip.close()
def merge_nearby_segments(segments, gap_threshold=5):
    if not segments:
        return []

    merged = [segments[0]]

    for curr in segments[1:]:
        prev = merged[-1]

        # If gap is small → merge
        if curr["start_time"] - prev["end_time"] <= gap_threshold:
            prev["end_time"] = max(prev["end_time"], curr["end_time"])
            prev["text"] += " " + curr["text"]
        else:
            merged.append(curr)

    return merged
# ----------------------------
# Headline generation
# ----------------------------
def generate_headline(selected_segments: List[Dict[str, Any]]) -> str:
    combined_text = "\n".join(
        f'- {s["text"]}' for s in selected_segments
    )

    prompt = f"""
You are writing a viral reel hook headline.

Write ONE short catchy title for these selected highlight segments.
Keep it punchy. Max 8 words.

Return ONLY JSON:
{{"headline": "Your title"}}

HIGHLIGHTS:
{combined_text}
""".strip()

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
    )

    parsed = parse_json_response(response.text)
    headline = parsed.get("headline", "Viral Highlight Reel")
    return str(headline)


# ----------------------------
# Main pipeline
# ----------------------------
def get_viral_clips_multimodal(video_path: str) -> Dict[str, Any]:
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"{video_path} not found.")

    print(f"--- Transcribing: {video_path} ---")
    segments = transcribe_video(video_path)

    with tempfile.TemporaryDirectory() as tmpdir:
        wav_path = os.path.join(tmpdir, "audio.wav")
        print("--- Extracting audio ---")
        extract_audio_wav(video_path, wav_path)

        print("--- Computing audio features ---")
        audio_features = compute_audio_features_for_segments(wav_path, segments)

    print("--- Scoring transcript emotion with Gemini ---")
    text_scores = score_text_emotion_with_gemini(segments)

    print("--- Fusing multimodal scores ---")
    ranked = fuse_multimodal_scores(segments, audio_features, text_scores)

    print("--- Selecting non-continuous highlight segments ---")
    selected = select_top_segments(ranked)

    if not selected:
        print("No valid highlight segments selected. Using fallback top clips.")
        selected = []
        for seg in ranked[:3]:
            start = int(seg["start_time"])
            end = int(seg["end_time"])

            if end - start < 4:
                end = start + 4
            if end - start > 20:
                end = start + 20

            selected.append({
                "start_time": start,
                "end_time": end,
                "score": seg["final_score"],
                "source_index": seg["index"],
                "text": seg.get("text", "")
            })

    total_duration = sum(s["end_time"] - s["start_time"] for s in selected)
    print(f"--- Total selected duration: {total_duration}s ---")

    print("--- Generating headline ---")
    headline = generate_headline(selected)

    return {
        "headline": headline,
        "segments": selected,
        "whisper_segments": segments,
        "debug_top_ranked": ranked[:10]
    }

if __name__ == "__main__":
    video_file = "test_video.mp4"

    try:
        result = get_viral_clips_multimodal(video_file)

        print("\n--- MULTIMODAL VIRAL CLIP DATA ---")
        print(json.dumps(result, indent=2))

        print("\n--- Exporting final reel ---")
        output_video = export_highlight_reel(
        video_path=video_file,
        selected_segments=result["segments"],
        whisper_segments=result["whisper_segments"],
        headline=result["headline"],
        output_path="viral_reel_styled.mp4")
        print(f"\n Final reel exported: {output_video}")

    except Exception as e:
        print(f"Error: {type(e).__name__}: {e}")
