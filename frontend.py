import os
import tempfile
import streamlit as st

# Allow larger uploads during local demo
# You can also set this in .streamlit/config.toml


# Import your existing backend functions from app.py
# Make sure app.py contains these functions.
from app import get_viral_clips_multimodal, export_highlight_reel

st.set_page_config(page_title="AttentionX AI", page_icon="🎬", layout="wide")

st.markdown(
    """
    <style>
        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
            max-width: 1200px;
        }
        .hero {
            padding: 1.4rem 1.6rem;
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 22px;
            background: linear-gradient(135deg, rgba(83,109,254,0.16), rgba(0,0,0,0.08));
            margin-bottom: 1rem;
        }
        .metric-card {
            padding: 1rem 1.2rem;
            border-radius: 18px;
            border: 1px solid rgba(255,255,255,0.08);
            background: rgba(255,255,255,0.03);
        }
        .section-card {
            padding: 1rem 1.2rem;
            border-radius: 18px;
            border: 1px solid rgba(255,255,255,0.08);
            background: rgba(255,255,255,0.02);
            margin-top: 0.8rem;
        }
        .small-muted {
            opacity: 0.8;
            font-size: 0.95rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
        <h1 style="margin-bottom:0.3rem;">🎬 AttentionX AI</h1>
        <h3 style="margin-top:0; margin-bottom:0.8rem;">Automated Content Repurposing Engine</h3>
        <p class="small-muted" style="margin-bottom:0.4rem;">
            Upload a long-form video and generate a short-form viral reel using multimodal AI.
        </p>
        <p class="small-muted" style="margin-bottom:0;">
            Whisper + Gemini + Librosa + MoviePy → emotional peak detection → vertical reel with captions.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Settings")
    st.caption("Tune export behavior for the demo.")
    padding = st.slider("Clip padding (seconds)", 0.0, 3.0, 1.0, 0.5)
    output_name = st.text_input("Output filename", value="viral_reel_styled.mp4")
    show_debug = st.checkbox("Show debug ranking", value=True)
    st.markdown("---")
    st.markdown("**Upload note**")
    st.caption("Default Streamlit upload cap is 200 MB. This app raises it for local demo use.")

uploaded_file = st.file_uploader(
    "Upload video",
    type=["mp4", "mov", "mkv", "avi", "webm"],
    help="For fastest demo, use a 5–10 minute clip. Long videos work, but processing will be slower."
)

if uploaded_file is not None:
    left, right = st.columns([1.45, 1])

    with left:
        st.markdown("### 📥 Input Preview")
        st.video(uploaded_file)

    with right:
        size_mb = uploaded_file.size / (1024 * 1024)
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Input size", f"{size_mb:.1f} MB")
        st.metric("Padding", f"{padding:.1f}s")
        st.markdown("</div>", unsafe_allow_html=True)
        st.info("Tip: For live demo, use shorter inputs for faster turnaround.")

    if st.button("🚀 Generate Reel", type="primary"):
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = os.path.join(tmpdir, uploaded_file.name)
            output_path = os.path.join(tmpdir, output_name)

            with open(input_path, "wb") as f:
                f.write(uploaded_file.read())

            try:
                with st.spinner("Analyzing emotional peaks, audio spikes, and transcript sentiment..."):
                    result = get_viral_clips_multimodal(input_path)

                st.success("Analysis complete")

                col1, col2 = st.columns([1.8, 1])

                with col1:
                    st.markdown("### 🔥 Generated Headline")
                    st.success(result["headline"])

                    st.markdown("### ✂️ Selected Segments")
                    for i, seg in enumerate(result["segments"], start=1):
                        with st.container(border=True):
                            st.write(
                                f"**Clip {i}:** {seg['start_time']}s → {seg['end_time']}s"
                            )
                            if "text" in seg and seg["text"]:
                                st.caption(seg["text"])

                with col2:
                    total_duration = sum(
                        s["end_time"] - s["start_time"] for s in result["segments"]
                    )
                    st.metric("Total Duration", f"{total_duration:.1f}s")
                    st.metric("Segments", len(result["segments"]))

                with st.spinner("Rendering final vertical reel with headline and captions..."):
                    final_video_path = export_highlight_reel(
                        video_path=input_path,
                        selected_segments=result["segments"],
                        whisper_segments=result["whisper_segments"],
                        headline=result["headline"],
                        output_path=output_path,
                        padding=padding,
                    )

                st.success("Final reel generated successfully")

                st.markdown("### 🎥 Final Reel")
                st.video(final_video_path)
                st.caption("Vertical reel with selected highlights, headline, and caption overlays.")

                with open(final_video_path, "rb") as f:
                    st.download_button(
                        label="⬇️ Download Reel",
                        data=f,
                        file_name=os.path.basename(final_video_path),
                        mime="video/mp4"
                    )

                if show_debug and "debug_top_ranked" in result:
                    st.markdown("### 🧠 Debug: Top Ranked Segments")
                    st.json(result["debug_top_ranked"])

            except Exception as e:
                st.error(f"Error: {type(e).__name__}: {e}")
else:
    st.info("Upload a video to begin.")

st.markdown("---")
st.caption(
    "Powered by Whisper, Gemini, Librosa, and MoviePy"
)
