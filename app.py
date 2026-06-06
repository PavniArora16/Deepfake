import streamlit as st
import numpy as np
import cv2
import os
import tempfile
import torch
import torch.nn as nn
import torch.nn.functional as F
from model_definitions import FuNetM
from my_models import extract_faces_from_video, image_to_graph

# ─── Page Config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Deepfake Analyzer",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─── CSS ─────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;700&family=Syne:wght@400;600;700;800&display=swap');

/* ── Base ── */
html, body, [class*="css"] {
    font-family: 'Syne', sans-serif;
    background-color: #0a0a0f;
    color: #e2e8f0;
}
.main { background-color: #0a0a0f; }
.block-container { padding: 2rem 3rem; max-width: 1100px; }

/* ── Header ── */
.header-wrap {
    display: flex;
    align-items: center;
    justify-content: space-between;
    border-bottom: 1px solid #1e293b;
    padding-bottom: 1.2rem;
    margin-bottom: 2.5rem;
}
.header-title {
    font-size: 1.6rem;
    font-weight: 800;
    letter-spacing: -0.5px;
    color: #f8fafc;
}
.header-badge {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem;
    background: #1a1a2e;
    border: 1px solid #334155;
    color: #94a3b8;
    padding: 4px 10px;
    border-radius: 4px;
    letter-spacing: 1px;
}

/* ── Upload Card ── */
.card {
    background: #111827;
    border: 1px solid #1e293b;
    border-radius: 12px;
    padding: 1.8rem;
}
.card-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: #475569;
    margin-bottom: 0.6rem;
}
.card-title {
    font-size: 1.1rem;
    font-weight: 700;
    color: #f1f5f9;
    margin-bottom: 0.4rem;
}
.card-sub {
    font-size: 0.85rem;
    color: #64748b;
    margin-bottom: 1.2rem;
}

/* ── File uploader override ── */
[data-testid="stFileUploader"] {
    background: #0f172a;
    border: 1.5px dashed #1e293b;
    border-radius: 10px;
    padding: 1rem;
    transition: border-color 0.2s;
}
[data-testid="stFileUploader"]:hover {
    border-color: #3b82f6;
}

/* ── Result boxes ── */
.result-real {
    background: #052e16;
    border: 1px solid #166534;
    border-left: 4px solid #22c55e;
    border-radius: 10px;
    padding: 1.4rem 1.6rem;
}
.result-fake {
    background: #1c0a0a;
    border: 1px solid #7f1d1d;
    border-left: 4px solid #ef4444;
    border-radius: 10px;
    padding: 1.4rem 1.6rem;
}
.result-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-bottom: 0.5rem;
}
.result-verdict {
    font-size: 1.5rem;
    font-weight: 800;
    margin-bottom: 0.3rem;
}
.result-conf {
    font-size: 0.9rem;
    color: #94a3b8;
    font-family: 'JetBrains Mono', monospace;
}
.real-accent { color: #4ade80; }
.fake-accent { color: #f87171; }

/* ── Confidence bar ── */
.conf-bar-wrap {
    background: #1e293b;
    border-radius: 4px;
    height: 6px;
    margin-top: 1rem;
    overflow: hidden;
}
.conf-bar-real {
    height: 100%;
    background: linear-gradient(90deg, #16a34a, #4ade80);
    border-radius: 4px;
    transition: width 0.8s ease;
}
.conf-bar-fake {
    height: 100%;
    background: linear-gradient(90deg, #dc2626, #f87171);
    border-radius: 4px;
    transition: width 0.8s ease;
}

/* ── Stats row ── */
.stat-box {
    background: #111827;
    border: 1px solid #1e293b;
    border-radius: 8px;
    padding: 1rem 1.2rem;
    text-align: center;
}
.stat-num {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.4rem;
    font-weight: 700;
    color: #f1f5f9;
}
.stat-lbl {
    font-size: 0.75rem;
    color: #475569;
    margin-top: 2px;
}

/* ── About section ── */
.about-wrap {
    background: #111827;
    border: 1px solid #1e293b;
    border-radius: 12px;
    padding: 1.8rem;
    margin-top: 2rem;
}
.about-wrap h4 {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: #475569;
    margin-bottom: 1rem;
}
.about-wrap p {
    color: #94a3b8;
    line-height: 1.75;
    font-size: 0.9rem;
}
.pill {
    display: inline-block;
    background: #1e293b;
    color: #94a3b8;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    padding: 3px 10px;
    border-radius: 999px;
    margin: 2px;
}

/* ── Spinner override ── */
[data-testid="stSpinner"] { color: #3b82f6 !important; }

/* ── Error / Success override ── */
[data-testid="stAlert"] { border-radius: 8px; }
</style>
""", unsafe_allow_html=True)

# ─── Load Model ───────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    m = FuNetM()
    m.load_state_dict(torch.load("funet_M_full.pth", map_location="cpu"))
    m.eval()
    return m

model = load_model()

# ─── Header ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="header-wrap">
    <div class="header-title">🔍 Deepfake Analyzer</div>
    <div class="header-badge">FuNet-M · v1.0</div>
</div>
""", unsafe_allow_html=True)

# ─── Main Layout ──────────────────────────────────────────────────────────────
left, right = st.columns([1, 1], gap="large")

with left:
    st.markdown("""
    <div class="card">
        <div class="card-label">Input</div>
        <div class="card-title">Upload Video</div>
        <div class="card-sub">Supported formats: MP4 · AVI · MOV</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "video_upload",
        label_visibility="collapsed",
        type=["mp4", "avi", "mov"]
    )

    if uploaded_file:
        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        file_size_kb = len(uploaded_file.getvalue()) / 1024
        st.markdown(f"""
        <div style='font-family:JetBrains Mono,monospace; font-size:0.75rem; color:#475569;'>
            📄 {uploaded_file.name} &nbsp;·&nbsp; {file_size_kb:.1f} KB
        </div>
        """, unsafe_allow_html=True)

with right:
    st.markdown("""
    <div class="card">
        <div class="card-label">Output</div>
        <div class="card-title">Analysis Results</div>
        <div class="card-sub">Detection verdict and confidence score</div>
    </div>
    """, unsafe_allow_html=True)
    result_placeholder = st.empty()

# ─── Processing ───────────────────────────────────────────────────────────────
if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp:
        tmp.write(uploaded_file.read())
        temp_path = tmp.name

    with st.spinner("Extracting faces and running inference…"):
        faces = extract_faces_from_video(temp_path, max_faces=10)

    if not faces:
        result_placeholder.error("⚠️ No faces detected. Try a clearer video.")
    else:
        faces_np = np.array(faces) / 255.0
        all_probs = []

        progress_bar = st.progress(0, text="Analyzing faces…")

        with torch.no_grad():
            for i, face in enumerate(faces_np):
                face_tensor = (
                    torch.tensor(face, dtype=torch.float32)
                    .unsqueeze(0)
                    .permute(0, 3, 1, 2)
                )
                graph = image_to_graph(face_tensor.squeeze(0))
                if graph is None:
                    continue
                output = model(face_tensor, graph)
                prob = torch.softmax(output, dim=1)[:, 1]
                all_probs.append(prob.item())
                progress_bar.progress(
                    int((i + 1) / len(faces_np) * 100),
                    text=f"Analyzed {i+1}/{len(faces_np)} faces"
                )

        progress_bar.empty()
        os.remove(temp_path)

        if len(all_probs) == 0:
            result_placeholder.error("⚠️ Could not process any faces. Try a clearer video.")
        else:
            avg_prob = float(np.mean(all_probs))
            is_fake = avg_prob > 0.5
            conf = avg_prob if is_fake else (1 - avg_prob)
            bar_width = int(conf * 100)

            if is_fake:
                result_placeholder.markdown(f"""
                <div class="result-fake">
                    <div class="result-label fake-accent">⚠ Verdict</div>
                    <div class="result-verdict fake-accent">DEEPFAKE DETECTED</div>
                    <div class="result-conf">Confidence: {conf:.1%}</div>
                    <div class="conf-bar-wrap">
                        <div class="conf-bar-fake" style="width:{bar_width}%"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                result_placeholder.markdown(f"""
                <div class="result-real">
                    <div class="result-label real-accent">✓ Verdict</div>
                    <div class="result-verdict real-accent">AUTHENTIC VIDEO</div>
                    <div class="result-conf">Confidence: {conf:.1%}</div>
                    <div class="conf-bar-wrap">
                        <div class="conf-bar-real" style="width:{bar_width}%"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            # ── Stats row ──
            st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
            s1, s2, s3 = st.columns(3)
            with s1:
                st.markdown(f"""
                <div class="stat-box">
                    <div class="stat-num">{len(faces)}</div>
                    <div class="stat-lbl">Faces Extracted</div>
                </div>""", unsafe_allow_html=True)
            with s2:
                st.markdown(f"""
                <div class="stat-box">
                    <div class="stat-num">{len(all_probs)}</div>
                    <div class="stat-lbl">Faces Analyzed</div>
                </div>""", unsafe_allow_html=True)
            with s3:
                st.markdown(f"""
                <div class="stat-box">
                    <div class="stat-num">{avg_prob:.2f}</div>
                    <div class="stat-lbl">Avg Fake Score</div>
                </div>""", unsafe_allow_html=True)

# ─── About ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="about-wrap">
    <h4>About This Tool</h4>
    <p>
        This analyzer uses <strong style="color:#e2e8f0">FuNet-M</strong>, a graph neural network that models spatial
        relationships between facial landmarks to detect AI-generated or manipulated faces.
        Each face in the video is extracted, converted to a graph representation, and passed
        through the model independently. The final verdict is based on the average deepfake
        probability across all detected faces.
    </p>
    <div style="margin-top:1rem">
        <span class="pill">FuNet-M</span>
        <span class="pill">Graph Neural Network</span>
        <span class="pill">Face Detection</span>
        <span class="pill">PyTorch</span>
        <span class="pill">Streamlit</span>
    </div>
</div>
""", unsafe_allow_html=True)