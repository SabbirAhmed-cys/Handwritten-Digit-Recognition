import io
import json
import time
import os
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps, ImageFilter

# Optional drawing component. Install it with requirements.txt.
try:
    from streamlit_drawable_canvas import st_canvas
    CANVAS_AVAILABLE = True
except Exception:
    CANVAS_AVAILABLE = False

# ---------------------------- Page setup ------------------------------------
st.set_page_config(
    page_title="DigitLab | AI Recognition Studio",
    page_icon="🔢",
    layout="wide",
    initial_sidebar_state="expanded",
)

IMG_SIZE = 128
MODEL_PATH = "digit_model.keras"
HISTORY_LIMIT = 100

# ------------------------------- Styling ------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&family=Space+Grotesk:wght@500;600;700&display=swap');

    :root { --ink:#172033; --muted:#69758b; --line:#e7ebf2; --blue:#4263eb; }
    html, body, [class*="css"] { font-family:'DM Sans',sans-serif; }
    .stApp { background: linear-gradient(135deg,#f8faff 0%,#f2f6ff 48%,#fbfcff 100%); color:var(--ink); }
    [data-testid="stHeader"] { background:rgba(248,250,255,.82); }
    [data-testid="stSidebar"] { background:#ffffff; border-right:1px solid var(--line); }
    [data-testid="stSidebar"] * { color:#26334b; }
    h1,h2,h3 { font-family:'Space Grotesk',sans-serif !important; color:#172033; letter-spacing:-.035em; }
    .hero {
      padding:30px 34px; border:1px solid #e6ebf7; border-radius:24px;
      background:linear-gradient(115deg,#ffffff 0%,#eef3ff 68%,#f8faff 100%);
      box-shadow:0 12px 35px rgba(48,73,130,.07); margin-bottom:22px;
    }
    .eyebrow { color:#4263eb; text-transform:uppercase; letter-spacing:.14em; font-size:11px; font-weight:800; }
    .hero-title { font-family:'Space Grotesk',sans-serif; font-size:clamp(30px,4vw,48px); line-height:1.08; font-weight:700; color:#172033; margin:8px 0; }
    .hero-sub { color:#69758b; font-size:15px; max-width:700px; }
    .metric-card {
      background:#fff; border:1px solid #e7ebf2; border-radius:18px; padding:18px 20px;
      box-shadow:0 8px 24px rgba(40,60,110,.045); min-height:112px;
    }
    .metric-label { color:#748097; font-size:12px; font-weight:700; text-transform:uppercase; letter-spacing:.08em; }
    .metric-value { color:#172033; font-family:'Space Grotesk',sans-serif; font-size:30px; font-weight:700; margin-top:8px; }
    .metric-note { color:#8b95a8; font-size:12px; margin-top:2px; }
    .panel { background:#fff; border:1px solid #e7ebf2; border-radius:20px; padding:20px; box-shadow:0 8px 24px rgba(40,60,110,.04); }
    .small-muted { color:#748097; font-size:13px; }
    .result-digit { font-family:'Space Grotesk',sans-serif; font-size:100px; font-weight:700; line-height:1; color:#4263eb; text-align:center; }
    .pill { display:inline-block; padding:5px 10px; border-radius:99px; background:#edf2ff; color:#4263eb; font-size:11px; font-weight:800; }
    div.stButton > button { border-radius:12px; font-weight:700; min-height:42px; }
    div[data-testid="stFileUploader"] { background:#fff; border:1px dashed #b9c7e8; border-radius:16px; padding:12px; }
    /* Custom cursor: decorative follower plus normal system cursor fallback. */
    html, body, button, a, input, textarea { cursor: crosshair !important; }
    #digitlab-cursor {
      position:fixed; left:0; top:0; width:18px; height:18px; border:2px solid #4263eb;
      border-radius:50%; pointer-events:none; z-index:999999; transform:translate(-50%,-50%);
      transition:width .16s ease,height .16s ease,background .16s ease; background:rgba(66,99,235,.08);
    }
    #digitlab-cursor-dot {
      position:fixed; left:0; top:0; width:4px; height:4px; border-radius:50%;
      background:#4263eb; pointer-events:none; z-index:1000000; transform:translate(-50%,-50%);
    }
    @media (hover:none) { #digitlab-cursor,#digitlab-cursor-dot { display:none; } }
    </style>
    <div id="digitlab-cursor"></div><div id="digitlab-cursor-dot"></div>
    <script>
    (function(){
      if(window.__digitlabCursorReady) return;
      window.__digitlabCursorReady=true;
      document.addEventListener('mousemove',function(e){
        const c=document.getElementById('digitlab-cursor');
        const d=document.getElementById('digitlab-cursor-dot');
        if(c){c.style.left=e.clientX+'px';c.style.top=e.clientY+'px';}
        if(d){d.style.left=e.clientX+'px';d.style.top=e.clientY+'px';}
      });
      document.addEventListener('mouseover',function(e){
        const c=document.getElementById('digitlab-cursor');
        if(c && e.target.closest('button,a,[role=button]')) {
          c.style.width='30px';c.style.height='30px';c.style.background='rgba(66,99,235,.16)';
        } else if(c) { c.style.width='18px';c.style.height='18px';c.style.background='rgba(66,99,235,.08)'; }
      });
    })();
    </script>
    """,
    unsafe_allow_html=True,
)

# ------------------------------- Model --------------------------------------
@st.cache_resource(show_spinner="Loading your trained CNN model...")
def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Could not find {MODEL_PATH}. Put digit_model.keras in the same folder as app.py."
        )
    return tf.keras.models.load_model(MODEL_PATH)

# ---------------------------- Preprocessing ---------------------------------
def preprocess_image(image: Image.Image):
    """Convert a photo/canvas image into the model's 128x128 MNIST-like input."""
    image = ImageOps.grayscale(image)
    image = ImageOps.autocontrast(image)
    arr = np.asarray(image).astype(np.uint8)

    # Determine whether the source has a light or dark background.
    border = np.concatenate([arr[0, :], arr[-1, :], arr[:, 0], arr[:, -1]])
    if float(border.mean()) > float(arr.mean()):
        image = ImageOps.invert(image)

    # Remove faint background noise, then crop around the foreground digit.
    image = image.point(lambda p: 255 if p > 35 else 0)
    bbox = image.getbbox()
    if bbox is None:
        return None, None

    image = image.crop(bbox)
    image.thumbnail((92, 92), Image.Resampling.LANCZOS)

    canvas = Image.new("L", (IMG_SIZE, IMG_SIZE), 0)
    x = (IMG_SIZE - image.width) // 2
    y = (IMG_SIZE - image.height) // 2
    canvas.paste(image, (x, y))

    image_array = np.asarray(canvas).astype("float32") / 255.0

    # Centre the foreground by its centre of mass.
    total = float(image_array.sum())
    if total > 0:
        rows, cols = np.indices(image_array.shape)
        center_y = float((rows * image_array).sum() / total)
        center_x = float((cols * image_array).sum() / total)
        shift_y = int(round(63.5 - center_y))
        shift_x = int(round(63.5 - center_x))
        image_array = np.roll(image_array, (shift_y, shift_x), axis=(0, 1))
        if shift_y > 0: image_array[:shift_y, :] = 0
        elif shift_y < 0: image_array[shift_y:, :] = 0
        if shift_x > 0: image_array[:, :shift_x] = 0
        elif shift_x < 0: image_array[:, shift_x:] = 0

    return image_array.reshape(1, IMG_SIZE, IMG_SIZE, 1), canvas

def predict(image: Image.Image):
    tensor, processed = preprocess_image(image)
    if tensor is None:
        return None
    model = load_model()
    probabilities = model.predict(tensor, verbose=0)[0]
    digit = int(np.argmax(probabilities))
    return {
        "digit": digit,
        "confidence": float(probabilities[digit]) * 100,
        "probabilities": probabilities.astype(float),
        "processed": processed,
    }

def add_history(source, result):
    if "history" not in st.session_state:
        st.session_state.history = []
    st.session_state.history.insert(0, {
        "time": datetime.now().strftime("%d %b %Y, %I:%M:%S %p"),
        "source": source,
        "digit": result["digit"],
        "confidence": round(result["confidence"], 2),
    })
    st.session_state.history = st.session_state.history[:HISTORY_LIMIT]

def show_result(result, key_prefix="result"):
    if result is None:
        st.error("No digit was detected. Try a clearer image with one large digit.")
        return
    left, right = st.columns([1, 1.3], gap="large")
    with left:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown('<div class="small-muted" style="text-align:center">PREDICTED DIGIT</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="result-digit">{result["digit"]}</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div style="text-align:center"><span class="pill">{result["confidence"]:.2f}% confidence</span></div>',
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)
        st.write("")
        st.image(result["processed"], caption="128 × 128 model input", width=220)
    with right:
        st.markdown("#### Probability by digit")
        chart_df = pd.DataFrame({
            "Digit": [str(i) for i in range(10)],
            "Probability": result["probabilities"] * 100,
        }).set_index("Digit")
        st.bar_chart(chart_df, y="Probability", height=300)
        st.caption("The model scores are probabilities produced by its softmax output.")

# ------------------------------- Sidebar ------------------------------------
with st.sidebar:
    st.markdown("## 🔢 DigitLab")
    st.caption("AI Recognition Studio")
    page = st.radio(
        "NAVIGATION",
        ["Overview", "Recognize", "Analytics", "History", "About"],
        label_visibility="visible",
    )
    st.divider()
    st.markdown("**MODEL STATUS**")
    if os.path.exists(MODEL_PATH):
        st.success("Model file detected")
    else:
        st.error("Model file missing")
    st.caption("CNN · 128 × 128 · 10 classes")
    st.divider()
    st.caption("Class XII Artificial Intelligence Project · 2026–27")

# ------------------------------- Header -------------------------------------
st.markdown(
    """
    <div class="hero">
      <div class="eyebrow">AI · COMPUTER VISION · CNN</div>
      <div class="hero-title">Handwriting, understood.</div>
      <div class="hero-sub">A clean workspace for recognising handwritten digits with your trained convolutional neural network.</div>
    </div>
    """,
    unsafe_allow_html=True,
)

history = st.session_state.get("history", [])

# -------------------------------- Overview ----------------------------------
if page == "Overview":
    total = len(history)
    avg_conf = np.mean([x["confidence"] for x in history]) if total else 0
    most_common = max(set(x["digit"] for x in history), key=[x["digit"] for x in history].count) if total else "—"
    c1, c2, c3 = st.columns(3)
    for col, label, value, note in [
        (c1, "TOTAL PREDICTIONS", total, "This session"),
        (c2, "AVERAGE CONFIDENCE", f"{avg_conf:.1f}%" if total else "—", "Across saved predictions"),
        (c3, "MOST PREDICTED", most_common, "This session"),
    ]:
        with col:
            st.markdown(f'<div class="metric-card"><div class="metric-label">{label}</div><div class="metric-value">{value}</div><div class="metric-note">{note}</div></div>', unsafe_allow_html=True)
    st.write("")
    a, b = st.columns([1.15, .85], gap="large")
    with a:
        st.markdown("### Start recognising")
        st.markdown('<div class="panel"><h4>Turn a handwritten digit into a prediction</h4><p class="small-muted">Draw a digit on the canvas or upload an image. The app preprocesses it and sends a 128 × 128 grayscale tensor to your CNN.</p></div>', unsafe_allow_html=True)
        if st.button("Open recognition studio →", type="primary", use_container_width=True):
            st.session_state["nav_to_recognize"] = True
            st.rerun()
    with b:
        st.markdown("### Model snapshot")
        st.markdown('<div class="panel"><div class="metric-label">ARCHITECTURE</div><h3 style="margin:6px 0">Convolutional Neural Network</h3><p class="small-muted">Input: 128 × 128 × 1<br>Output: 10 digit classes<br>Dataset: MNIST</p></div>', unsafe_allow_html=True)
    if history:
        st.markdown("### Recent activity")
        st.dataframe(pd.DataFrame(history).head(5), use_container_width=True, hide_index=True)
    else:
        st.info("Your prediction activity will appear here after your first recognition.")

# -------------------------------- Recognize ---------------------------------
elif page == "Recognize":
    st.markdown("### Recognition studio")
    st.caption("Choose a method. For best results, use one clear digit with a dark background and a bright handwritten stroke.")
    tab_draw, tab_upload = st.tabs(["✍️ Draw a digit", "⬆️ Upload an image"])

    with tab_draw:
        if not CANVAS_AVAILABLE:
            st.warning("Drawing is unavailable because streamlit-drawable-canvas is not installed. Install requirements.txt and restart the app.")
        else:
            draw_col, help_col = st.columns([1.1, .9], gap="large")
            with draw_col:
                st.markdown('<div class="panel">', unsafe_allow_html=True)
                canvas_result = st_canvas(
                    fill_color="rgba(255, 255, 255, 1)",
                    stroke_width=15,
                    stroke_color="#FFFFFF",
                    background_color="#080D1B",
                    height=300,
                    width=300,
                    drawing_mode="freedraw",
                    key="digit_canvas",
                    update_streamlit=True,
                )
                st.markdown("</div>", unsafe_allow_html=True)
                if st.button("Recognise drawn digit", type="primary", use_container_width=True, key="predict_canvas"):
                    if canvas_result.image_data is None or not np.any(canvas_result.image_data[:, :, 3] > 0):
                        st.warning("Draw a digit on the canvas first.")
                    else:
                        rgba = np.asarray(canvas_result.image_data).astype(np.uint8)
                        # Canvas background is dark and stroke is bright.
                        drawn = Image.fromarray(rgba, mode="RGBA").convert("RGB")
                        with st.spinner("Your CNN is analysing the drawing..."):
                            try:
                                result = predict(drawn)
                                if result:
                                    add_history("Canvas", result)
                                    st.session_state.last_result = result
                                    st.success("Prediction complete.")
                                else:
                                    st.warning("No digit detected. Draw a larger, clearer digit.")
                            except Exception as exc:
                                st.error(f"Prediction failed: {exc}")
            with help_col:
                st.markdown('<div class="panel"><h4>Drawing tips</h4><p class="small-muted">• Draw one digit at a time.<br>• Keep the digit near the centre.<br>• Use a continuous, clear stroke.<br>• Avoid extra marks and symbols.</p><span class="pill">Canvas input</span></div>', unsafe_allow_html=True)

    with tab_upload:
        uploaded = st.file_uploader("Choose a digit image", type=["png", "jpg", "jpeg", "webp"], key="digit_upload")
        if uploaded:
            try:
                original = Image.open(uploaded).convert("RGB")
                preview_col, action_col = st.columns([1, 1], gap="large")
                with preview_col:
                    st.image(original, caption="Uploaded image", use_container_width=True)
                with action_col:
                    st.markdown("#### Ready to analyse?")
                    st.caption("The app will convert the image to grayscale, isolate the digit, resize it and centre it.")
                    if st.button("Recognise uploaded digit", type="primary", use_container_width=True):
                        with st.spinner("Your CNN is analysing the image..."):
                            try:
                                result = predict(original)
                                if result:
                                    add_history(uploaded.name, result)
                                    st.session_state.last_result = result
                                    st.success("Prediction complete.")
                                else:
                                    st.warning("No digit detected. Try another image.")
                            except Exception as exc:
                                st.error(f"Prediction failed: {exc}")
            except Exception as exc:
                st.error(f"Could not open that image: {exc}")

    if st.session_state.get("last_result") is not None:
        st.divider()
        st.markdown("### Latest prediction")
        show_result(st.session_state.last_result)

# -------------------------------- Analytics ---------------------------------
elif page == "Analytics":
    st.markdown("### Prediction analytics")
    if not history:
        st.info("No predictions yet. Visit Recognize and analyse a few digits to populate your analytics.")
    else:
        df = pd.DataFrame(history)
        c1, c2, c3 = st.columns(3)
        c1.metric("Predictions", len(df))
        c2.metric("Mean confidence", f"{df['confidence'].mean():.1f}%")
        c3.metric("Unique digits", df["digit"].nunique())
        left, right = st.columns(2, gap="large")
        with left:
            st.markdown("#### Digit frequency")
            counts = df["digit"].value_counts().reindex(range(10), fill_value=0).sort_index()
            st.bar_chart(counts)
        with right:
            st.markdown("#### Confidence distribution")
            st.area_chart(df.iloc[::-1].reset_index(drop=True)[["confidence"]], height=280)
        st.markdown("#### Prediction breakdown")
        breakdown = df.groupby("digit").agg(
            Predictions=("digit", "size"),
            Average_Confidence=("confidence", "mean"),
        ).reset_index()
        st.dataframe(breakdown, use_container_width=True, hide_index=True)
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("Download analytics as CSV", csv, "digitlab_analytics.csv", "text/csv")

# -------------------------------- History -----------------------------------
elif page == "History":
    st.markdown("### Prediction history")
    if not history:
        st.info("No predictions saved in this session yet.")
    else:
        df = pd.DataFrame(history)
        st.dataframe(df, use_container_width=True, hide_index=True)
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("Export history as CSV", csv, "digitlab_history.csv", "text/csv")
        if st.button("Clear session history", type="secondary"):
            st.session_state.history = []
            st.session_state.last_result = None
            st.rerun()
        st.caption("History is stored in Streamlit session state. It resets when the session ends or the page is refreshed in a new session.")

# -------------------------------- About -------------------------------------
elif page == "About":
    st.markdown("### About this project")
    st.markdown(
        """
        <div class="panel">
          <h3>Handwritten Digit Recognition using AI</h3>
          <p class="small-muted">An educational computer-vision project that recognises handwritten digits (0–9) using image processing and a Convolutional Neural Network.</p>
          <hr>
          <b>Model specifications</b><br>
          Input size: 128 × 128 pixels<br>
          Colour mode: Grayscale<br>
          Output classes: 10 (0–9)<br>
          Dataset: MNIST<br>
          Framework: TensorFlow / Keras
          <hr>
          <b>Developed by</b><br>
          Sabbir Ahmed · Class XII · Artificial Intelligence · Academic Session 2026–27
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption("This is an educational model. Confidence is the model's score, not a guarantee that every prediction is correct.")
