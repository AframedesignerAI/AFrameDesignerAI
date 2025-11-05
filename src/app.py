# src/app.py
import io
import os
import time
import streamlit as st
from PIL import Image
from ai_pipeline import AIDesigner

st.set_page_config(
    page_title="A-Frame Designer AI (Demo)",
    page_icon="🌿",
    layout="wide",
)

# --- Sidebar (settings) ---
with st.sidebar:
    st.header("⚙️ Settings")
    model_id = st.text_input(
        "Model ID",
        value=os.getenv("AFRAME_MODEL_ID", "stabilityai/sdxl-turbo"),
        help="Hugging Face model repo id. Default: stabilityai/sdxl-turbo",
    )
    steps = st.slider("Inference steps", 1, 8, 2, help="SDXL-Turbo works best with 1–4 steps")
    guidance = st.slider("Guidance scale", 0.0, 3.0, 0.0, 0.1, help="0–1 for Turbo; higher for other models")
    seed = st.number_input("Seed (optional)", min_value=0, value=0, step=1, help="0 = random")
    width = st.select_slider("Width", options=[512, 640, 768, 896, 1024], value=768)
    height = st.select_slider("Height", options=[512, 576, 640, 720, 768], value=512)

st.title("🏗️ A-Frame Designer AI")
st.caption("🌍 Open-source assistant for sustainable architecture and landscape design.")

col1, col2 = st.columns([1, 1])

with col1:
    prompt = st.text_area(
        "Prompt",
        value="a minimalist A-frame cabin in a serene garden, Japanese-Scandinavian style, soft daylight, wooden textures, architectural visualization",
        height=140,
        help="Décris ce que tu veux visualiser (maison A-frame, jardin, ambiance, matériaux…).",
    )
    negative_prompt = st.text_input("Negative prompt (optionnel)", value="low quality, blurry, text, watermark")

    gen_btn = st.button("🎨 Generate", type="primary")

with col2:
    st.markdown("### Output")
    placeholder = st.empty()

# --- Lazy init pipeline (on first use) ---
@st.cache_resource
def get_pipeline(_model_id: str):
    return AIDesigner(model_id=_model_id)

# --- Generate on click ---
if gen_btn:
    try:
        t0 = time.time()
        pipe = get_pipeline(model_id)
        image: Image.Image = pipe.generate_image(
            prompt=prompt,
            negative_prompt=negative_prompt or None,
            steps=steps,
            guidance=guidance,
            seed=(None if seed == 0 else seed),
            width=width,
            height=height,
        )
        dt = time.time() - t0

        st.success(f"✅ Image generated in {dt:.1f}s with {model_id}")
        placeholder.image(image, use_container_width=True)

        # Download buttons
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        st.download_button(
            "⬇️ Download PNG",
            data=buf.getvalue(),
            file_name="aframe_designer_ai.png",
            mime="image/png",
        )

    except Exception as e:
        st.error(f"Generation error: {e}")
        # Fallback: show a neutral placeholder
        fallback = Image.new("RGB", (width, height), color=(230, 235, 230))
        placeholder.image(fallback, use_container_width=True)
        st.info("Tip: if you're on CPU, try smaller sizes or another model (e.g., `stabilityai/sd-turbo`).")

st.markdown("---")
st.caption("Made with 🌿 by MaisonENA-Labs — Open-source • SDXL Turbo • Streamlit")
