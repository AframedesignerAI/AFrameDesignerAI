# src/app.py
import sys, os
import io
import time
import streamlit as st
from PIL import Image

# 🧩 S'assure que Python trouve ai_pipeline.py dans le même dossier
sys.path.append(os.path.dirname(__file__))
from ai_pipeline import AIDesigner

# 🪴 Configuration de la page Streamlit
st.set_page_config(
    page_title="A-Frame Designer AI (Demo)",
    page_icon="🌿",
    layout="wide",
)

# --- Barre latérale (paramètres) ---
with st.sidebar:
    st.header("⚙️ Paramètres")
    model_id = st.text_input(
        "Model ID",
        value=os.getenv("AFRAME_MODEL_ID", "stabilityai/sdxl-turbo"),
        help="Modèle IA à utiliser (par défaut : stabilityai/sdxl-turbo)",
    )
    steps = st.slider("Étapes d'inférence", 1, 8, 2, help="SDXL-Turbo fonctionne bien entre 1 et 4 étapes")
    guidance = st.slider("Échelle de guidance", 0.0, 3.0, 0.0, 0.1, help="0–1 pour Turbo, plus haut pour d'autres modèles")
    seed = st.number_input("Seed (facultatif)", min_value=0, value=0, step=1, help="0 = aléatoire")
    width = st.select_slider("Largeur", options=[512, 640, 768, 896, 1024], value=768)
    height = st.select_slider("Hauteur", options=[512, 576, 640, 720, 768], value=512)

# --- Titre principal ---
st.title("🏗️ A-Frame Designer AI")
st.caption("🌍 Assistant open-source pour la conception architecturale et paysagère durable.")

col1, col2 = st.columns([1, 1])

# --- Zone de saisie ---
with col1:
    prompt = st.text_area(
        "Prompt",
        value="a minimalist A-frame cabin in a serene garden, Japanese-Scandinavian style, soft daylight, wooden textures, architectural visualization",
        height=140,
        help="Décris ta scène : maison A-frame, matériaux, ambiance, style...",
    )
    negative_prompt = st.text_input("Negative prompt (optionnel)", value="low quality, blurry, text, watermark")
    gen_btn = st.button("🎨 Générer l'image", type="primary")

with col2:
    st.markdown("### 🖼️ Résultat")
    placeholder = st.empty()

# --- Cache le pipeline IA pour éviter les rechargements ---
@st.cache_resource
def get_pipeline(_model_id: str):
    return AIDesigner(model_id=_model_id)

# --- Quand on clique sur Générer ---
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

        st.success(f"✅ Image générée en {dt:.1f}s avec {model_id}")
        placeholder.image(image, use_container_width=True)

        # Bouton de téléchargement
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        st.download_button(
            "⬇️ Télécharger l'image PNG",
            data=buf.getvalue(),
            file_name="aframe_designer_ai.png",
            mime="image/png",
        )

    except Exception as e:
        st.error(f"Erreur de génération : {e}")
        fallback = Image.new("RGB", (width, height), color=(230, 235, 230))
        placeholder.image(fallback, use_container_width=True)
        st.info("💡 Astuce : si tu es sur CPU, réduis la taille (512x512) ou essaie le modèle 'stabilityai/sd-turbo'.")

st.markdown("---")
st.caption("Made with 🌿 by MaisonENA-Labs — Open-source • SDXL Turbo • Streamlit")

