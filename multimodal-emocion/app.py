import numpy as np
import torch
import gradio as gr
import librosa

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoFeatureExtractor,
    ASTForAudioClassification,
)

# ==========================
# CONFIG
# ==========================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TEXT_MODEL_ID = "rafiiitaaaaa/sonentia-text"
AUDIO_MODEL_ID = "rafiiitaaaaa/sonentia-audio"

TARGET_SR = 16000
MAX_SECONDS = 5.0


# ==========================
# STYLE
# ==========================
CSS = """
.gradio-container {
    max-width: none !important;
    width: 100% !important;
    min-height: 100vh !important;
    padding: 0 !important;
    background:
        radial-gradient(circle at top left, rgba(99,102,241,0.18), transparent 28%),
        radial-gradient(circle at top right, rgba(139,92,246,0.14), transparent 30%),
        linear-gradient(180deg, #020617 0%, #030712 100%) !important;
    color: #e5e7eb !important;
    font-family: -apple-system, BlinkMacSystemFont, "Inter", sans-serif !important;
}

footer {
    display: none !important;
}

.main-wrap {
    max-width: 1120px;
    margin: 0 auto;
    padding: 36px 28px 28px 28px;
}

/* HERO */
.hero {
    text-align: center;
    padding: 58px 28px;
    border-radius: 32px;
    background:
        linear-gradient(135deg, rgba(15,23,42,0.92), rgba(30,27,75,0.78)),
        radial-gradient(circle at top right, rgba(99,102,241,0.22), transparent 30%);
    border: 1px solid rgba(148,163,184,0.18);
    box-shadow: 0 24px 70px rgba(0,0,0,0.42);
    margin-bottom: 34px;
    animation: fadeIn 0.8s ease both;
}

.hero h1 {
    font-size: 62px !important;
    font-weight: 850 !important;
    color: #f8fafc !important;
    letter-spacing: -2px !important;
    margin: 18px 0 12px 0 !important;
    animation: floatTitle 3.2s ease-in-out infinite;
}

.hero p {
    color: #cbd5e1 !important;
    font-size: 17px !important;
    margin: 0 auto;
    max-width: 700px;
}

.badge-row {
    margin-bottom: 10px;
}

.badge {
    display: inline-block;
    padding: 8px 14px;
    margin: 4px;
    border-radius: 999px;
    background: rgba(99,102,241,0.18);
    color: #c7d2fe;
    border: 1px solid rgba(199,210,254,0.20);
    font-size: 13px;
    font-weight: 600;
}

.status-dot {
    display: inline-block;
    width: 10px;
    height: 10px;
    border-radius: 999px;
    background: #22c55e;
    margin-right: 8px;
    box-shadow: 0 0 0 6px rgba(34,197,94,0.12);
    animation: pulse 1.4s infinite;
}

.model-status {
    margin-top: 18px !important;
}

/* TITULOS SIN RECUADRO Y CENTRADOS */
.plain-title,
.plain-desc {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    padding: 0 !important;
}

.plain-title h4 {
    color: #e5e7eb !important;
    text-align: center !important;
    margin: 0 0 12px 0 !important;
    font-weight: 750 !important;
}

.plain-desc h3 {
    color: #f8fafc !important;
    text-align: center !important;
    margin: 0 0 22px 0 !important;
    font-weight: 750 !important;
}

/* OCULTAR LABELS NATIVOS */
.label-wrap,
label {
    display: none !important;
}

/* TARJETAS SOLO PARA COMPONENTES */
.glass-card {
    border-radius: 24px !important;
    background: rgba(15,23,42,0.70) !important;
    border: 1px solid rgba(148,163,184,0.16) !important;
    box-shadow: 0 14px 42px rgba(0,0,0,0.22) !important;
    padding: 14px !important;
}

/* INPUTS */
textarea,
input {
    background: rgba(2,6,23,0.50) !important;
    color: #f8fafc !important;
    border-radius: 18px !important;
    border: 1px solid rgba(148,163,184,0.18) !important;
}

textarea:disabled,
input:disabled {
    opacity: 1 !important;
    background: rgba(2,6,23,0.50) !important;
    color: #f8fafc !important;
}

/* AUDIO */
.audio-container,
.upload-container {
    border-radius: 24px !important;
    background: rgba(15,23,42,0.70) !important;
    border: 1px solid rgba(148,163,184,0.16) !important;
}

/* TABS */
.tabs {
    background: transparent !important;
    border: none !important;
}

.tab-nav {
    border-bottom: 1px solid rgba(148,163,184,0.18) !important;
}

.tab-nav button {
    border-radius: 999px !important;
    color: #cbd5e1 !important;
    font-weight: 650 !important;
}

/* BOTONES */
button {
    border-radius: 18px !important;
    font-weight: 750 !important;
    transition: 0.18s ease !important;
}

button:hover {
    transform: translateY(-2px);
    box-shadow: 0 14px 32px rgba(99,102,241,0.35) !important;
}

/* FOOTER */
.custom-footer {
    margin-top: 42px;
    padding: 22px 0 8px 0;
    text-align: center;
    color: #94a3b8;
    font-size: 14px;
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
}

.footer-logo {
    display: inline-flex;
    width: 28px;
    height: 28px;
    border-radius: 10px;
    align-items: center;
    justify-content: center;
    margin-right: 8px;
    background: linear-gradient(135deg, #6366f1, #8b5cf6);
    color: white;
    font-weight: 800;
}

/* ANIMACIONES */
@keyframes fadeIn {
    from { opacity: 0; transform: translateY(12px); }
    to { opacity: 1; transform: translateY(0); }
}

@keyframes floatTitle {
    0%,100% { transform: translateY(0); }
    50% { transform: translateY(-5px); }
}

@keyframes pulse {
    0% { box-shadow: 0 0 0 0 rgba(34,197,94,0.55); }
    70% { box-shadow: 0 0 0 10px rgba(34,197,94,0); }
    100% { box-shadow: 0 0 0 0 rgba(34,197,94,0); }
}
"""


# ==========================
# UTILS
# ==========================
def softmax(x):
    x = x - np.max(x)
    e = np.exp(x)
    return e / (np.sum(e) + 1e-9)


def normalize_id2label(d):
    return {int(k): v for k, v in d.items()}


def load_audio(path):
    y, _ = librosa.load(path, sr=TARGET_SR, mono=True)
    return y.astype(np.float32)


def pad_or_trim(y, target_len):
    if len(y) > target_len:
        return y[:target_len]
    if len(y) < target_len:
        return np.pad(y, (0, target_len - len(y)))
    return y


# ==========================
# LOAD MODELS
# ==========================
text_tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL_ID)
text_model = AutoModelForSequenceClassification.from_pretrained(TEXT_MODEL_ID).to(DEVICE)
text_model.eval()
TEXT_ID2LABEL = normalize_id2label(text_model.config.id2label)

audio_fe = AutoFeatureExtractor.from_pretrained(AUDIO_MODEL_ID)
audio_model = ASTForAudioClassification.from_pretrained(AUDIO_MODEL_ID).to(DEVICE)
audio_model.eval()
AUDIO_ID2LABEL = normalize_id2label(audio_model.config.id2label)


# ==========================
# INFERENCE
# ==========================
@torch.no_grad()
def predict_text(text):
    if not text or not text.strip():
        return "Introduce texto", {}

    inputs = text_tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128,
    )
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    logits = text_model(**inputs).logits[0].cpu().numpy()
    probs = softmax(logits)

    pred = int(np.argmax(probs))
    label = TEXT_ID2LABEL[pred]

    return label, {
        TEXT_ID2LABEL[i]: float(probs[i])
        for i in range(len(probs))
    }


@torch.no_grad()
def predict_audio(path):
    if path is None:
        return "Sube audio", {}

    y = load_audio(path)
    y = pad_or_trim(y, int(TARGET_SR * MAX_SECONDS))

    inputs = audio_fe(
        y,
        sampling_rate=TARGET_SR,
        return_tensors="pt",
        padding=True,
    )
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    logits = audio_model(**inputs).logits[0].cpu().numpy()
    probs = softmax(logits)

    pred = int(np.argmax(probs))
    label = AUDIO_ID2LABEL[pred]

    return label, {
        AUDIO_ID2LABEL[i]: float(probs[i])
        for i in range(len(probs))
    }


# ==========================
# UI
# ==========================
with gr.Blocks(
    title="Sonentia",
    theme=gr.themes.Soft(primary_hue="indigo", neutral_hue="slate"),
    css=CSS,
) as demo:

    with gr.Column(elem_classes="main-wrap"):

        gr.HTML("""
        <section class="hero">
            <div class="badge-row">
                <span class="badge">🎧 Audio AI</span>
                <span class="badge">🧠 NLP</span>
                <span class="badge">⚡ Transformers</span>
            </div>

            <h1>🎭 Sonentia</h1>

            <p>
                IA multimodal para detectar emociones en voz y analizar sentimiento en texto.
            </p>

            <p class="model-status">
                <span class="status-dot"></span>
                Modelos activos desde Hugging Face
            </p>
        </section>
        """)

        with gr.Tabs():

            with gr.Tab("🎧 Audio → Emoción"):
                gr.Markdown(
                    "### Analiza una grabación de voz y predice la emoción predominante.",
                    elem_classes="plain-desc",
                )

                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown(
                            "#### Sube un archivo de audio",
                            elem_classes="plain-title",
                        )

                        with gr.Column(elem_classes="glass-card"):
                            audio_in = gr.Audio(
                                type="filepath",
                                show_label=False,
                            )

                        btn_a = gr.Button("Analizar audio", variant="primary")

                    with gr.Column(scale=1):
                        gr.Markdown(
                            "#### Resultado",
                            elem_classes="plain-title",
                        )

                        with gr.Column(elem_classes="glass-card"):
                            out_a = gr.Textbox(
                                show_label=False,
                                placeholder="Aquí aparecerá la emoción detectada...",
                            )

                        gr.Markdown(
                            "#### Probabilidades",
                            elem_classes="plain-title",
                        )

                        with gr.Column(elem_classes="glass-card"):
                            prob_a = gr.Label(
                                show_label=False,
                                num_top_classes=6,
                            )

                btn_a.click(
                    fn=predict_audio,
                    inputs=audio_in,
                    outputs=[out_a, prob_a],
                    show_progress="full",
                )

            with gr.Tab("✍️ Texto → Sentimiento"):
                gr.Markdown(
                    "### Introduce un texto y analiza su polaridad emocional.",
                    elem_classes="plain-desc",
                )

                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown(
                            "#### Texto",
                            elem_classes="plain-title",
                        )

                        with gr.Column(elem_classes="glass-card"):
                            text_in = gr.Textbox(
                                lines=7,
                                show_label=False,
                                placeholder="Ejemplo: Me encanta este proyecto, ha quedado genial.",
                            )

                        btn_t = gr.Button("Analizar texto", variant="primary")

                    with gr.Column(scale=1):
                        gr.Markdown(
                            "#### Resultado",
                            elem_classes="plain-title",
                        )

                        with gr.Column(elem_classes="glass-card"):
                            out_t = gr.Textbox(
                                show_label=False,
                                placeholder="Aquí aparecerá el sentimiento detectado...",
                            )

                        gr.Markdown(
                            "#### Probabilidades",
                            elem_classes="plain-title",
                        )

                        with gr.Column(elem_classes="glass-card"):
                            prob_t = gr.Label(
                                show_label=False,
                                num_top_classes=3,
                            )

                btn_t.click(
                    fn=predict_text,
                    inputs=text_in,
                    outputs=[out_t, prob_t],
                    show_progress="full",
                )

        gr.HTML("""
        <div class="custom-footer">
            <span class="footer-logo">S</span>
            © 2026 Sonentia · rafiitaaa
        </div>
        """)


if __name__ == "__main__":
    demo.queue().launch()