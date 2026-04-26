# app.py
# Ejecuta: python app.py
# Estructura esperada:
#   multimodal-emocion/
#     app.py
#     models/
#       text_beto_v1/best/   (Transformers: config + tokenizer + model.safetensors)
#       ast_es_best.pt       (checkpoint .pt de audio)  <-- puedes cambiarlo
#       (opcional) export_transformers/  (audio HuggingFace save_pretrained)

import os
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

# Texto (según tu captura)
TEXT_MODEL_DIR = os.path.join("models", "text_beto_v1", "best")

# Audio: primero intenta HF export (si existe), si no usa .pt
AUDIO_HF_DIR = os.path.join("models", "export_transformers")  # opcional
AUDIO_PT_PATH = os.path.join("models", "ast_es_best.pt")      # cámbialo si quieres

# Base AST (para reconstruir el modelo cuando el audio está en .pt)
AST_BASE = "MIT/ast-finetuned-audioset-10-10-0.4593"

# Audio settings (en tu pipeline normalmente 16k)
TARGET_SR = 16000
MAX_SECONDS = 5.0  # recorta o rellena a 5s para estabilidad

DEFAULT_EMOTIONS = ["anger", "disgust", "fear", "joy", "neutral", "sadness"]


# ==========================
# UTILS
# ==========================
def softmax(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x)
    e = np.exp(x)
    return e / (np.sum(e) + 1e-9)


def load_audio_mono_16k(path: str) -> np.ndarray:
    # librosa es robusto con wav/mp3/etc (si tu entorno lo permite)
    y, sr = librosa.load(path, sr=TARGET_SR, mono=True)
    y = y.astype(np.float32)
    return y


def pad_or_trim(y: np.ndarray, target_len: int) -> np.ndarray:
    if len(y) > target_len:
        return y[:target_len]
    if len(y) < target_len:
        return np.pad(y, (0, target_len - len(y)))
    return y


# ==========================
# LOAD TEXT MODEL
# ==========================
text_tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL_DIR)
text_model = AutoModelForSequenceClassification.from_pretrained(TEXT_MODEL_DIR).to(DEVICE)
text_model.eval()
TEXT_ID2LABEL = text_model.config.id2label


# ==========================
# LOAD AUDIO MODEL
# ==========================
def load_audio_stack():
    """
    Devuelve: (feature_extractor, model, id2label)
    Soporta:
      - HuggingFace save_pretrained en models/export_transformers
      - .pt con state_dict (y opcionalmente id2label/label2id dentro)
    """
    # 1) HF export
    if os.path.isdir(AUDIO_HF_DIR) and os.path.isfile(os.path.join(AUDIO_HF_DIR, "config.json")):
        fe = AutoFeatureExtractor.from_pretrained(AUDIO_HF_DIR)
        m = ASTForAudioClassification.from_pretrained(AUDIO_HF_DIR).to(DEVICE)
        m.eval()
        return fe, m, m.config.id2label

    # 2) .pt
    if not os.path.isfile(AUDIO_PT_PATH):
        raise FileNotFoundError(
            f"No encuentro el modelo de audio. Revisa:\n"
            f"- Carpeta HF: {AUDIO_HF_DIR}\n"
            f"- Checkpoint .pt: {AUDIO_PT_PATH}"
        )

    ckpt = torch.load(AUDIO_PT_PATH, map_location="cpu")

    # Intentamos detectar el formato del checkpoint
    if isinstance(ckpt, dict):
        # formatos típicos: {"model": state_dict, "id2label":..., "label2id":...}
        # o {"state_dict": ...}
        if "model" in ckpt and isinstance(ckpt["model"], dict):
            state_dict = ckpt["model"]
        elif "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
            state_dict = ckpt["state_dict"]
        else:
            # quizá ya sea un state_dict puro pero dentro de dict con otras cosas
            # probamos a usarlo como state_dict si parece compatible
            state_dict = ckpt
    else:
        # ckpt es directamente state_dict
        state_dict = ckpt

    # labels
    if isinstance(ckpt, dict) and "id2label" in ckpt and isinstance(ckpt["id2label"], dict):
        id2label = ckpt["id2label"]
        num_labels = len(id2label)
        label2id = ckpt.get("label2id", {v: int(k) for k, v in id2label.items()})
    else:
        label2id = {lab: i for i, lab in enumerate(DEFAULT_EMOTIONS)}
        id2label = {i: lab for lab, i in label2id.items()}
        num_labels = len(id2label)

    fe = AutoFeatureExtractor.from_pretrained(AST_BASE)

    # reconstruimos AST con las etiquetas
    m = ASTForAudioClassification.from_pretrained(
        AST_BASE,
        num_labels=num_labels,
        label2id=label2id,
        id2label=id2label,
        ignore_mismatched_sizes=True,
    ).to(DEVICE)

    # Cargamos pesos
    missing, unexpected = m.load_state_dict(state_dict, strict=False)
    # Si quieres “estricto”, cambia strict=True arriba.
    if missing or unexpected:
        print("[WARN] load_state_dict no fue 100% estricto.")
        if missing:
            print("  Missing keys:", missing[:10], "..." if len(missing) > 10 else "")
        if unexpected:
            print("  Unexpected keys:", unexpected[:10], "..." if len(unexpected) > 10 else "")

    m.eval()
    return fe, m, id2label


audio_feature_extractor, audio_model, AUDIO_ID2LABEL = load_audio_stack()


# ==========================
# INFERENCE
# ==========================
@torch.no_grad()
def predict_text(text: str):
    if not text or not text.strip():
        return "Escribe un texto.", {}

    inputs = text_tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128,
    )
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    out = text_model(**inputs)
    logits = out.logits.squeeze(0).detach().cpu().numpy()
    probs = softmax(logits)

    pred_id = int(np.argmax(probs))
    label = TEXT_ID2LABEL.get(pred_id, str(pred_id))

    # dict de probabilidades
    prob_dict = {TEXT_ID2LABEL.get(i, str(i)): float(probs[i]) for i in range(len(probs))}
    return f"Sentimiento: {label}", prob_dict


@torch.no_grad()
def predict_audio(audio_filepath: str):
    if audio_filepath is None:
        return "Sube un .wav.", {}

    y = load_audio_mono_16k(audio_filepath)
    y = pad_or_trim(y, int(TARGET_SR * MAX_SECONDS))

    inputs = audio_feature_extractor(
        y,
        sampling_rate=TARGET_SR,
        return_tensors="pt",
        padding=True,
    )
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    out = audio_model(**inputs)
    logits = out.logits.squeeze(0).detach().cpu().numpy()
    probs = softmax(logits)

    pred_id = int(np.argmax(probs))
    label = AUDIO_ID2LABEL.get(pred_id, str(pred_id))

    prob_dict = {AUDIO_ID2LABEL.get(i, str(i)): float(probs[i]) for i in range(len(probs))}
    return f"Emoción: {label}", prob_dict


# ==========================
# UI (GRADIO)
# ==========================
with gr.Blocks() as demo:
    gr.Markdown("# 🎭 Demo IA: Emoción (Audio) + Sentimiento (Texto)")
    gr.Markdown(f"**Device:** `{DEVICE}`")

    with gr.Tab("Audio → Emoción"):
        audio_in = gr.Audio(type="filepath", label="Sube un archivo (wav recomendado)")
        audio_out = gr.Textbox(label="Predicción")
        audio_probs = gr.Label(label="Probabilidades")
        gr.Button("Analizar audio").click(
            predict_audio,
            inputs=audio_in,
            outputs=[audio_out, audio_probs],
        )

    with gr.Tab("Texto → Sentimiento"):
        text_in = gr.Textbox(lines=4, placeholder="Escribe un texto...", label="Texto")
        text_out = gr.Textbox(label="Predicción")
        text_probs = gr.Label(label="Probabilidades")
        gr.Button("Analizar texto").click(
            predict_text,
            inputs=text_in,
            outputs=[text_out, text_probs],
        )

if __name__ == "__main__":
    demo.launch()