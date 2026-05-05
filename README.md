# 🎭 Sonentia

**Sonentia** es un sistema de Inteligencia Artificial multimodal capaz de analizar emociones humanas a partir de dos tipos de entrada:

- 🎤 **Audio (voz)** → detección de emociones  
- 📝 **Texto** → análisis de sentimiento  

El proyecto combina modelos avanzados basados en Transformers para ofrecer una solución integrada y funcional.

---

## 🚀 Características

- Clasificación de emociones en audio (6 clases):
  - anger, disgust, fear, joy, neutral, sadness  
- Análisis de sentimiento en texto:
  - POS, NEU, NEG  
- Uso de modelos Transformer:
  - AST (Audio Spectrogram Transformer)  
  - BETO (modelo NLP en español)  
- Interfaz interactiva con Gradio  
- Sistema completamente funcional y reproducible  

---

## 🧠 Arquitectura

### 🎤 Audio → Emoción
- Preprocesado: audio mono, 16kHz  
- Feature extraction: espectrogramas  
- Modelo: **AST (Audio Spectrogram Transformer)**  
- Output: emoción  

### 📝 Texto → Sentimiento
- Tokenización  
- Modelo: **Transformer (BETO)**  
- Output: sentimiento  

---

## 🔗 Modelos

Los modelos no están incluidos en el repositorio debido a su tamaño.  
Se descargan automáticamente desde Hugging Face:

- 🎤 Audio:  
  https://huggingface.co/rafiitaaa/sonentia-audio  

- 📝 Texto:  
  https://huggingface.co/rafiitaaa/sonentia-text  

---

## ⚙️ Instalación

```bash
git clone https://github.com/rafiitaaa/Sonentia
cd Sonentia
pip install -r requirements.txt
