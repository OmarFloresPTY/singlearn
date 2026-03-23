"""
SignLearn - Flask Backend
Servidor para predicción de lenguaje de señas con MediaPipe + Keras
"""

from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import numpy as np
import base64
import cv2
import mediapipe as mp
from tensorflow.keras.models import load_model
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# ─── CONFIGURACIÓN ────────────────────────────────────────────────
MODEL_PATH = os.environ.get('MODEL_PATH', 'models/train18.h5')
NO_FRAMES = 30

# Señas que reconoce el modelo (mismo orden del entrenamiento)
CATEGORY = {
    'VOCALES': ['A', 'E', 'I', 'O', 'U'],
    'PALABRAS': ['BONITO', 'CAFE', 'BUENOS DIAS', 'PANAMA', 'INGENIERO']
}
ACTIONS = [v for values in CATEGORY.values() for v in values]

# ─── MEDIAPIPE ────────────────────────────────────────────────────
mp_holistic = mp.solutions.holistic
mp_hands    = mp.solutions.hands

# ─── CARGA DEL MODELO ─────────────────────────────────────────────
model = None

def load_keras_model():
    global model
    if not os.path.exists(MODEL_PATH):
        logger.warning(f"Modelo no encontrado en: {MODEL_PATH}")
        return False
    try:
        model = load_model(MODEL_PATH)
        logger.info(f"✅ Modelo cargado desde {MODEL_PATH}")
        return True
    except Exception as e:
        logger.error(f"❌ Error cargando modelo: {e}")
        return False

# ─── FUNCIONES DE PROCESAMIENTO ───────────────────────────────────

def decode_frame(b64_string: str) -> np.ndarray:
    """Decodifica un frame base64 a imagen OpenCV (BGR)."""
    header, data = b64_string.split(',', 1) if ',' in b64_string else ('', b64_string)
    img_bytes = base64.b64decode(data)
    arr = np.frombuffer(img_bytes, np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def mediapipe_detection(image, holistic):
    """Convierte BGR→RGB, procesa con Holistic, regresa BGR + resultados."""
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    rgb.flags.writeable = False
    results = holistic.process(rgb)
    rgb.flags.writeable = True
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), results


def extract_keypoints(results) -> np.ndarray:
    """
    Extrae keypoints solo de manos (izquierda + derecha).
    Vector resultante: 126 valores (21*3 + 21*3)
    Debe coincidir con el modelo entrenado (train18.h5).
    """
    lh = (np.array([[r.x, r.y, r.z] for r in results.left_hand_landmarks.landmark]).flatten()
          if results.left_hand_landmarks else np.zeros(21 * 3))
    rh = (np.array([[r.x, r.y, r.z] for r in results.right_hand_landmarks.landmark]).flatten()
          if results.right_hand_landmarks else np.zeros(21 * 3))
    return np.concatenate([lh, rh])


def process_frames(frames_b64: list) -> dict:
    """
    Recibe lista de frames base64, extrae keypoints con MediaPipe,
    y predice con el modelo Keras.
    """
    if len(frames_b64) != NO_FRAMES:
        return {"error": f"Se esperaban {NO_FRAMES} frames, se recibieron {len(frames_b64)}"}

    sequence = []

    with mp_holistic.Holistic(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as holistic:
        for b64 in frames_b64:
            frame = decode_frame(b64)
            if frame is None:
                sequence.append(np.zeros(126))
                continue
            _, results = mediapipe_detection(frame, holistic)
            keypoints = extract_keypoints(results)
            sequence.append(keypoints)

    sequence_np = np.array(sequence)  # (30, 126)

    if model is None:
        # Modo DEMO: predicción aleatoria si no hay modelo
        logger.warning("Modelo no cargado — usando predicción demo")
        idx = np.random.randint(0, len(ACTIONS))
        confidence = round(float(np.random.uniform(0.6, 0.99)), 4)
    else:
        input_data = np.expand_dims(sequence_np, axis=0)  # (1, 30, 126)
        prediction = model.predict(input_data, verbose=0)[0]
        idx = int(np.argmax(prediction))
        confidence = round(float(prediction[idx]), 4)

    predicted_action = ACTIONS[idx]

    # Determinar categoría
    cat = next((k for k, v in CATEGORY.items() if predicted_action in v), 'DESCONOCIDO')

    return {
        "prediction": predicted_action,
        "confidence": confidence,
        "index": idx,
        "category": cat,
        "all_scores": {
            action: round(float(score), 4)
            for action, score in zip(ACTIONS,
                np.random.dirichlet(np.ones(len(ACTIONS))) if model is None
                else model.predict(np.expand_dims(sequence_np, axis=0), verbose=0)[0]
            )
        }
    }


# ─── RUTAS ────────────────────────────────────────────────────────

@app.route('/')
def index():
    return render_template('index.html', actions=ACTIONS, categories=CATEGORY)


@app.route('/api/predict', methods=['POST'])
def predict():
    """
    Endpoint principal de predicción.
    Recibe JSON: { "frames": ["data:image/jpeg;base64,...", ...] }
    Devuelve JSON: { "prediction": "A", "confidence": 0.97, ... }
    """
    try:
        data = request.get_json()
        if not data or 'frames' not in data:
            return jsonify({"error": "Se requiere el campo 'frames'"}), 400

        frames = data['frames']
        if len(frames) != NO_FRAMES:
            return jsonify({
                "error": f"Se requieren exactamente {NO_FRAMES} frames",
                "received": len(frames)
            }), 400

        result = process_frames(frames)

        if "error" in result:
            return jsonify(result), 500

        logger.info(f"Predicción: {result['prediction']} ({result['confidence']*100:.1f}%)")
        return jsonify(result)

    except Exception as e:
        logger.error(f"Error en /api/predict: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/status', methods=['GET'])
def status():
    """Verifica el estado del servidor y del modelo."""
    return jsonify({
        "status": "ok",
        "model_loaded": model is not None,
        "model_path": MODEL_PATH,
        "actions": ACTIONS,
        "categories": CATEGORY,
        "frames_required": NO_FRAMES
    })


@app.route('/api/actions', methods=['GET'])
def get_actions():
    """Devuelve la lista de señas disponibles."""
    return jsonify({"actions": ACTIONS, "categories": CATEGORY})


# ─── INICIO ───────────────────────────────────────────────────────

if __name__ == '__main__':
    load_keras_model()
    app.run(debug=True, host='0.0.0.0', port=5000, ssl_context='adhoc')
