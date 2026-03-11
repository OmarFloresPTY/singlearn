# 🤟 SignLearn — Lenguaje de Señas con IA

App web educativa para aprender lenguaje de señas panameño usando tu cámara y un modelo LSTM entrenado con MediaPipe.

---

## 📁 Estructura del proyecto

```
signlearn/
├── app.py                  ← Servidor Flask (backend)
├── requirements.txt        ← Dependencias Python
├── models/
│   └── train18.h5          ← Tu modelo Keras entrenado ← COLÓCALO AQUÍ
└── templates/
    └── index.html          ← Frontend web
```

---

## ⚙️ Instalación

### 1. Clona / copia los archivos
Coloca todos los archivos en una carpeta llamada `signlearn/`.

### 2. Crea un entorno virtual (recomendado)
```bash
cd signlearn
python -m venv venv

# Windows:
venv\Scripts\activate

# Mac/Linux:
source venv/bin/activate
```

### 3. Instala dependencias
```bash
pip install -r requirements.txt
```

### 4. Coloca tu modelo
Copia tu archivo `train18.h5` dentro de la carpeta `models/`:
```
signlearn/
└── models/
    └── train18.h5
```

---

## 🚀 Ejecutar el servidor

```bash
python app.py
```

Abre tu navegador en: **http://localhost:5000**

---

## 🔌 API Endpoints

| Método | Ruta | Descripción |
|--------|------|-------------|
| `GET`  | `/` | Interfaz web principal |
| `GET`  | `/api/status` | Estado del servidor y modelo |
| `POST` | `/api/predict` | Predice una seña a partir de 30 frames |
| `GET`  | `/api/actions` | Lista de señas disponibles |

### Ejemplo de petición a `/api/predict`
```json
POST /api/predict
{
  "frames": ["data:image/jpeg;base64,...", "..."]  // exactamente 30 frames
}
```

### Respuesta
```json
{
  "prediction": "A",
  "confidence": 0.9732,
  "index": 0,
  "category": "VOCALES",
  "all_scores": {
    "A": 0.9732,
    "E": 0.0121,
    ...
  }
}
```

---

## 🧠 Modelo

El modelo `train18.h5` usa:
- **Entrada:** secuencia de 30 frames × 126 keypoints (mano izquierda + mano derecha)
- **Arquitectura:** LSTM
- **Señas:** A, E, I, O, U, BONITO, CAFE, BUENOS DIAS, PANAMA, INGENIERO

Si tu modelo usa una extracción de keypoints diferente (ej. con pose o cara), edita la función `extract_keypoints()` en `app.py`.

---

## 🎮 Cómo usar la app

1. Abre **http://localhost:5000**
2. Selecciona una seña de la cuadrícula
3. Observa la **referencia visual** de cómo hacer la seña
4. Presiona **Iniciar cámara** y permite el acceso
5. Realiza la seña frente a la cámara
6. Presiona **Capturar seña** — el sistema grabará 30 frames (~3 segundos)
7. El modelo de IA analiza los frames y muestra el resultado
8. Gana XP, mantén tu racha y completa todas las señas 🏆

---

## 🔧 Personalización

### Cambiar el modelo
```python
# En app.py, línea 17:
MODEL_PATH = 'models/tu_nuevo_modelo.h5'
```

### Agregar nuevas señas
```python
# En app.py:
CATEGORY = {
    'VOCALES': ['A','E','I','O','U'],
    'PALABRAS': ['BONITO','CAFE','BUENOS DIAS','PANAMA','INGENIERO'],
    'NUEVA_CATEGORIA': ['SEÑA1', 'SEÑA2']  # ← agregar aquí
}
```
Y también actualizar el array `SIGNS` en `templates/index.html`.

---

## ⚠️ Notas importantes

- La cámara se accede directamente desde el navegador (requiere HTTPS en producción, localhost funciona con HTTP)
- El modelo debe coincidir con la función `extract_keypoints()` en `app.py`
- Para producción, usa `gunicorn` en lugar de `python app.py`

```bash
pip install gunicorn
gunicorn -w 2 -b 0.0.0.0:5000 app:app
```
