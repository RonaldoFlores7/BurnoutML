from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import uvicorn
import numpy as np
import pandas as pd

# ---------- Esquemas Pydantic ----------

class PredictRequest(BaseModel):
    # Recibimos un diccionario con todas las respuestas
    respuestas: dict

class PredictResponse(BaseModel):
    resultado: str         # "SI" o "N"
    probabilidad: float | None = None
    model_version: str | None = None


# ---------- Carga de artefactos ----------

app = FastAPI(title="API ML Burnout")

modelo = joblib.load("modelo_RandomForest.joblib")
onehot = joblib.load("onehot_v1.joblib")
scaler = joblib.load("scaler_v1.joblib")
column_order = joblib.load("column_order_v1.joblib")  # lista de columnas que espera el modelo
model_version = "v1"

# ---------- Mapeos y configuración ----------

mapa_frecuencia = {
    "Siempre": 100,
    "A menudo": 75,
    "Algunas veces": 50,
    "Rara vez": 25,
    "Nunca o casi nunca": 0
}

mapa_grado = {
    "En un grado muy alto": 100,
    "En un grado alto": 75,
    "En cierto grado": 50,
    "En un grado bajo": 25,
    "En un grado muy bajo": 0
}

categoricas_nominales = ["genero", "facultad", "practicasprepro"]

escala_frecuencia = [
    "pregunta1", "pregunta2", "pregunta3", "pregunta4", "pregunta5", "pregunta6",
    "pregunta10", "pregunta11", "pregunta12", "pregunta13", "pregunta18", "pregunta19"
]

escala_percepcion = [
    "pregunta7", "pregunta8", "pregunta9", "pregunta14", "pregunta15", "pregunta16", "pregunta17"
]

columnas_a_escalar = ["ciclo"] + escala_frecuencia + escala_percepcion

# categorías que realmente vio el encoder en entrenamiento
encoder_categories = dict(zip(categoricas_nominales, onehot.categories_))


# ---------- Funciones auxiliares ----------

def normalize_text(s: str | None) -> str | None:
    """
    Normaliza texto:
    - Convierte a str
    - Reemplaza espacios no separables (\xa0) por espacio normal
    - Aplica strip()
    """
    if s is None:
        return None
    return str(s).replace("\xa0", " ").strip()


def preprocess_input(respuestas: dict) -> np.ndarray:
    """
    Toma el diccionario de respuestas de la API y lo transforma
    en el array 2D que espera el modelo.
    """
    # 1) DataFrame de una sola fila
    df = pd.DataFrame([respuestas])

    # 2) Verificar que las categóricas necesarias existen
    for col in categoricas_nominales:
        if col not in df.columns or pd.isna(df.at[0, col]):
            raise HTTPException(
                status_code=422,
                detail=f"El campo '{col}' es requerido en 'respuestas'."
            )

    # 3) Normalizar texto en categóricas
    for col in categoricas_nominales:
        df[col] = df[col].apply(normalize_text)

    # 4) Alinear cada categórica con lo que conoce el encoder
    for col in categoricas_nominales:
        valor_api = df.at[0, col]
        if valor_api is None:
            raise HTTPException(
                status_code=422,
                detail=f"El campo '{col}' no puede ser nulo."
            )

        # Intentar hacer match con las categorías del encoder normalizadas
        candidatos = encoder_categories[col]
        valor_norm = normalize_text(valor_api)

        match = None
        for cat in candidatos:
            if normalize_text(cat) == valor_norm:
                match = cat
                break

        if match is None:
            # No se encontró ninguna categoría equivalente
            valores_permitidos = sorted({normalize_text(c) for c in candidatos})
            raise HTTPException(
                status_code=422,
                detail=(
                    f"Valor no permitido en '{col}': '{valor_api}'. "
                    f"Valores permitidos (formato texto): {valores_permitidos}"
                )
            )

        # Forzamos el valor exactamente igual al que conoce el encoder
        df.at[0, col] = match

    # 5) Aplicar mapeos de escalas de frecuencia y grado
    for col in escala_frecuencia:
        if col in df.columns:
            df[col] = df[col].replace(mapa_frecuencia)
        else:
            df[col] = 0  # valor por defecto si no viene

    for col in escala_percepcion:
        if col in df.columns:
            df[col] = df[col].replace(mapa_grado)
        else:
            df[col] = 0

    # 6) One-Hot Encoding para categóricas nominales
    cat_ohe = onehot.transform(df[categoricas_nominales])
    cat_cols = onehot.get_feature_names_out(categoricas_nominales)
    df_cat_ohe = pd.DataFrame(cat_ohe, columns=cat_cols, index=[0])

    # 7) Preparar numéricas a escalar
    df_num = pd.DataFrame(columns=columnas_a_escalar, index=[0])

    for c in columnas_a_escalar:
        if c in df.columns:
            df_num.at[0, c] = df.at[0, c]
        else:
            df_num.at[0, c] = 0  # por defecto

    df_num = df_num.astype(float)

    # 8) Escalar numéricas
    df_num_scaled = pd.DataFrame(
        scaler.transform(df_num),
        columns=columnas_a_escalar,
        index=[0]
    )

    # 9) Concatenar categóricas OHE + numéricas escaladas
    X = pd.concat([df_cat_ohe, df_num_scaled], axis=1)

    # 10) Alinear al orden de columnas que espera el modelo
    for col in column_order:
        if col not in X.columns:
            X[col] = 0  # columnas faltantes en 0

    X = X[column_order]

    return X.values  # array 2D (1, n_features)


# ---------- Endpoint de predicción ----------

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    try:
        X = preprocess_input(req.respuestas)

        # Probabilidades
        proba = modelo.predict_proba(X)[0]

        # Intentar detectar la columna de la clase "positiva" (burnout)
        clases = list(modelo.classes_)
        if "S" in clases:
            idx_pos = clases.index("S")
        elif 1 in clases:
            idx_pos = clases.index(1)
        else:
            # fallback: tomamos la mayor probabilidad
            idx_pos = int(np.argmax(proba))

        prob_si = float(proba[idx_pos])

        # Predicción de clase
        pred = modelo.predict(X)[0]
        pred_str = str(pred).upper()

        # Normalizamos resultado a "SI" / "N"
        if pred_str.startswith("S") or pred_str == "1":
            resultado = "SI"
        else:
            resultado = "N"

        return PredictResponse(
            resultado=resultado,
            probabilidad=prob_si,
            model_version=model_version
        )

    except HTTPException:
        # Re-lanzar errores controlados tal cual
        raise
    except Exception as e:
        # Cualquier otro error interno
        raise HTTPException(status_code=500, detail=str(e))


# ---------- Punto de entrada local ----------

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
