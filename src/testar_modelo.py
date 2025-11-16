# testar_modelo.py
import os
import joblib

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
modelo = joblib.load(os.path.join(MODEL_DIR, "modelo_email.joblib"))
vetorizador = joblib.load(os.path.join(MODEL_DIR, "vetorizador.joblib"))

def prever(texto):
    texto_limpo = texto.lower()
    
    texto_limpo = texto_limpo  
    vec = vetorizador.transform([texto_limpo])
    return modelo.predict(vec)[0], modelo.predict_proba(vec)[0]

if __name__ == "__main__":
    texto = "Clique aqui para atualizar sua conta bancária."
    pred, proba = prever(texto)
    print("Texto:", texto)
    print("Predição:", pred)
    print("Probabilidades:", proba)
