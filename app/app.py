from flask import Flask, request, render_template_string
import os, joblib
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
modelo = joblib.load(BASE / "models" / "modelo_email.joblib")
vetorizador = joblib.load(BASE / "models" / "vetorizador.joblib")

app = Flask(__name__)

HTML = """
<form method="post">
<textarea name="texto" rows="6" cols="60"></textarea><br>
<button type="submit">Analisar</button>
</form>
{% if resultado %}
  <p>Resultado: {{ resultado }}</p>
{% endif %}
"""

@app.route("/", methods=["GET", "POST"])
def home():
    resultado = None
    if request.method == "POST":
        texto = request.form["texto"]
        # ideal: aplicar mesma limpeza aqui também
        vec = vetorizador.transform([texto])
        resultado = modelo.predict(vec)[0]
    return render_template_string(HTML, resultado=resultado)

if __name__ == "__main__":
    app.run(debug=True)
