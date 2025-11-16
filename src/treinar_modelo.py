# treinar_modelo.py
import os
import re
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix


# Função: limpar_texto

def limpar_texto(texto):
    
    if not isinstance(texto, str):
        texto = str(texto)
    texto = texto.lower()
    texto = re.sub(r"http\S+|www\S+", " url ", texto)
    texto = re.sub(r"\S+@\S+", " email ", texto)
    texto = re.sub(r"[^a-zA-ZÀ-ÿ0-9\s]", " ", texto)  # mantem letras acentuadas
    texto = re.sub(r"\s+", " ", texto).strip()
    return texto


# Carregar dados

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "emails.csv")
print("Procurando arquivo de dados em:", DATA_PATH)
dados = pd.read_csv(DATA_PATH)


# Pré-processamento de texto

print("Tamanho inicial do dataset:", len(dados))
dados["text"] = dados["text"].apply(limpar_texto)

# Verificar distribuição de classes
print("Distribuição de rótulos:\n", dados["label"].value_counts())


# Separar X e y, dividir treino/teste

X = dados["text"]
y = dados["label"]
X_treino, X_teste, y_treino, y_teste = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


# Vetorização TF-IDF

vetorizador = TfidfVectorizer(stop_words="portuguese", ngram_range=(1,2), min_df=2)
X_treino_tfidf = vetorizador.fit_transform(X_treino)
X_teste_tfidf = vetorizador.transform(X_teste)

print("Dimensão TF-IDF (treino):", X_treino_tfidf.shape)


# Modelo 1: Naive Bayes

nb = MultinomialNB()
nb.fit(X_treino_tfidf, y_treino)
pred_nb = nb.predict(X_teste_tfidf)
print("\n--- Naive Bayes ---")
print("Acurácia NB:", accuracy_score(y_teste, pred_nb))
print(classification_report(y_teste, pred_nb))
print("Matriz de confusão:\n", confusion_matrix(y_teste, pred_nb))


# Modelo 2: Logistic Regression + GridSearch

parametros = {
    "C": [0.1, 1, 10],
    "solver": ["liblinear", "lbfgs"]
}
grid = GridSearchCV(LogisticRegression(max_iter=1000), parametros, cv=3, scoring="f1_macro", n_jobs=-1)
grid.fit(X_treino_tfidf, y_treino)

melhor_lr = grid.best_estimator_
pred_lr = melhor_lr.predict(X_teste_tfidf)

print("\n--- Logistic Regression (GridSearch) ---")
print("Melhores parâmetros:", grid.best_params_)
print("Acurácia LR:", accuracy_score(y_teste, pred_lr))
print(classification_report(y_teste, pred_lr))
print("Matriz de confusão:\n", confusion_matrix(y_teste, pred_lr))


# Escolher melhor modelo (simples regra: F1 macro a favor do LR ou NB)

# Aqui uso f1 macro do relatório para decidir 
report_lr = classification_report(y_teste, pred_lr, output_dict=True)
report_nb = classification_report(y_teste, pred_nb, output_dict=True)

f1_lr = report_lr["macro avg"]["f1-score"]
f1_nb = report_nb["macro avg"]["f1-score"]

if f1_lr >= f1_nb:
    melhor_modelo = melhor_lr
    print("\nEscolhido: Logistic Regression (melhor F1_macro: {:.3f})".format(f1_lr))
else:
    melhor_modelo = nb
    print("\nEscolhido: Naive Bayes (melhor F1_macro: {:.3f})".format(f1_nb))


# Salvar modelo e vetorizador

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
os.makedirs(MODEL_DIR, exist_ok=True)
joblib.dump(melhor_modelo, os.path.join(MODEL_DIR, "modelo_email.joblib"))
joblib.dump(vetorizador, os.path.join(MODEL_DIR, "vetorizador.joblib"))
print("Modelo e vetorizador salvos em:", MODEL_DIR)
