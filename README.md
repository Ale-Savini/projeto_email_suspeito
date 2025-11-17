# 🛡️ Sistema de Detecção de E-mails Suspeitos com Inteligência Artificial

Este projeto implementa um sistema completo de **classificação de e-mails suspeitos**, utilizando técnicas de **Processamento de Linguagem Natural (NLP)** e **Machine Learning**.  
O modelo identifica conteúdos potencialmente maliciosos, como:

- phishing  
- spam  
- golpes digitais  
- mensagens suspeitas em geral  

O sistema inclui:
- Pipeline de treinamento
- Modelo treinado com TF-IDF + Machine Learning
- API/Interface web criada com Flask
- Arquitetura organizada para estudos, testes e deploy

---

## 📌 Objetivo

O objetivo deste projeto é demonstrar como a Inteligência Artificial pode ser aplicada para **melhorar a segurança digital**, ajudando usuários e empresas a detectarem e-mails potencialmente perigosos antes que causem danos.

---

## 🧠 Tecnologias Utilizadas

### **Linguagem**
- Python 3.12

### **Bibliotecas principais**
- Pandas — leitura e manipulação do dataset  
- Scikit-learn — treinamento dos modelos (Naive Bayes e Logistic Regression)  
- TF-IDF Vectorizer — transformação do texto em vetores  
- Flask — criação da interface web  
- Joblib — salvar e carregar modelos  
- Regex e NLTK — pré-processamento de texto  

---

## 🏗️ Estrutura do Projeto
projeto_email_suspeito/
│
├── app/
│ └── app.py # Interface web Flask
│
├── src/
│ └── treinar_modelo.py # Script de treinamento do modelo
│
├── models/
│ ├── modelo_email.joblib # Modelo treinado
│ └── vetorizador.joblib # TF-IDF salvo
│
├── data/
│ └── emails.csv # Base de dados utilizada
│
└── README.md


---

## ⚙️ Como o Sistema Funciona

1. **Carregamento dos dados**  
   A base `emails.csv` contém textos rotulados como *suspeitos* ou *normais*.

2. **Pré-processamento**  
   O texto é limpo e preparado para análise.

3. **Geração dos vetores (TF-IDF)**  
   As palavras dos e-mails são convertidas em números.

4. **Treinamento do modelo**  
   Modelos como Naive Bayes ou Logistic Regression aprendem a identificar padrões que caracterizam e-mails suspeitos.

5. **Avaliação do modelo**  
   São geradas métricas como:
   - acurácia  
   - matriz de confusão  
   - precisão  
   - recall  

6. **Interface Web**  
   O usuário insere o texto de um e-mail e recebe a classificação em tempo real.

---

## ▶️ Como Executar o Projeto

### **1. Clonar o repositório**

git clone https://github.com/SEU_USUARIO/projeto_email_suspeito.git
cd projeto_email_suspeito

### **2 Criar um ambiente profissional.**

python -m venv venv

### **3 Instalar dependencias.**

pip install -r requirements.txt

### **4. Rodar o treinamento.**

cd src
python treinar_modelo.py

### **5. Rodar o servidor Flask.**
cd ../app
python app.py


Acesse no navegador:

👉http://localhost:5000

👤 Autores

Alexandre Savini                                     RA:24000563
Julio Cesar Azevedo Souza                            RA: 24001773
Gabriel Henrique dos Reis Diunizio                   RA: RA:24000541

Projeto acadêmico focado em Inteligência Artificial aplicada à Segurança da Informação.



