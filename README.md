# 📰 Fake News Validator  

A machine learning powered web application that validates whether a news article is **Fake** or **True**.  
The project combines **FastAPI** (backend), **Streamlit** (frontend), and a **Retrieval-Augmented Generation (RAG)** pipeline to classify news and provide supporting articles.  

---

## 🚀 Features  
- Classifies news articles as **Fake** or **True**  
- Provides **supporting articles** from a reference dataset (RAG-based retrieval)  
- Simple **web interface** built with Streamlit  
- FastAPI backend serving predictions and article retrieval  
- Modular and extensible architecture  

---

## 🛠️ Tech Stack  
- **Backend**: FastAPI, scikit-learn, pickle, pandas  
- **Frontend**: Streamlit  
- **Model**: TF-IDF + Logistic Regression (baseline, can be upgraded later)  
- **Dataset**: [Fake and real news dataset on Kaggle](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)  

---

## 📂 Project Structure  

```

Fake News Validator - 2/
│
├── backend/               # FastAPI backend
│   ├── server.py          # Main API server
│   ├── meta.pkl           # TF-IDF metadata (vectorizer)
│   ├── model.pkl          # Trained classification model
│   └── requirements.txt   # Backend dependencies
│
├── frontend/              # Streamlit frontend
│   └── app.py             # Streamlit UI
│
├── requirements.txt       # Combined project dependencies
├── Sample Queries.txt     # Example test queries
├── .gitignore
└── README.md

````

---

## ⚙️ Installation  

1. **Clone the repository**  
```bash
git clone https://github.com/azkatahir10/Fake-News-Validator.git
cd Fake-News-Validator
````

2. **Create virtual environment**

```bash
python -m venv venv
```

3. **Activate environment**

* Windows (PowerShell):

  ```bash
  venv\Scripts\activate
  ```
* Linux/Mac:

  ```bash
  source venv/bin/activate
  ```

4. **Install dependencies**

```bash
pip install -r requirements.txt
```

---

## ▶️ Running the Project

### Start the Backend (FastAPI)

From the `backend` folder:

```bash
uvicorn server:app --reload
```

Backend runs on: **[http://127.0.0.1:8000](http://127.0.0.1:8000)**

---

### Start the Frontend (Streamlit)

From the `frontend` folder:

```bash
streamlit run app.py
```

Frontend runs on: **[http://localhost:8501](http://localhost:8501)**

---

## 🧪 Example Queries

You can try queries like:

* *"Trump campaign had undisclosed contacts with Russians"*
* *"Obama secretly planned a coup in the US"*
* *"NASA announces discovery of water on Mars"*
* *"Bill Gates owns all farmland in the US"*

Check `Sample Queries.txt` for more.

---

## 📊 Dataset Credits

This project uses the **Fake and Real News Dataset** from Kaggle:
👉 [Fake and real news dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset) by *Clément Bisaillon*

* **Fake.csv** → Collection of fake news articles
* **True.csv** → Collection of true/verified news articles

Please cite the dataset if you use this project in research or academic work.

---






Do you want me to also add **screenshots of the Streamlit UI** (once you run it) to the README for GitHub presentation?
```
