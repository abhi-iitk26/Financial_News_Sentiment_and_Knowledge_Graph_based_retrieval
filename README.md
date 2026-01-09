# Financial News Sentiment Analysis & Knowledge Graph

Problem Statement:
Financial news contains valuable insights that can influence investment and business decisions, but manually extracting sentiment and understanding relationships between entities is time-consuming. 
This project automates sentiment classification and builds an interactive Knowledge Graph to efficiently explore entities and their related news.

This project combines **financial news sentiment analysis** with **graph-based information retrieval**, providing an interactive system for exploring relationships between entities and related news records.

---

## 🔹 Features

- **Exploratory Data Analysis & Sentiment Classification:**  
  - Performed EDA on **5,842 financial news records**.  
  - Classified news as **positive, negative, or neutral** using **NLTK** for preprocessing and **TF-IDF** vectorization.  
  - Trained models like **Logistic Regression**, **Linear SVC**, **Random Forest**, and **Naive Bayes** with **Stratified K-Fold cross-validation**, achieving **70% accuracy** and **0.71 F1-score** (best with Logistic Regression).

- **Deep Learning Sentiment Classifiers:**  
  - Built **LSTM, Bi-LSTM, and GRU** classifiers using **Word2Vec embeddings (gensim)**.  
  - Incorporated **Dropout layers** and **Keras callbacks** to prevent overfitting.  
  - **Bi-LSTM achieved the best performance: 72% accuracy.**

- **Named Entity Recognition (NER) & Knowledge Graph:**  
  - Extracted entities using **spaCy**.  
  - Constructed a **Knowledge Graph** in a **Dockerized Neo4j** container.  
  - Interactive **Streamlit frontend** allows users to query entities: for each query, the relevant subgraph is generated using **Cypher**, and corresponding news records are retrieved.

---

## 🔹 Tech Stack

- **Python**, **NLTK**, **TextBlob**, **spaCy**  
- **Gensim (Word2Vec)** for embeddings  
- **Neo4j + Cypher** for graph database  
- **Docker** for Neo4j containerization  
- **Streamlit** for interactive frontend  
- **PyVis** for graph visualization  
- **Pandas**, **NumPy** for data handling  

---
## 🔹 Project Structure
```
Project Structure
│
├── data.csv
│   └── Original financial news dataset
│
├── corpus_with_entities.csv
│   └── Dataset with extracted named entities
│
├── sentiment_analysis_ml.ipynb
│   └── ML models: Logistic Regression, SVC, etc.
│
├── sentiment_analysis_dl.ipynb
│   └── Deep Learning models: LSTM, Bi-LSTM, GRU
│
├── ner_extraction.py
│   └── Named Entity Recognition using spaCy
│
├── kg_builder.py
│   └── Knowledge Graph construction in Neo4j
│
├── kg_frontend.py
│   └── Streamlit frontend for querying & visualization
│
├── requirements.txt
│   └── Python dependencies
│
├── README.md
│   └── Project documentation
│
└── lib/
    └── Frontend libraries
    │
    ├── bindings/
    │   └── utils.js
    │
    ├── tom-select/
    │   ├── tom-select.complete.min.js
    │   └── tom-select.css
    │
    └── vis-9.1.2/
        ├── vis-network.min.js
        └── vis-network.css
```

---

⭐ **Star this repository if you found it useful!**
