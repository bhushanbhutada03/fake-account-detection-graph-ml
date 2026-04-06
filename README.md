#  Fake Account Detection in Social Media
### Using Graph Analytics + Machine Learning | BDA Mini Project

![Python](https://img.shields.io/badge/Python-3.8+-blue?style=flat-square&logo=python)
![NetworkX](https://img.shields.io/badge/NetworkX-Graph%20Analytics-orange?style=flat-square)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML%20Models-red?style=flat-square)
![Accuracy](https://img.shields.io/badge/ML%20Accuracy-98.2%25-brightgreen?style=flat-square)
![Graph Accuracy](https://img.shields.io/badge/Graph%20Accuracy-87%25-yellow?style=flat-square)

---

##  Problem Statement

Social media platforms are flooded with **fake accounts (bots)** that:
- Spread misinformation
- Manipulate trends
- Inflate engagement artificially

Manual detection is impossible at scale.

This project builds an **automated detection system** using:
- **Graph Analytics (main approach)**
- **Machine Learning (enhancement)**

---

##  Project Goal

> Detect fake accounts using graph structure first, then improve accuracy using machine learning.

---

## 📁 Project Structure

```
fake_account_project/
│
├── data/
│   └── dataset.csv
│
├── generate_data.py
├── graph_build.py
├── feature_engineering.py
├── model.py
└── main.py
```

##  Dataset

- 5000 users (2500 bots, 2500 real)
- Synthetic but realistic behaviour

### Bot Behaviour:
- High retweets & mentions  
- Low followers  
- New accounts  

### Real Users:
- Moderate activity  
- Higher followers  
- Older accounts  

---

##  Graph Construction

- Each user = Node  
- Interaction = Edge  

Edges are created using:
- Retweet behaviour  
- Mention behaviour  

---

##  Graph Analytics

| Metric | Meaning | Bot Pattern |
|--------|--------|------------|
| PageRank | Importance of user | Low |
| In-degree | Incoming links | Low |
| Out-degree | Outgoing links | High |
| Clustering | Community connectivity | Low |

---

##  Graph-Based Detection

graph_score = follower_retweet_ratio  
            + pagerank  
            - mention_retweet_ratio  

- Low score → Bot  
- High score → Real user  

Accuracy: ~87%

---

##  Community Analysis

- 1500+ communities detected  
- Many isolated nodes  

Bots are mostly isolated  
Real users form connected clusters  

---

##  Feature Engineering

- follower_retweet_ratio  
- mention_retweet_ratio  
- account_age_days  
- hashtags presence  

---

##  Machine Learning Models

- Random Forest  
- Gradient Boosting  
- Logistic Regression  

---

##  Results

| Method | Accuracy |
|--------|---------|
| Graph Detection | ~87% |
| Random Forest | ~98% |
| Gradient Boosting | ~98% |
| Logistic Regression | ~97% |

Graph gives structure  
ML improves accuracy  

---

##  How to Run

pip install pandas networkx scikit-learn matplotlib  

python generate_data.py  
python main.py  

---

##  Limitations

- Synthetic dataset  
- Approximate graph  
- No temporal behaviour analysis  

---

##  Future Work

- Real Twitter API data  
- NLP features  
- Temporal analysis  
- Real-time detection  

---

## 🛠️ Tech Stack

- Python  
- NetworkX  
- scikit-learn  
- pandas  
- matplotlib  

---
 
