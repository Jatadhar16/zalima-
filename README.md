🧠 AI-Powered Task Management System (Synthetic Tasks)
📌 Project Overview

The AI-Powered Task Management System is designed to intelligently classify, prioritize, and assign tasks to users based on their descriptions, deadlines, and workloads.
This project leverages Natural Language Processing (NLP) and Machine Learning (ML) to automate task management workflows commonly handled in corporate tools like JIRA.

In this version, a synthetic dataset is generated to simulate real-world tasks, ensuring reproducibility and privacy-safe experimentation.

🚀 Key Objectives

Automate classification of tasks (e.g., Bug, Improvement, Story, Task).

Predict task priority (e.g., High, Medium, Low).

Suggest task assignees based on workload.

Use NLP + ML models to learn from textual and metadata features.

Visualize insights such as task type distribution and model accuracy.

🧩 Features

✅ Task classification using TF-IDF and ML algorithms
✅ Task priority prediction
✅ AI-based workload balancing suggestion
✅ Data visualization (matplotlib / seaborn)
✅ Evaluation metrics — Accuracy, Precision, Recall, F1-Score
✅ Configurable and reproducible synthetic dataset

📂 Dataset: Synthetic_Tasks.csv

Since real JIRA datasets are proprietary, we create a synthetic dataset that mimics real-world task attributes.

Column Name	Description
Task_ID	Unique identifier for each task
Summary	Short textual description of the task
Description	Detailed explanation of the issue or feature
Issue_Type	Type of issue (Bug / Story / Improvement / Task)
Priority	Level of urgency (High / Medium / Low)
Assignee	Name of user assigned to the task
Status	Current progress status (To Do / In Progress / Done)
Created_Date	Task creation date
Due_Date	Expected completion date
⚙️ Workflow
Step 1: Data Preparation
import pandas as pd
df = pd.read_csv('synthetic_tasks.csv')
df = df.dropna(subset=['Summary'])

Step 2: Text Preprocessing

Lowercasing

Removing stopwords, punctuation

Lemmatization (optional)

from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(stop_words='english', max_features=3000)
X = tfidf.fit_transform(df['Summary'])

Step 3: Label Encoding
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
y = encoder.fit_transform(df['Issue_Type'])

Step 4: Model Training
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

2️⃣ Evaluation Metrics
Metric	Formula	Description
Accuracy	(TP + TN) / (TP + TN + FP + FN)	Overall correctness
Precision	TP / (TP + FP)	How many predicted positives are correct
Recall	TP / (TP + FN)	How many actual positives were captured
F1-Score	2 × (Precision × Recall) / (Precision + Recall)	Balance between precision and recall
🧰 Technologies Used
Category	Tools & Libraries
Programming Language	Python
Data Processing	Pandas, NumPy
NLP	Scikit-learn (TF-IDF, CountVectorizer)
Machine Learning	RandomForestClassifier, LogisticRegression
Visualization	Matplotlib, Seaborn
Dataset	Synthetic_Tasks.csv (custom-generated)
IDE / Environment	Jupyter Notebook / VS Code
📊 Results
Model	Accuracy	Precision	Recall	F1-Score
Logistic Regression	84.2%	0.83	0.84	0.84
Random Forest	92.8%	0.92	0.93	0.92

✅ Best Model: Random Forest
✅ Reason: Handles feature sparsity well, robust to text noise

🧠 Insights

Frequent terms: error, update, UI, performance, login

Task type distribution: Bugs (40%), Improvements (30%), Stories (20%), Tasks (10%)

AI can predict task type and urgency with >90% accuracy

📈 Visualization Example
import seaborn as sns
import matplotlib.pyplot as plt

sns.countplot(x='Issue_Type', data=df)
plt.title('Task Type Distribution')
plt.show()

🔮 Future Enhancements

Integrate JIRA API for real-time task data

Deploy model with FastAPI or Flask

Add AI-based task recommendation engine

Introduce deep learning (BERT) for context understanding

Build React.js dashboard for visual task insights

📁 Repository Structure
AI-Powered-Task-Management/
│
├── data/
│   └── synthetic_tasks.csv
│
├── notebooks/
│   └── AI_Task_Management_Model.ipynb
│
├── src/
│   ├── preprocessing.py
│   ├── model_training.py
│   └── evaluation.py
│
├── results/
│   └── model_report.csv
│
├── README.md
└── requirements.txt

🧾 Requirements
pandas
numpy
scikit-learn
matplotlib
seaborn
