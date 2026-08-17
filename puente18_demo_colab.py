
# PUENTE 18+ — Mini laboratorio de ML
# Ejecutable en Google Colab. Todo es sintético.
# No usar datos identificables ni historias clínicas reales.

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, classification_report

rng = np.random.default_rng(18)
n = 1600

df = pd.DataFrame({
    "age_years": rng.choice([16,17,18], n, p=[.3,.5,.2]),
    "months_to_18": rng.uniform(0,24,n),
    "complexity": rng.choice(["Baja","Media","Alta"], n, p=[.1,.45,.45]),
    "readiness_score": rng.integers(20,97,n),
    "caregiver_support": rng.integers(25,100,n),
    "no_shows_last_12m": rng.poisson(.8,n),
    "adult_receiver_identified": rng.choice([0,1], n, p=[.48,.52]),
    "appointment_confirmed": rng.choice([0,1], n, p=[.42,.58]),
    "summary_complete": rng.choice([0,1], n, p=[.28,.72]),
})

# Etiqueta sintética para probar el pipeline.
lin = (
    0.08*(df["complexity"]=="Alta").astype(int)
    + .95*(1-df["adult_receiver_identified"])
    + .78*(1-df["appointment_confirmed"])
    + .56*(1-df["summary_complete"])
    + .055*(50-df["readiness_score"]).clip(lower=0)
    + .055*(df["no_shows_last_12m"].clip(upper=4))
    + .55*(df["months_to_18"]<2).astype(int)
    + rng.normal(0,.45,n)
)
prob = 1/(1+np.exp(-lin+1.1))
df["transition_interruption"] = rng.binomial(1, np.clip(prob,.04,.96))

X = df.drop(columns="transition_interruption")
y = df["transition_interruption"]

cat = ["complexity"]
num = [c for c in X.columns if c not in cat]

pre = ColumnTransformer([
    ("num", StandardScaler(), num),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat)
])

pipe = Pipeline([
    ("pre", pre),
    ("clf", LogisticRegression(max_iter=1000, class_weight="balanced"))
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=.25, random_state=18, stratify=y
)

pipe.fit(X_train, y_train)
pred = pipe.predict_proba(X_test)[:,1]

print("AUC sintético:", round(roc_auc_score(y_test,pred),3))
print(classification_report(y_test, (pred>=.5).astype(int)))
