import joblib
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, log_loss
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.ensemble import VotingClassifier

# 1. Load the Decoupled Data
print("Loading pre-vectorized matrices...")
x_tr_v = joblib.load('x_training_vector.pkl')
x_te_v = joblib.load('x_testing_vector.pkl')
y_tr = joblib.load('y_training_vector.pkl')
y_te = joblib.load('y_testing_vector.pkl')

# 2. Dimensionality Reduction (Chi-Square)
print("Running Chi-Square Feature Selection...")
sel = SelectKBest(chi2, k=1000) 
x_tr_s = sel.fit_transform(x_tr_v, y_tr)
x_te_s = sel.transform(x_te_v)

# 3. Model Tuning Tournament 
models = [
    {'name': 'LR', 'est': LogisticRegression(max_iter=1000), 'grid': {'C': [0.1, 1, 10]}},
    {'name': 'SVM', 'est': SVC(probability=True), 'grid': {'kernel': ['linear', 'rbf'], 'C': [0.1, 1, 10]}}
]

best_mdls = {}
x_samp, y_samp = x_tr_s[:2000], y_tr[:2000]

print("Optimizing models...")
for m in models:
    grid = GridSearchCV(m['est'], m['grid'], cv=2, n_jobs=-1)
    grid.fit(x_samp, y_samp)
    best_mdls[m['name']] = grid.best_estimator_

# 4. Ensemble Training
print("Training Ensemble Classifier...")
ensemble = VotingClassifier(
    estimators=[('lr', best_mdls['LR']), ('svm', best_mdls['SVM'])],
    voting='soft'
)

ensemble.fit(x_tr_s, y_tr)

# 5. Evaluation
y_pred = ensemble.predict(x_te_s)
y_prob = ensemble.predict_proba(x_te_s) 

acc = accuracy_score(y_te, y_pred)
loss = log_loss(y_te, y_prob)

print(f"\nAccuracy: {acc * 100:.2f}%")
print(f"Log Loss: {loss:.4f}")
print(f"Confusion Matrix:\n{confusion_matrix(y_te, y_pred)}")

# 6. Save Final Artifacts
joblib.dump(ensemble, 'model.pkl')
joblib.dump(sel, 'sel.pkl')
print("Saved final model.pkl and sel.pkl!")
