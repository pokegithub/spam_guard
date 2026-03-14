import joblib
from sklearn.ensemble import VotingClassifier
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, log_loss

print("Loading vectors...")
x_tr = joblib.load('x_training_vector.pkl')
x_te = joblib.load('x_testing_vector.pkl')
y_tr = joblib.load('y_training_vector.pkl')
y_te = joblib.load('y_testing_vector.pkl')

# Chi-Square feature selection - keep top 1000 features
print("Running feature selection...")
sel   = SelectKBest(chi2, k=1000)
x_tr_s = sel.fit_transform(x_tr, y_tr)
x_te_s = sel.transform(x_te)

# Tune models on a sample (full dataset makes GridSearch very slow)
x_sample = x_tr_s[:5000]
y_sample = y_tr.iloc[:5000]

print("Tuning models...")
lr_grid  = GridSearchCV(LogisticRegression(max_iter=1000, class_weight='balanced'), {'C': [0.1, 1, 10]}, cv=5, scoring='f1', n_jobs=-1)
svm_grid = GridSearchCV(SVC(probability=True, class_weight='balanced'), {'kernel': ['linear', 'rbf'], 'C': [0.1, 1, 10]}, cv=5, scoring='f1', n_jobs=-1)
nb_grid  = GridSearchCV(MultinomialNB(), {'alpha': [0.1, 0.5, 1.0]}, cv=5, scoring='f1', n_jobs=-1)

lr_grid.fit(x_sample, y_sample)
svm_grid.fit(x_sample, y_sample)
nb_grid.fit(x_sample, y_sample)

print("LR  best params:", lr_grid.best_params_)
print("SVM best params:", svm_grid.best_params_)
print("NB  best params:", nb_grid.best_params_)

# Build and train the ensemble on the full training set
print("\nTraining ensemble...")
ensemble = VotingClassifier(
    estimators=[
        ('lr',  lr_grid.best_estimator_),
        ('svm', svm_grid.best_estimator_),
        ('nb',  nb_grid.best_estimator_)
    ],
    voting='soft'
)
ensemble.fit(x_tr_s, y_tr)

# Evaluate
y_pred = ensemble.predict(x_te_s)
y_prob = ensemble.predict_proba(x_te_s)

print("\nAccuracy:", round(accuracy_score(y_te, y_pred) * 100, 2), "%")
print("Log Loss:", round(log_loss(y_te, y_prob), 4))
print("\nClassification Report:\n", classification_report(y_te, y_pred, target_names=['Ham', 'Spam']))
print("Confusion Matrix:\n", confusion_matrix(y_te, y_pred))

joblib.dump(ensemble, 'model.pkl')
joblib.dump(sel, 'sel.pkl')
print("\nSaved model.pkl and sel.pkl")
