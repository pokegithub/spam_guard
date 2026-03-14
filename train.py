import os
import logging
import joblib

from sklearn.ensemble import VotingClassifier
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
log = logging.getLogger(__name__)

BASE        = os.path.dirname(__file__)
X_TRAIN_PKL = os.path.join(BASE, 'x_training_vector.pkl')
X_TEST_PKL  = os.path.join(BASE, 'x_testing_vector.pkl')
Y_TRAIN_PKL = os.path.join(BASE, 'y_training_vector.pkl')
Y_TEST_PKL  = os.path.join(BASE, 'y_testing_vector.pkl')
MODEL_PKL   = os.path.join(BASE, 'model.pkl')
SEL_PKL     = os.path.join(BASE, 'sel.pkl')

# Number of top TF-IDF features to keep after Chi-Square selection
K_BEST = 1000

# Fraction of training data used for hyperparameter tuning.
# GridSearchCV on the full dataset with an SVM is very slow, so we tune on a
# representative sample and then refit on all data with the best params.
# If runtime is not a concern, set TUNE_SAMPLE_SIZE = None to use everything.
TUNE_SAMPLE_SIZE = 5000


def load_artefacts():
    for path in (X_TRAIN_PKL, X_TEST_PKL, Y_TRAIN_PKL, Y_TEST_PKL):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Vector file not found: {path}. Run data_vectorizer.py first.")

    log.info("Loading pre-vectorized matrices...")
    x_tr = joblib.load(X_TRAIN_PKL)
    x_te = joblib.load(X_TEST_PKL)
    y_tr = joblib.load(Y_TRAIN_PKL)
    y_te = joblib.load(Y_TEST_PKL)
    return x_tr, x_te, y_tr, y_te


def select_features(x_tr, x_te, y_tr):
    log.info("Running Chi-Square feature selection (k=%d)...", K_BEST)
    sel = SelectKBest(chi2, k=K_BEST)
    x_tr_s = sel.fit_transform(x_tr, y_tr)
    x_te_s = sel.transform(x_te)
    log.info("Feature matrix reduced to shape: %s", x_tr_s.shape)
    return x_tr_s, x_te_s, sel


def tune_model(name, estimator, param_grid, x_tune, y_tune):
    log.info("Tuning %s on %d samples (cv=5)...", name, len(y_tune))
    grid = GridSearchCV(estimator, param_grid, cv=5, n_jobs=-1, scoring='f1')
    grid.fit(x_tune, y_tune)
    log.info("%s best params: %s | best F1: %.4f", name, grid.best_params_, grid.best_score_)
    return grid.best_estimator_


def build_ensemble(x_tr_s, y_tr):
    if TUNE_SAMPLE_SIZE and TUNE_SAMPLE_SIZE < x_tr_s.shape[0]:
        x_tune = x_tr_s[:TUNE_SAMPLE_SIZE]
        y_tune = y_tr.iloc[:TUNE_SAMPLE_SIZE]
    else:
        x_tune, y_tune = x_tr_s, y_tr

    model_specs = [
        {
            'name': 'LR',
            'est':  LogisticRegression(max_iter=1000, class_weight='balanced'),
            'grid': {'C': [0.1, 1, 10]},
        },
        {
            'name': 'SVM',
            # class_weight='balanced' adjusts for spam/ham imbalance automatically
            'est':  SVC(probability=True, class_weight='balanced'),
            'grid': {'kernel': ['linear', 'rbf'], 'C': [0.1, 1, 10]},
        },
        {
            # MultinomialNB is classically strong on TF-IDF spam data.
            # It has no class_weight param; class_prior implicitly handles balance.
            'name': 'NB',
            'est':  MultinomialNB(),
            'grid': {'alpha': [0.1, 0.5, 1.0]},
        },
    ]

    best_models = {}
    for spec in model_specs:
        best_models[spec['name']] = tune_model(
            spec['name'], spec['est'], spec['grid'], x_tune, y_tune
        )

    log.info("Training ensemble on full training set...")
    ensemble = VotingClassifier(
        estimators=[
            ('lr',  best_models['LR']),
            ('svm', best_models['SVM']),
            ('nb',  best_models['NB']),
        ],
        voting='soft'
    )
    ensemble.fit(x_tr_s, y_tr)
    return ensemble


def evaluate(ensemble, x_te_s, y_te) -> None:
    y_pred = ensemble.predict(x_te_s)
    y_prob = ensemble.predict_proba(x_te_s)

    acc    = accuracy_score(y_te, y_pred)
    loss   = log_loss(y_te, y_prob)
    auc    = roc_auc_score(y_te, y_prob[:, 1])
    cm     = confusion_matrix(y_te, y_pred)
    report = classification_report(y_te, y_pred, target_names=['Ham', 'Spam'])

    log.info("\n" + "=" * 50)
    log.info("EVALUATION RESULTS")
    log.info("=" * 50)
    log.info("Accuracy : %.4f", acc)
    log.info("Log Loss : %.4f", loss)
    log.info("ROC-AUC  : %.4f", auc)
    log.info("Confusion Matrix:\n%s", cm)
    log.info("\nClassification Report:\n%s", report)

    tn, fp, fn, tp = cm.ravel()
    log.info("False Positive Rate (ham marked as spam): %.4f", fp / (fp + tn))
    log.info("False Negative Rate (spam missed):        %.4f", fn / (fn + tp))


def main() -> None:
    x_tr, x_te, y_tr, y_te = load_artefacts()
    x_tr_s, x_te_s, sel    = select_features(x_tr, x_te, y_tr)
    ensemble                = build_ensemble(x_tr_s, y_tr)

    evaluate(ensemble, x_te_s, y_te)

    joblib.dump(ensemble, MODEL_PKL)
    joblib.dump(sel,      SEL_PKL)
    log.info("Saved model to: %s", MODEL_PKL)
    log.info("Saved selector to: %s", SEL_PKL)


if __name__ == '__main__':
    main()
    
