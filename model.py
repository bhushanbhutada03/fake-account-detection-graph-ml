"""
Trains & compares 3 classifiers. Prints full metrics + feature importance plot.

Features used (matching actual dataset columns + engineered):
  Profile : Retweet Count, Mention Count, Follower Count,
            Verified, account_age_days, has_hashtags
  Graph   : pagerank, in_degree, out_degree, clustering
  Engineered: follower_retweet_ratio, mention_retweet_ratio
Label: Bot Label (0 = Real, 1 = Bot)
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, confusion_matrix,
    classification_report, roc_auc_score
)

FEATURES = [
    # Profile features (from dataset)
    'Retweet Count', 'Mention Count', 'Follower Count',
    'Verified', 'account_age_days', 'has_hashtags',
    # Graph features
    'pagerank', 'in_degree', 'out_degree', 'clustering',
    # Engineered features
    'follower_retweet_ratio', 'mention_retweet_ratio',
]

LABEL = 'Bot Label'


def train_models(df: pd.DataFrame):
    X = df[FEATURES].fillna(0)
    y = df[LABEL]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    models = {
        "Random Forest":       RandomForestClassifier(n_estimators=100, random_state=42),
        "Gradient Boosting":   GradientBoostingClassifier(n_estimators=100, random_state=42),
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    }

    results = {} 
    print("\n" + "="*60)
    print("         MODEL COMPARISON RESULTS")
    print("="*60)

    for name, clf in models.items():
        clf.fit(X_train, y_train)
        preds = clf.predict(X_test)
        proba = clf.predict_proba(X_test)[:, 1]

        acc  = accuracy_score(y_test, preds)
        auc  = roc_auc_score(y_test, proba)
        cv   = cross_val_score(clf, X_scaled, y, cv=5, scoring='accuracy').mean()

        print(f"\n── {name} ──")
        print(f"  Accuracy      : {acc:.4f}")
        print(f"  ROC-AUC Score : {auc:.4f}")
        print(f"  5-Fold CV Acc : {cv:.4f}")
        print(f"  Confusion Matrix:\n{confusion_matrix(y_test, preds)}")
        print(classification_report(y_test, preds, target_names=['Real User', 'Bot']))

        results[name] = {'model': clf, 'acc': acc, 'auc': auc}

    # Feature Importance Plot
    rf = results["Random Forest"]['model']
    importances = pd.Series(rf.feature_importances_, index=FEATURES).sort_values()

    plt.figure(figsize=(9, 5))
    importances.plot(kind='barh', color='steelblue')
    plt.title('Feature Importance — Random Forest', fontsize=13)
    plt.xlabel('Importance Score')
    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=150)
    plt.show()
    print("[model] Saved → feature_importance.png")

    best = max(results, key=lambda k: results[k]['auc'])
    print(f"\n✅ Best Model: {best} (AUC = {results[best]['auc']:.4f})")
    return results[best]['model']

