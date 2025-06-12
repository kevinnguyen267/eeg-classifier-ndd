from itertools import combinations

import lightgbm as lgb
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import LeaveOneOut, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC


def find_optimal_features_and_model(groups, df):
    features = df.drop("Group", axis=1).columns

    models = {
        "LightGBM": lgb.LGBMClassifier(random_state=42),
        "SVM": SVC(kernel="poly", random_state=42),
        "KNN": KNeighborsClassifier(n_neighbors=3),
        "MLP": MLPClassifier((3,), random_state=42),
        "Random Forest": RandomForestClassifier(random_state=42),
    }

    features_models_df = pd.DataFrame().rename_axis("Features")
    y = df["Group"]
    cv = LeaveOneOut()

    # Iterate through every combination of feature subsets and perform a Leave-One-Subject-Out (LOSO) cross-validation for each combination
    for k in range(1, len(features) + 1):
        for c in combinations(features, k):
            X = df[list(c)]
            for name, model in models.items():
                scores = cross_val_score(model, X, y, cv=cv)
                print(f"Completed: {c}, {name}")
                features_models_df.loc[", ".join(c), name] = f"{scores.mean():.4f}"

    features_models_df.to_pickle(
        f"../../data/interim/02_{groups}_optimal_features_and_model.pkl"
    )

    # Find the optimal features subset and model (max accuracy)
    max_col = features_models_df.max().idxmax()
    max_row = features_models_df[max_col].idxmax()
    max_val = features_models_df.loc[max_row, max_col]
    print(
        f"Optimal Features and Model for {groups.upper()} Classification:\nFeatures: {max_row}\nModel: {max_col}\nAccuracy:{max_val}"
    )

    return (max_col, max_row, max_val)


df = pd.read_pickle("../../data/interim/01_data_processed.pkl")

ad_cn_df = df[df["Group"] != "F"]
ftd_cn_df = df[df["Group"] != "A"]

ad_cn_optimal_model, ad_cn_optimal_features, ad_cn_accuracy = (
    find_optimal_features_and_model("ad_cn", ad_cn_df)
)
ftd_cn_optimal_model, ftd_cn_optimal_features, ftd_cn_accuracy = (
    find_optimal_features_and_model("ftd_cn", ftd_cn_df)
)
