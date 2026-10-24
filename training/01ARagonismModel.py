Python 3.11.6 (tags/v3.11.6:8b6ee5b, Oct  2 2023, 14:57:12) [MSC v.1935 64 bit (AMD64)] on win32
Type "help", "copyright", "credits" or "license()" for more information.
>>> # ========================================================
... # 1-Predictive Modeling of Androgen Receptor (AR) Agonism
... # ========================================================
... 
... import warnings
... import pandas as pd
... import numpy as np
... import random
... import os
... from typing import Tuple, Dict, Any, Optional
... 
... # RDKit for cheminformatics
... from rdkit import Chem
... from rdkit.Chem import Descriptors, AllChem, MACCSkeys
... 
... # Scikit-learn for machine learning modeling and metrics
... from sklearn.model_selection import GridSearchCV, train_test_split
... from sklearn.preprocessing import MinMaxScaler
... from sklearn.feature_selection import VarianceThreshold
... from sklearn.ensemble import RandomForestClassifier
... from sklearn.svm import SVC
... from sklearn.decomposition import PCA
... from sklearn.metrics import (
...     roc_auc_score, matthews_corrcoef, recall_score,
...     accuracy_score, precision_score, f1_score,
...     roc_curve, confusion_matrix
... )
... from sklearn.neighbors import NearestNeighbors
... 
... # Boruta for feature selection
... from boruta import BorutaPy
... from imblearn.ensemble import BalancedRandomForestClassifier
... 
... import matplotlib.pyplot as plt
... import seaborn as sns
... 
... 
... def set_global_seed(seed: int = 42) -> None:
    """Fixes all random seeds to ensure reproducibility."""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)


# Suppress warnings for cleaner console output during model training
warnings.filterwarnings('ignore')


class FeatureGenerator:
    """
    Extracts comprehensive molecular features from SMILES strings.
    """
    def get_features(self, smiles: str) -> Optional[np.ndarray]:
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None

            # 1. Morgan Fingerprints (Radius=2) - 2048 bits
            morgan = np.array(AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048))
            # 2. MACCS Keys
            maccs = np.array(MACCSkeys.GenMACCSKeys(mol))
            # 3. Physicochemical Descriptors
            desc = np.array([
                Descriptors.MolWt(mol),
                Descriptors.MolLogP(mol),
                Descriptors.TPSA(mol),
                Descriptors.NumHDonors(mol),
                Descriptors.NumHAcceptors(mol),
                Descriptors.NumRotatableBonds(mol)
            ])
            return np.concatenate([morgan, maccs, desc])
        except Exception:
            return None


def get_model_config(algo: str) -> Tuple[Any, Dict]:
    """Returns the base estimator and hyperparameter grid for tuning."""
    if algo == 'BRF':
        model = BalancedRandomForestClassifier(
            random_state=42,
            n_jobs=-1,
            sampling_strategy='auto'
        )
        params = {
            'n_estimators': [100, 200, 300],
            'max_features': ['sqrt'],
            'max_depth': [10, 20, None]
        }
    elif algo == 'SVM':
        model = SVC(
            probability=True,
            random_state=42,
            kernel='rbf',
            class_weight='balanced'
        )
        params = {
            'C': [0.01, 0.1, 1, 10, 20],
            'gamma': ['scale', 0.01, 0.1]
        }
    else:
        raise ValueError("Unsupported algorithm selected.")
    return model, params


def safe_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """
    Safely computes AUC.
    Returns np.nan if only one class is present in y_true.
    """
    unique_classes = np.unique(y_true)
    if len(unique_classes) < 2:
        return np.nan
    return roc_auc_score(y_true, y_prob)


def safe_specificity(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Safely computes specificity = recall of negative class.
    Returns np.nan if no negative samples are present.
    """
    unique_classes = np.unique(y_true)
    if 0 not in unique_classes:
        return np.nan
    return recall_score(y_true, y_pred, pos_label=0)


def safe_mcc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Safely computes MCC.
    If only one class exists in y_true, MCC is not informative -> np.nan.
    """
    unique_classes = np.unique(y_true)
    if len(unique_classes) < 2:
        return np.nan
    return matthews_corrcoef(y_true, y_pred)


def safe_sensitivity(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Safely computes sensitivity = recall of positive class.
    Returns np.nan if no positive samples are present.
    """
    unique_classes = np.unique(y_true)
    if 1 not in unique_classes:
        return np.nan
    return recall_score(y_true, y_pred)


def calculate_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
    dataset_type: str
) -> Dict[str, Any]:
    return {
        'Model': model_name,
        'Dataset': dataset_type,
        'N_Samples': len(y_true),
        'N_Active': int(np.sum(y_true == 1)),
        'N_Inactive': int(np.sum(y_true == 0)),
        'AUC': safe_auc(y_true, y_prob),
        'MCC': safe_mcc(y_true, y_pred),
        'Accuracy': accuracy_score(y_true, y_pred) if len(y_true) > 0 else np.nan,
        'Precision': precision_score(y_true, y_pred, zero_division=0) if len(y_true) > 0 else np.nan,
        'Sensitivity': safe_sensitivity(y_true, y_pred),
        'Specificity': safe_specificity(y_true, y_pred),
        'F1_Score': f1_score(y_true, y_pred, zero_division=0) if len(y_true) > 0 else np.nan
    }


# =====================================================================
# Applicability Domain (AD) functions
# =====================================================================

def build_knn_applicability_domain(
    X_train: np.ndarray,
    k: int = 5,
    percentile: float = 95.0
) -> Dict[str, Any]:

    n_train = X_train.shape[0]
    k_eff = min(k, max(1, n_train - 1))

    nn = NearestNeighbors(n_neighbors=k_eff + 1, metric='euclidean', n_jobs=-1)
    nn.fit(X_train)

    distances_train, _ = nn.kneighbors(X_train)
    # remove self-distance in column 0
    mean_train_dist = distances_train[:, 1:k_eff + 1].mean(axis=1)

    threshold = np.percentile(mean_train_dist, percentile)

    return {
        'nn_model': nn,
        'k': k_eff,
        'percentile': percentile,
        'threshold': threshold,
        'train_mean_knn_dist': mean_train_dist
    }


def predict_applicability_domain(
    X_query: np.ndarray,
    ad_model: Dict[str, Any]
) -> Tuple[np.ndarray, np.ndarray]:

    nn = ad_model['nn_model']
    k = ad_model['k']
    threshold = ad_model['threshold']

    distances_query, _ = nn.kneighbors(X_query, n_neighbors=k)
    mean_query_dist = distances_query.mean(axis=1)
    in_ad_mask = mean_query_dist <= threshold

    return mean_query_dist, in_ad_mask


def calculate_subset_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    y_pred: np.ndarray,
    mask: np.ndarray,
    model_name: str,
    subset_name: str
) -> Dict[str, Any]:
    """
    Calculates metrics for a masked subset (e.g., AD_in, AD_out).
    """
    if mask.sum() == 0:
        return {
            'Model': model_name,
            'Dataset': subset_name,
            'N_Samples': 0,
            'N_Active': 0,
            'N_Inactive': 0,
            'AUC': np.nan,
            'MCC': np.nan,
            'Accuracy': np.nan,
            'Precision': np.nan,
            'Sensitivity': np.nan,
            'Specificity': np.nan,
            'F1_Score': np.nan
        }

    return calculate_metrics(
        y_true=y_true[mask],
        y_prob=y_prob[mask],
        y_pred=y_pred[mask],
        model_name=model_name,
        dataset_type=subset_name
    )


# =====================================================================
# Plotting
# =====================================================================

def plot_and_save_roc_curve(
    roc_data_dict: Dict[str, Dict],
    filename: str = 'Figure_1_ROC_Curve.png'
) -> None:
    plt.figure(figsize=(8, 6))
    colors = {'BRF': '#1f77b4', 'SVM': '#ff7f0e'}

    for algo, data in roc_data_dict.items():
        if np.isnan(data['auc']):
            label = f"{algo} (AUC = NA)"
        else:
            label = f"{algo} (AUC = {data['auc']:.4f})"

        plt.plot(
            data['fpr'], data['tpr'],
            color=colors.get(algo, 'green'),
            lw=2.5,
            label=label
        )

    plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--', label='Random Guess')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=13)
    plt.ylabel('True Positive Rate', fontsize=13)
    plt.title('ROC Curve - AR agonism', fontsize=15)
    plt.legend(loc="lower right", fontsize=12)
    plt.grid(alpha=0.3, linestyle='--')
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()


def plot_and_save_confusion_matrices(
    y_test: np.ndarray,
    predictions_dict: Dict[str, np.ndarray],
    filename: str = 'Figure_2_Confusion_Matrices.png'
) -> None:
    """Plots confusion matrices for all algorithms side-by-side."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for i, (algo, y_pred) in enumerate(predictions_dict.items()):
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues', ax=axes[i], cbar=False,
            xticklabels=['Inactive (0)', 'Active (1)'],
            yticklabels=['Inactive (0)', 'Active (1)'],
            annot_kws={"size": 14, "weight": "bold"}
        )
        axes[i].set_title(f'{algo} Confusion Matrix', fontsize=14, fontweight='bold')
        axes[i].set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
        axes[i].set_ylabel('True Label', fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()


def plot_and_save_applicability_domain(
    X_train: np.ndarray,
    X_test: np.ndarray,
    test_in_ad_mask: np.ndarray,
    filename: str = 'Figure_3_Applicability_Domain.png'
) -> None:
    """
    Plots the Applicability Domain using PCA of the chemical space.
    Training set is shown as reference chemical space,
    while test set is split into AD-in and AD-out.
    """
    pca = PCA(n_components=2, random_state=42)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)

    plt.figure(figsize=(9, 7))

    plt.scatter(
        X_train_pca[:, 0], X_train_pca[:, 1],
        alpha=0.35, c='#2ca02c',
        label='Training Set',
        edgecolors='w', s=60
    )

    if np.sum(test_in_ad_mask) > 0:
        plt.scatter(
            X_test_pca[test_in_ad_mask, 0], X_test_pca[test_in_ad_mask, 1],
            alpha=0.85, c='#1f77b4', marker='^',
            label='Test Set (AD in)',
            edgecolors='w', s=80
        )

    if np.sum(~test_in_ad_mask) > 0:
        plt.scatter(
            X_test_pca[~test_in_ad_mask, 0], X_test_pca[~test_in_ad_mask, 1],
            alpha=0.95, c='#d62728', marker='X',
            label='Test Set (AD out)',
            edgecolors='w', s=100
        )

    var_1, var_2 = pca.explained_variance_ratio_ * 100
    plt.xlabel(f'Principal Component 1 ({var_1:.1f}%)', fontsize=13, fontweight='bold')
    plt.ylabel(f'Principal Component 2 ({var_2:.1f}%)', fontsize=13, fontweight='bold')
    plt.title('Applicability Domain Analysis (Chemical Space)', fontsize=15, fontweight='bold')
    plt.legend(loc='best', fontsize=11)
    plt.grid(alpha=0.3, linestyle='--')

    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()


# =====================================================================
# Main pipeline
# =====================================================================
def run_modeling_pipeline(cmp_path: str, end_path: str) -> None:
    set_global_seed(42)

    print("=" * 85)
    print("Initializing QSAR Modeling Pipeline for AR Agonism Prediction")
    print("=" * 85)

    # 1. Data Loading
    print("[1/6] Loading datasets and defining activity labels...")
    df_end = pd.read_csv(end_path)
    df_cmp = pd.read_csv(cmp_path)[['CID', 'SMILES']]
    df = pd.merge(df_end, df_cmp, on='CID')

    # -----------------------------------------------------------------
    # Use pre-defined labels from ARagonism instead of generating labels
    # from AC50
    # ARagonism: 0 = inactive, 1 = active, NA = remove
    # -----------------------------------------------------------------
    if 'ARagonism' not in df.columns:
        raise KeyError("Column 'ARagonism' not found in endpoint file.")

    # Remove missing labels
    df = df[df['ARagonism'].notna()].copy()

    # Keep only valid binary labels: 0 and 1
    df = df[df['ARagonism'].isin([0, 1])].copy()

    # Assign final modeling label
    df['True_Label'] = df['ARagonism'].astype(int)

    print(f"      Total compounds after removing NA labels: {len(df)}")
    print(f"      Inactive (0): {(df['True_Label'] == 0).sum()}")
    print(f"      Active (1): {(df['True_Label'] == 1).sum()}")

    gen = FeatureGenerator()
    valid_indices = []
    features_list = []
    print("      Generating hybrid molecular descriptors...")

    for idx, row in df.iterrows():
        feat = gen.get_features(row['SMILES'])
        if feat is not None:
            features_list.append(feat)
            valid_indices.append(idx)

    df_valid = df.loc[valid_indices].reset_index(drop=True)
    X = np.array(features_list)
    y = df_valid['True_Label'].values

    print(f"      Compounds with valid molecular features: {len(df_valid)}")

    # 2. Train/Test Split
    print("[2/6] Splitting data into stratified training and independent test sets...")
    indices = np.arange(len(df_valid))
    X_train, X_test, y_train, y_test, train_idx, test_idx = train_test_split(
        X, y, indices, test_size=0.2, stratify=y, random_state=42
    )
    df_valid['Dataset_Split'] = 'Train'
    df_valid.loc[test_idx, 'Dataset_Split'] = 'Test'

    # 3. Data Preprocessing
    print("[3/6] Performing data scaling, variance, and correlation filtering...")
    var_thresh = VarianceThreshold(threshold=0)
    X_train = var_thresh.fit_transform(X_train)
    X_test = var_thresh.transform(X_test)

    print("      Removing highly correlated features (>0.9) to reduce redundancy...")
    corr_matrix = pd.DataFrame(X_train).corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > 0.90)]

    X_train = np.delete(X_train, to_drop, axis=1)
    X_test = np.delete(X_test, to_drop, axis=1)
    print(f"      Dropped {len(to_drop)} redundant features due to high correlation.")

    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    print(f"      Feature space dimension after all filtering: {X_train.shape[1]}")

    # 4. Feature Selection via Boruta
    print("[4/6] Executing Boruta algorithm for rigorous feature selection...")
    rf_boruta = RandomForestClassifier(
        n_jobs=-1,
        class_weight='balanced',
        max_depth=8,
        random_state=42
    )
    boruta = BorutaPy(
        rf_boruta,
        n_estimators='auto',
        verbose=0,
        max_iter=200,
        perc=80,
        random_state=42
    )
    boruta.fit(X_train, y_train)

    X_train_sel = boruta.transform(X_train)
    X_test_sel = boruta.transform(X_test)

    print(f"      Selected features after Boruta: {X_train_sel.shape[1]}")

    # 5. Applicability Domain
    print("[5/6] Building applicability domain (AD) model using training set only...")
    ad_model = build_knn_applicability_domain(X_train_sel, k=5, percentile=95.0)
    test_ad_distance, test_in_ad_mask = predict_applicability_domain(X_test_sel, ad_model)

    print(f"      AD method: kNN distance")
    print(f"      k = {ad_model['k']}")
    print(f"      Threshold percentile = {ad_model['percentile']}")
    print(f"      AD distance threshold = {ad_model['threshold']:.6f}")
    print(f"      Test samples in AD: {np.sum(test_in_ad_mask)} / {len(test_in_ad_mask)}")
    print(f"      Test samples out of AD: {np.sum(~test_in_ad_mask)} / {len(test_in_ad_mask)}")

    # store AD info
    df_valid['AD_Status'] = np.nan
    df_valid['AD_Mean_kNN_Distance'] = np.nan
    df_valid.loc[test_idx, 'AD_Status'] = np.where(test_in_ad_mask, 'In_AD', 'Out_AD')
    df_valid.loc[test_idx, 'AD_Mean_kNN_Distance'] = test_ad_distance

    # 6. Model Training, Hyperparameter Tuning, and Evaluation
    print("[6/6] Training models using Balanced Random Forest and SVM...")
    metrics_results = []
    ad_metrics_results = []
    probability_threshold = 0.50

    test_roc_data = {}
    test_predictions_dict = {}
    representative_test_in_ad_mask = None

    for algo in ['BRF', 'SVM']:
        print(f"\n      -> Initiating {algo} modeling pipeline")
        base_model, params = get_model_config(algo)

        # Grid Search with 3-fold cross-validation on Training Set
        grid = GridSearchCV(base_model, params, cv=3, scoring='roc_auc', n_jobs=-1)
        grid.fit(X_train_sel, y_train)
        best_model = grid.best_estimator_
        print(f"         Best hyperparameters: {grid.best_params_}")

        # Evaluate on Training Set
        y_train_prob = best_model.predict_proba(X_train_sel)[:, 1]
        y_train_pred = (y_train_prob >= probability_threshold).astype(int)
        train_metrics = calculate_metrics(y_train, y_train_prob, y_train_pred, algo, 'Train')
        metrics_results.append(train_metrics)

        # Evaluate on Independent Test Set
        y_test_prob = best_model.predict_proba(X_test_sel)[:, 1]
        y_test_pred = (y_test_prob >= probability_threshold).astype(int)
        test_metrics = calculate_metrics(y_test, y_test_prob, y_test_pred, algo, 'Test')
        metrics_results.append(test_metrics)

        # AD in/out metrics on test set
        test_in_metrics = calculate_subset_metrics(
            y_true=y_test,
            y_prob=y_test_prob,
            y_pred=y_test_pred,
            mask=test_in_ad_mask,
            model_name=algo,
            subset_name='Test_In_AD'
        )
        ad_metrics_results.append(test_in_metrics)

        test_out_metrics = calculate_subset_metrics(
            y_true=y_test,
            y_prob=y_test_prob,
            y_pred=y_test_pred,
            mask=~test_in_ad_mask,
            model_name=algo,
            subset_name='Test_Out_AD'
        )
        ad_metrics_results.append(test_out_metrics)

        # ROC for full test set
        if len(np.unique(y_test)) >= 2:
            fpr, tpr, _ = roc_curve(y_test, y_test_prob)
        else:
            fpr, tpr = np.array([0, 1]), np.array([0, 1])

        test_roc_data[algo] = {'fpr': fpr, 'tpr': tpr, 'auc': test_metrics['AUC']}
        test_predictions_dict[algo] = y_test_pred

        # Record predictions
        df_valid.loc[train_idx, f'{algo}_Prob'] = y_train_prob
        df_valid.loc[train_idx, f'{algo}_Pred'] = y_train_pred
        df_valid.loc[test_idx, f'{algo}_Prob'] = y_test_prob
        df_valid.loc[test_idx, f'{algo}_Pred'] = y_test_pred

        print(
            f"         Test AUC: {test_metrics['AUC']:.4f} | "
            f"Test MCC: {test_metrics['MCC']:.4f} | "
            f"Test F1: {test_metrics['F1_Score']:.4f}"
        )

        if representative_test_in_ad_mask is None:
            representative_test_in_ad_mask = test_in_ad_mask

    # Save metric tables
    metrics_df = pd.DataFrame(metrics_results)
    ad_metrics_df = pd.DataFrame(ad_metrics_results)

    metrics_df.to_csv("Table_1_Model_Performance.csv", index=False)
    ad_metrics_df.to_csv("Table_2_AD_Subset_Performance.csv", index=False)
    df_valid.to_csv("Table_3_Predictions_with_AD.csv", index=False)

    # Save plots
    plot_and_save_roc_curve(test_roc_data, filename='Figure_1_ROC_Curve.png')
    plot_and_save_confusion_matrices(y_test, test_predictions_dict, filename='Figure_2_Confusion_Matrices.png')

    if representative_test_in_ad_mask is not None:
        plot_and_save_applicability_domain(
            X_train_sel,
            X_test_sel,
            representative_test_in_ad_mask,
            filename='Figure_3_Applicability_Domain.png'
        )

    print("\n" + "=" * 85)
    print("Modeling completed successfully.")
    print("Generated files:")
    print("  - Table_1_Model_Performance.csv")
    print("  - Table_2_AD_Subset_Performance.csv")
    print("  - Table_3_Predictions_with_AD.csv")
    print("  - Figure_1_ROC_Curve.png")
    print("  - Figure_2_Confusion_Matrices.png")
    print("  - Figure_3_Applicability_Domain.png")
    print("=" * 85)
    

if __name__ == "__main__":
    COMPOUNDS_FILE = 'compounds.csv'
    ENDPOINTS_FILE = 'endpoints.csv'
