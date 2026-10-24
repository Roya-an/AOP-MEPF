"""
==============================================================================
3-Project: Machine Learning-based QSAR Modeling for Estradiol (E2) Prediction
==============================================================================

"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem, MACCSkeys, Draw
from sklearn.model_selection import GridSearchCV, cross_val_predict, StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import VarianceThreshold, RFE
from sklearn.metrics import (roc_auc_score, matthews_corrcoef, accuracy_score, 
                             roc_curve, confusion_matrix)
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.inspection import permutation_importance
from scipy.spatial import distance
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.neighbors import NearestNeighbors
from PIL import Image, ImageDraw, ImageFont 
import warnings

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-whitegrid')

# ==========================================
# 0. Global Configuration
# ==========================================
TARGET_NAME = 'Estradiol'
N_FEATURES_TO_SELECT = 24
OUTPUT_DIR = f'{TARGET_NAME}_Final_Report'

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
    os.makedirs(os.path.join(OUTPUT_DIR, 'Feature_Images'))

# ================================================================================
# 1. Complete MACCS dictionary (for structural visualization and feature analysis)
# ================================================================================
MACCS_DICT = {
    1: ('ISOTOPE', '?', 0),
    2: ('ISOTOPE Not complete', '[#103,#104]', 0),
    3: ('Group IVa,Va,VIa Periods 4-6 (Ge...)', '[Ge,As,Se,Sn,Sb,Te,Tl,Pb,Bi]', 0),
    4: ('Actinide', '[Ac,Th,Pa,U,Np,Pu,Am,Cm,Bk,Cf,Es,Fm,Md,No,Lr]', 0),
    5: ('Group IIIB,IVB (Sc...)', '[Sc,Ti,Y,Zr,Hf]', 0),
    6: ('Lanthanide', '[La,Ce,Pr,Nd,Pm,Sm,Eu,Gd,Tb,Dy,Ho,Er,Tm,Yb,Lu]', 0),
    7: ('Group VB,VIB,VIIB (V...)', '[V,Cr,Mn,Nb,Mo,Tc,Ta,W,Re]', 0),
    8: ('QAAA@1', '[!#6;!#1]1~*~*~*~1', 0),
    9: ('Group VIII (Fe...)', '[Fe,Co,Ni,Ru,Rh,Pd,Os,Ir,Pt]', 0),
    10: ('Group IIa (Alkaline earth)', '[Be,Mg,Ca,Sr,Ba,Ra]', 0),
    11: ('4M Ring', '*1~*~*~*~1', 0),
    12: ('Group IB,IIB (Cu..)', '[Cu,Zn,Ag,Cd,Au,Hg]', 0),
    13: ('ON(C)C', '[#8]~[#7](~[#6])~[#6]', 0),
    14: ('S-S', '[#16]-[#16]', 0),
    15: ('OC(O)O', '[#8]~[#6](~[#8])~[#8]', 0),
    16: ('QAA@1', '[!#6;!#1]1~*~*~1', 0),
    17: ('CTC (C#C)', '[#6]#[#6]', 0),
    18: ('Group IIIA (B...)', '[B,Al,Ga,In,Tl]', 0),
    19: ('7M Ring', '*1~*~*~*~*~*~*~1', 0),
    20: ('Si', '[Si]', 0),
    21: ('C=C(Q)Q', '[#6]=[#6](~[!#6;!#1])~[!#6;!#1]', 0),
    22: ('3M Ring', '*1~*~*~1', 0),
    23: ('NC(O)O', '[#7]~[#6](~[#8])~[#8]', 0),
    24: ('N-O', '[#7]-[#8]', 0),
    25: ('NC(N)N', '[#7]~[#6](~[#7])~[#7]', 0),
    26: ('C$=C($A)$A', '[#6]=;@[#6](@*)@*', 0),
    27: ('I', '[I]', 0),
    28: ('QCH2Q', '[!#6;!#1]~[CH2]~[!#6;!#1]', 0),
    29: ('P', '[#15]', 0),
    30: ('CQ(C)(C)A', '[#6]~[!#6;!#1](~[#6])(~[#6])~*', 0),
    31: ('QX', '[!#6;!#1]~[F,Cl,Br,I]', 0),
    32: ('CSN', '[#6]~[#16]~[#7]', 0),
    33: ('NS', '[#7]~[#16]', 0),
    34: ('CH2=A', '[CH2]=*', 0),
    35: ('Group IA (Alkali Metal)', '[Li,Na,K,Rb,Cs,Fr]', 0),
    36: ('S Heterocycle', '[#16R]', 0),
    37: ('NC(O)N', '[#7]~[#6](~[#8])~[#7]', 0),
    38: ('NC(C)N', '[#7]~[#6](~[#6])~[#7]', 0),
    39: ('OS(O)O', '[#8]~[#16](~[#8])~[#8]', 0),
    40: ('S-O', '[#16]-[#8]', 0),
    41: ('CTN (C#N)', '[#6]#[#7]', 0),
    42: ('F', 'F', 0),
    43: ('QHAQH', '[!C;!c;!#1;!H0]~*~[!C;!c;!#1;!H0]', 0),
    44: ('OTHER', '[!#1;!#6;!#7;!#8;!#9;!#14;!#15;!#16;!#17;!#35;!#53]', 0),
    45: ('C=CN', '[#6]=[#6]~[#7]', 0),
    46: ('BR', 'Br', 0),
    47: ('SAN', '[#16]~*~[#7]', 0),
    48: ('OQ(O)O', '[#8]~[!#6;!#1](~[#8])(~[#8])', 0),
    49: ('CHARGE', '[!+0]', 0),
    50: ('C=C(C)C', '[#6]=[#6](~[#6])~[#6]', 0),
    51: ('CSO', '[#6]~[#16]~[#8]', 0),
    52: ('NN', '[#7]~[#7]', 0),
    53: ('QHAAAQH', '[!#6;!#1;!H0]~*~*~*~[!#6;!#1;!H0]', 0),
    54: ('QHAAQH', '[!#6;!#1;!H0]~*~*~[!#6;!#1;!H0]', 0),
    55: ('OSO', '[#8]~[#16]~[#8]', 0),
    56: ('ON(O)C', '[#8]~[#7](~[#8])~[#6]', 0),
    57: ('O Heterocycle', '[#8R]', 0),
    58: ('QSQ', '[!#6;!#1]~[#16]~[!#6;!#1]', 0),
    59: ('Snot%A%A', '[#16]!:*:*', 0),
    60: ('S=O', '[#16]=[#8]', 0),
    61: ('AS(A)A', '*~[#16](~*)~*', 0),
    62: ('A$!A$A', '*@*!@*@*', 0),
    63: ('N=O', '[#7]=[#8]', 0),
    64: ('A$A!S', '*@*!@[#16]', 0),
    65: ('C%N (c:n)', 'c:n', 0),
    66: ('CC(C)(C)A', '[#6]~[#6](~[#6])(~[#6])~*', 0),
    67: ('QS', '[!#6;!#1]~[#16]', 0),
    68: ('QHQH', '[!#6;!#1;!H0]~[!#6;!#1;!H0]', 0),
    69: ('QQH', '[!#6;!#1]~[!#6;!#1;!H0]', 0),
    70: ('QNQ', '[!#6;!#1]~[#7]~[!#6;!#1]', 0),
    71: ('NO', '[#7]~[#8]', 0),
    72: ('OAAO', '[#8]~*~*~[#8]', 0),
    73: ('S=A', '[#16]=*', 0),
    74: ('CH3ACH3', '[CH3]~*~[CH3]', 0),
    75: ('A!N$A', '*!@[#7]@*', 0),
    76: ('C=C(A)A', '[#6]=[#6](~*)~*', 0),
    77: ('NAN', '[#7]~*~[#7]', 0),
    78: ('C=N', '[#6]=[#7]', 0),
    79: ('NAAN', '[#7]~*~*~[#7]', 0),
    80: ('NAAAN', '[#7]~*~*~*~[#7]', 0),
    81: ('SA(A)A', '[#16]~*(~*)~*', 0),
    82: ('ACH2QH', '*~[CH2]~[!#6;!#1;!H0]', 0),
    83: ('QAAAA@1', '[!#6;!#1]1~*~*~*~*~1', 0),
    84: ('NH2', '[NH2]', 0),
    85: ('CN(C)C', '[#6]~[#7](~[#6])~[#6]', 0),
    86: ('CH2QCH2', '[C;H2,H3][!#6;!#1][C;H2,H3]', 0),
    87: ('X!A$A', '[F,Cl,Br,I]!@*@*', 0),
    88: ('S', '[#16]', 0),
    89: ('OAAAO', '[#8]~*~*~*~[#8]', 0),
    90: ('QHAACH2A', '[$([!#6;!#1;!H0]~*~*~[CH2]~*),$([!#6;!#1;!H0;R]1@[R]@[R]@[CH2;R]1),$([!#6;!#1;!H0]~[R]1@[R]@[CH2;R]1)]', 0),
    91: ('QHAAACH2A', '[$([!#6;!#1;!H0]~*~*~*~[CH2]~*),$([!#6;!#1;!H0;R]1@[R]@[R]@[R]@[CH2;R]1),$([!#6;!#1;!H0]~[R]1@[R]@[R]@[CH2;R]1),$([!#6;!#1;!H0]~*~[R]1@[R]@[CH2;R]1)]', 0),
    92: ('OC(N)C', '[#8]~[#6](~[#7])~[#6]', 0),
    93: ('QCH3', '[!#6;!#1]~[CH3]', 0),
    94: ('QN', '[!#6;!#1]~[#7]', 0),
    95: ('NAAO', '[#7]~*~*~[#8]', 0),
    96: ('5M Ring', '*1~*~*~*~*~1', 0),
    97: ('NAAAO', '[#7]~*~*~*~[#8]', 0),
    98: ('QAAAAA@1', '[!#6;!#1]1~*~*~*~*~*~1', 0),
    99: ('C=C', '[#6]=[#6]', 0),
    100: ('ACH2N', '*~[CH2]~[#7]', 0),
    101: ('8M Ring or larger', '[$([R]@1@[R]@[R]@[R]@[R]@[R]@[R]@[R]1),$([R]@1@[R]@[R]@[R]@[R]@[R]@[R]@[R]@[R]1),$([R]@1@[R]@[R]@[R]@[R]@[R]@[R]@[R]@[R]@[R]1),$([R]@1@[R]@[R]@[R]@[R]@[R]@[R]@[R]@[R]@[R]@[R]1),$([R]@1@[R]@[R]@[R]@[R]@[R]@[R]@[R]@[R]@[R]@[R]@[R]1),$([R]@1@[R]@[R]@[R]@[R]@[R]@[R]@[R]@[R]@[R]@[R]@[R]@[R]1),$([R]@1@[R]@[R]@[R]@[R]@[R]@[R]@[R]@[R]@[R]@[R]@[R]@[R]@[R]1)]', 0),
    102: ('QO', '[!#6;!#1]~[#8]', 0),
    103: ('CL', 'Cl', 0),
    104: ('QHACH2A', '[!#6;!#1;!H0]~*~[CH2]~*', 0),
    105: ('A$A($A)$A', '*@*(@*)@*', 0),
    106: ('QA(Q)Q', '[!#6;!#1]~*(~[!#6;!#1])~[!#6;!#1]', 0),
    107: ('XA(A)A', '[F,Cl,Br,I]~*(~*)~*', 0),
    108: ('CH3AAACH2A', '[CH3]~*~*~*~[CH2]~*', 0),
    109: ('ACH2O', '*~[CH2]~[#8]', 0),
    110: ('NCO', '[#7]~[#6]~[#8]', 0),
    111: ('NACH2A', '[#7]~*~[CH2]~*', 0),
    112: ('AA(A)(A)A', '*~*(~*)(~*)~*', 0),
    113: ('Onot%A%A', '[#8]!:*:*', 0),
    114: ('CH3CH2A', '[CH3]~[CH2]~*', 0),
    115: ('CH3ACH2A', '[CH3]~*~[CH2]~*', 0),
    116: ('CH3AACH2A', '[$([CH3]~*~*~[CH2]~*),$([CH3]~*1~*~[CH2]1)]', 0),
    117: ('NAO', '[#7]~*~[#8]', 0),
    118: ('ACH2CH2A > 1', '[$(*~[CH2]~[CH2]~*),$(*1~[CH2]~[CH2]1)]', 1),
    119: ('N=A', '[#7]=*', 0),
    120: ('Heterocyclic atom > 1', '[!#6;R]', 1),
    121: ('N Heterocycle', '[#7;R]', 0),
    122: ('AN(A)A', '*~[#7](~*)~*', 0),
    123: ('OCO', '[#8]~[#6]~[#8]', 0),
    124: ('QQ', '[!#6;!#1]~[!#6;!#1]', 0),
    125: ('Aromatic Ring > 1', '?', 0),
    126: ('A!O!A', '*!@[#8]!@*', 0),
    127: ('A$A!O > 1', '*@*!@[#8]', 1),
    128: ('ACH2AAACH2A', '[$(*~[CH2]~*~*~*~[CH2]~*),$([R]1@[CH2;R]@[R]@[R]@[R]@[CH2;R]1),$(*~[CH2]~[R]1@[R]@[R]@[CH2;R]1),$(*~[CH2]~*~[R]1@[R]@[CH2;R]1)]', 0),
    129: ('ACH2AACH2A', '[$(*~[CH2]~*~*~[CH2]~*),$([R]1@[CH2]@[R]@[R]@[CH2;R]1),$(*~[CH2]~[R]1@[R]@[CH2;R]1)]', 0),
    130: ('QQ > 1', '[!#6;!#1]~[!#6;!#1]', 1),
    131: ('QH > 1', '[!#6;!#1;!H0]', 1),
    132: ('OACH2A', '[#8]~*~[CH2]~*', 0),
    133: ('A$A!N', '*@*!@[#7]', 0),
    134: ('X (HALOGEN)', '[F,Cl,Br,I]', 0),
    135: ('Nnot%A%A', '[#7]!:*:*', 0),
    136: ('O=A > 1', '[#8]=*', 1),
    137: ('Heterocycle', '[!C;!c;R]', 0),
    138: ('QCH2A > 1', '[!#6;!#1]~[CH2]~*', 1),
    139: ('OH', '[O;!H0]', 0),
    140: ('O > 3', '[#8]', 3),
    141: ('CH3 > 2', '[CH3]', 2),
    142: ('N > 1', '[#7]', 1),
    143: ('A$A!O', '*@*!@[#8]', 0),
    144: ('Anot%A%Anot%A', '*!:*:*!:*', 0),
    145: ('6M ring > 1', '*1~*~*~*~*~*~1', 1),
    146: ('O > 2', '[#8]', 2),
    147: ('ACH2CH2A', '[$(*~[CH2]~[CH2]~*),$([R]1@[CH2;R]@[CH2;R]1)]', 0),
    148: ('AQ(A)A', '*~[!#6;!#1](~*)~*', 0),
    149: ('CH3 > 1', '[C;H3,H4]', 1),
    150: ('A!A$A!A', '*!@*@*!@*', 0),
    151: ('NH', '[#7;!H0]', 0),
    152: ('OC(C)C', '[#8]~[#6](~[#6])~[#6]', 0),
    153: ('QCH2A', '[!#6;!#1]~[CH2]~*', 0),
    154: ('C=O', '[#6]=[#8]', 0),
    155: ('A!CH2!A', '*!@[CH2]!@*', 0),
    156: ('NA(A)A', '[#7]~*(~*)~*', 0),
    157: ('C-O', '[#6]-[#8]', 0),
    158: ('C-N', '[#6]-[#7]', 0),
    159: ('O > 1', '[#8]', 1),
    160: ('CH3', '[C;H3,H4]', 0),
    161: ('N', '[#7]', 0),
    162: ('Aromatic', 'a', 0),
    163: ('6M Ring', '*1~*~*~*~*~*~1', 0),
    164: ('O', '[#8]', 0),
    165: ('Ring', '[R]', 0),
    166: ('Fragments', '?', 0)
}

# ======================
# 2. Feature Generation
# ======================
class FeatureGenerator:
    def get_features(self, smiles):
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None: return None
            morgan = np.array(AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048))
            maccs = np.array(MACCSkeys.GenMACCSKeys(mol))
            desc = np.array([
                Descriptors.MolWt(mol), Descriptors.MolLogP(mol), Descriptors.TPSA(mol),
                Descriptors.NumHDonors(mol), Descriptors.NumHAcceptors(mol),
                Descriptors.NumRotatableBonds(mol)
            ])
            return np.concatenate([morgan, maccs, desc])
        except: return None

    def get_feature_names(self):
        names = [f'Morgan_{i}' for i in range(2048)]
        names += [f'MACCS_{i}' for i in range(167)]
        names += ['MolWt', 'LogP', 'TPSA', 'HDonors', 'HAcceptors', 'RotBonds']
        return np.array(names)

# ==========================================
# 3. Plotting Function
# ==========================================
def plot_roc_curve(y_true, y_prob, auc_score, output_dir):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='#d62728', lw=2, label=f'ROC curve (AUC = {auc_score:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title(f'ROC Curve - {TARGET_NAME}', fontsize=14)
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'Fig1_ROC_Curve.png'), dpi=300)
    plt.close()

def plot_confusion_matrix(y_true, y_pred, output_dir):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Inactive (0)', 'Active (1)'], 
                yticklabels=['Inactive (0)', 'Active (1)'])
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.title('Confusion Matrix', fontsize=14)
    plt.savefig(os.path.join(output_dir, 'Fig2_Confusion_Matrix.png'), dpi=300)
    plt.close()

def plot_feature_importance(importances, feature_names, output_dir):
    indices = np.argsort(importances)[::-1]
    
    plt.figure(figsize=(12, 8))
    plt.title(f"Feature Importance of Selected {len(feature_names)} Features (SVM Permutation)", fontsize=14)
    plt.bar(range(len(feature_names)), importances[indices], align="center", color='#1f77b4')
    plt.xticks(range(len(feature_names)), feature_names[indices], rotation=90, fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'Fig3_Feature_Importance.png'), dpi=300)
    plt.close()

def analyze_applicability_domain(X_train, X_pred, output_dir):
    print("   Performing application domain (AD) analysis (based on k-NN algorithm)...")
    
    # 1. Fitting the k-NN model (k=5)
    k_neighbors = 5
    neigh = NearestNeighbors(n_neighbors=k_neighbors)
    neigh.fit(X_train)
    
    # 2. Calculate the average k-NN distance of the training set
    # Note: The distance between the samples in the training set is 0 (in the first column), so take the mean starting from the second column
    distances_train, _ = neigh.kneighbors(X_train, n_neighbors=k_neighbors+1)
    dist_train = np.mean(distances_train[:, 1:], axis=1)
    
    # 3. Calculate the k-NN average distance of the prediction set (if the predicted samples are not in the training set, simply take the average of all k neighbors)
    distances_pred, _ = neigh.kneighbors(X_pred, n_neighbors=k_neighbors)
    dist_pred = np.mean(distances_pred, axis=1)
    
    # 4. Set AD threshold (mean + 3 times standard deviation)
    dist_mean = np.mean(dist_train)
    dist_std = np.std(dist_train)
    ad_threshold = dist_mean + 3 * dist_std
    
    # 5. Plotting
    plt.figure(figsize=(10, 6))
    plt.scatter(range(len(dist_train)), dist_train, c='#1f77b4', alpha=0.6, label='Training Set', s=20)
    plt.scatter(range(len(dist_train), len(dist_train)+len(dist_pred)), dist_pred, c='#2ca02c', alpha=0.6, label='Prediction Set', s=20)
    plt.axhline(y=ad_threshold, color='r', linestyle='--', label=f'Threshold (k-NN, k={k_neighbors}): {ad_threshold:.2f}')
    
    plt.xlabel('Compound Index', fontsize=12)
    plt.ylabel(f'Mean Euclidean Distance to {k_neighbors} Nearest Neighbors', fontsize=12)
    plt.title('Applicability Domain (k-NN Distance)', fontsize=14)
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'Fig4_Applicability_Domain.png'), dpi=300)
    plt.close()
    
    out_of_domain = np.array(dist_pred) > ad_threshold
    return out_of_domain, dist_pred

# ==========================================
# 4. Feature Structure Analysis
# ==========================================
def add_label_to_image(pil_img, label_text):
    
    w, h = pil_img.size
    new_h = h + 40
    new_img = Image.new("RGB", (w, new_h), "white")
    new_img.paste(pil_img, (0, 0))
    
    draw = ImageDraw.Draw(new_img)
    draw.text((10, h + 10), label_text, fill="black")
    
    return new_img

def interpret_and_visualize_features(selected_features, df_train, output_dir):
    """
    Traverse the selected features, generate an explanatory dictionary, and draw the structure diagrams of Morgan and MACCS.
    """
    print("   [Processing] Generate feature dictionary and structure diagram (with labels)...")
    img_dir = os.path.join(output_dir, 'Feature_Images')
    
    feature_data = []

    for feat in selected_features:
        info = {
            'ID': feat,
            'Type': 'Unknown',
            'Description': '-',
            'SMARTS_or_Bit': '-',
            'Representative_SMILES': '-',
            'Image_File': 'N/A'
        }
        
        # --- A: Morgan Fingerprint ---
        if feat.startswith('Morgan_'):
            info['Type'] = 'Morgan Fingerprint'
            bit_id = int(feat.split('_')[1])
            info['Description'] = f'Circular Substructure (Radius=2, Bit={bit_id})'
            info['SMARTS_or_Bit'] = f'Bit {bit_id}'
            
            found_mol = False
            for smi in df_train['SMILES']:
                mol = Chem.MolFromSmiles(smi)
                if not mol: continue
                bi = {}
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048, bitInfo=bi)
                if bit_id in bi:
                    img_path = os.path.join(img_dir, f'{feat}.png')
                    try:
                        # 1. Generate basic fingerprint image
                        img = Draw.DrawMorganBit(mol, bit_id, bi, useSVG=False)
                        # 2. Add tags (feature names)
                        labeled_img = add_label_to_image(img, feat)
                        # 3. Save
                        labeled_img.save(img_path)
                        
                        info['Representative_SMILES'] = smi
                        info['Image_File'] = f'{feat}.png'
                        found_mol = True
                        break
                    except: pass
            if not found_mol:
                info['Description'] += " (No examples found in the training subset)"

        # --- B: MACCS Fingerprint ---
        elif feat.startswith('MACCS_'):
            info['Type'] = 'MACCS Key'
            try:
                key_id = int(feat.split('_')[1])
                
                if key_id in MACCS_DICT:
                    desc, smarts, threshold = MACCS_DICT[key_id]
                    info['Description'] = desc
                    info['SMARTS_or_Bit'] = f"{smarts} (Count > {threshold})"
                    
                    found_mol = False
                    
                    for smi in df_train['SMILES']:
                        mol = Chem.MolFromSmiles(smi)
                        if not mol: continue
                        
                        # 【Core Logic 1: Seeking Truth】 Directly calculate the standard MACCS fingerprint to determine if the true value is 1
                        maccs_fp = MACCSkeys.GenMACCSKeys(mol)
                        
                        if maccs_fp.GetBit(key_id): # If the value is 1, it indicates that the condition (such as quantity>2) has been strictly met
                            img_path = os.path.join(img_dir, f'{feat}.png')
                            highlight_atoms = []
                            
                            # 【Core Logic 2: Visualization】 Highlighting using SMARTS
                            # Note: There are some complex features in the official documents (such as ring counting) that SMARTS is marked as'? '
                            if smarts not in ['?', '', None]: 
                                pattern = Chem.MolFromSmarts(smarts)
                                if pattern and mol.HasSubstructMatch(pattern):
                                    matches = mol.GetSubstructMatches(pattern) #Get all matching functional groups
                                    #Flatten and deduplicate, highlighting all atoms belonging to this feature
                                    highlight_atoms = list(set([atom_idx for match in matches for atom_idx in match]))
                            
                            custom_legend = f"{feat}: {desc}\n(Condition: Count > {threshold})"
                            
                            try:
                                #If there are matching atoms, highlight them; What if it is? Type, just draw the original molecule
                                if highlight_atoms:
                                    img = Draw.MolToImage(mol, highlightAtoms=highlight_atoms, size=(350, 350), legend=custom_legend)
                                else:
                                    img = Draw.MolToImage(mol, size=(350, 350), legend=f"{custom_legend}\n(No SMARTS for highlight)")
                                
                                img.save(img_path)
                                
                                info['Representative_SMILES'] = smi
                                info['Image_File'] = f'{feat}.png'
                                found_mol = True
                                break # After finding the representative molecule, jump out of the loop
                                
                            except Exception as e:
                                print(f"   [Warning] error drawing {feat} : {e}")
                                pass
                                
                    if not found_mol:
                         info['Description'] += " (No positive samples were found in the training set)"
                         
                else:
                    info['Description'] = 'MACCS Key not in Dictionary'
            except Exception as e:
                info['Description'] = f'Error parsing MACCS ID: {e}'
        
        # --- C: Physical and chemical descriptors ---
        else:
            info['Type'] = 'Descriptor'
            desc_map = {
                'MolWt': 'Molecular Weight (Daltons)',
                'LogP': 'Octanol-Water Partition Coefficient',
                'TPSA': 'Topological Polar Surface Area (Å²)',
                'HDonors': 'Hydrogen Bond Donors Count',
                'HAcceptors': 'Hydrogen Bond Acceptors Count',
                'RotBonds': 'Number of Rotatable Bonds'
            }
            info['Description'] = desc_map.get(feat, 'Physicochemical Property')
            info['Image_File'] = 'N/A (Whole Molecule Property)'
        
        feature_data.append(info)

    # Save Dictionary CSV
    pd.DataFrame(feature_data).to_csv(os.path.join(output_dir, 'Feature_Interpretation.csv'), index=False)
    print(f"   [Done] The feature dictionary has been saved to Feature_Interpretation.csv")

# ==========================================
# 5. Main Process
# ==========================================
def run_full_pipeline(cmp_path, end_path):
    print(f"\n{'='*80}\n{TARGET_NAME} Prediction process (Consistency Fixed + Labeled Images)\n{'='*80}")
    
    # --- A. Data loading ---
    try:
        df_cmp = pd.read_csv(cmp_path)[['CID', 'SMILES']]
        df_end = pd.read_csv(end_path)
    except KeyError:
        print(f"Error: Please check the column names in the input CSV file. Need CID, SMILES, and {TARGET_NAME}")
        return

    # Merge data
    df = pd.merge(df_end, df_cmp, on='CID')
    
    if TARGET_NAME not in df.columns:
        print(f"Error: Column '{TARGET_NAME}' not found in endpoints file")
        return

    print(f"Total number of data rows: {len(df)}")
    
    # --- B. Feature calculation---
    print("[1/6] Calculate features (Morgan + MACCS + Descriptors)...")
    gen = FeatureGenerator()
    X_all_raw, y_all, valid_indices = [], [], []
    for idx, row in df.iterrows():
        f = gen.get_features(row['SMILES'])
        if f is not None:
            X_all_raw.append(f)
            y_all.append(row[TARGET_NAME])
            valid_indices.append(idx)
            
    X_all = np.array(X_all_raw)
    y_all = np.array(y_all)
    names = gen.get_feature_names()
    meta_df_all = df.iloc[valid_indices][['CID', 'SMILES', TARGET_NAME]].reset_index(drop=True)
    
    # --- C. Data splitting ---
    train_mask = (y_all != -1)
    pred_mask = (y_all == -1)
    
    X_train = X_all[train_mask]
    y_train = y_all[train_mask]
    X_pred = X_all[pred_mask]
    
    df_train_out = meta_df_all[train_mask].copy()
    df_pred_out = meta_df_all[pred_mask].copy()
    
    n_pos = np.sum(y_train == 1)
    n_neg = np.sum(y_train == 0)
    print(f"   Total number of training sets: {len(y_train)} (Pos:{n_pos}, Neg:{n_neg})")
    
    # --- D. Unsupervised cleaning ---
    print(f"[2/6] Unsupervised cleaning...")
    var_selector = VarianceThreshold(threshold=0)
    X_train_var = var_selector.fit_transform(X_train)
    X_pred_var = var_selector.transform(X_pred)
    names = names[var_selector.get_support()]
    
    df_feat = pd.DataFrame(X_train_var)
    corr_matrix = df_feat.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [col for col in upper.columns if any(upper[col] > 0.85)]
    keep_indices = [i for i in range(X_train_var.shape[1]) if i not in to_drop]
    
    X_train_clean = X_train_var[:, keep_indices]
    X_pred_clean = X_pred_var[:, keep_indices]
    names = names[keep_indices]
    
    print(f"   Remaining number of features after cleaning: {X_train_clean.shape[1]}")

    # ==========================================
    # --- E. Modeling and Evaluation (Pipeline) ---
    # ==========================================
    print("[3/6] Model Training and Cross Validation (SVM-RFE Pipeline)...")
    
    rfe_estimator = RandomForestClassifier(n_jobs=-1, class_weight='balanced', max_depth=10, random_state=42)
    
    pipeline = ImbPipeline([
        ('scaler', MinMaxScaler()),
        ('rfe', RFE(estimator=rfe_estimator, n_features_to_select=N_FEATURES_TO_SELECT, step=0.1)),
        ('smote', SMOTE(random_state=42, k_neighbors=3)), 
        ('classifier', SVC(probability=True, class_weight='balanced', random_state=42))
    ])
    
    param_grid = {
        'classifier__C': [0.01, 1, 10, 20, 30, 40],
        'classifier__gamma': ['scale', 0.1, 0.4, 0.6, 1.0],
        'classifier__kernel': ['rbf']
    }
    
    search_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    grid = GridSearchCV(pipeline, param_grid, cv=search_cv, scoring='roc_auc', n_jobs=-1)
    grid.fit(X_train_clean, y_train)
    best_model = grid.best_estimator_
    
    print(f"   optimal parameters: {grid.best_params_}")
    
    # CV prediction
    eval_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    y_cv_prob = cross_val_predict(best_model, X_train_clean, y_train, cv=eval_cv, method='predict_proba', n_jobs=-1)[:, 1]
    
    # Threshold optimization
    best_th, max_mcc = 0.5, -1
    for th in np.arange(0.3, 0.8, 0.01):
        pred_bin = (y_cv_prob >= th).astype(int)
        mcc = matthews_corrcoef(y_train, pred_bin)
        if mcc > max_mcc: max_mcc = mcc; best_th = th
            
    y_cv_pred = (y_cv_prob >= best_th).astype(int)
    auc_val = roc_auc_score(y_train, y_cv_prob)
    tn, fp, fn, tp = confusion_matrix(y_train, y_cv_pred).ravel()
    sensitivity = tp / (tp + fn) 
    specificity = tn / (tn + fp) 
    acc = accuracy_score(y_train, y_cv_pred)
    
    print(f"\n   Verification set results (5-Fold CV):")
    print(f"   AUC: {auc_val:.3f}, MCC: {max_mcc:.3f}")
    print(f"   Sensitivity (Active accuracy): {sensitivity:.3f}")
    print(f"   Specificity (Inactive accuracy): {specificity:.3f}")
    print(f"   optimal threshold: {best_th:.2f}")
    
    plot_roc_curve(y_train, y_cv_prob, auc_val, OUTPUT_DIR)
    plot_confusion_matrix(y_train, y_cv_pred, OUTPUT_DIR)

    # ==========================================
    # --- F. Feature consistency analysis ---
    # ==========================================
    print("[4/6] Final model analysis (feature importance & structural analysis) ")
    
    best_model.fit(X_train_clean, y_train)
    
    final_rfe = best_model.named_steps['rfe']
    selected_mask = final_rfe.support_
    final_names = names[selected_mask] 
    
    print(f"   The final selected {len(final_names)} features have been extracted.")
    
    scaler = best_model.named_steps['scaler']
    svc_step = best_model.named_steps['classifier']
    
    X_train_selected = final_rfe.transform(scaler.transform(X_train_clean))
    perm_result = permutation_importance(svc_step, X_train_selected, y_train, n_repeats=10, random_state=42, n_jobs=-1, scoring='roc_auc')
    
    plot_feature_importance(perm_result.importances_mean, final_names, OUTPUT_DIR)
    
    interpret_and_visualize_features(final_names, df_train_out, OUTPUT_DIR)
    
    # ==========================================
    # --- G. Application Domain & Save---
    # ==========================================
    print("[5/6] Applicability Domain Analysis")
    
    X_pred_selected = final_rfe.transform(scaler.transform(X_pred_clean))
    is_out_domain, dist_pred = analyze_applicability_domain(X_train_selected, X_pred_selected, OUTPUT_DIR)
    
    print(f"  Number of samples outside the domain: {sum(is_out_domain)} / {len(X_pred)}")

    # ==========================================
    # --- H. Save Results ---
    # ==========================================
    print("[6/6] Save Final Results ")
    
    y_pred_prob = best_model.predict_proba(X_pred_clean)[:, 1]
    
    df_train_out['Prob_Estradiol'] = y_cv_prob
    df_train_out['Pred_Label'] = y_cv_pred
    df_train_out['Dataset'] = 'Training_CV'
    df_train_out['In_Domain'] = 'Yes'
    df_train_out['Distance_to_Centroid'] = 'N/A'
    
    df_pred_out['Prob_Estradiol'] = y_pred_prob
    df_pred_out['Pred_Label'] = (y_pred_prob >= best_th).astype(int)
    df_pred_out['Dataset'] = 'Prediction'
    df_pred_out['In_Domain'] = np.where(is_out_domain, 'No', 'Yes')
    df_pred_out['Distance_to_Centroid'] = dist_pred
    
    final_df = pd.concat([df_train_out, df_pred_out], ignore_index=True)
    final_df.to_csv(os.path.join(OUTPUT_DIR, 'Final_Predictions_with_AD.csv'), index=False)
    
    with open(os.path.join(OUTPUT_DIR, 'Manuscript_Report.txt'), 'w') as f:
        f.write(f"=== QSAR Model Report: {TARGET_NAME} ===\n")
        f.write(f"Class Balance (Train): {n_pos} Positive vs {n_neg} Negative\n")
        f.write(f"Model: SVM (RBF) with RFE Feature Selection & SMOTE\n")
        f.write(f"Features Selected: {len(final_names)}\n\n")
        
        f.write(f"--- Validation Metrics (5-Fold CV) ---\n")
        f.write(f"AUC: {auc_val:.4f}\n")
        f.write(f"MCC: {max_mcc:.4f}\n")
        f.write(f"Accuracy: {acc:.4f}\n")
        f.write(f"Sensitivity (Active): {sensitivity:.4f}\n")
        f.write(f"Specificity (Inactive): {specificity:.4f}\n")
        f.write(f"Optimal Threshold: {best_th:.2f}\n\n")
        
        f.write(f"--- Selected Features List (Sorted by Importance) ---\n")
        sorted_idx = np.argsort(perm_result.importances_mean)[::-1]
        for idx in sorted_idx:
            f.write(f"{final_names[idx]} : {perm_result.importances_mean[idx]:.4f}\n")
            
    print(f"\n[Completed] Result saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    cmp_file = 'compounds.csv'
    end_file = 'endpoints.csv'
    
    if os.path.exists(cmp_file) and os.path.exists(end_file):
        run_full_pipeline(cmp_file, end_file)
    else:
        print(f"Error：input file {cmp_file} or {end_file} not found")
