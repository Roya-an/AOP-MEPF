"""
==============================================================================
4-Project: Machine Learning-based QSAR Modeling for Testosterone (T) Prediction
==============================================================================

"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem, MACCSkeys, Draw
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import VarianceThreshold, SelectFromModel
from sklearn.metrics import (roc_auc_score, matthews_corrcoef, accuracy_score, 
                             roc_curve, confusion_matrix, recall_score)
from sklearn.ensemble import RandomForestClassifier
from imblearn.ensemble import BalancedRandomForestClassifier 
from imblearn.pipeline import Pipeline as ImbPipeline
from scipy.spatial import distance
from PIL import Image, ImageDraw, ImageFont 
from sklearn.neighbors import NearestNeighbors
import warnings

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-whitegrid')

# ==========================================
# 0. Global Configuration
# ==========================================
TARGET_NAME = 'Testosterone'
N_FEATURES_TO_SELECT = 26
OUTPUT_DIR = f'{TARGET_NAME}_Final_Report'

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
    os.makedirs(os.path.join(OUTPUT_DIR, 'Feature_Images'))

# ==========================================
# 1. MACCS dictionary 
# ==========================================
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

# ==========================================
# 2. Feature Generation
# ==========================================
class FeatureGenerator:
    def get_features(self, smiles):
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None: return None
            morgan = np.array(AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048))
            maccs = np.array(MACCSkeys.GenMACCSKeys(mol))
            desc = np.array([Descriptors.MolWt(mol), Descriptors.MolLogP(mol), Descriptors.TPSA(mol),
                             Descriptors.NumHDonors(mol), Descriptors.NumHAcceptors(mol), Descriptors.NumRotatableBonds(mol)])
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
def add_label_to_image(pil_img, label_text):
    w, h = pil_img.size
    new_img = Image.new("RGB", (w, h + 40), "white")
    new_img.paste(pil_img, (0, 0))
    draw = ImageDraw.Draw(new_img)
    draw.text((10, h + 10), label_text, fill="black")
    return new_img

def plot_roc_curve(y_true, y_prob, auc_score, output_dir):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='#d62728', lw=2, label=f'ROC curve (AUC = {auc_score:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate', fontsize=12); plt.ylabel('True Positive Rate', fontsize=12)
    plt.title(f'ROC Curve - {TARGET_NAME}', fontsize=14); plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3); plt.savefig(os.path.join(output_dir, 'Fig1_ROC_Curve.png'), dpi=300); plt.close()

def plot_confusion_matrix(y_true, y_pred, output_dir):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Inactive (0)', 'Active (1)'], yticklabels=['Inactive (0)', 'Active (1)'])
    plt.xlabel('Predicted Label', fontsize=12); plt.ylabel('True Label', fontsize=12)
    plt.title('Confusion Matrix', fontsize=14); plt.savefig(os.path.join(output_dir, 'Fig2_Confusion_Matrix.png'), dpi=300); plt.close()

def analyze_applicability_domain(X_train, X_pred, output_dir):
    print("   Performing application domain (AD) analysis (based on k-NN algorithm)...")
    
    # 1.  Fitting the k-NN model (k=5)
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

def generate_manuscript_report(metrics, params, output_dir, feature_importances=None):
    with open(os.path.join(output_dir, 'Manuscript_Report.txt'), 'w', encoding='utf-8') as f:
        f.write(f"MODEL FINAL REPORT - {TARGET_NAME}\n" + "="*40 + "\n")
        f.write(f"1. Dataset Info:\n   Total Training: {metrics['train_size']} (Pos:{metrics['pos']}, Neg:{metrics['neg']})\n\n")
        f.write(f"2. Validation (5-Fold CV):\n   AUC: {metrics['auc']:.3f}\n   MCC: {metrics['mcc']:.3f}\n")
        f.write(f"   Accuracy: {metrics.get('accuracy', 0.0):.3f}\n") 
        f.write(f"   Sensitivity: {metrics['sens']:.3f}\n   Specificity: {metrics['spec']:.3f}\n")
        f.write(f"   Threshold: {metrics['threshold']:.2f}\n\n")
        f.write(f"3. Parameters:\n")
        for k, v in params.items(): 
            f.write(f"   {k}: {v}\n")
        if feature_importances:
            f.write(f"\n4. Selected Features List (Sorted by Importance):\n")
            for idx, (feat, imp) in enumerate(feature_importances, 1):
                f.write(f"   {idx}. {feat} (Importance: {imp:.4f})\n")
                
    print(f"   [Done] The report has been saved to Manuscript_Report.txt")
    
# ==========================================
# 4. Feature structure visualization
# ==========================================
def interpret_and_visualize_features(selected_features, df_train, output_dir):
    print("   [Processing] Generate feature dictionary and feature images...")
    img_dir = os.path.join(output_dir, 'Feature_Images')
    feature_data = []

    for feat in selected_features:
        info = {'ID': feat, 'Type': 'Unknown', 'Description': '-', 'SMARTS_or_Bit': '-', 'Representative_SMILES': '-', 'Image_File': 'N/A'}
        
        # --- A: Morgan Fingerprint ---
        if feat.startswith('Morgan_'):
            info['Type'] = 'Morgan Fingerprint'
            bit_id = int(feat.split('_')[1])
            info['Description'] = f'Circular Substructure (Bit={bit_id})'
            for smi in df_train['SMILES']:
                mol = Chem.MolFromSmiles(smi)
                if mol:
                    bi = {}; fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048, bitInfo=bi)
                    if bit_id in bi:
                        try:
                            img = Draw.DrawMorganBit(mol, bit_id, bi, useSVG=False)
                            add_label_to_image(img, feat).save(os.path.join(img_dir, f'{feat}.png'))
                            info['Representative_SMILES'] = smi
                            info['Image_File'] = f'{feat}.png'
                            break
                        except: pass
                        
        # --- B: MACCS Keys ---
        elif feat.startswith('MACCS_'):
            info['Type'] = 'MACCS Key'
            try:
                kid = int(feat.split('_')[1])
                if kid in MACCS_DICT: 
                    desc, smarts, threshold = MACCS_DICT[kid]
                    info['Description'] = desc
                    info['SMARTS_or_Bit'] = f"{smarts} (Count > {threshold})"
                    
                    for smi in df_train['SMILES']:
                        mol = Chem.MolFromSmiles(smi)
                        if not mol: continue
                        
                        # 【Core Logic 1: Seeking Truth】 Directly calculate the standard MACCS keys to determine if the true value is 1
                        maccs_fp = MACCSkeys.GenMACCSKeys(mol)
                        if maccs_fp.GetBit(kid):
                            img_path = os.path.join(img_dir, f'{feat}.png')
                            highlight_atoms = []
                            
                            # 【Core Logic 2: Visualization】 Highlighting using SMARTS
                            if smarts not in ['?', '', None]:
                                pattern = Chem.MolFromSmarts(smarts)
                                if pattern and mol.HasSubstructMatch(pattern):
                                    matches = mol.GetSubstructMatches(pattern)
                                    # Flatten and deduplicate, highlighting all atoms belonging to this feature
                                    highlight_atoms = list(set([atom_idx for match in matches for atom_idx in match]))
                            
                            custom_legend = f"{feat}: {desc}\n(Condition: Count > {threshold})"
                            try:
                                if highlight_atoms:
                                    img = Draw.MolToImage(mol, highlightAtoms=highlight_atoms, size=(350, 350), legend=custom_legend)
                                else:
                                    img = Draw.MolToImage(mol, size=(350, 350), legend=f"{custom_legend}\n(No SMARTS for highlight)")
                                img.save(img_path)
                                info['Representative_SMILES'] = smi
                                info['Image_File'] = f'{feat}.png'
                                break
                            except Exception as e:
                                print(f"     [Warning] error drawing {feat} : {e}")
                                pass
            except Exception as e:
                info['Description'] = f'Error parsing MACCS ID: {e}'

        # --- C: Physical and chemical descriptors  ---
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
        
    pd.DataFrame(feature_data).to_csv(os.path.join(output_dir, 'Feature_Interpretation.csv'), index=False)
    print(f"   [Done] feature dictionary and {len(selected_features)} images have been generated")
# ==========================================
# 5. Main Process
# ==========================================
def run_full_pipeline(cmp_path, end_path):
    print(f"\n{'='*80}\n{TARGET_NAME} Prediction process (Balanced RF + AD + Manuscript)\n{'='*80}")
    
    # --- A. Data loading ---
    print("[0/5] Loading data and generating features...")
    df = pd.merge(pd.read_csv(end_path), pd.read_csv(cmp_path)[['CID', 'SMILES']], on='CID')
    gen = FeatureGenerator()
    X_all, y_all, valid_idx = [], [], []
    
    for idx, row in df.iterrows():
        f = gen.get_features(row['SMILES'])
        if f is not None: 
            X_all.append(f)
            y_all.append(row[TARGET_NAME])
            valid_idx.append(idx)
        else:
            print(f"   [Warning] Invalid SMILES skipped at index {idx}: {row['SMILES']}")
    
    X_all, y_all = np.array(X_all), np.array(y_all)
    names = gen.get_feature_names()
    meta_df = df.iloc[valid_idx].reset_index(drop=True)
    
    # --- B. Split training and prediction sets ---
    X_train = X_all[y_all != -1].astype(float)
    y_train = y_all[y_all != -1].astype(int)
    X_pred = X_all[y_all == -1].astype(float)
    
    df_train_out = meta_df[y_all != -1].copy()
    df_pred_out = meta_df[y_all == -1].copy()

    # --- C. Feature cleaning and screening ---
    print("[1/5] Feature cleaning and screening...")
    print("   -> Removing zero-variance features...")
    var_sel = VarianceThreshold(threshold=0.0) 
    X_train_var = var_sel.fit_transform(X_train)
    X_pred_var = var_sel.transform(X_pred)     
    names_var = names[var_sel.get_support()]

    print("   -> Removing highly correlated features (Pearson > 0.90)...")
    df_X_train = pd.DataFrame(X_train_var, columns=names_var)
    corr_matrix = df_X_train.corr().abs()
    
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > 0.90)]
    keep_cols = [c for c in df_X_train.columns if c not in to_drop]
    
    X_train_corr = df_X_train[keep_cols].values
    X_pred_corr = pd.DataFrame(X_pred_var, columns=names_var)[keep_cols].values
    names_corr = np.array(keep_cols)
    print(f"      Dropped {len(to_drop)} correlated features. Remaining: {len(keep_cols)}")

    print("   -> Scaling features to [0, 1] using MinMaxScaler...")
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train_corr)
    X_pred_scaled = scaler.transform(X_pred_corr)
    
    print(f"   -> Selecting top {N_FEATURES_TO_SELECT} features using RandomForest...")
    sfm = SelectFromModel(
        RandomForestClassifier(n_estimators=100, class_weight='balanced', max_depth=5, random_state=42, n_jobs=-1), 
        max_features=N_FEATURES_TO_SELECT, 
        threshold=-np.inf 
    )
    
    X_train_reduced = sfm.fit_transform(X_train_scaled, y_train)
    X_pred_reduced = sfm.transform(X_pred_scaled)
    final_names = names_corr[sfm.get_support()]
    
    print(f"   -> Final features selected: {final_names.tolist()}")
    # --- D. Model Training and CV ---
    print("[2/5] train Balanced Random Forest and conduct 5-Fold CV...")
    brf_model = BalancedRandomForestClassifier(n_estimators=300, max_depth=6, min_samples_leaf=2, 
                                               sampling_strategy='majority', random_state=42, n_jobs=-1)
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    y_cv_prob = cross_val_predict(brf_model, X_train_reduced, y_train, cv=skf, method='predict_proba')[:, 1]
    
    best_th, max_gmean = 0.5, -1
    for th in np.arange(0.2, 0.8, 0.01):
        p = (y_cv_prob >= th).astype(int)
        g = np.sqrt(recall_score(y_train, p) * recall_score(y_train, p, pos_label=0))
        if g > max_gmean: max_gmean = g; best_th = th
    
    y_cv_pred = (y_cv_prob >= best_th).astype(int)
    metrics = {'train_size': len(y_train), 'pos': sum(y_train==1), 'neg': sum(y_train==0),
               'auc': roc_auc_score(y_train, y_cv_prob), 'mcc': matthews_corrcoef(y_train, y_cv_pred), 
               'sens': recall_score(y_train, y_cv_pred), 'spec': recall_score(y_train, y_cv_pred, pos_label=0),
               'threshold': best_th,
               'accuracy': accuracy_score(y_train, y_cv_pred)}
    
    df_train_out['Prob_Testosterone'] = y_cv_prob
    df_train_out['Pred_Label'] = y_cv_pred
    df_train_out['Dataset'] = 'Training_CV'
    df_train_out['In_Domain'] = 'Yes'                     
    df_train_out['Distance_to_Centroid'] = 'N/A'          

    # --- E. Output results and visualization ---
    print("[3/5] Generate visual charts and Manuscript reports...")
    plot_roc_curve(y_train, y_cv_prob, metrics['auc'], OUTPUT_DIR)
    plot_confusion_matrix(y_train, y_cv_pred, OUTPUT_DIR)
    
    brf_model.fit(X_train_reduced, y_train)
    indices = np.argsort(brf_model.feature_importances_)[::-1]
    plt.figure(figsize=(12, 8)); plt.title(f"Feature Importance (Top {len(final_names)})")
    plt.bar(range(len(final_names)), brf_model.feature_importances_[indices], align="center", color='#1f77b4')
    plt.xticks(range(len(final_names)), final_names[indices], rotation=90)
    plt.tight_layout(); plt.savefig(os.path.join(OUTPUT_DIR, 'Fig3_Feature_Importance.png'), dpi=300); plt.close()

    params = {"Model": "BalancedRandomForest", "n_estimators": 300, "max_depth": 6, "min_samples_leaf": 2, "Features": N_FEATURES_TO_SELECT}
    feature_imp_list = [(final_names[i], brf_model.feature_importances_[i]) for i in indices]
    generate_manuscript_report(metrics, params, OUTPUT_DIR, feature_importances=feature_imp_list)

    print("[4/5] Feature analysis...")
    interpret_and_visualize_features(final_names, df_train_out, OUTPUT_DIR)

    # --- F. Applicability domain analysis and save the final prediction file---
    print("[5/5] Applicability domain analysis and save the final prediction file...")
    if len(X_pred_reduced) > 0:
        is_out, dists = analyze_applicability_domain(X_train_reduced, X_pred_reduced, OUTPUT_DIR)
        y_pred_prob = brf_model.predict_proba(X_pred_reduced)[:, 1]
        
        # Build a prediction set DataFrame (with column names consistent with the training set)
        pred_df = pd.DataFrame({
            'CID': df_pred_out['CID'],
            'SMILES': df_pred_out['SMILES'],
            'Testosterone': df_pred_out[TARGET_NAME],           
            'Prob_Testosterone': np.round(y_pred_prob, 4),
            'Pred_Label': (y_pred_prob >= best_th).astype(int),
            'Dataset': 'Prediction',
            'In_Domain': np.where(~is_out, 'Yes', 'No'),        
            'Distance_to_Centroid': np.round(dists, 4)
        })
        
        # Merge training set and prediction set
        final_df = pd.concat([df_train_out, pred_df], ignore_index=True)
    else:
        final_df = df_train_out.copy()

    column_order = ['CID', 'SMILES', 'Testosterone', 'Prob_Testosterone', 
                    'Pred_Label', 'Dataset', 'In_Domain', 'Distance_to_Centroid']
    final_df = final_df[column_order]
    
    print(f"  Number of samples outside the domain: {sum(is_out)} / {len(X_pred)}")
    # Save CSV
    final_csv_name = 'Final_Predictions_with_AD.csv'
    final_df.to_csv(os.path.join(OUTPUT_DIR, final_csv_name), index=False)
    print(f"   [Done] The final prediction file has been successfully saved as: {os.path.join(OUTPUT_DIR, final_csv_name)}")

if __name__ == "__main__":
    cmp_file = 'compounds.csv'
    end_file = 'endpoints.csv'
    
    if os.path.exists(cmp_file) and os.path.exists(end_file):
        run_full_pipeline(cmp_file, end_file)
    else:
        print(f"Error：input file {cmp_file} or {end_file} not found")
