#AOP-MEPF

=============================================================================
README for Prediction Models
=============================================================================
Manuscript Title: Multi-endpoint screening framework of reproductive toxicants in fish with machine learning
Authors: Xuan Liu, Yanlun Zhang, Jiazhen Chen, Hongxia Yu, Hongling Liu
Corresponding Contact: hlliu@nju.edu.cn
Date: October 2026

=============================================================================
1. REMARKS / OVERVIEW
=============================================================================
This repository contains the Python source code used to develop and evaluate
machine learning-based Quantitative Structure-Activity Relationship (QSAR)
models for multiple in vitro and in vivo reproductive toxicity-related
endpoints in fish.

The repository includes endpoint-specific scripts, for example:

- 01ARModel.py
- 02VTGModel.py
- 03E2Model.py
- 04TModel.py
- 05HISModel.py
- 06REPModel.py

Each script follows a generally consistent modeling framework, while certain
details (e.g., endpoint column name, classifier type, feature selection
strategy, imbalance handling approach, or applicability domain settings) may
vary slightly depending on the endpoint and final optimized modeling protocol.

In general, the modeling workflow includes:
- Molecular feature generation from SMILES strings using RDKit
- Feature cleaning and preprocessing
- Optional or endpoint-specific feature selection
- Machine learning model training and hyperparameter optimization
- Independent or cross-validated model evaluation
- Applicability Domain (AD) analysis
- Export of predictions, metrics, and visualization outputs

Key characteristics of the repository include:
- Reproducible machine learning workflows with fixed random seeds
- Molecular representation using fingerprints and/or molecular descriptors
- Feature filtering to reduce redundancy and noise
- Support for endpoint-specific optimization strategies
- Explicit Applicability Domain (AD) analysis to define reliable prediction space
- Exportable prediction tables and publication-ready visualizations

=============================================================================
2. SYSTEM REQUIREMENTS & DEPENDENCIES
=============================================================================
The scripts were developed and tested using Python 3.11+.

The following Python packages are required to run the modeling scripts:

- numpy
- pandas
- scipy
- matplotlib
- seaborn
- scikit-learn
- imbalanced-learn (imblearn)
- RDKit
- boruta (required for scripts that use Boruta-based feature selection)
- Pillow (recommended for image processing compatibility)

Installation recommendation via conda:

    conda create -n qsar_env python=3.11
    conda activate qsar_env
    conda install -c conda-forge rdkit scikit-learn pandas numpy scipy \
        matplotlib seaborn imbalanced-learn pillow
    pip install boruta

For strict manuscript reproducibility, the following package versions are
recommended:

# Core Data Manipulation
pandas==1.5.3
numpy==1.23.5
scipy==1.10.1

# Cheminformatics Engine
rdkit==2022.09.5

# Machine Learning
scikit-learn==1.2.2
imbalanced-learn==0.10.1
boruta==0.3

# Visualization
matplotlib==3.7.1
seaborn==0.12.2
Pillow==9.4.0

=============================================================================
3. INPUT DATA FORMAT
=============================================================================
Each modeling script typically requires two CSV files located in the working
directory or specified by the user:

1. compounds.csv
   Required columns:
   - CID      : Compound identifier
   - SMILES   : Canonical or valid SMILES string

2. endpoints.csv
   Required columns:
   - CID
   - Endpoint label column corresponding to the target model

Examples of endpoint columns may include:
- ARagonism
- VTG-related endpoint column
- E2-related endpoint column
- T-related endpoint column
- HIS-related endpoint column
- REP-related endpoint column

Important notes:
- The exact endpoint column name depends on the specific script being run.
- Endpoint labels are typically binary (0/1), where:
    0 = inactive / negative
    1 = active / positive
- Missing labels (NA) are generally removed before model training.
- Only compounds with valid molecular structures and valid endpoint labels are
  retained for modeling.

=============================================================================
4. GENERAL MODELING WORKFLOW
=============================================================================
Although details may differ slightly among endpoint-specific scripts, the
common workflow generally includes the following steps:

[1] Data loading and endpoint label assignment
    - Load compound and endpoint files
    - Merge datasets using CID
    - Remove compounds with missing or invalid labels

[2] Molecular feature generation
    - Generate molecular representations from SMILES using RDKit
    - Depending on the script, features may include:
      * Morgan fingerprints
      * MACCS keys
      * Physicochemical descriptors
      * Hybrid descriptor combinations

[3] Feature preprocessing
    - Remove invalid or constant features
    - Apply variance filtering where needed
    - Remove highly correlated features where needed
    - Scale or normalize features where required by the model

[4] Feature selection (script-dependent)
    - Some scripts may use Boruta
    - Some scripts may use recursive elimination or other feature selection
      procedures
    - Some scripts may proceed without an additional wrapper-based selection step

[5] Model training and optimization
    - Train endpoint-specific machine learning models
    - Depending on the script, models may include:
      * Balanced Random Forest
      * Support Vector Machine
      * Other optimized classifiers if defined in the corresponding script
    - Hyperparameter tuning is performed using internal cross-validation when
      implemented

[6] Evaluation and reporting
    - Evaluate model performance using internal validation and/or an independent
      test set
    - Generate prediction outputs, summary metrics, and figures

[7] Applicability Domain (AD) analysis
    - Determine whether compounds fall inside or outside the model’s reliable
      chemical space
    - AD methods may vary by script, but are explicitly implemented for
      prediction reliability assessment

=============================================================================
5. USAGE / EXECUTION
=============================================================================
To execute the modeling pipeline, run the endpoint-specific Python script from
the command line or terminal. Example commands:

    python 01ARModel.py
    python 02VTGModel.py
    python 03E2Model.py
    python 04TModel.py
    python 05HISModel.py
    python 06REPModel.py

Each script is designed to run independently for its corresponding endpoint.

Before execution, please ensure:
- The required input CSV files are available
- File paths in the script are correct
- The endpoint column required by the script exists in the endpoint file
- All required dependencies are installed

=============================================================================
6. EXPECTED OUTPUT FILES
=============================================================================
Upon successful execution, each script may generate an output directory or a set
of output files in the working directory. The exact filenames may differ among
scripts, but typical outputs include:

- Final_Predictions_with_AD.csv
  Prediction results for training, validation, and/or test compounds, including
  predicted labels, probability scores, and AD status where applicable

- Model_Performance_Summary.csv or Manuscript_Report.txt
  Summary of model performance metrics, dataset composition, and important
  modeling settings

- Visualization files (*.png)
  Depending on the script, these may include:
  - ROC curve
  - Confusion matrix
  - Applicability Domain plot
  - Feature importance plot
  - Other endpoint-specific visualizations

- Intermediate or supporting files
  Some scripts may also export:
  - selected features
  - feature rankings
  - AD distances
  - train/test split annotations
  - tuned parameter summaries

=============================================================================
7. PERFORMANCE METRICS
=============================================================================
Depending on the endpoint-specific script, model performance may be assessed
using one or more of the following metrics:

- AUC (Area Under the ROC Curve)
- MCC (Matthews Correlation Coefficient)
- Accuracy
- Precision
- Recall / Sensitivity
- Specificity
- F1-score

Some scripts may additionally report:
- sample counts by class
- in-AD vs out-of-AD performance
- threshold-dependent prediction summaries

=============================================================================
8. REPRODUCIBILITY
=============================================================================
To improve reproducibility, the scripts generally fix random seeds for Python,
NumPy, and machine learning procedures whenever supported.

Nevertheless, minor numerical differences may still occur across operating
systems, package builds, and parallel computation environments.

For the most consistent manuscript reproduction, please use:
- the package versions listed above
- the same Python version
- the same script-specific parameter settings
- the same input data files used in the study

=============================================================================
9. IMPORTANT NOTES
=============================================================================
- This repository contains endpoint-specific scripts; therefore, not every
  script will use the same preprocessing, feature selection, imbalance
  handling, or AD method.
- The exact endpoint label column name must match the code in the selected
  script.
- Invalid SMILES strings or missing endpoint labels are typically removed during
  preprocessing.
- Users are encouraged to inspect the corresponding script for endpoint-specific
  implementation details before reproducing results.

=============================================================================
10. LICENSE
=============================================================================
This code is provided for peer-review and academic reproduction purposes.
All rights reserved pending publication.
