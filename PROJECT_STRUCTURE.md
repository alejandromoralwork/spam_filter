# Project Structure Overview

## Directory Organization (CRISP-DM Standard)

```
spam_filter/
├── src/                                # Source Code (CRISP-DM: Implementation)
│   ├── spam_filter.py                 # Core classification engine (1,100+ lines)
│   │                                   # - SpamFilter class with fit/predict methods
│   │                                   # - Risk-level management
│   │                                   # - Business confidence scoring
│   ├── gui_interface.py               # Interactive GUI for service teams (900+ lines)
│   │                                   # - Data Management tab
│   │                                   # - Message Classification tab
│   │                                   # - Business Analytics tab
│   │                                   # - Review Queue tab
│   └── report_generator.py            # Report generation utilities (600+ lines)
│                                       # - LaTeX report generation
│                                       # - JSON data persistence
│                                       # - Performance metrics
│
├── data/                               # Datasets (CRISP-DM: Data Understanding)
│   ├── raw/                           # Original, unprocessed data
│   │   └── SMSSpamCollection.csv      # 5,572 labeled SMS messages
│   │                                   # Columns: label, message
│   │                                   # Classes: spam (747), ham (4,825)
│   └── processed/                     # Processed datasets
│       └── training_data.json         # Training statistics and metrics
│
├── models/                             # Trained ML Models (CRISP-DM: Modeling)
│   ├── spam_filter_low_risk.pkl       # Conservative model (threshold: 0.30)
│   ├── spam_filter_medium_risk.pkl    # Balanced model (threshold: 0.50)
│   └── spam_filter_high_risk.pkl      # Permissive model (threshold: 0.70)
│
├── reports/                            # Documentation & Reports (CRISP-DM: Evaluation)
│   ├── FINAL_PROJECT_REPORT.md        # Comprehensive project report (6,000+ words)
│   │                                   # - Business context
│   │                                   # - CRISP-DM implementation details
│   │                                   # - Error analysis
│   │                                   # - Deployment guidelines
│   ├── project_report.tex             # LaTeX technical report
│   └── automated_project_report.tex   # Generated technical report
│
├── figures/                            # Generated Visualizations
│   └── [Confusion matrices, ROC curves, data distributions]
│
├── notebooks/                          # Jupyter Notebooks (for exploration)
│   └── [Optional: Analysis notebooks]
│
├── README.md                          # Main project documentation
│                                       # - Quick start guide
│                                       # - Project overview
│                                       # - Usage instructions
│                                       # - Deployment guidelines
│
├── requirements.txt                   # Python dependencies
│                                       # - pandas, numpy
│                                       # - scikit-learn
│                                       # - matplotlib, seaborn
│                                       # - kagglehub
│
├── task.txt                           # Project assignment/requirements
│
└── .gitignore                         # Git ignore configuration
```

---

## File Descriptions

### Source Code Files

#### `src/spam_filter.py` (1,100+ lines)
**Purpose**: Core spam classification engine

**Key Components**:
- `create_sample_data()`: Generates sample training data
- `SpamFilter` class: Main classification engine
  - `__init__()`: Initialize with model type and risk level
  - `fit()`: Train on labeled data
  - `predict()`: Classify messages with probabilities
  - `evaluate()`: Generate performance metrics
  - `save_model()`: Persist model to disk
  - `load_model()`: Load saved model
- `train_multiple_models()`: Train for all risk levels
- `main()`: CRISP-DM pipeline orchestration

**Features**:
- TF-IDF vectorization
- Multinomial Naive Bayes classification
- Risk-adjusted thresholds
- Business confidence scoring
- Command-line interface

#### `src/gui_interface.py` (900+ lines)
**Purpose**: Interactive GUI for service team operations

**Key Components**:
- `SpamFilterGUI` class: Main GUI application
- Four main tabs:
  1. **Data & Configuration**: Load data, select risk levels, train models
  2. **Message Classification**: Test messages, view predictions
  3. **Business Analytics**: Visualize data and model performance
  4. **Review Queue**: Manage messages requiring human review

**Features**:
- Model training and loading
- Real-time message classification
- Confidence-based review workflow
- Business metrics visualization
- Interactive plots and analysis

#### `src/report_generator.py` (600+ lines)
**Purpose**: Automated report generation and data persistence

**Key Functions**:
- `update_latex_report_from_json()`: Generate LaTeX reports
- `compile_latex_report()`: PDF compilation
- `load_training_data_from_json()`: Load metrics from file
- Report formatting and metrics organization

---

### Data Files

#### `data/raw/SMSSpamCollection.csv`
**Format**: CSV (label, message)
**Size**: ~478 KB
**Records**: 5,572 messages
**Distribution**:
- Spam: 747 messages (13.4%)
- Legitimate: 4,825 messages (86.6%)

**Columns**:
```
label      : str ("spam" or "ham")
message    : str (SMS text message)
```

#### `data/processed/training_data.json`
**Purpose**: Stores training statistics and metrics
**Format**: JSON
**Size**: ~2 KB
**Contents**:
- Dataset metadata
- Model performance metrics
- Training/test split information
- Message statistics (length, word count)

---

### Model Files

#### `models/spam_filter_*.pkl`
**Format**: Python pickle serialization
**Size**: 100-350 KB per model
**Contents**: Trained Multinomial Naive Bayes model, TF-IDF vectorizer, configuration

**Three Models**:
1. **Low Risk** (threshold 0.30): Conservative, blocks uncertain messages
2. **Medium Risk** (threshold 0.50): Balanced, general use
3. **High Risk** (threshold 0.70): Permissive, maximizes reach

---

### Documentation Files

#### `README.md`
**Purpose**: Main project documentation and quick reference
**Sections**:
- Project overview
- Quick start guide
- Project structure
- Model overview and performance
- Service team integration
- Deployment guidelines
- Troubleshooting

#### `reports/FINAL_PROJECT_REPORT.md`
**Purpose**: Comprehensive technical and business report
**Sections** (6,000+ words):
- Executive summary
- Business context
- CRISP-DM implementation (Phases 1-6)
- Data understanding and preparation
- Modeling details
- Error analysis
- Business strategy for error handling
- Deployment guidelines
- Conclusions and recommendations

#### `requirements.txt`
**Purpose**: Python package dependencies
**Packages**:
- pandas>=1.3.0: Data processing
- numpy>=1.21.0: Numerical computing
- scikit-learn>=1.0.0: Machine learning
- matplotlib>=3.3.0: Visualization
- seaborn>=0.11.0: Statistical plots
- kagglehub>=0.2.0: Dataset acquisition

---

## CRISP-DM Mapping

| CRISP-DM Phase | Implementation | Files |
|---|---|---|
| **1. Business Understanding** | Stakeholder analysis, problem definition | Reports section, README |
| **2. Data Understanding** | Data exploration, quality assessment | data/raw/SCC, analysis in reports |
| **3. Data Preparation** | Cleaning, preprocessing, feature engineering | src/spam_filter.py (preprocess_text) |
| **4. Modeling** | Algorithm selection, training, configuration | src/spam_filter.py, models/ |
| **5. Evaluation** | Metrics, error analysis, validation | src/spam_filter.py (evaluate), reports |
| **6. Deployment** | GUI, integration, monitoring | src/gui_interface.py, README |

---

## File Statistics

### Code Files
- **Total Lines of Code**: ~2,600 lines
- **Python Files**: 3 main files
- **Documentation**: 2 comprehensive reports

### Data Files
- **Raw Dataset**: 5,572 messages
- **Training Data**: 4,458 messages (80%)
- **Test Data**: 1,114 messages (20%)

### Models
- **Total Trained Models**: 3 (low, medium, high risk)
- **Total Model Size**: ~650 KB
- **Model Format**: Pickle (Python 3.7+)

---

## Workflow: File Dependencies

```
1. Data Loading
   └── data/raw/SMSSpamCollection.csv
   └── data/processed/training_data.json

2. Model Training
   ├── Input: Data files
   ├── Process: src/spam_filter.py
   └── Output: models/*.pkl

3. Evaluation & Reporting
   ├── Input: models/*.pkl
   ├── Process: src/report_generator.py
   └── Output: reports/FINAL_PROJECT_REPORT.md

4. Service Deployment
   ├── Models: models/*.pkl
   ├── GUI: src/gui_interface.py
   └── Documentation: README.md
```

---

## How to Use This Structure

### For Training
1. Run `python src/spam_filter.py`
2. Models saved to `models/`
3. Metrics saved to `data/processed/training_data.json`
4. Reports generated to `reports/`

### For Testing
1. Load model: `models/spam_filter_medium_risk.pkl`
2. Use GUI: `python src/gui_interface.py`
3. Or command-line: `python src/spam_filter.py --test-model <path> <message>`

### For Documentation
1. Read `README.md` for quick start
2. See `reports/FINAL_PROJECT_REPORT.md` for comprehensive details
3. Check `requirements.txt` for dependencies

---

## Clean-Up Summary

### Files Removed
- ❌ Business_Requirements.md (consolidated into final report)
- ❌ CRISP_DM_Documentation.md (consolidated into final report)
- ❌ Implementation_Summary.md (consolidated into final report)
- ❌ LaTeX_Report_Updates_Summary.md (archived)
- ❌ README_CRISP_DM.md (consolidated into main README)
- ❌ metrics_summary.md (consolidated into reports)
- ❌ test_accuracy.py (functionality in src/spam_filter.py)
- ❌ Temporary files (tmpclaude-*)

### Files Reorganized
- ✅ Source code → `src/` directory
- ✅ Data files → `data/raw/` and `data/processed/`
- ✅ Reports → `reports/` directory
- ✅ Models kept in `models/` directory

### Files Created
- ✅ `README.md` (comprehensive main documentation)
- ✅ `reports/FINAL_PROJECT_REPORT.md` (6,000+ word technical report)

---

## Version Information
- **Project Version**: 2.0
- **Date**: March 14, 2026
- **Status**: ✅ Production Ready
- **Python**: 3.7+
- **Last Updated**: March 14, 2026

---

## Quick Navigation

**Want to get started?** → See `README.md`
**Need technical details?** → See `reports/FINAL_PROJECT_REPORT.md`
**Want to train models?** → Run `python src/spam_filter.py`
**Want to use GUI?** → Run `python src/gui_interface.py`
**Need dependencies?** → Check `requirements.txt`
