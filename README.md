# Spam Filter Project

## Overview

This project implements an intelligent spam email filter using machine learning techniques. The system employs TF-IDF vectorization and Naive Bayes classification with configurable risk levels to balance between spam detection accuracy and false positive rates.

## Features

### Core Functionality
- **Machine Learning Classification**: Naive Bayes classifier with TF-IDF features
- **Risk-Level Adjustment**: Three configurable sensitivity levels (low, medium, high)
- **Automated Dataset Acquisition**: Kaggle integration using kagglehub API
- **Real-time Classification**: Individual message testing and batch processing
- **Interactive GUI**: User-friendly Tkinter interface for easy operation

### Advanced Features
- **Automated LaTeX Reporting**: Dynamic report generation with real performance data
- **Comprehensive Visualizations**: Data distribution plots, confusion matrices, ROC curves
- **Performance Analytics**: Detailed metrics across multiple evaluation criteria
- **Flexible Data Input**: Support for custom CSV datasets and sample data

## Installation

### Prerequisites
- Python 3.7 or higher
- pip package manager
- (Optional) LaTeX distribution for PDF report compilation

### Setup Instructions

1. **Clone or download the project files**:
   ```
   spam_filter.py
   gui_interface.py
   project_report.tex
   requirements.txt
   ```

2. **Install required packages**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Install additional packages if needed**:
   ```bash
   pip install kagglehub pandas scikit-learn matplotlib seaborn numpy tkinter
   ```

4. **For LaTeX reporting (optional)**:
   - Install a LaTeX distribution (TeX Live, MiKTeX, or MacTeX)
   - Ensure `pdflatex` is available in your system PATH

## Usage

### Command Line Interface

**Basic spam filter analysis**:
```bash
python spam_filter.py
```

This will:
- Download a spam dataset from Kaggle automatically
- Train models with all three risk levels
- Generate comprehensive visualizations
- Create an automated LaTeX report
- Display detailed performance metrics

### Graphical User Interface

**Launch the interactive GUI**:
```bash
python gui_interface.py
```

GUI Features:
- **Data Loading**: Load sample data or import CSV files
- **Model Training**: Configure and train models with different risk levels
- **Message Testing**: Test individual messages in real-time
- **Visualizations**: Interactive plots and analysis charts
- **Export Capabilities**: Save results and visualizations

## Project Structure

```
spam_filter/
├── spam_filter.py          # Main classification engine
├── gui_interface.py        # Interactive user interface
├── project_report.tex      # LaTeX report template
├── requirements.txt        # Python dependencies
├── README.md              # This file
├── task.txt               # Project objectives
├── data/                  # Dataset storage (auto-created)
├── figures/               # Generated visualizations (auto-created)
└── models/                # Saved models (auto-created)
```

## Risk Level Configuration

### Low Risk (Conservative)
- **Threshold**: 30% spam probability
- **Use Case**: Business-critical communications
- **Characteristics**: Minimizes false positives, may miss some spam

### Medium Risk (Balanced)
- **Threshold**: 50% spam probability  
- **Use Case**: General email filtering
- **Characteristics**: Optimal balance between precision and recall

### High Risk (Aggressive)
- **Threshold**: 70% spam probability
- **Use Case**: High-volume environments
- **Characteristics**: Maximizes spam detection, higher false positive rate

## Dataset Information

### Automatic Download
The system automatically downloads the "Spam Email Dataset" from Kaggle using the kagglehub API. The dataset provides professionally curated and labeled spam/ham messages for training and evaluation.

### Custom Data Support
You can also use your own datasets by ensuring they have the following format:
- CSV file with 'label' and 'message' columns
- Labels should be either 'spam' or 'ham'
- UTF-8 encoding recommended

### Sample Data
If Kaggle download fails, the system falls back to built-in sample data for demonstration purposes.

## Performance Metrics

The system evaluates models using comprehensive metrics:

- **Accuracy**: Overall classification correctness
- **Precision**: Percentage of spam predictions that are correct
- **Recall**: Percentage of actual spam messages detected
- **F1-Score**: Harmonic mean of precision and recall
- **False Positive Rate**: Legitimate emails incorrectly flagged as spam
- **ROC-AUC**: Area under the receiver operating characteristic curve

## Output Files

### Automatically Generated
- `data/kaggle_spam_data.csv` - Downloaded dataset
- `figures/data_exploration.png` - Dataset analysis visualizations
- `figures/confusion_matrix_*.png` - Confusion matrices for each risk level
- `figures/roc_curves.png` - ROC curve comparison
- `automated_project_report.tex` - LaTeX report with real data
- `metrics_summary.md` - Quick performance summary
- `automated_project_report.pdf` - Compiled report (if LaTeX available)

## Technical Details

### Machine Learning Pipeline
1. **Text Preprocessing**: Lowercase conversion, URL/phone removal, whitespace normalization
2. **Feature Extraction**: TF-IDF vectorization with optimized parameters
3. **Model Training**: Multinomial Naive Bayes with Laplace smoothing
4. **Risk Adjustment**: Configurable probability thresholds
5. **Evaluation**: Comprehensive metrics across test data

### Key Parameters
- **Max Features**: 3,000 (TF-IDF vocabulary limit)
- **Min Document Frequency**: 2 (rare term filtering)
- **Max Document Frequency**: 0.8 (common term filtering)
- **Alpha**: 0.1 (Naive Bayes smoothing)
- **Test Split**: 20% (train-test ratio)

## Troubleshooting

### Common Issues

**Kaggle Download Fails**:
- Check internet connection
- Verify kagglehub installation: `pip install kagglehub`
- System will automatically use sample data as fallback

**LaTeX Compilation Errors**:
- Ensure LaTeX distribution is installed
- Verify `pdflatex` is in system PATH
- The .tex file is still generated for manual compilation

**GUI Display Issues**:
- Update tkinter: Part of Python standard library
- Check display settings and screen resolution
- Ensure matplotlib backend supports GUI

**Memory Issues with Large Datasets**:
- Reduce max_features parameter in TF-IDF
- Use data sampling for very large datasets
- Close other applications to free memory

### Performance Tips

- Use SSD storage for faster data loading
- Ensure sufficient RAM (4GB+ recommended)
- Close unnecessary applications during training
- Use virtual environments to avoid conflicts

## Contributing

### Code Style
- Follow PEP 8 Python style guidelines
- Use descriptive variable names
- Include docstrings for functions and classes
- Comment complex logic sections

### Testing
- Test with different dataset formats
- Verify GUI functionality across platforms
- Check LaTeX compilation with various distributions
- Validate performance across risk levels

## License

This project is provided for educational and research purposes. Please ensure proper attribution when using the code or methodology.

## Acknowledgments

- **Dataset**: Spam Email Dataset by jackksoncsie on Kaggle
- **Libraries**: scikit-learn, pandas, matplotlib, tkinter, kagglehub
- **Methodology**: CRISP-DM framework for data mining projects

## Contact

For questions, issues, or suggestions, please create an issue in the project repository or contact the development team.

---

**Version**: 2.0  
**Last Updated**: March 2026  
**Python Version**: 3.7+  
**Status**: Production Ready