#!/usr/bin/env python3
"""
Populate Case Study Report with Actual JSON Data
This script reads training_data.json and populates the case study LaTeX report
with real metrics without needing to retrain the models.
"""

import sys
import os
import io

# Fix encoding for Windows
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Add parent directory to path to import report_generator
sys.path.insert(0, os.path.dirname(__file__))

from report_generator import populate_case_study_with_json_data, load_training_data_from_json

def main():
    """Populate case study report with actual JSON data"""
    print("="*70)
    print("CASE STUDY REPORT POPULATION WITH ACTUAL DATA")
    print("="*70)

    # Load the JSON data
    json_data = load_training_data_from_json()

    if json_data is None:
        print("\n❌ Error: No training_data.json found!")
        print("Please run training first: python src/spam_filter.py")
        return False

    # Display data that will be used
    print("\n📊 Data from training_data.json:")
    print(f"  Dataset source: {json_data['metadata']['dataset_source']}")
    print(f"  Total messages: {json_data['metadata']['total_messages']}")
    print(f"  Spam: {json_data['metadata']['spam_count']} ({json_data['dataset_stats']['spam_percentage']:.1f}%)")
    print(f"  Legitimate: {json_data['metadata']['ham_count']} ({json_data['dataset_stats']['ham_percentage']:.1f}%)")
    print(f"  Training set: {json_data['metadata']['training_size']}")
    print(f"  Test set: {json_data['metadata']['test_size']}")

    print("\n📈 Model Performance (from JSON):")
    for risk_level, model_data in json_data['model_results'].items():
        metrics = model_data['metrics']
        print(f"  {risk_level.upper()}:")
        print(f"    - Accuracy: {metrics['accuracy']:.0f}%")
        print(f"    - Precision: {metrics['precision']:.0f}%")
        print(f"    - Recall: {metrics['recall']:.0f}%")
        print(f"    - F1-Score: {metrics['f1_score']:.0f}%")
        print(f"    - False Positive Rate: {metrics['fp_rate']:.1f}%")

    print("\n" + "="*70)
    print("Populating case study report...")
    print("="*70)

    # Populate the case study report
    success = populate_case_study_with_json_data(json_data)

    if success:
        print("\n✅ SUCCESS!")
        print("\nCase study report has been populated with ACTUAL data from:")
        print("  - reports/spam_filter_case_study.tex")
        print("\nYou can now:")
        print("  1. Generate PDF: cd reports && pdflatex spam_filter_case_study.tex")
        print("  2. Or use Overleaf: Upload spam_filter_case_study.tex to Overleaf.com")
        print("\n📝 The report now contains:")
        print("  - Real dataset statistics")
        print("  - Actual model performance metrics")
        print("  - Real confusion matrices")
        print("  - Calculated daily performance estimates")
        return True
    else:
        print("\n❌ Error populating report")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
