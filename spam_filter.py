import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve
from sklearn.linear_model import LogisticRegression
import re
import string
import os
import kagglehub
from datetime import datetime

class SpamFilter:
    def __init__(self, model_type='naive_bayes', risk_level='medium'):
        """
        Initialize spam filter with adjustable risk level
        risk_level: 'low' (restrictive), 'medium', 'high' (permissive)
        """
        self.model_type = model_type
        self.risk_level = risk_level
        self.vectorizer = None
        self.model = None
        self.threshold = self._get_threshold()
        
    def _get_threshold(self):
        """Define spam probability thresholds based on risk level"""
        thresholds = {
            'low': 0.3,      # Very restrictive - flag if >30% spam probability
            'medium': 0.5,   # Balanced - flag if >50% spam probability
            'high': 0.7      # Permissive - flag if >70% spam probability
        }
        return thresholds.get(self.risk_level, 0.5)
    
    def preprocess_text(self, text):
        """Clean and preprocess text messages"""
        # Convert to lowercase
        text = text.lower()
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        # Remove phone numbers
        text = re.sub(r'\d{10,}', '', text)
        # Remove extra whitespace
        text = ' '.join(text.split())
        return text
    
    def fit(self, X_train, y_train):
        """Train the spam filter model"""
        # Preprocess texts
        X_train_processed = X_train.apply(self.preprocess_text)
        
        # Vectorize text using TF-IDF
        self.vectorizer = TfidfVectorizer(max_features=3000, min_df=2, max_df=0.8)
        X_train_vec = self.vectorizer.fit_transform(X_train_processed)
        
        # Train model
        if self.model_type == 'naive_bayes':
            self.model = MultinomialNB(alpha=0.1)
        else:
            self.model = LogisticRegression(max_iter=1000, random_state=42)
            
        self.model.fit(X_train_vec, y_train)
        
    def predict(self, X_test):
        """Predict spam with risk-adjusted threshold"""
        X_test_processed = X_test.apply(self.preprocess_text)
        X_test_vec = self.vectorizer.transform(X_test_processed)
        
        # Get probability predictions
        proba = self.model.predict_proba(X_test_vec)[:, 1]
        
        # Apply risk-adjusted threshold
        predictions = (proba >= self.threshold).astype(int)
        
        return predictions, proba
    
    def evaluate(self, X_test, y_test):
        """Comprehensive model evaluation"""
        predictions, probabilities = self.predict(X_test)
        
        print(f"\n{'='*60}")
        print(f"Evaluation Results (Risk Level: {self.risk_level.upper()})")
        print(f"Threshold: {self.threshold}")
        print(f"{'='*60}")
        print("\nClassification Report:")
        print(classification_report(y_test, predictions, target_names=['Ham', 'Spam']))
        
        cm = confusion_matrix(y_test, predictions)
        
        return {
            'predictions': predictions,
            'probabilities': probabilities,
            'confusion_matrix': cm
        }


def download_kaggle_dataset():
    """Download spam email dataset from Kaggle using kagglehub"""
    print("Downloading dataset from Kaggle...")
    
    # Download latest version
    path = kagglehub.dataset_download("jackksoncsie/spam-email-dataset")
    
    print("Path to dataset files:", path)
    return path


def standardize_dataset_format(df):
    """Standardize dataset format to have 'label' and 'message' columns"""
    print("\nStandardizing dataset format...")
    
    # Common column name mappings for spam datasets
    label_mappings = {
        'Category': 'label',
        'category': 'label',
        'class': 'label',
        'Class': 'label',
        'target': 'label',
        'Target': 'label',
        'spam': 'label',
        'Spam': 'label'
    }
    
    message_mappings = {
        'Message': 'message',
        'text': 'message',
        'Text': 'message',
        'email': 'message',
        'Email': 'message',
        'content': 'message',
        'Content': 'message',
        'body': 'message',
        'Body': 'message'
    }
    
    # Rename columns
    for old_name, new_name in label_mappings.items():
        if old_name in df.columns:
            df = df.rename(columns={old_name: new_name})
            break
    
    for old_name, new_name in message_mappings.items():
        if old_name in df.columns:
            df = df.rename(columns={old_name: new_name})
            break
    
    # Check if we have the required columns
    if 'label' not in df.columns or 'message' not in df.columns:
        print("Warning: Could not find standard 'label' and 'message' columns")
        print("Available columns:", df.columns.tolist())
        print("Please manually specify the correct column names")
        
        # Take the first two columns as label and message
        if len(df.columns) >= 2:
            df = df.rename(columns={df.columns[0]: 'label', df.columns[1]: 'message'})
            print(f"Using '{df.columns[0]}' as label and '{df.columns[1]}' as message")
    
    # Standardize label values (spam/ham)
    if 'label' in df.columns:
        # Convert to lowercase and map common variations
        df['label'] = df['label'].astype(str).str.lower()
        
        # Map different spam/ham representations
        spam_variations = ['spam', '1', 'true', 'yes', 'positive']
        ham_variations = ['ham', 'legitimate', '0', 'false', 'no', 'negative']
        
        df['label'] = df['label'].apply(lambda x: 'spam' if x in spam_variations else 'ham')
    
    # Keep only the required columns
    df = df[['label', 'message']].copy()
    
    # Remove any rows with missing values
    df = df.dropna()
    
    print(f"Dataset standardized: {len(df)} rows with columns {df.columns.tolist()}")
    print("Label distribution:")
    print(df['label'].value_counts())
    
    return df


def load_kaggle_dataset(dataset_path):
    """Load the Kaggle spam email dataset"""
    # Look for CSV files in the downloaded path
    csv_files = []
    for root, dirs, files in os.walk(dataset_path):
        for file in files:
            if file.endswith('.csv'):
                csv_files.append(os.path.join(root, file))
    
    if not csv_files:
        raise FileNotFoundError("No CSV files found in the downloaded dataset")
    
    # Load the first CSV file found
    df = pd.read_csv(csv_files[0])
    print(f"Loaded dataset with {len(df)} rows from {csv_files[0]}")
    
    # Display basic info about the dataset
    print("\nOriginal dataset columns:")
    print(df.columns.tolist())
    print("\nFirst few rows:")
    print(df.head())
    
    # Standardize the dataset format
    df = standardize_dataset_format(df)
    
    return df


def generate_latex_stats(df, models_results):
    """Generate LaTeX content with actual statistics"""
    stats = {}
    
    # Dataset statistics
    stats['total_messages'] = len(df)
    stats['spam_count'] = len(df[df['label'] == 'spam'])
    stats['ham_count'] = len(df[df['label'] == 'ham'])
    stats['spam_percentage'] = (stats['spam_count'] / stats['total_messages']) * 100
    stats['ham_percentage'] = (stats['ham_count'] / stats['total_messages']) * 100
    
    # Message length statistics
    df_temp = df.copy()
    df_temp['message_length'] = df_temp['message'].apply(len)
    df_temp['word_count'] = df_temp['message'].apply(lambda x: len(x.split()))
    
    stats['avg_spam_length'] = df_temp[df_temp['label'] == 'spam']['message_length'].mean()
    stats['avg_ham_length'] = df_temp[df_temp['label'] == 'ham']['message_length'].mean()
    stats['avg_spam_words'] = df_temp[df_temp['label'] == 'spam']['word_count'].mean()
    stats['avg_ham_words'] = df_temp[df_temp['label'] == 'ham']['word_count'].mean()
    
    # Model performance statistics
    for risk_level, results in models_results.items():
        cm = results['confusion_matrix']
        tn, fp, fn, tp = cm.ravel()
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        
        stats[f'{risk_level}_precision'] = precision * 100
        stats[f'{risk_level}_recall'] = recall * 100
        stats[f'{risk_level}_f1'] = f1 * 100
        stats[f'{risk_level}_accuracy'] = accuracy * 100
        stats[f'{risk_level}_fp_rate'] = (fp / (fp + tn)) * 100 if (fp + tn) > 0 else 0
    
    return stats


def create_performance_table_latex(models_results):
    """Generate LaTeX table with performance metrics"""
    latex_table = """
\\begin{table}[H]
\\centering
\\caption{Model Performance Across Risk Levels}
\\begin{tabular}{@{}lcccc@{}}
\\toprule
\\textbf{Risk Level} & \\textbf{Accuracy (\\%)} & \\textbf{Precision (\\%)} & \\textbf{Recall (\\%)} & \\textbf{FP Rate (\\%)} \\\\
\\midrule
"""
    
    for risk_level, results in models_results.items():
        cm = results['confusion_matrix']
        tn, fp, fn, tp = cm.ravel()
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        fp_rate = fp / (fp + tn) if (fp + tn) > 0 else 0
        
        latex_table += f"{risk_level.capitalize()} & {accuracy*100:.1f} & {precision*100:.1f} & {recall*100:.1f} & {fp_rate*100:.1f} \\\\\n"
    
    latex_table += """\\bottomrule
\\end{tabular}
\\end{table}
"""
    return latex_table


def create_dataset_overview_latex(stats):
    """Generate LaTeX content for dataset overview"""
    return f"""
\\subsection{{Dataset Statistics}}

The dataset analysis revealed the following characteristics:

\\begin{{itemize}}
    \\item \\textbf{{Total Messages:}} {stats['total_messages']:,}
    \\item \\textbf{{Spam Messages:}} {stats['spam_count']:,} ({stats['spam_percentage']:.1f}\\%)
    \\item \\textbf{{Ham Messages:}} {stats['ham_count']:,} ({stats['ham_percentage']:.1f}\\%)
\\end{{itemize}}

\\subsubsection{{Message Length Analysis}}

Analysis of message characteristics shows distinct patterns between spam and legitimate messages:

\\begin{{table}}[H]
\\centering
\\caption{{Message Characteristics by Type}}
\\begin{{tabular}}{{@{{}}lcc@{{}}}}
\\toprule
\\textbf{{Metric}} & \\textbf{{Spam}} & \\textbf{{Ham}} \\\\
\\midrule
Average Length (chars) & {stats['avg_spam_length']:.0f} & {stats['avg_ham_length']:.0f} \\\\
Average Word Count & {stats['avg_spam_words']:.1f} & {stats['avg_ham_words']:.1f} \\\\
\\bottomrule
\\end{{tabular}}
\\end{{table}}

These statistics indicate that {'spam messages tend to be longer' if stats['avg_spam_length'] > stats['avg_ham_length'] else 'ham messages tend to be longer'}, which provides valuable features for classification.
"""


def update_latex_report(df, models_results, dataset_source="Kaggle"):
    """Update the LaTeX report with actual results"""
    print("Generating automated LaTeX report...")
    
    # Generate statistics
    stats = generate_latex_stats(df, models_results)
    
    # Read the existing LaTeX template
    try:
        with open('project_report.tex', 'r', encoding='utf-8') as f:
            latex_content = f.read()
    except Exception as e:
        print(f"Error reading LaTeX file: {e}")
        return
    
    # Create updated sections
    dataset_overview = create_dataset_overview_latex(stats)
    performance_table = create_performance_table_latex(models_results)
    
    # Update date
    current_date = datetime.now().strftime("%B %d, %Y")
    latex_content = re.sub(r'March 3, 2026', current_date, latex_content)
    
    # Update dataset source information
    if "kaggle" in dataset_source.lower():
        dataset_info = f"""
\\subsection{{Dataset Source}}

The dataset was automatically downloaded from Kaggle using the kagglehub API. The specific dataset used is the "Spam Email Dataset" by jackksoncsie, which provides a comprehensive collection of labeled spam and ham messages.

\\textbf{{Dataset Details:}}
\\begin{{itemize}}
    \\item \\textbf{{Source:}} Kaggle - jackksoncsie/spam-email-dataset
    \\item \\textbf{{Download Date:}} {current_date}
    \\item \\textbf{{Total Messages:}} {stats['total_messages']:,}
    \\item \\textbf{{Quality:}} Professionally curated and labeled
\\end{{itemize}}
"""
    else:
        dataset_info = f"""
\\subsection{{Dataset Source}}

The analysis was performed on a sample dataset containing {stats['total_messages']:,} messages for demonstration purposes.
"""
    
    # Insert dataset overview after "Data Quality Assessment and Exploration" section
    section_target = '\\section{Data Quality Assessment and Exploration}'
    if section_target in latex_content:
        latex_content = latex_content.replace(section_target, 
                                            f'{section_target}\n{dataset_info}\n{dataset_overview}', 1)
    
    # Insert performance table after "Performance Metrics Overview"
    metrics_target = '\\subsection{Performance Metrics Overview}'
    if metrics_target in latex_content:
        latex_content = latex_content.replace(metrics_target,
                                            f'{metrics_target}\n\nThe following table summarizes the model performance across different risk levels:\n{performance_table}', 1)
    
    # Add results summary
    best_accuracy = max(stats.get('high_accuracy', 0), stats.get('medium_accuracy', 0), stats.get('low_accuracy', 0))
    best_level = 'High' if stats.get('high_accuracy', 0) == best_accuracy else 'Medium' if stats.get('medium_accuracy', 0) == best_accuracy else 'Low'
    
    results_summary = f"""
\\subsection{{Key Results Summary}}

The automated analysis of {stats['total_messages']:,} messages yielded the following key insights:

\\begin{{enumerate}}
    \\item \\textbf{{Dataset Composition:}} {stats['spam_percentage']:.1f}\\% spam, {stats['ham_percentage']:.1f}\\% legitimate messages
    \\item \\textbf{{Best Performance:}} {best_level} risk level achieved highest accuracy ({best_accuracy:.1f}\\%)
    \\item \\textbf{{False Positive Control:}} Low risk level maintains FP rate at {stats.get('low_fp_rate', 0):.1f}\\%
    \\item \\textbf{{Message Patterns:}} {'Spam messages are longer on average' if stats['avg_spam_length'] > stats['avg_ham_length'] else 'Ham messages are longer on average'}
\\end{{enumerate}}
"""
    
    # Insert results summary after performance table
    confidence_target = '\\subsection{Model Confidence Analysis}'
    if confidence_target in latex_content:
        latex_content = latex_content.replace(confidence_target,
                                            f'{results_summary}\n{confidence_target}', 1)
    
    # Save updated LaTeX file
    output_file = 'automated_project_report.tex'
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(latex_content)
        print(f"Updated LaTeX report saved as: {output_file}")
        
        # Also create a summary file with key metrics
        create_metrics_summary(stats, models_results)
        
    except Exception as e:
        print(f"Error saving LaTeX file: {e}")


def create_metrics_summary(stats, models_results):
    """Create a simple metrics summary file"""
    best_accuracy = max(stats.get('high_accuracy', 0), stats.get('medium_accuracy', 0), stats.get('low_accuracy', 0))
    best_accuracy_level = 'High' if stats.get('high_accuracy', 0) == best_accuracy else 'Medium' if stats.get('medium_accuracy', 0) == best_accuracy else 'Low'
    best_recall = max(stats.get('high_recall', 0), stats.get('medium_recall', 0), stats.get('low_recall', 0))
    best_recall_level = 'High' if stats.get('high_recall', 0) == best_recall else 'Medium' if stats.get('medium_recall', 0) == best_recall else 'Low'
    
    summary_content = f"""# Automated Spam Filter Analysis Summary

Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Dataset Overview
- Total Messages: {stats['total_messages']:,}
- Spam Messages: {stats['spam_count']:,} ({stats['spam_percentage']:.1f}%)
- Ham Messages: {stats['ham_count']:,} ({stats['ham_percentage']:.1f}%)

## Message Characteristics
- Average Spam Length: {stats['avg_spam_length']:.0f} characters
- Average Ham Length: {stats['avg_ham_length']:.0f} characters
- Average Spam Word Count: {stats['avg_spam_words']:.1f} words
- Average Ham Word Count: {stats['avg_ham_words']:.1f} words

## Model Performance by Risk Level

### Low Risk (Restrictive)
- Accuracy: {stats.get('low_accuracy', 0):.1f}%
- Precision: {stats.get('low_precision', 0):.1f}%
- Recall: {stats.get('low_recall', 0):.1f}%
- False Positive Rate: {stats.get('low_fp_rate', 0):.1f}%

### Medium Risk (Balanced)
- Accuracy: {stats.get('medium_accuracy', 0):.1f}%
- Precision: {stats.get('medium_precision', 0):.1f}%
- Recall: {stats.get('medium_recall', 0):.1f}%
- False Positive Rate: {stats.get('medium_fp_rate', 0):.1f}%

### High Risk (Permissive)
- Accuracy: {stats.get('high_accuracy', 0):.1f}%
- Precision: {stats.get('high_precision', 0):.1f}%
- Recall: {stats.get('high_recall', 0):.1f}%
- False Positive Rate: {stats.get('high_fp_rate', 0):.1f}%

## Recommendations
- Best overall performance: {best_accuracy_level} risk level
- Lowest false positive rate: Low risk level ({stats.get('low_fp_rate', 0):.1f}%)
- Best recall: {best_recall_level} risk level ({best_recall:.1f}%)
"""
    
    try:
        with open('metrics_summary.md', 'w', encoding='utf-8') as f:
            f.write(summary_content)
        print("Metrics summary saved as: metrics_summary.md")
    except Exception as e:
        print(f"Error saving metrics summary: {e}")


def compile_latex_report():
    """Attempt to compile the LaTeX report to PDF"""
    import subprocess
    
    try:
        # Try to compile the LaTeX document
        result = subprocess.run(['pdflatex', 'automated_project_report.tex'], 
                              capture_output=True, text=True, cwd='.')
        
        if result.returncode == 0:
            print("LaTeX report successfully compiled to PDF!")
            print("Generated: automated_project_report.pdf")
            
            # Run again for proper references
            subprocess.run(['pdflatex', 'automated_project_report.tex'], 
                          capture_output=True, text=True, cwd='.')
            
        else:
            print("LaTeX compilation failed. Make sure pdflatex is installed.")
            print("You can manually compile using: pdflatex automated_project_report.tex")
            
    except FileNotFoundError:
        print("pdflatex not found. Please install LaTeX to compile the report.")
        print("The .tex file has been generated and can be compiled manually.")
    except Exception as e:
        print(f"Error during LaTeX compilation: {e}")


def load_and_explore_data(filepath='spam_data.csv'):
    """Load and perform initial data exploration"""
    df = pd.read_csv(filepath, encoding='latin-1')
    
    print("Dataset Overview:")
    print(f"Total messages: {len(df)}")
    print(f"\nClass distribution:")
    print(df['label'].value_counts())
    print(f"\nClass percentages:")
    print(df['label'].value_counts(normalize=True) * 100)
    
    # Basic statistics
    df['message_length'] = df['message'].apply(len)
    df['word_count'] = df['message'].apply(lambda x: len(x.split()))
    
    return df


def visualize_data(df):
    """Create visualizations for EDA"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Class distribution
    df['label'].value_counts().plot(kind='bar', ax=axes[0, 0], color=['green', 'red'])
    axes[0, 0].set_title('Class Distribution')
    axes[0, 0].set_xlabel('Class')
    axes[0, 0].set_ylabel('Count')
    
    # Message length distribution
    df.boxplot(column='message_length', by='label', ax=axes[0, 1])
    axes[0, 1].set_title('Message Length by Class')
    axes[0, 1].set_xlabel('Class')
    axes[0, 1].set_ylabel('Length')
    
    # Word count distribution
    df.boxplot(column='word_count', by='label', ax=axes[1, 0])
    axes[1, 0].set_title('Word Count by Class')
    axes[1, 0].set_xlabel('Class')
    axes[1, 0].set_ylabel('Word Count')
    
    # Histogram of message lengths
    axes[1, 1].hist([df[df['label']=='ham']['message_length'], 
                     df[df['label']=='spam']['message_length']], 
                    label=['Ham', 'Spam'], bins=30, alpha=0.7)
    axes[1, 1].set_title('Message Length Distribution')
    axes[1, 1].set_xlabel('Length')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig('figures/data_exploration.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_confusion_matrix(cm, risk_level):
    """Plot confusion matrix"""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Ham', 'Spam'], 
                yticklabels=['Ham', 'Spam'])
    plt.title(f'Confusion Matrix - {risk_level.upper()} Risk Level')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(f'figures/confusion_matrix_{risk_level}.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_roc_curves(models_results, y_test):
    """Plot ROC curves for different risk levels"""
    plt.figure(figsize=(10, 8))
    
    for risk_level, results in models_results.items():
        fpr, tpr, _ = roc_curve(y_test, results['probabilities'])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'{risk_level.upper()} Risk (AUC = {roc_auc:.3f})')
    
    plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves - Different Risk Levels')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.savefig('figures/roc_curves.png', dpi=300, bbox_inches='tight')
    plt.close()


def create_sample_data():
    """Create sample spam dataset for demonstration"""
    spam_messages = [
        "Congratulations! You've won a $1000 gift card. Call now!",
        "URGENT: Click here to claim your prize money now!!!",
        "Free entry in 2 a weekly comp to win FA Cup final tickets",
        "You have been selected to receive a cash prize. Text WIN to 12345",
        "Limited time offer! Get 50% off on all products. Visit our website now!",
        "WINNER!! As a valued customer you have been selected to receive £900 prize reward!",
        "This is the 2nd time we have tried to contact you. Call 09061701461",
        "Claim your free holiday package now! Limited spots available.",
        "You've been chosen to test our new product for FREE! Reply YES",
        "Urgent! Your account needs verification. Click link immediately."
    ]
    
    ham_messages = [
        "Hey, are you free for dinner tonight?",
        "Can you pick up some milk on your way home?",
        "The meeting has been rescheduled to 3 PM tomorrow.",
        "Thanks for your help with the project yesterday!",
        "I'll be there in 10 minutes.",
        "Did you receive my email about the proposal?",
        "Let me know when you're available to discuss this.",
        "Great job on the presentation today!",
        "Can we reschedule our appointment to next week?",
        "I need some help with this issue, when can we talk?"
    ]
    
    # Create more samples
    spam_messages = spam_messages * 10
    ham_messages = ham_messages * 15
    
    labels = ['spam'] * len(spam_messages) + ['ham'] * len(ham_messages)
    messages = spam_messages + ham_messages
    
    df = pd.DataFrame({
        'label': labels,
        'message': messages
    })
    
    # Shuffle the dataset
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    return df


def main():
    """Main execution function"""
    print("Spam Filter Project - CRISP-DM Implementation")
    print("="*60)
    
    # Create directories
    os.makedirs('figures', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    os.makedirs('data', exist_ok=True)
    
    # Download and load Kaggle dataset
    try:
        print("\nDownloading spam email dataset from Kaggle...")
        dataset_path = download_kaggle_dataset()
        df = load_kaggle_dataset(dataset_path)
        
        # Save a copy locally for future use
        df.to_csv('data/kaggle_spam_data.csv', index=False)
        print("Dataset saved locally as data/kaggle_spam_data.csv")
        
    except Exception as e:
        print(f"Error downloading Kaggle dataset: {e}")
        print("Falling back to local or sample dataset...")
        
        # Try to load existing Kaggle dataset first
        if os.path.exists('data/kaggle_spam_data.csv'):
            print("\nLoading existing Kaggle dataset...")
            df = pd.read_csv('data/kaggle_spam_data.csv')
        elif os.path.exists('data/spam_data.csv'):
            print("\nLoading existing sample dataset...")
            df = pd.read_csv('data/spam_data.csv')
        else:
            print("\nCreating sample dataset...")
            df = create_sample_data()
            df.to_csv('data/spam_data.csv', index=False)
            print(f"Sample dataset created with {len(df)} messages")
    
    # Data exploration
    print("\n" + "="*60)
    print("DATA EXPLORATION")
    print("="*60)
    
    print(f"\nTotal messages: {len(df)}")
    print(f"\nClass distribution:")
    print(df['label'].value_counts())
    print(f"\nClass percentages:")
    print(df['label'].value_counts(normalize=True) * 100)
    
    # Add message statistics
    df['message_length'] = df['message'].apply(len)
    df['word_count'] = df['message'].apply(lambda x: len(x.split()))
    
    print(f"\nMessage length statistics:")
    print(df.groupby('label')['message_length'].describe())
    
    # Visualize data
    print("\nGenerating visualizations...")
    visualize_data(df)
    print("Saved: figures/data_exploration.png")
    
    # Prepare data
    print("\n" + "="*60)
    print("MODEL TRAINING")
    print("="*60)
    
    # Convert labels to binary
    df['label_binary'] = (df['label'] == 'spam').astype(int)
    
    # Split data
    X = df['message']
    y = df['label_binary']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\nTraining set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")
    
    # Train models with different risk levels
    models_results = {}
    
    for risk_level in ['low', 'medium', 'high']:
        print(f"\n{'='*60}")
        print(f"Training model with {risk_level.upper()} risk level...")
        print(f"{'='*60}")
        
        model = SpamFilter(model_type='naive_bayes', risk_level=risk_level)
        model.fit(X_train, y_train)
        results = model.evaluate(X_test, y_test)
        
        models_results[risk_level] = results
        
        # Plot confusion matrix
        plot_confusion_matrix(results['confusion_matrix'], risk_level)
        print(f"Saved: figures/confusion_matrix_{risk_level}.png")
    
    # Plot ROC curves for all models
    print("\nGenerating ROC curves...")
    plot_roc_curves(models_results, y_test)
    print("Saved: figures/roc_curves.png")
    
    # Generate automated LaTeX report
    print("\n" + "="*60)
    print("AUTOMATED REPORT GENERATION")
    print("="*60)
    
    # Determine dataset source
    dataset_source = "Kaggle" if os.path.exists('data/kaggle_spam_data.csv') else "Sample"
    
    # Update LaTeX report with actual results
    update_latex_report(df, models_results, dataset_source)
    
    # Attempt to compile LaTeX to PDF
    compile_latex_report()
    
    print("\n" + "="*60)
    print("PROJECT COMPLETION")
    print("="*60)
    print("\nAll models trained and evaluated successfully!")
    print("\nGenerated files:")
    print("  - figures/data_exploration.png")
    print("  - figures/confusion_matrix_low.png")
    print("  - figures/confusion_matrix_medium.png")
    print("  - figures/confusion_matrix_high.png")
    print("  - figures/roc_curves.png")
    print("  - automated_project_report.tex (updated with actual results)")
    print("  - metrics_summary.md (quick reference)")
    if os.path.exists('automated_project_report.pdf'):
        print("  - automated_project_report.pdf (compiled report)")
    
    print("\nNext steps:")
    print("  1. Review the automated report: automated_project_report.tex")
    print("  2. Check the metrics summary: metrics_summary.md")
    if not os.path.exists('automated_project_report.pdf'):
        print("  3. Compile LaTeX manually: pdflatex automated_project_report.tex")
    print("  4. Run the GUI interface: python gui_interface.py")


if __name__ == "__main__":
    main()
