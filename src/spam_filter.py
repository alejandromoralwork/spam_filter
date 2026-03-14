"""
Short Message Spam Filter - CRISP-DM Implementation

This module implements a comprehensive spam filter for short message classification
in customer service channels, following CRISP-DM (Cross-Industry Standard Process 
for Data Mining) methodology.

Focus: SHORT MESSAGES (SMS, chat, social media) - NOT email classification
Purpose: Customer service channel protection with adjustable risk levels
Target Users: Service teams managing customer communication channels

CRISP-DM Phase 4: Modeling - Core classification algorithms and risk management

Author: Data Science Team
Date: March 2026
Version: 2.0 - CRISP-DM Aligned
"""

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
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import report generation functionality
import report_generator


def create_sample_data():
    """Create sample short message dataset for customer service channel testing
    
    Focus: Short messages typical of customer service channels
    Context: Customer feedback, support requests, and product inquiries
    Business Use: Service team training and system validation
    
    Returns:
        DataFrame: Sample short messages with business context labels
    """
    
    # Spam short messages (50 examples) - Customer service channel context
    spam_messages = [
        "Congratulations! You've won $1000! Click here now!",
        "URGENT: Your account will be closed. Verify now: fake-link.com",
        "Make money fast! Work from home! Call 555-SCAM",
        "You've won a FREE iPhone! Claim it now!!!",
        "CLICK HERE FOR AMAZING DEALS! Limited time offer!",
        "Your credit score needs attention. Get approved now!",
        "Hot singles in your area want to meet you!",
        "Get rich quick scheme! Guaranteed returns!",
        "Act now! Limited time offer expires today!",
        "Free lottery tickets! You could win millions!",
        "LOSE 30 POUNDS IN 30 DAYS! Miracle diet pill!",
        "Nigerian prince needs your help! $10 million reward!",
        "CLICK NOW! Free vacation to Hawaii! No strings attached!",
        "Urgent! Your PayPal account suspended! Verify immediately!",
        "Make $5000 per week working from home! No experience needed!",
        "CONGRATULATIONS! You're our 1 millionth visitor! Claim prize now!",
        "FREE VIAGRA! Discreet shipping! Order now!",
        "Your computer is infected! Download our antivirus now!",
        "REFINANCE NOW! Lowest rates ever! Bad credit OK!",
        "Win cash prizes! Enter our sweepstakes! Limited time!",
        "URGENT: IRS notice! You owe back taxes! Pay now!",
        "Free trial! Cancel anytime! (Hidden fees apply)",
        "WEIGHT LOSS MIRACLE! Doctors hate this trick!",
        "Investment opportunity! Double your money in 30 days!",
        "FREE GIFT CARDS! Amazon, iTunes, Google Play!",
        "Your warranty is about to expire! Extend now!",
        "CLICK HERE to claim your inheritance! $50,000 waiting!",
        "FREE CREDIT REPORT! Check your score now! No fees!",
        "AMAZING OPPORTUNITY! Be your own boss! Work from home!",
        "WIN BIG! Online casino! First bet free!",
        "URGENT NOTICE: Your email will be deleted! Verify now!",
        "GET RICH QUICK! Bitcoin investment! Guaranteed profits!",
        "FREE SAMPLES! Beauty products! Pay only shipping!",
        "YOUR PACKAGE IS WAITING! Pay customs fee to release!",
        "CONGRATULATIONS! You qualified for $10,000 loan!",
        "CLICK NOW! Secret to making millions revealed!",
        "URGENT: Bank security alert! Verify your account!",
        "FREE iPAD! Limited quantity! First come first serve!",
        "LOSE WEIGHT WITHOUT DIET OR EXERCISE! Magic pill!",
        "WIN A NEW CAR! Enter our contest! No purchase necessary!",
        "URGENT: Microsoft security warning! Your PC is at risk!",
        "FREE MONEY! Government grants available! Apply now!",
        "CONGRATULATIONS! You've been selected! Claim your prize!",
        "URGENT: Your subscription expires today! Renew now!",
        "MIRACLE CURE! Doctors shocked! Big pharma hates this!",
        "WIN $1000 CASH! Complete our survey! Takes 2 minutes!",
        "URGENT: Social Security suspension notice! Call now!",
        "FREE TRIAL OFFER! Cancel anytime! (Automatic billing)",
        "CONGRATULATIONS! You're approved! $5000 cash advance!",
        "CLICK HERE! Secret method to make money online!"
    ]
    
    # Ham messages (50 examples)
    ham_messages = [
        "Hi, can we reschedule our meeting to 3 PM?",
        "Thanks for the presentation slides. Very helpful!",
        "Reminder: Team standup at 9 AM tomorrow.",
        "Your order has been shipped. Tracking: 123456789",
        "Happy birthday! Hope you have a wonderful day.",
        "The project deadline has been extended to next Friday.",
        "Please review the attached document when you have time.",
        "Lunch at the usual place? 12:30 PM works for me.",
        "Flight confirmation: Your boarding pass is attached.",
        "Thank you for your purchase. Receipt is attached.",
        "Could you please send me the quarterly report?",
        "Meeting minutes from yesterday are now available.",
        "Don't forget about the client call at 2 PM.",
        "Your subscription renewal is due next month.",
        "Welcome to our newsletter! Thank you for subscribing.",
        "Password reset requested for your account.",
        "Your appointment has been confirmed for Monday.",
        "Weekly team update: All projects on track.",
        "Invoice #12345 is attached for your records.",
        "System maintenance scheduled for this weekend.",
        "Conference room booking confirmed for Thursday.",
        "Please complete your timesheet by Friday.",
        "New employee orientation starts Monday at 9 AM.",
        "Your package was delivered successfully.",
        "Reminder: Health insurance enrollment deadline approaching.",
        "Monthly sales figures are now available in the portal.",
        "Your request has been approved by management.",
        "Thank you for attending the workshop yesterday.",
        "Project status update: Phase 1 completed.",
        "Your vacation request has been processed.",
        "Security update installed on all company devices.",
        "Budget review meeting scheduled for next Tuesday.",
        "Your feedback survey response has been received.",
        "Office will be closed on Monday for the holiday.",
        "New policy document uploaded to the company intranet.",
        "Training session rescheduled to next Wednesday.",
        "Your expense report has been approved.",
        "Quarterly goals review scheduled for next week.",
        "System backup completed successfully last night.",
        "Your profile information has been updated.",
        "Committee meeting minutes attached for review.",
        "Annual performance review scheduled for next month.",
        "Your request for additional resources is being reviewed.",
        "Staff parking permits are available at reception.",
        "New safety guidelines posted on the notice board.",
        "Your certificate of completion is ready for pickup.",
        "Monthly newsletter published on the company website.",
        "Your access privileges have been updated.",
        "Vendor contract renewal requires your signature.",
        "Your suggestion has been forwarded to management."
    ]
    
    # Combine messages and create labels
    all_messages = spam_messages + ham_messages
    all_labels = ['spam'] * len(spam_messages) + ['ham'] * len(ham_messages)
    
    # Shuffle the data to mix spam and ham messages
    combined_data = list(zip(all_messages, all_labels))
    np.random.seed(42)  # For reproducible results
    np.random.shuffle(combined_data)
    
    # Separate back into messages and labels
    shuffled_messages, shuffled_labels = zip(*combined_data)
    
    sample_data = {
        'message': list(shuffled_messages),
        'label': list(shuffled_labels)
    }
    
    df = pd.DataFrame(sample_data)
    return df


class SpamFilter:
    """Short Message Spam Filter for Customer Service Channels
    
    CRISP-DM Phase 4: Modeling Implementation
    
    Designed specifically for short message classification in customer service
    environments. Provides adjustable risk levels to balance customer experience
    with spam protection based on business requirements.
    
    Business Context:
    - Customer service channel protection
    - Adjustable risk tolerance for different communication channels
    - Service team workflow integration
    - Business confidence and error analysis
    """
    
    def __init__(self, model_type='naive_bayes', risk_level='medium'):
        """
        Initialize short message spam filter with business risk configuration
        
        Args:
            model_type (str): Algorithm type ('naive_bayes' recommended for short text)
            risk_level (str): Business risk tolerance
                - 'low': Very restrictive (premium customer channels)
                - 'medium': Balanced (general customer communication)
                - 'high': Permissive (open feedback channels)
        
        Business Impact:
            Low risk: Minimizes spam pass-through, accepts more false positives
            High risk: Minimizes false positives, accepts some spam pass-through
        """
        self.model_type = model_type
        self.risk_level = risk_level
        self.vectorizer = None
        self.model = None
        self.threshold = self._get_threshold()
        self.business_context = self._get_business_context()
        
    def _get_threshold(self):
        """Define spam probability thresholds based on business risk level
        
        Business Logic:
        - Lower threshold = more restrictive = less spam passes through
        - Higher threshold = more permissive = fewer false positives
        
        Returns:
            float: Probability threshold for spam classification
        """
        thresholds = {
            'low': 0.3,      # Very restrictive - Premium customer channels
            'medium': 0.5,   # Balanced - General customer communication  
            'high': 0.7      # Permissive - Open feedback channels
        }
        return thresholds.get(self.risk_level, 0.5)
    
    def _get_business_context(self):
        """Get business context and implications for risk level
        
        Returns:
            dict: Business context information for service team
        """
        contexts = {
            'low': {
                'description': 'Very Restrictive - Premium Customer Channels',
                'use_case': 'VIP customers, high-value communications',
                'trade_off': 'Higher false positives acceptable to block all spam',
                'review_strategy': 'All flagged messages reviewed immediately',
                'business_impact': 'Protects premium customer experience'
            },
            'medium': {
                'description': 'Balanced - General Customer Communication',
                'use_case': 'Standard customer service, product feedback',
                'trade_off': 'Optimal balance between protection and experience',
                'review_strategy': 'Sample-based review of flagged messages',
                'business_impact': 'Best overall performance for most channels'
            },
            'high': {
                'description': 'Permissive - Open Feedback Channels',
                'use_case': 'Public feedback, social media integration',
                'trade_off': 'Customer experience prioritized over perfect blocking',
                'review_strategy': 'Minimal review, focus on obvious spam',
                'business_impact': 'Maximizes customer communication opportunity'
            }
        }
        return contexts.get(self.risk_level, contexts['medium'])
    
    def preprocess_text(self, text):
        """Clean and preprocess short messages for customer service classification
        
        Optimized for short message characteristics:
        - Preserves important short message patterns
        - Handles customer service terminology
        - Maintains readability for business review
        
        Args:
            text (str): Raw short message text
            
        Returns:
            str: Preprocessed text ready for classification
        """
        # Convert to lowercase for consistency
        text = text.lower()
        
        # Handle URLs (common in spam but preserve for analysis)
        text = re.sub(r'http\S+|www\S+|https\S+', '[URL]', text, flags=re.MULTILINE)
        
        # Handle phone numbers (preserve pattern for business analysis)
        text = re.sub(r'\d{10,}', '[PHONE]', text)
        
        # Handle excessive punctuation (common in short message spam)
        text = re.sub(r'[!]{2,}', '!!', text)
        text = re.sub(r'[?]{2,}', '??', text)
        
        # Preserve important customer service indicators
        # (help, support, question, etc. are important for legitimate messages)
        
        # Remove extra whitespace but preserve message structure
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
        """Predict spam with risk-adjusted threshold and business confidence
        
        Business Features:
        - Risk-adjusted classification based on business context
        - Confidence scoring for manual review prioritization
        - Detailed prediction metadata for service team
        
        Args:
            X_test: Test messages (pandas Series or list)
            
        Returns:
            tuple: (predictions, probabilities, confidence_metadata)
        """
        X_test_processed = X_test.apply(self.preprocess_text)
        X_test_vec = self.vectorizer.transform(X_test_processed)
        
        # Get probability predictions
        proba = self.model.predict_proba(X_test_vec)[:, 1]
        
        # Apply risk-adjusted threshold
        predictions = (proba >= self.threshold).astype(int)
        
        # Calculate business confidence scores
        confidence_scores = self._calculate_business_confidence(proba)
        
        return predictions, proba, confidence_scores
    
    def _calculate_business_confidence(self, probabilities):
        """Calculate business confidence scores for service team review
        
        Provides actionable confidence metrics for business decision making:
        - High confidence: Automated processing recommended
        - Medium confidence: Sample review suggested  
        - Low confidence: Manual review required
        
        Args:
            probabilities (np.array): Spam probabilities from model
            
        Returns:
            list: Business confidence metadata for each prediction
        """
        confidence_metadata = []
        
        for prob in probabilities:
            # Calculate distance from threshold (uncertainty indicator)
            threshold_distance = abs(prob - self.threshold)
            
            # Determine confidence level based on business risk context
            if threshold_distance > 0.3:
                confidence_level = 'high'
                review_recommendation = 'automated_processing'
            elif threshold_distance > 0.15:
                confidence_level = 'medium'  
                review_recommendation = 'sample_review'
            else:
                confidence_level = 'low'
                review_recommendation = 'manual_review_required'
                
            # Business-friendly confidence score (0-100)
            confidence_score = min(100, int(threshold_distance * 200))
            
            metadata = {
                'confidence_level': confidence_level,
                'confidence_score': confidence_score,
                'review_recommendation': review_recommendation,
                'business_context': self.business_context['description'],
                'threshold_used': self.threshold,
                'probability': prob
            }
            
            confidence_metadata.append(metadata)
            
        return confidence_metadata
    
    def get_business_explanation(self, message, prediction, probability, confidence):
        """Generate business-friendly explanation for classification decision
        
        Provides clear, actionable explanations for service team:
        - Why the message was classified as spam/legitimate
        - Confidence level and business implications
        - Recommended next actions
        
        Args:
            message (str): Original message text
            prediction (int): Classification result (0=ham, 1=spam)
            probability (float): Spam probability
            confidence (dict): Confidence metadata
            
        Returns:
            dict: Business-friendly explanation
        """
        explanation = {
            'classification': 'SPAM' if prediction == 1 else 'LEGITIMATE',
            'confidence_level': confidence['confidence_level'],
            'business_recommendation': '',
            'risk_context': self.business_context['description'],
            'next_actions': [],
            'probability_explanation': '',
        }
        
        # Generate probability explanation
        if prediction == 1:
            explanation['probability_explanation'] = f"Message has {probability*100:.1f}% spam probability, above {self.threshold*100:.0f}% threshold for {self.risk_level} risk level."
        else:
            explanation['probability_explanation'] = f"Message has {probability*100:.1f}% spam probability, below {self.threshold*100:.0f}% threshold for {self.risk_level} risk level."
            
        # Business recommendations based on confidence
        if confidence['confidence_level'] == 'high':
            explanation['business_recommendation'] = 'High confidence classification. Safe for automated processing.'
            explanation['next_actions'] = ['Process automatically', 'Include in performance monitoring']
        elif confidence['confidence_level'] == 'medium':
            explanation['business_recommendation'] = 'Medium confidence. Consider for sample review.'
            explanation['next_actions'] = ['Include in random review sample', 'Monitor for pattern changes']
        else:
            explanation['business_recommendation'] = 'Low confidence classification. Manual review recommended.'
            explanation['next_actions'] = ['Priority manual review', 'Investigate classification factors', 'Consider for model improvement']
            
        return explanation
    
    def evaluate(self, X_test, y_test):
        """Comprehensive model evaluation with business context
        
        Provides both technical metrics and business impact assessment
        for service team confidence and decision making.
        
        Args:
            X_test: Test messages
            y_test: True labels
            
        Returns:
            dict: Comprehensive evaluation results with business context
        """
        predictions, probabilities, confidence_metadata = self.predict(X_test)
        
        print(f"\n{'='*60}")
        print(f"Short Message Classification Results")
        print(f"Business Context: {self.business_context['description']}")
        print(f"Risk Level: {self.risk_level.upper()}")
        print(f"Classification Threshold: {self.threshold*100:.0f}%")
        print(f"{'='*60}")
        
        print(f"\nBusiness Impact Analysis:")
        print(f"Use Case: {self.business_context['use_case']}")
        print(f"Trade-off Strategy: {self.business_context['trade_off']}")
        print(f"Review Strategy: {self.business_context['review_strategy']}")
        
        print("\nTechnical Classification Report:")
        print(classification_report(y_test, predictions, target_names=['Legitimate', 'Spam']))
        
        # Business confidence analysis
        high_conf = sum(1 for c in confidence_metadata if c['confidence_level'] == 'high')
        medium_conf = sum(1 for c in confidence_metadata if c['confidence_level'] == 'medium')
        low_conf = sum(1 for c in confidence_metadata if c['confidence_level'] == 'low')
        
        print(f"\nBusiness Confidence Distribution:")
        print(f"High Confidence (Automated): {high_conf} messages ({high_conf/len(confidence_metadata)*100:.1f}%)")
        print(f"Medium Confidence (Sample Review): {medium_conf} messages ({medium_conf/len(confidence_metadata)*100:.1f}%)")
        print(f"Low Confidence (Manual Review): {low_conf} messages ({low_conf/len(confidence_metadata)*100:.1f}%)")
        
        cm = confusion_matrix(y_test, predictions)
        
        return {
            'predictions': predictions,
            'probabilities': probabilities,
            'confidence_metadata': confidence_metadata,
            'confusion_matrix': cm,
            'business_context': self.business_context
        }
    
    def save_model(self, filepath):
        """Save the trained model and vectorizer to disk"""
        import pickle
        import os
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        model_data = {
            'model': self.model,
            'vectorizer': self.vectorizer,
            'risk_level': self.risk_level,
            'threshold': self.threshold,
            'model_type': self.model_type
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Model saved: {filepath}")
    
    def load_model(self, filepath):
        """Load a trained model and vectorizer from disk"""
        import pickle
        
        try:
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
            
            self.model = model_data['model']
            self.vectorizer = model_data['vectorizer']
            self.risk_level = model_data['risk_level']
            self.threshold = model_data['threshold']
            self.model_type = model_data['model_type']
            
            print(f"Model loaded: {filepath}")
            return True
            
        except FileNotFoundError:
            print(f"Model file not found: {filepath}")
            return False
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    @classmethod
    def load_from_file(cls, filepath):
        """Class method to create a SpamFilter instance from saved model"""
        instance = cls()
        if instance.load_model(filepath):
            return instance
        return None


def save_training_data_to_json(df, models_results, X_train, y_train, X_test, y_test, dataset_source):
    """Save all training data, statistics, and results to JSON file"""
    import numpy as np
    
    # Prepare data for JSON serialization
    training_data = {
        'metadata': {
            'timestamp': datetime.now().isoformat(),
            'dataset_source': dataset_source,
            'total_messages': len(df),
            'training_size': len(X_train),
            'test_size': len(X_test),
            'spam_count': len(df[df['label'] == 'spam']),
            'ham_count': len(df[df['label'] == 'ham'])
        },
        'dataset_stats': {
            'spam_percentage': (len(df[df['label'] == 'spam']) / len(df)) * 100,
            'ham_percentage': (len(df[df['label'] == 'ham']) / len(df)) * 100,
        },
        'model_results': {}
    }
    
    # Add message length statistics
    df_temp = df.copy()
    df_temp['message_length'] = df_temp['message'].apply(len)
    df_temp['word_count'] = df_temp['message'].apply(lambda x: len(x.split()))
    
    training_data['message_stats'] = {
        'avg_spam_length': float(df_temp[df_temp['label'] == 'spam']['message_length'].mean()),
        'avg_ham_length': float(df_temp[df_temp['label'] == 'ham']['message_length'].mean()),
        'avg_spam_words': float(df_temp[df_temp['label'] == 'spam']['word_count'].mean()),
        'avg_ham_words': float(df_temp[df_temp['label'] == 'ham']['word_count'].mean())
    }
    
    # Process model results
    for risk_level, results in models_results.items():
        cm = results['confusion_matrix']
        tn, fp, fn, tp = cm.ravel()
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        fp_rate = fp / (fp + tn) if (fp + tn) > 0 else 0
        
        training_data['model_results'][risk_level] = {
            'confusion_matrix': {
                'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)
            },
            'metrics': {
                'accuracy': round(accuracy * 100, 0),
                'precision': round(precision * 100, 0),
                'recall': round(recall * 100, 0),
                'f1_score': round(f1 * 100, 0),
                'fp_rate': round(fp_rate * 100, 1)
            },
            'threshold': 0.3 if risk_level == 'low' else (0.5 if risk_level == 'medium' else 0.7),
            'model_path': f'models/spam_filter_{risk_level}_risk.pkl'
        }
    
    # Add training data composition
    y_train_labels = ['spam' if val == 1 else 'ham' for val in y_train]
    train_spam_count = y_train_labels.count('spam')
    train_ham_count = y_train_labels.count('ham')
    
    training_data['training_composition'] = {
        'total_messages': len(y_train),
        'spam_count': train_spam_count,
        'ham_count': train_ham_count,
        'spam_percentage': (train_spam_count / len(y_train)) * 100,
        'ham_percentage': (train_ham_count / len(y_train)) * 100
    }
    
    # Save to JSON file
    os.makedirs('data', exist_ok=True)
    with open('data/training_data.json', 'w', encoding='utf-8') as f:
        json.dump(training_data, f, indent=2, ensure_ascii=False)
    
    print("Training data saved to: data/training_data.json")
    return training_data


def load_training_data_from_json():
    """Load training data from JSON file"""
    return report_generator.load_training_data_from_json()


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


def train_multiple_models(X_train, y_train, X_test, y_test, df):
    """Train models with different risk levels"""
    risk_levels = ['low_risk', 'medium_risk', 'high_risk']
    models_results = {}
    
    for risk_level in risk_levels:
        risk_setting = risk_level.replace('_risk', '')
        
        print(f"\nTraining {risk_level.upper()} model...")
        
        # Initialize spam filter with different risk levels
        spam_filter = SpamFilter(model_type='naive_bayes', risk_level=risk_setting)
        
        # Adjust vectorizer parameters based on risk level
        if risk_setting == 'low':
            max_features = 5000  # More features for better precision
        elif risk_setting == 'medium':
            max_features = 3000  # Balanced
        else:  # high
            max_features = 1500  # Fewer features for simpler model
        
        # Preprocess training data
        X_train_processed = X_train.apply(spam_filter.preprocess_text)
        
        # Custom vectorizer for this risk level
        vectorizer = TfidfVectorizer(
            max_features=max_features,
            min_df=2,
            max_df=0.8,
            ngram_range=(1, 2) if risk_setting != 'high' else (1, 1)
        )
        
        X_train_vec = vectorizer.fit_transform(X_train_processed)
        spam_filter.vectorizer = vectorizer
        
        # Train model
        spam_filter.model = MultinomialNB(alpha=0.1)
        spam_filter.model.fit(X_train_vec, y_train)
        
        # Evaluate model with business context
        results = spam_filter.evaluate(X_test, y_test)
        
        # Extract results for compatibility
        predictions = results['predictions']
        probabilities = results['probabilities']
        confidence_metadata = results['confidence_metadata']
        cm = results['confusion_matrix']
        
        tn, fp, fn, tp = cm.ravel()
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        # Calculate training accuracy
        train_predictions, _, _ = spam_filter.predict(X_train)
        train_accuracy = (train_predictions == y_train).mean()
        
        # Store results
        models_results[risk_level] = {
            'model': spam_filter,
            'predictions': predictions,
            'probabilities': probabilities,
            'confusion_matrix': cm,
            'accuracy': accuracy,
            'train_accuracy': train_accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'train_size': len(X_train),
            'test_size': len(X_test),
            'vectorizer_type': 'TfidfVectorizer',
            'max_features': max_features
        }
        
        # Save the trained model
        model_path = f'models/spam_filter_{risk_level}.pkl'
        spam_filter.save_model(model_path)
        
        print(f"✓ {risk_level.upper()} model trained and saved")
        print(f"  Accuracy: {accuracy*100:.1f}%")
        print(f"  Precision: {precision:.3f}")
        print(f"  Recall: {recall:.3f}")
        print(f"  F1-Score: {f1_score:.3f}")
    
    return models_results


def list_saved_models():
    """List all saved model files in the models directory"""
    models_dir = 'models'
    
    if not os.path.exists(models_dir):
        print("No models directory found. Train some models first!")
        return
    
    model_files = [f for f in os.listdir(models_dir) if f.endswith('.pkl')]
    
    if not model_files:
        print("No saved models found in models directory.")
        return
    
    print("\n" + "="*50)
    print("SAVED MODELS")
    print("="*50)
    
    for model_file in sorted(model_files):
        model_path = os.path.join(models_dir, model_file)
        file_size = os.path.getsize(model_path)
        creation_time = datetime.fromtimestamp(os.path.getctime(model_path))
        
        print(f"\nModel: {model_file}")
        print(f"  Path: {model_path}")
        print(f"  Size: {file_size/1024:.1f} KB")
        print(f"  Created: {creation_time.strftime('%Y-%m-%d %H:%M:%S')}")


def test_saved_model(model_path, test_message=None):
    """Test a saved model with a custom message"""
    print(f"\nTesting model: {model_path}")
    
    # Load the model
    spam_filter = SpamFilter.load_from_file(model_path)
    if spam_filter is None:
        print("Failed to load model.")
        return
    
    # Default test message if none provided
    if test_message is None:
        test_message = "Congratulations! You've won $1000! Click here now to claim your prize!"
    
    print(f"\nTest Message: '{test_message}'")
    
    # Test the message
    test_df = pd.Series([test_message])
    predictions, probabilities = spam_filter.predict(test_df)
    
    prediction = predictions[0]
    probability = probabilities[0]
    
    print(f"\nResults:")
    print(f"  Prediction: {'SPAM' if prediction == 1 else 'HAM (Legitimate)'}")
    print(f"  Spam Probability: {probability:.3f}")
    print(f"  Risk Level: {spam_filter.risk_level}")
    print(f"  Threshold: {spam_filter.threshold}")
    
    if prediction == 1:
        print(f"  ⚠️  This message is classified as SPAM")
    else:
        print(f"  ✅ This message is classified as legitimate")


def main():
    """Main training and evaluation pipeline - CRISP-DM Implementation

    CRISP-DM Phase 4-5: Modeling and Evaluation

    Implements comprehensive short message spam classification system
    for customer service channels with business-oriented evaluation.
    """
    try:
        print("="*60)
        print("SHORT MESSAGE SPAM FILTER - CRISP-DM IMPLEMENTATION")
        print("Focus: Customer Service Channel Protection")
        print("Methodology: CRISP-DM Phase 4-5 (Modeling & Evaluation)")
        print("="*60)

        # Load the labeled dataset from local storage (CRISP-DM Phase 2-3: Data Understanding & Preparation)
        print("\n📊 Data Understanding & Preparation (CRISP-DM Phase 2-3)")
        print("Loading SMS Spam Collection dataset...")

        # Load from local CSV file
        csv_path = 'data/raw/SMSSpamCollection.csv'
        try:
            df = pd.read_csv(csv_path, sep='\t', header=None, names=['label', 'message'], encoding='utf-8')
            print(f"✓ Dataset loaded from: {csv_path}")
        except FileNotFoundError:
            print(f"⚠️  File not found: {csv_path}")
            print("Attempting to load from alternative location...")
            # Try alternative path
            df = pd.read_csv('SMSSpamCollection.csv', sep='\t', header=None, names=['label', 'message'], encoding='utf-8')
        
        # Convert labels to binary for short message classification
        y = (df['label'] == 'spam').astype(int)
        X = df['message']
        
        # Split the data with stratification for balanced evaluation
        print(f"\n📈 Dataset Analysis for Short Message Classification:")
        print(f"Total messages: {len(df):,} (optimized for short message patterns)")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"Training set: {len(X_train):,} messages")
        print(f"Test set: {len(X_test):,} messages")
        print(f"Business Context: Customer service channel simulation")
        
        # Train models with business risk levels (CRISP-DM Phase 4: Modeling)
        print("\n🤖 Model Training with Business Risk Levels (CRISP-DM Phase 4)")
        models_results = train_multiple_models(X_train, y_train, X_test, y_test, df)
        
        # Business-oriented data persistence
        dataset_source = "SMS Spam Collection Dataset"
        training_data = save_training_data_to_json(df, models_results, X_train, y_train, X_test, y_test, dataset_source)

        # Generate business-oriented reports (CRISP-DM Phase 5: Evaluation)
        print("\n📋 Business Report Generation (CRISP-DM Phase 5)")
        print("Generating comprehensive evaluation for business stakeholders...")

        # Update LaTeX report with business context
        report_generator.update_latex_report_from_json(training_data, dataset_source)

        # Populate case study report with actual JSON data (NEW)
        print("📝 Populating case study report with actual model data...")
        report_generator.populate_case_study_with_json_data(training_data)

        # Compile business report
        report_generator.compile_latex_report()
        
        print("\n" + "="*60)
        print("CRISP-DM IMPLEMENTATION COMPLETE")
        print("Business Focus: Short Message Classification for Customer Service")
        print("="*60)
        print("\n📁 Generated Business Assets:")
        print("  - Trained models with risk level configuration")
        print("  - Business confidence scoring system")
        print("  - Service team integration ready")
        
        # Display saved models for business deployment
        list_saved_models()
        
        print("\n🚀 Next Steps for Service Team Integration:")
        print("  1. Review model performance across risk levels")
        print("  2. Select appropriate risk level for your channel")
        print("  3. Launch GUI for service team training: python gui_interface.py")
        print("  4. Implement review workflow based on confidence scores")
        
        print("\n💡 Business Commands:")
        print("  python spam_filter.py --list-models    # Review available models")
        print("  python spam_filter.py --test-model <path> <message>  # Test classification")
        print("  python spam_filter.py --report-only    # Regenerate business reports")
        
    except Exception as e:
        print(f"❌ Error during CRISP-DM implementation: {str(e)}")
        print("💡 Business Impact: System requires technical review before deployment")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    import sys
    
    # Check for command-line arguments
    if len(sys.argv) > 1:
        if sys.argv[1] == "--report-only":
            report_generator.generate_report_from_json()
        elif sys.argv[1] == "--list-models":
            list_saved_models()
        elif sys.argv[1] == "--test-model":
            if len(sys.argv) > 2:
                model_path = sys.argv[2]
                test_message = sys.argv[3] if len(sys.argv) > 3 else "Congratulations! You've won $1000! Click here now!"
                test_saved_model(model_path, test_message)
            else:
                print("Usage: python spam_filter.py --test-model <model_path> [test_message]")
        elif sys.argv[1] == "--help":
            print("Spam Filter Usage:")
            print("  python spam_filter.py                    # Train models and generate reports")
            print("  python spam_filter.py --report-only      # Generate report from existing JSON")
            print("  python spam_filter.py --list-models      # List saved models")
            print("  python spam_filter.py --test-model <path> [message]  # Test saved model")
            print("  python spam_filter.py --help             # Show this help")
        else:
            print(f"Unknown option: {sys.argv[1]}. Use --help for usage information.")
    else:
        main()