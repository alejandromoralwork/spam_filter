from spam_filter import *
import pandas as pd

# Load data and train models quickly
df = pd.read_csv('data/kaggle_spam_data.csv')
df['label_binary'] = (df['label'] == 'spam').astype(int)

from sklearn.model_selection import train_test_split
X = df['message']
y = df['label_binary']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

models_results = {}
import os
os.makedirs('figures', exist_ok=True)

print("Quick training and testing accuracy values...")
for risk_level in ['low', 'medium', 'high']:
    model = SpamFilter(model_type='naive_bayes', risk_level=risk_level)
    model.fit(X_train, y_train)
    results = model.evaluate(X_test, y_test)
    models_results[risk_level] = results

# Generate stats and check values
stats = generate_latex_stats(df, models_results)
print('\nKey accuracy values from stats:')
print(f'Low: {stats["low_accuracy"]}%')
print(f'Medium: {stats["medium_accuracy"]}%')
print(f'High: {stats["high_accuracy"]}%')