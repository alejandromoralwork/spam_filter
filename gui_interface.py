import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox, filedialog
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import seaborn as sns
import pandas as pd
import numpy as np
from spam_filter import SpamFilter, create_sample_data
from sklearn.model_selection import train_test_split
import os


class SpamFilterGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Spam Filter - Machine Learning Analysis")
        self.root.geometry("1200x800")
        
        # Initialize variables
        self.model = None
        self.df = None
        self.risk_level = tk.StringVar(value="medium")
        self.is_trained = False
        
        self.setup_gui()
        
    def setup_gui(self):
        """Setup the GUI components"""
        # Create notebook for tabs
        notebook = ttk.Notebook(self.root)
        notebook.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Tab 1: Data & Training
        self.data_tab = ttk.Frame(notebook)
        notebook.add(self.data_tab, text="Data & Training")
        
        # Tab 2: Testing
        self.test_tab = ttk.Frame(notebook)
        notebook.add(self.test_tab, text="Testing")
        
        # Tab 3: Visualizations
        self.viz_tab = ttk.Frame(notebook)
        notebook.add(self.viz_tab, text="Visualizations")
        
        # Setup each tab
        self.setup_data_tab()
        self.setup_test_tab()
        self.setup_viz_tab()
        
    def setup_data_tab(self):
        """Setup the data and training tab"""
        # Left frame for controls
        left_frame = ttk.Frame(self.data_tab)
        left_frame.pack(side="left", fill="y", padx=10, pady=10)
        
        # Data loading section
        data_frame = ttk.LabelFrame(left_frame, text="Data Loading", padding=10)
        data_frame.pack(fill="x", pady=5)
        
        ttk.Button(data_frame, text="Load Sample Data", 
                  command=self.load_sample_data).pack(pady=2)
        ttk.Button(data_frame, text="Load CSV File", 
                  command=self.load_csv_file).pack(pady=2)
        
        # Model configuration section
        config_frame = ttk.LabelFrame(left_frame, text="Model Configuration", padding=10)
        config_frame.pack(fill="x", pady=5)
        
        ttk.Label(config_frame, text="Risk Level:").pack()
        for level in ["low", "medium", "high"]:
            ttk.Radiobutton(config_frame, text=level.capitalize(), 
                           variable=self.risk_level, value=level).pack()
        
        # Training section
        train_frame = ttk.LabelFrame(left_frame, text="Training", padding=10)
        train_frame.pack(fill="x", pady=5)
        
        ttk.Button(train_frame, text="Train Model", 
                  command=self.train_model).pack(pady=2)
        
        # Status section
        status_frame = ttk.LabelFrame(left_frame, text="Status", padding=10)
        status_frame.pack(fill="x", pady=5)
        
        self.status_text = scrolledtext.ScrolledText(status_frame, height=15, width=40)
        self.status_text.pack(fill="both", expand=True)
        
        # Right frame for data preview
        right_frame = ttk.Frame(self.data_tab)
        right_frame.pack(side="right", fill="both", expand=True, padx=10, pady=10)
        
        preview_frame = ttk.LabelFrame(right_frame, text="Data Preview", padding=10)
        preview_frame.pack(fill="both", expand=True)
        
        self.data_tree = ttk.Treeview(preview_frame, columns=("Label", "Message"), show="headings")
        self.data_tree.heading("Label", text="Label")
        self.data_tree.heading("Message", text="Message")
        self.data_tree.column("Label", width=100)
        self.data_tree.column("Message", width=400)
        self.data_tree.pack(fill="both", expand=True)
        
    def setup_test_tab(self):
        """Setup the testing tab"""
        # Top frame for input
        input_frame = ttk.LabelFrame(self.test_tab, text="Test Message", padding=10)
        input_frame.pack(fill="x", padx=10, pady=10)
        
        ttk.Label(input_frame, text="Enter message to classify:").pack()
        self.test_input = scrolledtext.ScrolledText(input_frame, height=5)
        self.test_input.pack(fill="x", pady=5)
        
        # Buttons frame
        button_frame = ttk.Frame(input_frame)
        button_frame.pack(fill="x", pady=5)
        
        ttk.Button(button_frame, text="Classify Message", 
                  command=self.classify_message).pack(side="left", padx=5)
        ttk.Button(button_frame, text="Clear", 
                  command=self.clear_input).pack(side="left", padx=5)
        
        # Results frame
        results_frame = ttk.LabelFrame(self.test_tab, text="Results", padding=10)
        results_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        self.results_text = scrolledtext.ScrolledText(results_frame)
        self.results_text.pack(fill="both", expand=True)
        
    def setup_viz_tab(self):
        """Setup the visualization tab"""
        # Create matplotlib figure
        self.fig, self.axes = plt.subplots(2, 2, figsize=(12, 8))
        self.fig.suptitle("Spam Filter Analysis")
        
        # Create canvas
        self.canvas = FigureCanvasTkAgg(self.fig, self.viz_tab)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        
        # Control frame
        control_frame = ttk.Frame(self.viz_tab)
        control_frame.pack(fill="x", padx=10, pady=5)
        
        ttk.Button(control_frame, text="Update Visualizations", 
                  command=self.update_visualizations).pack(side="left", padx=5)
        ttk.Button(control_frame, text="Save Plots", 
                  command=self.save_plots).pack(side="left", padx=5)
        
    def log_message(self, message):
        """Add message to status log"""
        self.status_text.insert(tk.END, message + "\n")
        self.status_text.see(tk.END)
        self.root.update()
        
    def load_sample_data(self):
        """Load sample spam/ham dataset"""
        try:
            self.log_message("Loading sample dataset...")
            self.df = create_sample_data()
            self.update_data_preview()
            self.log_message(f"Sample data loaded: {len(self.df)} messages")
            self.log_message(f"Spam: {sum(self.df['label'] == 'spam')}, Ham: {sum(self.df['label'] == 'ham')}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load sample data: {str(e)}")
            self.log_message(f"Error loading sample data: {str(e)}")
    
    def load_csv_file(self):
        """Load dataset from CSV file"""
        try:
            file_path = filedialog.askopenfilename(
                title="Select CSV File",
                filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
            )
            
            if file_path:
                self.log_message(f"Loading CSV file: {file_path}")
                self.df = pd.read_csv(file_path)
                
                # Try to standardize column names
                if 'message' not in self.df.columns or 'label' not in self.df.columns:
                    self.log_message("Attempting to standardize column names...")
                    # Simple heuristic: take first two columns as label and message
                    if len(self.df.columns) >= 2:
                        self.df = self.df.rename(columns={
                            self.df.columns[0]: 'label',
                            self.df.columns[1]: 'message'
                        })
                
                self.update_data_preview()
                self.log_message(f"CSV data loaded: {len(self.df)} messages")
                
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load CSV file: {str(e)}")
            self.log_message(f"Error loading CSV: {str(e)}")
    
    def update_data_preview(self):
        """Update the data preview tree"""
        # Clear existing data
        for item in self.data_tree.get_children():
            self.data_tree.delete(item)
        
        if self.df is not None:
            # Show first 100 rows
            for idx, row in self.df.head(100).iterrows():
                label = row['label']
                message = row['message'][:80] + "..." if len(row['message']) > 80 else row['message']
                self.data_tree.insert("", "end", values=(label, message))
    
    def train_model(self):
        """Train the spam filter model"""
        if self.df is None:
            messagebox.showerror("Error", "Please load data first!")
            return
        
        try:
            self.log_message(f"Training model with {self.risk_level.get()} risk level...")
            
            # Prepare data
            df_temp = self.df.copy()
            df_temp['label_binary'] = (df_temp['label'] == 'spam').astype(int)
            
            X = df_temp['message']
            y = df_temp['label_binary']
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Train model
            self.model = SpamFilter(model_type='naive_bayes', 
                                   risk_level=self.risk_level.get())
            self.model.fit(X_train, y_train)
            
            # Evaluate
            results = self.model.evaluate(X_test, y_test)
            
            self.is_trained = True
            self.log_message("Model training completed!")
            self.log_message("See console for detailed evaluation results.")
            
            # Store test results for visualization
            self.test_results = results
            self.y_test = y_test
            
        except Exception as e:
            messagebox.showerror("Error", f"Training failed: {str(e)}")
            self.log_message(f"Training error: {str(e)}")
    
    def classify_message(self):
        """Classify a single message"""
        if not self.is_trained:
            messagebox.showerror("Error", "Please train a model first!")
            return
        
        message = self.test_input.get("1.0", tk.END).strip()
        if not message:
            messagebox.showerror("Error", "Please enter a message to classify!")
            return
        
        try:
            # Create a pandas Series for the message
            message_series = pd.Series([message])
            prediction, probability = self.model.predict(message_series)
            
            # Format results
            result = "SPAM" if prediction[0] == 1 else "HAM"
            confidence = probability[0] * 100 if prediction[0] == 1 else (1 - probability[0]) * 100
            
            result_text = f"""
Classification Result:
{'='*50}
Message: {message}
{'='*50}

Prediction: {result}
Confidence: {confidence:.2f}%
Spam Probability: {probability[0]:.3f}
Risk Level: {self.risk_level.get().upper()}
Threshold: {self.model.threshold}

Analysis:
- The message was classified as {result.lower()}
- Spam probability is {probability[0]:.3f}
- Using {self.risk_level.get()} risk threshold ({self.model.threshold})
"""
            
            self.results_text.delete("1.0", tk.END)
            self.results_text.insert("1.0", result_text)
            
        except Exception as e:
            messagebox.showerror("Error", f"Classification failed: {str(e)}")
    
    def clear_input(self):
        """Clear the input text"""
        self.test_input.delete("1.0", tk.END)
        self.results_text.delete("1.0", tk.END)
    
    def update_visualizations(self):
        """Update the visualization plots"""
        if self.df is None:
            messagebox.showinfo("Info", "Please load data first!")
            return
        
        try:
            # Clear previous plots
            for ax in self.axes.flat:
                ax.clear()
            
            # Plot 1: Class distribution
            self.df['label'].value_counts().plot(kind='bar', ax=self.axes[0, 0], 
                                                color=['green', 'red'])
            self.axes[0, 0].set_title('Class Distribution')
            self.axes[0, 0].set_xlabel('Class')
            self.axes[0, 0].set_ylabel('Count')
            self.axes[0, 0].tick_params(axis='x', rotation=0)
            
            # Plot 2: Message length distribution
            if 'message_length' not in self.df.columns:
                self.df['message_length'] = self.df['message'].apply(len)
            
            spam_lengths = self.df[self.df['label'] == 'spam']['message_length']
            ham_lengths = self.df[self.df['label'] == 'ham']['message_length']
            
            self.axes[0, 1].hist([ham_lengths, spam_lengths], label=['Ham', 'Spam'], 
                                bins=20, alpha=0.7, color=['green', 'red'])
            self.axes[0, 1].set_title('Message Length Distribution')
            self.axes[0, 1].set_xlabel('Length (characters)')
            self.axes[0, 1].set_ylabel('Frequency')
            self.axes[0, 1].legend()
            
            # Plot 3: Word count distribution
            if 'word_count' not in self.df.columns:
                self.df['word_count'] = self.df['message'].apply(lambda x: len(x.split()))
            
            spam_words = self.df[self.df['label'] == 'spam']['word_count']
            ham_words = self.df[self.df['label'] == 'ham']['word_count']
            
            self.axes[1, 0].boxplot([ham_words, spam_words], labels=['Ham', 'Spam'])
            self.axes[1, 0].set_title('Word Count by Class')
            self.axes[1, 0].set_ylabel('Word Count')
            
            # Plot 4: Confusion Matrix (if model is trained)
            if hasattr(self, 'test_results'):
                cm = self.test_results['confusion_matrix']
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                           xticklabels=['Ham', 'Spam'],
                           yticklabels=['Ham', 'Spam'],
                           ax=self.axes[1, 1])
                self.axes[1, 1].set_title(f'Confusion Matrix ({self.risk_level.get().capitalize()} Risk)')
                self.axes[1, 1].set_xlabel('Predicted')
                self.axes[1, 1].set_ylabel('Actual')
            else:
                self.axes[1, 1].text(0.5, 0.5, 'Train model first\nto see confusion matrix', 
                                    ha='center', va='center', transform=self.axes[1, 1].transAxes)
                self.axes[1, 1].set_title('Confusion Matrix')
            
            plt.tight_layout()
            self.canvas.draw()
            self.log_message("Visualizations updated!")
            
        except Exception as e:
            messagebox.showerror("Error", f"Visualization failed: {str(e)}")
            self.log_message(f"Visualization error: {str(e)}")
    
    def save_plots(self):
        """Save the current plots to file"""
        try:
            if not os.path.exists('figures'):
                os.makedirs('figures')
            
            self.fig.savefig('figures/gui_analysis.png', dpi=300, bbox_inches='tight')
            messagebox.showinfo("Success", "Plots saved to figures/gui_analysis.png")
            self.log_message("Plots saved to figures/gui_analysis.png")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save plots: {str(e)}")
    
    def run(self):
        """Start the GUI application"""
        # Initial welcome message
        welcome_msg = """Welcome to Spam Filter GUI!

Steps to get started:
1. Load data (sample or CSV file)
2. Choose risk level
3. Train the model
4. Test messages in the Testing tab
5. View visualizations

Ready to begin!
"""
        self.log_message(welcome_msg)
        
        self.root.mainloop()


if __name__ == "__main__":
    app = SpamFilterGUI()
    app.run()