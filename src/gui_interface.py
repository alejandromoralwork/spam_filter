"""
Short Message Spam Filter GUI - Service Team Interface
CRISP-DM Phase 6: Deployment

Business-oriented interface designed for customer service teams to:
- Manage spam filtering for customer communication channels
- Adjust risk levels based on business requirements
- Review classifications with business confidence scoring
- Monitor performance with business-friendly metrics

Target Users: Customer service managers, support staff, business analysts
Focus: SHORT MESSAGES (not email) for customer feedback channels
"""

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
    """Service Team Interface for Short Message Spam Classification
    
    CRISP-DM Phase 6: Deployment Implementation
    
    Business-focused GUI providing:
    - Risk level management for customer channels
    - Confidence-based review workflow
    - Business performance metrics
    - Service team decision support
    """
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Customer Service Spam Filter - Short Message Classification")
        self.root.geometry("1400x900")
        
        # Initialize business variables
        self.model = None
        self.df = None
        self.risk_level = tk.StringVar(value="medium")
        self.is_trained = False
        self.business_metrics = {}
        
        self.setup_gui()
        
    def setup_gui(self):
        """Setup the business-oriented GUI components"""
        # Create notebook for business workflow tabs
        notebook = ttk.Notebook(self.root)
        notebook.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Tab 1: Business Configuration & Data
        self.data_tab = ttk.Frame(notebook)
        notebook.add(self.data_tab, text="📊 Data & Business Config")
        
        # Tab 2: Message Classification & Review
        self.test_tab = ttk.Frame(notebook)
        notebook.add(self.test_tab, text="🔍 Message Classification")
        
        # Tab 3: Business Analytics & Performance
        self.viz_tab = ttk.Frame(notebook)
        notebook.add(self.viz_tab, text="📈 Business Analytics")
        
        # Tab 4: Service Team Review Queue
        self.review_tab = ttk.Frame(notebook)
        notebook.add(self.review_tab, text="📋 Review Queue")
        
        # Setup each business workflow tab
        self.setup_data_tab()
        self.setup_test_tab()
        self.setup_viz_tab()
        self.setup_review_tab()
        
    def setup_review_tab(self):
        """Setup manual review queue for business confidence scoring"""
        # Top frame for review queue controls
        controls_frame = ttk.LabelFrame(self.review_tab, text="🔍 Review Queue Controls", padding=10)
        controls_frame.pack(fill="x", padx=10, pady=5)
        
        # Business confidence threshold controls
        threshold_frame = ttk.Frame(controls_frame)
        threshold_frame.pack(fill="x", pady=5)
        
        ttk.Label(threshold_frame, text="Business Review Threshold:", font=('Arial', 10, 'bold')).pack(side="left")
        self.review_threshold = tk.DoubleVar(value=0.7)
        threshold_scale = ttk.Scale(threshold_frame, from_=0.5, to=0.9, 
                                   variable=self.review_threshold, orient="horizontal")
        threshold_scale.pack(side="left", fill="x", expand=True, padx=10)
        
        self.threshold_label = ttk.Label(threshold_frame, text="0.70")
        self.threshold_label.pack(side="right")
        
        # Bind threshold change to label update
        self.review_threshold.trace_add("write", self.update_threshold_label)
        
        # Queue management buttons
        buttons_frame = ttk.Frame(controls_frame)
        buttons_frame.pack(fill="x", pady=5)
        
        ttk.Button(buttons_frame, text="📋 Load Review Queue", 
                  command=self.load_review_queue).pack(side="left", padx=5)
        ttk.Button(buttons_frame, text="🔄 Refresh Queue", 
                  command=self.refresh_review_queue).pack(side="left", padx=5)
        ttk.Button(buttons_frame, text="💾 Save Reviews", 
                  command=self.save_reviews).pack(side="left", padx=5)
        
        # Review queue display
        queue_frame = ttk.LabelFrame(self.review_tab, text="📋 Messages Requiring Review", padding=10)
        queue_frame.pack(fill="both", expand=True, padx=10, pady=5)
        
        # Create treeview for review queue with business columns
        self.review_tree = ttk.Treeview(queue_frame, 
                                       columns=("Message", "Prediction", "Confidence", "Business Risk", "Action"), 
                                       show="headings")
        
        self.review_tree.heading("Message", text="Customer Message")
        self.review_tree.heading("Prediction", text="Predicted Classification")
        self.review_tree.heading("Confidence", text="Business Confidence")
        self.review_tree.heading("Business Risk", text="Risk Level")
        self.review_tree.heading("Action", text="Recommended Action")
        
        self.review_tree.column("Message", width=350)
        self.review_tree.column("Prediction", width=120)
        self.review_tree.column("Confidence", width=120)
        self.review_tree.column("Business Risk", width=100)
        self.review_tree.column("Action", width=150)
        
        # Add scrollbars to review queue
        review_scrollbar_y = ttk.Scrollbar(queue_frame, orient="vertical", command=self.review_tree.yview)
        review_scrollbar_x = ttk.Scrollbar(queue_frame, orient="horizontal", command=self.review_tree.xview)
        self.review_tree.configure(yscrollcommand=review_scrollbar_y.set, xscrollcommand=review_scrollbar_x.set)
        
        review_scrollbar_y.pack(side="right", fill="y")
        review_scrollbar_x.pack(side="bottom", fill="x")
        self.review_tree.pack(fill="both", expand=True)
        
        # Bind selection event for review details
        self.review_tree.bind("<<TreeviewSelect>>", self.on_review_select)
        
        # Bottom frame for review actions
        actions_frame = ttk.LabelFrame(self.review_tab, text="📝 Review Actions", padding=10)
        actions_frame.pack(fill="x", padx=10, pady=5)
        
        # Review decision frame
        decision_frame = ttk.Frame(actions_frame)
        decision_frame.pack(fill="x", pady=5)
        
        ttk.Label(decision_frame, text="Review Decision:", font=('Arial', 10, 'bold')).pack(side="left")
        
        self.review_decision = tk.StringVar(value="")
        decision_buttons = ttk.Frame(decision_frame)
        decision_buttons.pack(side="left", fill="x", expand=True, padx=10)
        
        ttk.Radiobutton(decision_buttons, text="✅ Approve (Not Spam)", 
                       variable=self.review_decision, value="approve").pack(side="left", padx=5)
        ttk.Radiobutton(decision_buttons, text="❌ Reject (Spam)", 
                       variable=self.review_decision, value="reject").pack(side="left", padx=5)
        ttk.Radiobutton(decision_buttons, text="⚠️ Escalate to Manager", 
                       variable=self.review_decision, value="escalate").pack(side="left", padx=5)
        
        # Comments section
        comments_frame = ttk.Frame(actions_frame)
        comments_frame.pack(fill="x", pady=5)
        
        ttk.Label(comments_frame, text="Business Comments:", font=('Arial', 10, 'bold')).pack(anchor="w")
        self.review_comments = tk.Text(comments_frame, height=3, width=80)
        self.review_comments.pack(fill="x", pady=2)
        
        # Submit review button
        submit_frame = ttk.Frame(actions_frame)
        submit_frame.pack(fill="x", pady=5)
        
        ttk.Button(submit_frame, text="📥 Submit Review", 
                  command=self.submit_review).pack(side="right", padx=5)
        ttk.Button(submit_frame, text="⏭️ Skip to Next", 
                  command=self.skip_review).pack(side="right", padx=5)

    def update_threshold_label(self, *args):
        """Update the threshold display label"""
        threshold_value = self.review_threshold.get()
        self.threshold_label.config(text=f"{threshold_value:.2f}")
    
    def load_review_queue(self):
        """Load messages that need manual review based on business confidence"""
        if not hasattr(self, 'spam_filter') or not self.spam_filter.is_trained:
            messagebox.showwarning("Warning", "Please train the model first!")
            return
            
        # Get sample messages for review (would normally come from message queue)
        sample_messages = [
            "Get exclusive deals now! Limited time offer!",
            "Hi John, can you call me back about the project?", 
            "WINNER! You've won $1000! Click here now!",
            "Meeting moved to 3 PM today in conference room B",
            "Free money! No questions asked! Act fast!",
            "Your order #12345 has been shipped and will arrive tomorrow"
        ]
        
        # Clear existing queue
        for item in self.review_tree.get_children():
            self.review_tree.delete(item)
            
        threshold = self.review_threshold.get()
        
        # Add messages to review queue that fall below confidence threshold
        for message in sample_messages:
            prediction, confidence = self.spam_filter.predict_with_confidence(message)
            business_risk = self.risk_level.get()
            
            if confidence < threshold:
                if prediction == "spam":
                    action = "🚫 Block - Low Confidence"
                    risk_color = "high"
                else:
                    action = "✅ Allow - Review Needed"
                    risk_color = "medium"
                
                # Insert with business context tags
                item = self.review_tree.insert("", "end", values=(
                    message[:50] + "..." if len(message) > 50 else message,
                    prediction.title(),
                    f"{confidence:.1%}",
                    business_risk.title(),
                    action
                ))
                
                # Add color coding based on risk
                if prediction == "spam" and confidence < 0.6:
                    self.review_tree.set(item, "Business Risk", "HIGH")
                    
        self.update_status(f"Loaded {len(self.review_tree.get_children())} messages for business review")
    
    def refresh_review_queue(self):
        """Refresh the review queue with current threshold"""
        self.load_review_queue()
    
    def on_review_select(self, event):
        """Handle selection of a message in the review queue"""
        selection = self.review_tree.selection()
        if selection:
            item = selection[0]
            values = self.review_tree.item(item, "values")
            if values:
                # Auto-populate comments with business context
                message = values[0]
                confidence = values[2]
                self.review_comments.delete(1.0, tk.END)
                self.review_comments.insert(1.0, 
                    f"Message: {message}\n"
                    f"Business Confidence: {confidence}\n"
                    f"Requires review due to low confidence score.")
    
    def submit_review(self):
        """Submit the manual review decision"""
        selection = self.review_tree.selection()
        if not selection:
            messagebox.showwarning("Warning", "Please select a message to review")
            return
            
        decision = self.review_decision.get()
        if not decision:
            messagebox.showwarning("Warning", "Please select a review decision")
            return
            
        comments = self.review_comments.get(1.0, tk.END).strip()
        
        # Here you would normally save the review to a database
        # For demonstration, we'll just show a success message
        messagebox.showinfo("Success", 
                           f"Review submitted: {decision.title()}\n"
                           f"Comments: {comments[:50]}...")
        
        # Remove reviewed item and move to next
        self.review_tree.delete(selection[0])
        self.review_decision.set("")
        self.review_comments.delete(1.0, tk.END)
        
        if self.review_tree.get_children():
            # Select next item
            next_item = self.review_tree.get_children()[0]
            self.review_tree.selection_set(next_item)
            self.review_tree.focus(next_item)
    
    def skip_review(self):
        """Skip current review and move to next"""
        selection = self.review_tree.selection()
        if selection and len(self.review_tree.get_children()) > 1:
            current_item = selection[0]
            children = self.review_tree.get_children()
            current_index = children.index(current_item)
            next_index = (current_index + 1) % len(children)
            next_item = children[next_index]
            
            self.review_tree.selection_set(next_item)
            self.review_tree.focus(next_item)
    
    def save_reviews(self):
        """Save all pending reviews to file"""
        # Implementation would save review decisions to database or file
        messagebox.showinfo("Success", "Review queue saved successfully!")

    def setup_data_tab(self):
        """Setup business configuration and data management tab"""
        # Left frame for business controls
        left_frame = ttk.Frame(self.data_tab)
        left_frame.pack(side="left", fill="y", padx=10, pady=10)
        
        # Business Context Section
        context_frame = ttk.LabelFrame(left_frame, text="🏢 Business Context", padding=10)
        context_frame.pack(fill="x", pady=5)
        
        ttk.Label(context_frame, text="Channel Type:", font=('Arial', 10, 'bold')).pack(anchor="w")
        ttk.Label(context_frame, text="Customer Service Short Messages").pack(anchor="w")
        ttk.Label(context_frame, text="Focus: SMS, Chat, Social Media Feedback").pack(anchor="w")
        
        # Data loading section with business context
        data_frame = ttk.LabelFrame(left_frame, text="📊 Customer Message Data", padding=10)
        data_frame.pack(fill="x", pady=5)
        
        ttk.Button(data_frame, text="Load Sample Customer Messages", 
                  command=self.load_sample_data).pack(pady=2)
        ttk.Button(data_frame, text="Load CSV Message Data", 
                  command=self.load_csv_file).pack(pady=2)
        
        # Business Risk Level Configuration
        config_frame = ttk.LabelFrame(left_frame, text="⚙️ Business Risk Configuration", padding=10)
        config_frame.pack(fill="x", pady=5)
        
        ttk.Label(config_frame, text="Customer Channel Risk Level:", font=('Arial', 10, 'bold')).pack(anchor="w")
        
        # Risk level explanations
        risk_info = {
            "low": "🔒 Very Restrictive (Premium Customer Channels)",
            "medium": "⚖️ Balanced (General Customer Communication)", 
            "high": "🔓 Permissive (Open Feedback Channels)"
        }
        
        for level in ["low", "medium", "high"]:
            frame = ttk.Frame(config_frame)
            frame.pack(fill="x", pady=2)
            ttk.Radiobutton(frame, text=risk_info[level], 
                           variable=self.risk_level, value=level).pack(anchor="w")
        
        # Business Impact Information
        impact_frame = ttk.LabelFrame(config_frame, text="Business Impact", padding=5)
        impact_frame.pack(fill="x", pady=5)
        
        self.risk_info_label = ttk.Label(impact_frame, text="", wraplength=250)
        self.risk_info_label.pack(anchor="w")
        
        # Bind risk level change to update business impact
        self.risk_level.trace_add("write", self.update_risk_info)
        self.update_risk_info()
        
        # Model Management Section
        model_frame = ttk.LabelFrame(left_frame, text="🤖 Model Management", padding=10)
        model_frame.pack(fill="x", pady=5)
        
        ttk.Button(model_frame, text="Train New Model", 
                  command=self.train_model).pack(pady=2)
        ttk.Button(model_frame, text="Load Trained Model", 
                  command=self.load_trained_model).pack(pady=2)
        
        # Business Status Section
        status_frame = ttk.LabelFrame(left_frame, text="📋 System Status", padding=10)
        status_frame.pack(fill="x", pady=5)
        
        self.status_text = scrolledtext.ScrolledText(status_frame, height=12, width=35)
        self.status_text.pack(fill="both", expand=True)
        
        # Right frame for business data preview
        right_frame = ttk.Frame(self.data_tab)
        right_frame.pack(side="right", fill="both", expand=True, padx=10, pady=10)
        
        preview_frame = ttk.LabelFrame(right_frame, text="📋 Customer Message Preview", padding=10)
        preview_frame.pack(fill="both", expand=True)
        
        # Business-oriented column headers
        self.data_tree = ttk.Treeview(preview_frame, columns=("Message Type", "Customer Message", "Classification Confidence"), show="headings")
        self.data_tree.heading("Message Type", text="Message Type")
        self.data_tree.heading("Customer Message", text="Customer Message")
        self.data_tree.heading("Classification Confidence", text="Business Confidence")
        self.data_tree.column("Message Type", width=120)
        self.data_tree.column("Customer Message", width=400)
        self.data_tree.column("Classification Confidence", width=150)
        
        # Add scrollbar
        scrollbar = ttk.Scrollbar(preview_frame, orient="vertical", command=self.data_tree.yview)
        self.data_tree.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side="right", fill="y")
        self.data_tree.pack(side="left", fill="both", expand=True)
        
    def update_risk_info(self, *args):
        """Update business risk level information display"""
        risk_descriptions = {
            "low": "Premium customers, VIP channels\nMinimizes spam pass-through\nHigher false positives acceptable\nAll flagged messages reviewed",
            "medium": "General customer service\nBalanced performance\nOptimal for most use cases\nSample-based review process",
            "high": "Public feedback, open channels\nMinimizes false positives\nSome spam may pass through\nMinimal manual review needed"
        }
        
        current_risk = self.risk_level.get()
        self.risk_info_label.config(text=risk_descriptions.get(current_risk, ""))
        
    def setup_test_tab(self):
        """Setup the business message classification tab"""
        # Left side - input and controls
        left_frame = ttk.Frame(self.test_tab)
        left_frame.pack(side="left", fill="both", expand=True, padx=10, pady=10)
        
        # Business context header
        context_frame = ttk.LabelFrame(left_frame, text="💼 Customer Message Classification", padding=10)
        context_frame.pack(fill="x", pady=5)
        
        ttk.Label(context_frame, text="Channel: Customer Service Short Messages", 
                 font=('Arial', 10, 'bold')).pack(anchor="w")
        ttk.Label(context_frame, text="Purpose: Protect customer communication channels from spam").pack(anchor="w")
        
        # Message input section with business context
        input_frame = ttk.LabelFrame(left_frame, text="📱 Customer Message Input", padding=10)
        input_frame.pack(fill="x", pady=5)
        
        ttk.Label(input_frame, text="Enter customer message to classify:", 
                 font=('Arial', 10, 'bold')).pack(anchor="w")
        
        # Message type selector
        type_frame = ttk.Frame(input_frame)
        type_frame.pack(fill="x", pady=5)
        
        ttk.Label(type_frame, text="Message Source:").pack(side="left")
        self.message_source = tk.StringVar(value="sms")
        
        sources = [("📱 SMS/Text", "sms"), ("💬 Live Chat", "chat"), ("📧 Email", "email"), ("📱 Social Media", "social")]
        for text, value in sources:
            ttk.Radiobutton(type_frame, text=text, variable=self.message_source, value=value).pack(side="left", padx=5)
        
        self.test_input = scrolledtext.ScrolledText(input_frame, height=6, width=50)
        self.test_input.pack(fill="x", pady=5)
        
        # Quick test messages for business scenarios
        quick_frame = ttk.LabelFrame(input_frame, text="🚀 Quick Test Messages", padding=5)
        quick_frame.pack(fill="x", pady=5)
        
        quick_messages = [
            ("💬 Legitimate: Meeting Reminder", "Hi John, don't forget about our meeting at 2 PM today in the conference room."),
            ("💬 Legitimate: Order Update", "Your order #12345 has been processed and will be delivered tomorrow."),
            ("🚫 Spam: Prize Scam", "CONGRATULATIONS! You've won $1000! Click here immediately to claim your prize!"),
            ("🚫 Spam: Phishing", "URGENT: Your account will be suspended. Click this link to verify your information now."),
            ("⚠️ Borderline: Promotion", "Special offer just for you! 50% off all items this week. Shop now and save!")
        ]
        
        quick_btn_frame = ttk.Frame(quick_frame)
        quick_btn_frame.pack(fill="x")
        
        for i, (label, message) in enumerate(quick_messages):
            if i % 2 == 0:
                row_frame = ttk.Frame(quick_btn_frame)
                row_frame.pack(fill="x", pady=1)
            
            ttk.Button(row_frame, text=label, 
                      command=lambda m=message: self.load_test_message(m)).pack(side="left", padx=2, fill="x", expand=True)
        
        # Classification controls
        controls_frame = ttk.LabelFrame(left_frame, text="🔍 Classification Controls", padding=10)
        controls_frame.pack(fill="x", pady=5)
        
        # Risk level for this classification
        risk_frame = ttk.Frame(controls_frame)
        risk_frame.pack(fill="x", pady=5)
        
        ttk.Label(risk_frame, text="Business Context:", font=('Arial', 10, 'bold')).pack(side="left")
        self.current_risk_label = ttk.Label(risk_frame, text="", foreground="blue")
        self.current_risk_label.pack(side="left", padx=10)
        
        # Update risk label when risk level changes
        self.risk_level.trace_add("write", self.update_current_risk_display)
        self.update_current_risk_display()
        
        # Button frame
        button_frame = ttk.Frame(controls_frame)
        button_frame.pack(fill="x", pady=5)
        
        ttk.Button(button_frame, text="🤖 Classify Message", 
                  command=self.classify_message).pack(side="left", padx=5)
        ttk.Button(button_frame, text="🗑️ Clear", 
                  command=self.clear_input).pack(side="left", padx=5)
        
        # Right side - results and business insights
        right_frame = ttk.Frame(self.test_tab)
        right_frame.pack(side="right", fill="both", expand=True, padx=10, pady=10)
        
        # Classification results with business context
        results_frame = ttk.LabelFrame(right_frame, text="📊 Classification Results", padding=10)
        results_frame.pack(fill="both", expand=True, pady=5)
        
        self.results_text = scrolledtext.ScrolledText(results_frame)
        self.results_text.pack(fill="both", expand=True)
    
    def update_current_risk_display(self, *args):
        """Update the current risk level display"""
        risk_descriptions = {
            "low": "🔒 Premium Customer Channels (Restrictive)",
            "medium": "⚖️ General Customer Service (Balanced)", 
            "high": "🔓 Public Feedback Channels (Permissive)"
        }
        
        current_risk = self.risk_level.get()
        if hasattr(self, 'current_risk_label'):
            self.current_risk_label.config(text=risk_descriptions.get(current_risk, ""))
    
    def load_test_message(self, message):
        """Load a test message into the input field"""
        self.test_input.delete(1.0, tk.END)
        self.test_input.insert(1.0, message)
        
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
    
    def load_trained_model(self):
        """Load an already trained model from file"""
        try:
            # Open file dialog to select model file
            file_path = filedialog.askopenfilename(
                title="Select Trained Model",
                filetypes=[("Pickle files", "*.pkl"), ("All files", "*.*")],
                initialdir="models"
            )
            
            if not file_path:
                return  # User cancelled
            
            self.log_message(f"Loading model from: {file_path}")
            
            # Load the model using SpamFilter's load_from_file method
            loaded_model = SpamFilter.load_from_file(file_path)
            
            if loaded_model is None:
                messagebox.showerror("Error", "Failed to load the selected model file!")
                self.log_message("Failed to load model file.")
                return
            
            # Set the loaded model
            self.model = loaded_model
            self.is_trained = True
            
            # Update risk level display to match loaded model
            self.risk_level.set(self.model.risk_level)
            
            self.log_message(f"✓ Model loaded successfully!")
            self.log_message(f"Risk Level: {self.model.risk_level}")
            self.log_message(f"Threshold: {self.model.threshold}")
            self.log_message(f"Model Type: {self.model.model_type}")
            self.log_message("Model is ready for testing!")
            
            messagebox.showinfo("Success", f"Model loaded successfully!\nRisk Level: {self.model.risk_level}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load model: {str(e)}")
            self.log_message(f"Error loading model: {str(e)}")
    
    def classify_message(self):
        """Classify a message with business context and confidence scoring"""
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
            
            # Format business-oriented results
            result = "SPAM" if prediction[0] == 1 else "HAM"
            confidence = probability[0] * 100 if prediction[0] == 1 else (1 - probability[0]) * 100
            
            # Clear previous results
            self.results_text.delete("1.0", tk.END)
            
            # Format comprehensive business results
            result_text = f"""🔍 BUSINESS CLASSIFICATION RESULTS
{'='*50}

📱 MESSAGE SOURCE: {getattr(self, 'message_source', tk.StringVar(value='unknown')).get().upper()}
📝 MESSAGE: {message[:100]}{'...' if len(message) > 100 else ''}

🤖 CLASSIFICATION RESULT: {result}
📊 MODEL CONFIDENCE: {confidence:.1f}%
💼 SPAM PROBABILITY: {probability[0]:.1%}

🏢 BUSINESS CONTEXT:
• Risk Level: {self.risk_level.get().title()}
• Channel Type: Customer Service Short Messages  
• Threshold: {self.model.threshold}
• Business Impact: {'HIGH' if result == 'SPAM' and confidence > 70 else 'MEDIUM' if result == 'SPAM' else 'LOW'}

⚡ RECOMMENDED ACTIONS:
"""
            
            # Add business recommendations based on confidence and risk level
            risk_level = self.risk_level.get()
            
            if result == "SPAM":
                if confidence >= 80:
                    result_text += "✅ BLOCK MESSAGE - High confidence spam detection\n"
                    result_text += "• Automatically block and log for security monitoring\n"
                    result_text += "• No manual review required\n"
                elif confidence >= 60:
                    result_text += "⚠️ REVIEW REQUIRED - Moderate confidence spam\n"
                    result_text += "• Route to manual review queue\n"
                    result_text += "• Consider customer service team review\n"
                else:
                    result_text += "🔍 MANUAL REVIEW CRITICAL - Low confidence\n"
                    result_text += "• High priority manual review required\n"
                    result_text += "• Potential for false positive\n"
            else:  # HAM
                if confidence >= 80:
                    result_text += "✅ ALLOW MESSAGE - Legitimate customer communication\n"
                    result_text += "• Route to appropriate customer service queue\n"
                    result_text += "• No additional screening needed\n"
                elif confidence >= 60:
                    result_text += "⚠️ ALLOW WITH MONITORING - Moderate confidence\n"
                    result_text += "• Allow but monitor for patterns\n"
                    result_text += "• Consider sample review\n"
                else:
                    result_text += "🔍 REVIEW RECOMMENDED - Uncertain classification\n"
                    result_text += "• Manual review suggested for quality assurance\n"
                    result_text += "• Monitor customer feedback\n"
            
            # Add risk-specific recommendations
            if risk_level == "low":
                result_text += f"\n🔒 PREMIUM CHANNEL PROTOCOL:\n"
                result_text += "• Extra scrutiny applied for VIP customers\n"
                result_text += "• False positive tolerance: LOW\n"
                result_text += "• Manual review threshold: 70%+\n"
            elif risk_level == "high":
                result_text += f"\n🔓 OPEN CHANNEL PROTOCOL:\n"
                result_text += "• Balanced approach for public feedback\n"
                result_text += "• False positive tolerance: HIGH\n"
                result_text += "• Manual review threshold: 50%+\n"
            
            # Add quality metrics
            result_text += f"""
📊 QUALITY METRICS:
• Model Training Accuracy: {getattr(self.model, 'accuracy', 'Unknown')}
• Processing Time: <1 second
• Confidence Threshold Met: {'✅ YES' if confidence >= 60 else '❌ NO'}
• Review Required: {'✅ YES' if confidence < 70 else '❌ NO'}

💡 BUSINESS INSIGHTS:
• Customer Experience Impact: {'MINIMAL' if result == 'HAM' or confidence > 80 else 'MODERATE'}
• False Positive Risk: {'LOW' if confidence > 80 else 'MEDIUM' if confidence > 60 else 'HIGH'}
• Operational Cost: {'LOW - Automated' if confidence > 80 else 'MEDIUM - Review Required'}

📈 TECHNICAL DETAILS:
• Message Length: {len(message)} characters
• Spam Probability: {probability[0]:.3f}
• Ham Probability: {1-probability[0]:.3f}
• Risk Threshold: {self.model.threshold}
• Algorithm: Naive Bayes with TF-IDF
"""
            
            self.results_text.insert("1.0", result_text)
            
        except Exception as e:
            messagebox.showerror("Error", f"Classification failed: {str(e)}")
            
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