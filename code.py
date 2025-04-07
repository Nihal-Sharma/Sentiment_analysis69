import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# Download necessary NLTK resources
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')

class SentimentAnalyzer:
    def __init__(self, classifier_type='svm'):
        """
        Initialize the sentiment analyzer with specified classifier
        
        Parameters:
        -----------
        classifier_type : str
            Type of classifier to use ('svm' or 'random_forest')
        """
        self.classifier_type = classifier_type.lower()
        self.vectorizer = TfidfVectorizer(max_features=5000)
        self.classifier = None
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        
    def preprocess_text(self, text):
        """
        Clean and preprocess text data
        
        Parameters:
        -----------
        text : str
            Raw text to be preprocessed
            
        Returns:
        --------
        str
            Preprocessed text
        """
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        
        # Remove HTML tags
        text = re.sub(r'<.*?>', '', text)
        
        # Remove punctuation and numbers
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\d+', '', text)
        
        # Tokenize
        tokens = nltk.word_tokenize(text)
        
        # Remove stopwords and lemmatize
        tokens = [self.lemmatizer.lemmatize(word) for word in tokens if word not in self.stop_words and len(word) > 2]
        
        return ' '.join(tokens)
    
    def prepare_data(self, df, text_column, label_column):
        """
        Prepare data for training
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame containing text data and sentiment labels
        text_column : str
            Name of column containing text data
        label_column : str
            Name of column containing sentiment labels
            
        Returns:
        --------
        tuple
            (X_train, X_test, y_train, y_test)
        """
        # Clean text data
        df['clean_text'] = df[text_column].apply(self.preprocess_text)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            df['clean_text'], 
            df[label_column], 
            test_size=0.2, 
            random_state=42
        )
        
        # Vectorize text data
        X_train_vec = self.vectorizer.fit_transform(X_train)
        X_test_vec = self.vectorizer.transform(X_test)
        
        return X_train_vec, X_test_vec, y_train, y_test
    
    def train(self, X_train, y_train):
        """
        Train the classifier
        
        Parameters:
        -----------
        X_train : scipy.sparse.csr.csr_matrix
            Vectorized training text data
        y_train : pandas.Series
            Training labels
        """
        if self.classifier_type == 'svm':
            # SVM with hyperparameter tuning
            param_grid = {
                'C': [0.1, 1, 10, 100],
                'gamma': ['scale', 'auto'],
                'kernel': ['linear', 'rbf']
            }
            base_model = SVC(probability=True)
            
        elif self.classifier_type == 'random_forest':
            # Random Forest with hyperparameter tuning
            param_grid = {
                'n_estimators': [100, 200],
                'max_depth': [None, 10, 20],
                'min_samples_split': [2, 5],
                'min_samples_leaf': [1, 2]
            }
            base_model = RandomForestClassifier(random_state=42)
            
        else:
            raise ValueError("Classifier type must be 'svm' or 'random_forest'")
        
        # Use GridSearchCV for hyperparameter optimization
        grid_search = GridSearchCV(
            base_model,
            param_grid,
            cv=5,
            scoring='accuracy',
            verbose=1,
            n_jobs=-1
        )
        
        grid_search.fit(X_train, y_train)
        self.classifier = grid_search.best_estimator_
        print(f"Best parameters: {grid_search.best_params_}")
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate the trained classifier
        
        Parameters:
        -----------
        X_test : scipy.sparse.csr.csr_matrix
            Vectorized test text data
        y_test : pandas.Series
            Test labels
            
        Returns:
        --------
        dict
            Dictionary containing evaluation metrics
        """
        y_pred = self.classifier.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        conf_matrix = confusion_matrix(y_test, y_pred)
        
        # Plot confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=sorted(set(y_test)), 
                    yticklabels=sorted(set(y_test)))
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title(f'Confusion Matrix - {self.classifier_type.upper()}')
        plt.tight_layout()
        plt.show()
        
        print(f"\nAccuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        return {
            'accuracy': accuracy,
            'classification_report': report,
            'confusion_matrix': conf_matrix
        }
    
    def predict(self, texts):
        """
        Predict sentiment for new texts
        
        Parameters:
        -----------
        texts : list or pandas.Series
            List of texts to analyze
            
        Returns:
        --------
        numpy.ndarray
            Predicted sentiment labels
        """
        if not self.classifier:
            raise ValueError("Classifier not trained. Call train() first.")
        
        # Preprocess texts
        clean_texts = [self.preprocess_text(text) for text in texts]
        
        # Vectorize
        X_vec = self.vectorizer.transform(clean_texts)
        
        # Predict
        return self.classifier.predict(X_vec)
    
    def predict_proba(self, texts):
        """
        Get probability estimates for each class
        
        Parameters:
        -----------
        texts : list or pandas.Series
            List of texts to analyze
            
        Returns:
        --------
        numpy.ndarray
            Probability estimates for each class
        """
        if not self.classifier:
            raise ValueError("Classifier not trained. Call train() first.")
        
        # Preprocess texts
        clean_texts = [self.preprocess_text(text) for text in texts]
        
        # Vectorize
        X_vec = self.vectorizer.transform(clean_texts)
        
        # Predict probabilities
        return self.classifier.predict_proba(X_vec)


# Example usage
if __name__ == "__main__":
    # Sample data - in a real scenario, you would load your own dataset
    data = {
        'text': [
            "This product is amazing, I love it!",
            "Worst purchase ever, terrible quality.",
            "It's okay, not great but not bad either.",
            "I'm very satisfied with this product, works perfectly.",
            "Disappointing experience, wouldn't recommend.",
            "Average product, does what it's supposed to do."
        ],
        'sentiment': [
            'positive',
            'negative',
            'neutral',
            'positive',
            'negative',
            'neutral'
        ]
    }
    
    df = pd.DataFrame(data)
    
    # Create and train SVM classifier
    print("Training SVM classifier...")
    svm_analyzer = SentimentAnalyzer(classifier_type='svm')
    X_train_vec, X_test_vec, y_train, y_test = svm_analyzer.prepare_data(df, 'text', 'sentiment')
    svm_analyzer.train(X_train_vec, y_train)
    svm_metrics = svm_analyzer.evaluate(X_test_vec, y_test)
    
    # Create and train Random Forest classifier
    print("\nTraining Random Forest classifier...")
    rf_analyzer = SentimentAnalyzer(classifier_type='random_forest')
    X_train_vec, X_test_vec, y_train, y_test = rf_analyzer.prepare_data(df, 'text', 'sentiment')
    rf_analyzer.train(X_train_vec, y_train)
    rf_metrics = rf_analyzer.evaluate(X_test_vec, y_test)
    
    # Test on new data
    new_texts = [
        "The customer service was exceptional!",
        "I regret buying this, it broke after a week.",
        "It's a reasonable product for the price."
    ]
    
    # Choose the better performing model
    better_model = svm_analyzer if svm_metrics['accuracy'] >= rf_metrics['accuracy'] else rf_analyzer
    print(f"\nUsing {better_model.classifier_type.upper()} for predictions on new data:")
    
    predictions = better_model.predict(new_texts)
    probabilities = better_model.predict_proba(new_texts)
    
    for text, pred, prob in zip(new_texts, predictions, probabilities):
        print(f"\nText: '{text}'")
        print(f"Prediction: {pred}")
        class_labels = better_model.classifier.classes_
        probs_formatted = {class_label: f"{probability:.4f}" for class_label, probability in zip(class_labels, prob)}
        print(f"Probabilities: {probs_formatted}")