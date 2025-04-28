import pandas as pd
import numpy as np
import nltk
from nltk.corpus import twitter_samples
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense, Dropout
import matplotlib.pyplot as plt

def load_online_data():
    """Load Twitter sentiment dataset from NLTK"""
    print("Downloading Twitter dataset from NLTK...")
    
    try:
        # Download necessary NLTK data
        nltk.download('twitter_samples', quiet=True)
        
        # Get positive and negative tweets
        positive_tweets = twitter_samples.strings('positive_tweets.json')
        negative_tweets = twitter_samples.strings('negative_tweets.json')
        
        # Create a DataFrame from these tweets
        tweets = []
        for tweet in positive_tweets:
            tweets.append((tweet, 'positive'))
        for tweet in negative_tweets:
            tweets.append((tweet, 'negative'))
        
        df = pd.DataFrame(tweets, columns=['review', 'sentiment'])
        print(f"Dataset successfully loaded! Total samples: {len(df)}")
        return df
    except Exception as e:
        print(f"Failed to download dataset: {str(e)}")
        
        # Backup option: Small hardcoded dataset for demonstration
        print("Using backup mini dataset...")
        # Adding more training examples to improve model performance
        mini_data = [
            ("This movie was fantastic! I really enjoyed it.", "positive"),
            ("Great acting and wonderful direction.", "positive"),
            ("I loved everything about this film!", "positive"),
            ("This was the best movie I've seen in years.", "positive"),
            ("What an amazing storyline and beautiful cinematography.", "positive"),
            ("The characters were well developed and the plot was engaging.", "positive"),
            ("I couldn't stop smiling throughout the entire movie.", "positive"),
            ("This film exceeded all my expectations.", "positive"),
            ("A true masterpiece of modern cinema.", "positive"),
            ("The screenplay was brilliant and the ending was perfect.", "positive"),
            ("This movie was terrible and a complete waste of time.", "negative"),
            ("I hated the plot and the acting was awful.", "negative"),
            ("Worst movie experience ever, would not recommend.", "negative"),
            ("The film was boring and predictable.", "negative"),
            ("Disappointing storyline and poor character development.", "negative"),
            ("I almost fell asleep during this dreadful movie.", "negative"),
            ("The dialogue was unnatural and the pacing was terrible.", "negative"),
            ("I regret watching this film. It was a waste of money.", "negative"),
            ("The special effects were cheap and unconvincing.", "negative"),
            ("The ending made no sense and left me frustrated.", "negative")
        ]
        return pd.DataFrame(mini_data, columns=['review', 'sentiment'])

def prepare_data(df, max_words=10000, max_len=100):
    """Prepare the data for LSTM model"""
    # Map sentiment labels to binary values
    sentiment_mapping = {'positive': 1, 'negative': 0}
    df['sentiment_label'] = df['sentiment'].map(sentiment_mapping)
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        df['review'], 
        df['sentiment_label'], 
        test_size=0.2, 
        random_state=42
    )
    
    # Tokenize the text data
    tokenizer = Tokenizer(num_words=max_words)
    tokenizer.fit_on_texts(X_train)
    
    # Convert text to sequences
    X_train_seq = tokenizer.texts_to_sequences(X_train)
    X_test_seq = tokenizer.texts_to_sequences(X_test)
    
    # Pad sequences to ensure uniform length
    X_train_padded = pad_sequences(X_train_seq, maxlen=max_len)
    X_test_padded = pad_sequences(X_test_seq, maxlen=max_len)
    
    return X_train_padded, X_test_padded, y_train, y_test, tokenizer, max_len

def build_and_train_model(X_train, y_train, X_test, y_test, vocab_size=10000, max_len=100):
    """Build and train Bidirectional LSTM model"""
    print("Building and training LSTM model...")
    
    # Set random seeds for reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)
    
    # Create the model
    model = Sequential()
    
    # Add layers
    model.add(Embedding(input_dim=vocab_size, output_dim=128, input_length=max_len))
    model.add(Bidirectional(LSTM(128, return_sequences=True)))
    model.add(Bidirectional(LSTM(64)))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    
    # Compile the model
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    # Display model summary
    model.summary()
    
    # Train the model
    history = model.fit(
        X_train, y_train,
        epochs=5,  # Reduced for demo but can be increased for better accuracy
        batch_size=32,
        validation_split=0.1,
        verbose=1
    )
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='lower right')
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')
    
    plt.tight_layout()
    
    # Save the plot as an image if needed
    # plt.savefig('training_history.png')
    
    print("Model training completed!")
    
    return model

def evaluate_model(model, X_test, y_test):
    """Evaluate the trained model"""
    # Make predictions
    y_pred_proba = model.predict(X_test)
    y_pred = (y_pred_proba > 0.5).astype(int)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    
    # Print evaluation metrics
    print("\nModel Evaluation Results:")
    print(f"Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Negative', 'Positive']))
    
    return accuracy

def predict_sentiment(model, tokenizer, max_len, text):
    """Predict sentiment for user input"""
    # Preprocess the input text
    sequence = tokenizer.texts_to_sequences([text])
    padded_sequence = pad_sequences(sequence, maxlen=max_len)
    
    # Predict sentiment
    prediction = model.predict(padded_sequence)[0][0]
    
    # Determine sentiment and confidence
    sentiment = "Positive" if prediction > 0.5 else "Negative"
    confidence = float(prediction) if prediction > 0.5 else float(1 - prediction)
    
    return sentiment, confidence

def main():
    # Load dataset
    df = load_online_data()
    
    if df is None or df.empty:
        print("Failed to load dataset. Exiting.")
        return
    
    print(f"Dataset shape: {df.shape}")
    
    # Prepare data
    max_words = 10000
    max_len = 100
    X_train, X_test, y_train, y_test, tokenizer, max_len = prepare_data(df, max_words, max_len)
    
    # Build and train model
    model = build_and_train_model(X_train, y_train, X_test, y_test, max_words, max_len)
    
    # Evaluate model
    evaluate_model(model, X_test, y_test)
    
    # User interaction loop
    while True:
        print("\n" + "="*50)
        user_input = input("Enter a review to analyze sentiment (or 'quit' to exit): ")
        
        if user_input.lower() == 'quit':
            print("Exiting program. Goodbye!")
            break
        
        sentiment, confidence = predict_sentiment(model, tokenizer, max_len, user_input)
        print(f"\nSentiment: {sentiment}")
        print(f"Confidence: {confidence:.2f}")

if __name__ == "__main__":
    main()