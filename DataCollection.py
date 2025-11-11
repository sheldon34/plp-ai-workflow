# Import necessary libraries
import pandas as pd
# Import necessary libraries
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
# Sample data: Customer reviews
data = {
    'review': [
        'I love this product!',
        'This is the worst product I have ever bought.',
        'Great quality and fast delivery.',
        'Not worth the money.',
        'Highly recommend this product.'
    ],
    'sentiment': ['positive', 'negative', 'positive', 'negative', 'positive']
}

# Create a DataFrame
df = pd.DataFrame(data)
print(df)
# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Sample text preprocessing function
def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Tokenize text
    tokens = word_tokenize(text)
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)

# Apply preprocessing to the DataFrame
df['cleaned_review'] = df['review'].apply(preprocess_text)
print(df[['review', 'cleaned_review']])