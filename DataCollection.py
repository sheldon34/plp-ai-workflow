# Import necessary libraries
import pandas as pd

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
