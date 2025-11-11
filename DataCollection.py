import pandas as pd

import string
data={
    'review':[
        'I Love this product',
        'This is  the worst product i have ever bought .',
        'Great quality and fast delivery .',
        'Not worth the money',
        'Highly recommend this product'
    ],
    'sentiment':['positive', 'negative','positive','negative','positive']

}
df=pd.DataFrame(data)
print(df)
