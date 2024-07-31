import pandas as pd
from collections import Counter
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import matplotlib.pyplot as plt
import seaborn as sns
import nltk

# Download necessary NLTK resources
nltk.download('stopwords')
nltk.download('punkt')

# Function to clean and tokenize text
def preprocess_text(text):
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(text.lower())
    tokens = [word for word in tokens if word.isalpha() and word not in stop_words]
    return tokens

# Function to get top word counts
def get_top_words(dataframe, num_words=10):
    all_words = []
    for text in dataframe['Original Text']:
        all_words.extend(preprocess_text(text))
    word_counts = Counter(all_words)
    return word_counts.most_common(num_words)

# List of files to process
files = [
    'GB/true_negatives_gb_only.csv', 'GB/false_positives_gb_only.csv', 'GB/true_positives_gb_only.csv',
    'GB/false_negatives_gb_only.csv',
    'OverlappedResults/false_negatives_overlap.csv', 'OverlappedResults/false_positives_overlap.csv', 'OverlappedResults/true_negatives_overlap.csv',
    'OverlappedResults/true_positives_overlap.csv',
    'RF/false_negatives_rf_only.csv', 'RF/false_positives_rf_only.csv', 'RF/true_negatives_rf_only.csv',
    'RF/true_positives_rf_only.csv'
]

# Dictionary to store top words for each file
top_words_dict = {}

# Process each file
for file in files:
    df = pd.read_csv(file)
    top_words_dict[file] = get_top_words(df)

# Outputting the results
for file, top_words in top_words_dict.items():
    print(f'Top Words in {file}:')
    for word, count in top_words:
        print(f'  {word}: {count}')
    print()  # Blank line for readability

# Commented out the plotting section
# for file, top_words in top_words_dict.items():
#     words, counts = zip(*top_words)
#     plt.figure(figsize=(10, 6))
#     sns.barplot(x=list(counts), y=list(words))
#     plt.title(f'Top Words in {file}')
#     plt.xlabel('Word Count')
#     plt.ylabel('Words')
#     plt.show()
