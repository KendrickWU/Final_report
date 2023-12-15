import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from IPython.core.display import HTML

# Load the dataset
data = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))
data_samples = data.data[:100]  # Train on 100 documents

# Feature extraction
tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2, max_features=1000, stop_words='english')
tf = tf_vectorizer.fit_transform(data_samples)

# Fit the LDA model
lda = LatentDirichletAllocation(n_components=5, random_state=0)
lda.fit(tf)

# Display the topics
print("LDA model topics:")
topic_colors = ['red', 'green', 'blue', 'yellow', 'purple']  # Colors for each topic

for topic_idx, topic in enumerate(lda.components_):
    message = "Topic #%d: " % topic_idx
    message += " ".join([tf_vectorizer.get_feature_names_out()[i]
                         for i in topic.argsort()[:-10 - 1:-1]])
    print(message)

# Testing on a new document
new_doc = data.data[101]  # Select a new document
words = new_doc.split()[:250]  # Get the first 250 words
word_topics = np.argmax(lda.transform(tf_vectorizer.transform(words)), axis=1)

# Generate HTML to color-code words
html = "".join([f'<span style="color: {topic_colors[topic_idx]};">{word}</span> ' for word, topic_idx in zip(words, word_topics)])

# Display the HTML with color-coded words
display(HTML(html))