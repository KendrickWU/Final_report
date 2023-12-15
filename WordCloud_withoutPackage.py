from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from wordcloud import WordCloud

# Load the 20 Newsgroups dataset
newsgroups = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))

# Using only a subset of 100 documents for training the LDA model
data_samples = newsgroups.data[:100]

# Split the dataset to create a hold-out set for testing
train_data, test_data = train_test_split(data_samples, test_size=0.1, random_state=42)

# Vectorize the training documents
tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2, max_features=1000, stop_words='english')
X_train = tf_vectorizer.fit_transform(train_data).toarray()

# Obtain the feature names, which serve as our vocabulary
vocab = tf_vectorizer.get_feature_names_out()

# Initialize the LDA model parameters
num_topics = 10
num_docs, num_words = X_train.shape
num_iterations = 100
alpha = 1.0
eta = 0.01

# Randomly initialize the document-topic (theta) distribution and topic-word (beta) distribution
theta = np.random.dirichlet(alpha=np.ones(num_topics) * alpha, size=num_docs)
beta = np.random.dirichlet(alpha=np.ones(num_words) * eta, size=num_topics)

# EM algorithm to learn LDA model parameters
for iteration in range(num_iterations):
    # E-step
    phi = np.zeros((num_docs, num_topics, num_words))
    for d in range(num_docs):
        for w in range(num_words):
            if X_train[d, w] > 0:
                phi[d, :, w] = theta[d, :] * beta[:, w]
                phi[d, :, w] /= phi[d, :, w].sum()

    # M-step
    for k in range(num_topics):
        for w in range(num_words):
            beta[k, w] = np.sum(phi[:, k, w] * X_train[:, w]) + eta
        beta[k, :] /= beta[k, :].sum()

    for d in range(num_docs):
        for k in range(num_topics):
            theta[d, k] = np.sum(phi[d, k, :] * X_train[d, :]) + alpha
        theta[d, :] /= theta[d, :].sum()

# Function to plot topics with their word clouds
def plot_topics_word_clouds(beta, feature_names):
    cols = [color for name, color in mcolors.TABLEAU_COLORS.items()]

    cloud = WordCloud(
        background_color='white',
        width=2500,
        height=1800,
        max_words=10,
        colormap='tab10',
        color_func=lambda *args, **kwargs: cols[i],
        prefer_horizontal=1.0
    )

    fig, axes = plt.subplots(2, 5, figsize=(10, 4), sharex=True, sharey=True)

    for i, ax in enumerate(axes.flatten()):
        fig.add_subplot(ax)
        topic_words = {feature_names[j]: beta[i, j] for j in beta[i].argsort()[:-11:-1]}
        cloud.generate_from_frequencies(topic_words, max_font_size=300)
        plt.gca().imshow(cloud)
        plt.gca().set_title('Topic ' + str(i), fontdict=dict(size=16))
        plt.gca().axis('off')

    plt.subplots_adjust(wspace=0, hspace=0)
    plt.axis('off')
    plt.margins(x=0, y=0)
    plt.tight_layout()
    plt.show()

# Call the function to plot the topics after the LDA model parameters are learned
plot_topics_word_clouds(beta, vocab)

# Your existing colorization code
# ...

# Use the first document from the hold-out test set as the new document for visualization
new_doc = test_data[0]  # This document was not used during parameter estimation

# Preprocess and vectorize the held-out document using the same vectorizer
X_new_doc = tf_vectorizer.transform([new_doc]).toarray()[0]

# Infer the topic distribution for the new document
new_doc_theta = np.random.dirichlet(alpha=np.ones(num_topics) * alpha)
phi_new_doc = np.zeros((num_topics, num_words))
for w in range(num_words):
    if X_new_doc[w] > 0:
        phi_new_doc[:, w] = new_doc_theta * beta[:, w]
        phi_new_doc[:, w] /= phi_new_doc[:, w].sum()

new_doc_theta = phi_new_doc.sum(axis=1) + alpha
new_doc_theta /= new_doc_theta.sum()

# Colorize the words in the new document based on the topic distribution
# Note: This is a simple approach and more sophisticated methods can be used
colors = list(mcolors.TABLEAU_COLORS.keys())
word_colors = {}

for w in range(num_words):
    if X_new_doc[w] > 0:
        topic = phi_new_doc[:, w].argmax()
        word = vocab[w]
        word_colors[word] = colors[topic % len(colors)]

# Print the new document with words colorized based on their assigned topic
for word in new_doc.split():
    clean_word = ''.join(char for char in word if char.isalnum())
    if clean_word in word_colors:
        print(f"\033[38;5;{mcolors.TABLEAU_COLORS[word_colors[clean_word]][1:]}m{word}\033[0m", end=' ')
    else:
        print(word, end=' ')