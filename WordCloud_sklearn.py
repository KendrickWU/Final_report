from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.model_selection import train_test_split
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# Fetch the dataset
newsgroups = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))

# Split the data into training and testing sets
data_train, data_test = train_test_split(newsgroups.data, test_size=0.99, random_state=42)

# Vectorize the training data
tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2, max_features=1000, stop_words='english')
tf = tf_vectorizer.fit_transform(data_train)

# Train the LDA model
lda = LatentDirichletAllocation(n_components=10, max_iter=5, learning_method='online', learning_offset=50., random_state=0)
lda.fit(tf)

# Prepare the unseen document (let's take the first document from the test set for simplicity)
new_doc = data_test[0]
tf_new_doc = tf_vectorizer.transform([new_doc])

# Use the LDA model to infer the topic distribution for the new document
topic_distribution = lda.transform(tf_new_doc)

# Print the topic distribution for the unseen document
print("Topic distribution for the unseen document:")
print(topic_distribution)

# Function to plot 10 topics with their word clouds
def plot_10_topic_word_clouds(lda_model, feature_names):
    cols = [color for name, color in mcolors.TABLEAU_COLORS.items()]  # more colors: 'mcolors.XKCD_COLORS'

    cloud = WordCloud(
        background_color='white',
        width=2500,
        height=1800,
        max_words=10,
        colormap='tab10',
        color_func=lambda *args, **kwargs: cols[i],
        prefer_horizontal=1.0
    )

    topics = lda_model.components_

    fig, axes = plt.subplots(2, 5, figsize=(10, 4), sharex=True, sharey=True)

    for i, ax in enumerate(axes.flatten()):
        fig.add_subplot(ax)
        topic_words = {feature_names[j]: topics[i][j] for j in topics[i].argsort()[:-11:-1]}
        cloud.generate_from_frequencies(topic_words, max_font_size=300)
        plt.gca().imshow(cloud)
        plt.gca().set_title('Topic ' + str(i), fontdict=dict(size=16))
        plt.gca().axis('off')

    plt.subplots_adjust(wspace=0, hspace=0)
    plt.axis('off')
    plt.margins(x=0, y=0)
    plt.tight_layout()
    plt.show()

# Call the function
plot_10_topic_word_clouds(lda, tf_vectorizer.get_feature_names_out())