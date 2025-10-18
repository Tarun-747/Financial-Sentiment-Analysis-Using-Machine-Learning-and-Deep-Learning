import seaborn as sns
import matplotlib.pyplot as plt

def plot_sentiment_distribution(df):
    sns.countplot(x='sentiment', data=df)
    plt.title('Sentiment Distribution')
    plt.savefig("outputs/sentiment_distribution.png")
    plt.show()
