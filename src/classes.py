import random
import requests
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
import xml.etree.ElementTree as ET
import dateutil.parser
import pandas as pd
import plotly.express as px
import numpy as np
import plotly.graph_objs as go


class ArxivAPI:
    """
    The ArxivAPI class is responsible for fetching papers from the ArXiv API based on a specific URL.
    
    Attributes:
        url (str): The URL to fetch papers from the ArXiv API.
        papers (list): A list to store the fetched papers, each represented as a dictionary.
    
    Methods:
        fetch_papers(): Makes a request to the ArXiv API, parses the response to extract required details 
                        from each paper, and stores the details in the 'papers' list. Returns a DataFrame
                        created from the 'papers' list.
    """
    def __init__(self, url):
        self.url = url
        self.papers = []
        
    def fetch_papers(self):
        response = requests.get(self.url)
        root = ET.fromstring(response.content)
        namespaces = {'atom': 'http://www.w3.org/2005/Atom', 'arxiv': 'http://arxiv.org/schemas/atom'}

        for entry in root.findall('atom:entry', namespaces):
            paper = {}
            paper['title'] = entry.find('atom:title', namespaces).text
            paper['summary'] = entry.find('atom:summary', namespaces).text
            paper['date'] = dateutil.parser.parse(entry.find('atom:published', namespaces).text)
            self.papers.append(paper)

        return pd.DataFrame(self.papers)


class ProcessingData:
    """
    The ProcessingData class is used to process a DataFrame of papers, including transforming paper 
    summaries into tf-idf vectors, fitting an NMF model to extract topics, and mapping each paper to its topic.

    Attributes:
        df (DataFrame): The DataFrame of papers to be processed.
        n_topics (int): The number of topics to extract from the NMF model.
        n_top_words (int): The number of top words to consider in each topic.
        n_connected_words (int): The number of words to include in the summary of each topic.
        stop_words (list): A list of words to exclude during the tf-idf transformation.

    Methods:
        process(): Performs the tf-idf transformation, fits the NMF model, assigns each paper to its topic, 
                   and creates a summary for each topic. Returns a DataFrame with additional columns for the topic
                   and the topic summary. The final grouping of words of the topic is based on a random start index
                   position of 5 connected words from the n_top_words.
    """
    def __init__(self, df, n_topics=10, n_top_words=10, n_connected_words=5):
        self.df = df
        self.n_topics = n_topics
        self.n_top_words = n_top_words
        self.n_connected_words = n_connected_words
        self.stop_words = list(ENGLISH_STOP_WORDS)
        self.stop_words.extend(['data','tasks','task','models','node','machine', 'datasets'])

    def process(self):
        vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, stop_words=self.stop_words)
        tfidf = vectorizer.fit_transform(self.df['summary'])

        nmf = NMF(n_components=self.n_topics, random_state=1, max_iter=2000).fit(tfidf)
        self.df['topic'] = np.argmax(nmf.transform(tfidf), axis=1) + 1
        self.df['year'] = self.df['date'].dt.year

        _dict = {}
        for topic_idx, topic in enumerate(nmf.components_):
            top_word_indices = topic.argsort()[::-1][:self.n_top_words]
            top_words = [vectorizer.get_feature_names_out()[i] for i in top_word_indices]
            start_index = random.randint(0, len(top_words) - self.n_connected_words)
            connected_words = top_words[start_index : start_index + self.n_connected_words]
            _dict[f"{topic_idx + 1}"] = ' '.join(connected_words)

        topics_df = pd.DataFrame(_dict, index=[0]).T.reset_index()
        topics_df.columns = ['topic','topics']
        topics_df.topic = topics_df.topic.astype('int')

        return self.df.merge(topics_df, on='topic', how='left')


class Visualize:
    """
    The Visualize class is used to create a visualization of the distribution of topics over time.

    Attributes:
        df (DataFrame): The DataFrame of papers, each with an assigned topic and a topic summary.

    Methods:
        plot(): Creates and displays a stacked bar chart showing the distribution of topics over time.
    """
    def __init__(self, df):
        self.df = df

    def plot(self,number):
        self.df['date'] = pd.to_datetime(self.df['date'])
        df_grouped = self.df.groupby([self.df['date'].dt.date, 'topics']).size().reset_index(name='count')
        df_wide = df_grouped.pivot(index='date', columns='topics', values='count').reset_index().fillna(0)

        fig = px.bar(df_wide, x='date', y=df_wide.columns[1:],
                     labels={'value':'Frequency', 'date':'Date', 'variable':'Topic Summary'},
                     title=f'Topic Distribution Over Time for the Last {number} Machine Learning Papers on ArXiv')
        fig.update_layout(barmode='stack')
        return fig