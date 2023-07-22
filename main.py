import streamlit as st
from src.classes import ArxivAPI, ProcessingData, Visualize

st.set_page_config(page_title="ML Paper Topic Viz",
        page_icon="chart_with_upwards_trend",
        layout="wide")

st.markdown("""
<style>
.bottom-bar {
  background-color: #f8f9fa;
  color: #343a40;
  padding: 0px; 
  position: fixed;
  width: 90%;
  margin: auto;
  bottom: 0;
  font-family: Arial, Helvetica, sans-serif;
  display: flex;
  justify-content: center;
  align-items: center;
  flex-direction: column;
}
</style>

<div class='bottom-bar'>
  <h3>ML Paper Explorer</h3>
  <p>This app is designed to help you understand and visualize the distribution of topics over time for <a href='https://arxiv.org/' target='_blank'>ArXiv</a> Machine Learning Papers.</p>
</div>
""", unsafe_allow_html=True)

st.sidebar.header("How topics are generated 📓")
st.sidebar.markdown("""
The topics are identified from the machine learning papers using a method called Non-negative Matrix Factorization (NMF). 

This method takes into account the following parameters controlled by the sliders in this app:

1. **Number of topics**: This is the number of topics that NMF will attempt to identify from the papers. 

2. **Number of top words**: For each topic, the algorithm will identify and display this number of the most representative words.

The identified topics and their top words are then displayed in the main area of the app, with their distribution over time visualized as a stacked bar chart.
                    
""")


@st.cache_data(show_spinner="Fetching data from ArXiV...")
def fetch_papers(number):
    api = ArxivAPI(f"http://export.arxiv.org/api/query?search_query=cat:cs.LG&start=0&max_results={number}&sortBy=submittedDate&sortOrder=descending")
    return api.fetch_papers()

@st.cache_data(show_spinner=f"NLP time!🚀...")
def process_papers(df, num_topics, num_top_words):
    processor = ProcessingData(df,num_topics,num_top_words,num_top_words)
    return processor.process()

# Create sliders in the sidebar
num_topics = st.sidebar.slider("Number of topics", 1, 10, 10)
num_top_words = st.sidebar.slider("Number of top words", 1, 10, 3)
number = st.sidebar.slider("Number of papers", 1, 10000, 3000)

st.title('Topic Distribution Over Time for ArXiv Machine Learning Papers')

# Fetch the papers from the API and store them in a dataframe
df = fetch_papers(number)

# Process the dataframe to extract and assign topics
df2 = process_papers(df, num_topics, num_top_words)

# Visualize the topic distribution over time
viz = Visualize(df2)
fig = viz.plot(number=number)

# Display the figure in Streamlit
st.plotly_chart(fig, use_container_width=True)




st.sidebar.markdown("""
Designed by [Youssef Sultan](https://www.linkedin.com/in/YoussefSultan)
""")

st.sidebar.markdown("""
<a href='https://youssefsultan.github.io'>
    <img src='https://raw.githubusercontent.com/YoussefSultan/youssefsultan.github.io/master/images/YSLOGO.png' alt='Youssef Sultan' width='50'>
</a>
""", unsafe_allow_html=True)
