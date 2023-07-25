import streamlit as st
from src.classes import ArxivAPI, ProcessingData, Visualize, LanguageModelConnection
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import plotly.colors
import time
import pandas as pd
from datetime import datetime
import sys
import subprocess
from git import Repo
import os

# >> Page Config
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
st.set_page_config(page_title="ML Paper Topic Viz",
        page_icon="chart_with_upwards_trend",
        layout="wide")
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 



# >> Download Model Specified in Secrets
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# if on streamlit download the model
if os.name == 'posix':
    try:
    
        path_to_clone_to = '.'
        git_url = st.secrets["connections"]["bert"]["model_location"]
        cached_download(git_url,path_to_clone_to)
    except:
        print(e)
        pass
else:
    st.write('Debugging...')

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 



# >> UI
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
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
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 



# >> Instantiate streamlit caching for our processing to ensure we don't call the API each time
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
@st.cache_data(show_spinner="Fetching data from ArXiV...")
def fetch_papers(number):
    api = ArxivAPI(f"http://export.arxiv.org/api/query?search_query=cat:cs.LG&start=0&max_results={number}&sortBy=submittedDate&sortOrder=descending")
    return api.fetch_papers()

@st.cache_data(show_spinner=f"NLP time!🚀...")
def process_papers(df, num_topics, num_top_words):
    processor = ProcessingData(df,num_topics,num_top_words,num_top_words)
    return processor.process()
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 



# >> SideBar Sliders
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
num_topics = st.sidebar.slider("Number of topics", 1, 10, 10)
num_top_words = st.sidebar.slider("Number of top words", 1, 10, 6)
number = st.sidebar.slider("Number of papers", 1, 10000, 3000)
color_options = {
    1: plotly.colors.qualitative.Plotly,
    2: plotly.colors.qualitative.Dark24,
    3: plotly.colors.qualitative.Light24,
    4: plotly.colors.qualitative.D3,
    5: plotly.colors.qualitative.G10,
    6: plotly.colors.qualitative.T10,
    7: plotly.colors.qualitative.Alphabet
}
color = st.slider("Graph color", 1, 7, 1)
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 



# >> API, Processing, Visualization
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
st.title('Topics Over Time for ArXiv Machine Learning Papers')        #
                                                                      #
# Fetch the papers from the API and store them in a dataframe         #
df = fetch_papers(number)                                             #
                                                                      #
# Process the dataframe to extract and assign topics                  #  
df2 = process_papers(df, num_topics, num_top_words)                   #
                                                                      #
# Visualize the topic distribution over time                          #
viz = Visualize(df2)                                                  #
fig1 = viz.plot(number,color_options[color])                          #
fig2 = viz.plot2(number,color_options[color])                         #  
                                                                      #
# Display the figure in Streamlit                                     #
st.plotly_chart(fig1, use_container_width=True)                       #
st.plotly_chart(fig2, use_container_width=True)                       # 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 



# >> Download
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
col1, col2 = st.columns(2)
with col2:
    st.download_button('Download the dataset', 
                       df2.to_csv(index=False).encode('utf-8'),
                       f'ArXiv_ML_Papers_{str(datetime.today())}.csv',
                        'text/csv',
                        key='download-csv')
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 



# >> Pick a random paper from the bunch
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
random_paper = df2.sample(n=1)
title = random_paper['title'].values[0]
summary = random_paper['summary'].values[0]
topics = random_paper["topics"].values[0]
date = pd.to_datetime(random_paper["date"].values[0])
days_ago = (datetime.today() - date).days
Intro = f"""A research paper {title} was published {days_ago} days ago. Based on its topic {[i for i in topics.split(' ')]} it talks about """
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 



# >> Chat Bot Setup
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
if "messages" not in st.session_state:
    st.session_state.messages = []

with col1:
    ask_bert = st.button('Ask bert about a random insight regarding this data')

if ask_bert:
    st.session_state.messages = []  # Clear previous messages
    with st.chat_message("Assistant"):
        message_placeholder = st.empty()
        full_response = ""
        conn = st.experimental_connection('bert', type=LanguageModelConnection)
        response = conn.generate(context=summary)
        response = Intro + response
        for char in response:
            full_response += char
            message_placeholder.markdown(full_response + "▌")
            time.sleep(0.02)  
        message_placeholder.markdown(full_response)
    st.session_state.messages.append({"role": "Assistant", "content": full_response})
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

    

# >> Signature
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
st.sidebar.markdown("""
Designed by [Youssef Sultan](https://www.linkedin.com/in/YoussefSultan)
""")

st.sidebar.markdown("""
<a href='https://youssefsultan.github.io'>
    <img src='https://raw.githubusercontent.com/YoussefSultan/youssefsultan.github.io/master/images/YSLOGO.png' alt='Youssef Sultan' width='50'>
</a>
""", unsafe_allow_html=True)
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
