import streamlit as st
from src.classes import ArxivAPI, ProcessingData, Visualize, checkDeploymentEnv
import plotly.colors
import os
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
st.set_page_config(page_title="ML Paper Topic Viz",
        page_icon="chart_with_upwards_trend",
        layout="wide")
model_name = st.secrets['connections']['my_model']['model_name']
hf_write = st.secrets['connections']['my_model']['hf_write']
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 


@st.cache_resource 
def download_model():
    hf_read = st.secrets['connections']['my_model']['hf_read']
    result = os.system(f"export HF_TOKEN={hf_read}; git clone https://huggingface.co/meta-llama/Llama-2-7b-chat-hf")
    return result

sys_check = checkDeploymentEnv()
st.write(sys_check)

if checkDeploymentEnv():
    result = download_model()
    if result == 0:
        st.success("Model downloaded successfully!")
    else:
        st.error("Error in downloading model.")

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
number = st.sidebar.slider("Number of papers", 1, 30000, 3000)
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

st.title('Topics Over Time for ArXiv Machine Learning Papers')

# Fetch the papers from the API and store them in a dataframe
df = fetch_papers(number)

# Process the dataframe to extract and assign topics
df2 = process_papers(df, num_topics, num_top_words)

# Visualize the topic distribution over time
viz = Visualize(df2)


fig1 = viz.plot(number,color_options[color])
fig2 = viz.plot2(number,color_options[color])

# Display the figure in Streamlit
st.plotly_chart(fig1, use_container_width=True)
st.plotly_chart(fig2, use_container_width=True)


conn = st.experimental_connection('my_model', type=HuggingFaceConnection)
result = conn.generate("Translate this text to French")

st.sidebar.markdown("""
Designed by [Youssef Sultan](https://www.linkedin.com/in/YoussefSultan)
""")

st.sidebar.markdown("""
<a href='https://youssefsultan.github.io'>
    <img src='https://raw.githubusercontent.com/YoussefSultan/youssefsultan.github.io/master/images/YSLOGO.png' alt='Youssef Sultan' width='50'>
</a>
""", unsafe_allow_html=True)
