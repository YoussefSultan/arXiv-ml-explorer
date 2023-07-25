# ML Research Radar 📈

ML Research Radar is an example web application designed to help users understand and visualize the distribution of topics over time for ArXiv Machine Learning Papers. The main purpose of this application is to showcase the new `st.experimental_connection` feature introduced in Streamlit version 1.25.0. 

This application is built using Python, Streamlit for the web interface, NMF for topic modeling, and Plotly for interactive data visualization.

## Features

- **Data Fetching**: ML Research Radar utilizes the ArXiv API to fetch the latest machine learning papers.

- **Data Processing**: The application processes the summaries of fetched papers to determine the most discussed topics in machine learning research. This involves transforming paper summaries into tf-idf vectors, fitting an NMF model to extract topics, and mapping each paper to its topic.

- **Data Visualization**: The processed data is then visualized using Plotly, showing the distribution of topics over time. Users can choose between different visualizations based on either count or proportions.

- **Experimental Connections**: ML Research Radar takes advantage of Streamlit's new feature, `st.experimental_connection`, to connect to a language model that can summarize paper abstracts.

## Code Structure

This application is divided into several classes, each responsible for a particular task:

1. **ArxivAPI**: This class is responsible for fetching papers from the ArXiv API.

2. **ProcessingData**: This class is used to process a DataFrame of papers. It transforms paper summaries into tf-idf vectors, fits an NMF model to extract topics, and assigns each paper to its topic.

3. **Visualize**: This class creates a visualization of the distribution of topics over time.

4. **LanguageModelConnection**: This class demonstrates the usage of Streamlit's `st.experimental_connection`. It is used to connect to a language model, specifically a question-answering model.

For a more detailed understanding, please look into the code and comments directly.

## Usage

The application is hosted online and can be accessed directly through a web browser. To use ML Research Radar, visit https://mlresearchradar.streamlit.app/



