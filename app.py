import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
import re
from sklearn.feature_extraction.text import CountVectorizer
from wordcloud import WordCloud

st.set_option('deprecation.showPyplotGlobalUse', False)

DATA_URL = ("Tweets.csv")

st.title("Sentiment Analysis of Tweets about US Airlines")
st.sidebar.title("Sentiment Analysis of Tweets")

st.sidebar.markdown("This application is a Streamlit dashboard used "
            "to analyze sentiments of tweets 🐦")

@st.cache(persist=True)
def load_data():
    data = pd.read_csv(DATA_URL)
    return data

data = load_data()


# Options for displaying the type of Analysis
analysis_type = st.sidebar.radio(
    "Select Analysis",
    options=["Number of Tweets","Display Tweets", "Tweets By Sentiment", 
             "Reasons for Negative Reviews", "Word Cloud"]) 


# Show the analysis results

# Number of tweets
if analysis_type == "Number of Tweets":
    tweet_counts = data.airline.value_counts().to_frame("Counts").rename_axis("Airlines").reset_index()
    fig = px.bar(tweet_counts, x = "Airlines",  y = "Counts", title="Number of Tweets by Airlines")
    st.plotly_chart(fig)
    
    
    
# Display Some Random Tweets   
if analysis_type == "Display Tweets":
    col1, col2 = st.beta_columns(2)
    
    with col1:
        airline = st.selectbox("Select Airlines", 
                               options = ['All',"American","Delta","Southwest","US Airways",
                                          "United","Virgin America"])
    with col2:
        sentiment = st.radio("Select Sentiment Type", options = ["Random","Negative","Positive"])
    
    # filter the data based on the ailrline selected
    if airline == "All":
        airline_df = data.copy()
    else:
        airline_df = data[data['airline'] == airline]
        
    # show 5 random tweets 
    # filter the sentiment
    if sentiment == "Random":
        tweets = airline_df['text']
    elif sentiment == "Negative":
        tweets = airline_df[airline_df['airline_sentiment']=='negative']['text']
    else:
        tweets = airline_df[airline_df['airline_sentiment']=='positive']['text'] 
        
    for tweet in np.random.choice(tweets, 5):
        st.markdown(tweet)
        


# Tweets by Sentiment Type
if analysis_type == "Tweets By Sentiment":
    
    col1, col2 = st.beta_columns(2)
    
    with col1:
        airline = st.selectbox("Select Airlines", options=['All',"American","Delta","Southwest","US Airways",
                                          "United","Virgin America"])
    with col2:
        chart_type = st.radio("Chart Type", options = ['Bar Chart', "Pie Chart"])
    
    if airline == "All":
        airline_df = data.copy()
    else:
        airline_df = data[data['airline'] == airline]
    
    tweet_by_sentiment = airline_df.airline_sentiment.value_counts().to_frame("Counts").rename_axis("Sentiment").reset_index()
    
    if chart_type == "Bar Chart":
         fig = px.bar(tweet_by_sentiment, x = "Sentiment", y = "Counts",
                 title  = "Tweets By sentiment")
    else:
        fig = px.pie(tweet_by_sentiment,values='Counts', names = "Sentiment")

    st.plotly_chart(fig)
    
    

# Reasons for Negative Reviews
if analysis_type == "Reasons for Negative Reviews":
    
    airline = st.selectbox("Select Airlines", options=['All',"American","Delta","Southwest","US Airways",
                                          "United","Virgin America"])
    
    if airline == "All":
        airline_df = data.copy()
    else:
       airline_df = data[data['airline'] == airline] 
    
    negative_reason_df = airline_df.negativereason.dropna()
    negative_reason_df = negative_reason_df.value_counts().to_frame(name = "Counts").rename_axis("Reason").reset_index()
    negative_reason_df.sort_index(axis = 0, level="Counts", ascending=False, inplace=True, )
    fig = px.bar(negative_reason_df, x = "Counts", y = "Reason", orientation='h')
    st.plotly_chart(fig)
    
    

# Wordcloud
# cleant text function
def clean_text(text):
   # remove uername and hastags
   text = re.sub(r'[@#]\w+', '', text)
   #remove urls
   text = re.sub(r'http\S+', '', text)
   # remove digits
   text = re.sub(r'[0-9]','', text)
   # remove any extra white space
   text = text.strip()
   return text

if analysis_type == "Word Cloud":
    
    col1, col2 = st.beta_columns(2)
    
    with col1:
        sentiment_type = st.radio("Select Sentiment Type", options=["Negative", "Positive"]) 
    
    with col2:
        ngram = st.radio("Select N-gram", options = [1, 2])
    
    df = data.copy()
    df['text'] = df['text'].apply(clean_text)
    
    # filtering out data for the sentiment type
    if sentiment_type =="Negative":
        df = df[df['airline_sentiment'] == 'negative']
    else:
         df = df[df['airline_sentiment'] == 'positive']
    
    
    # creating word vectors bases on the selected n-grams
    if ngram == 1:
        vectorizer = CountVectorizer(lowercase=True, stop_words='english', min_df=2, max_features=2000)
    else:
        vectorizer = CountVectorizer(lowercase=True, stop_words='english', min_df=2, max_features=2000, 
                                     ngram_range=(2,2))
    
    vecs = vectorizer.fit_transform(df['text'])
    # convert the to data frame
    count_frame = pd.DataFrame(vecs.todense(), columns = vectorizer.get_feature_names())
    # get theword counts for the words
    word_counts = count_frame.sum(axis = 0)
    
    cloud = WordCloud(background_color="white", max_words=100,
                      normalize_plurals=(True)).generate_from_frequencies(word_counts)
    
    
    # display the word cloud
    if st.button("Show World Cloud"):
        fig = plt.imshow(cloud)
        plt.axis('off')
        st.pyplot()
    
    
    
    
    