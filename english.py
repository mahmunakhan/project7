 #######################################  new updated code  #######################################

#date:27/02/2024


#####################################################
import streamlit as st
import matplotlib.pyplot as plt
from textblob import TextBlob  # Importing TextBlob for sentiment analysis
from transformers import pipeline

from langchain.prompts import PromptTemplate
from langchain.llms import CTransformers
import streamlit.components.v1 as components
#PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python


## Function To get response from Hugging Face model

def getHuggingFaceResponse(input_text, model_name):
    # Load the model from Hugging Face Model Hub
    classifier = pipeline("sentiment-analysis", model=model_name)
    # Perform sentiment analysis
    result = classifier(input_text)
    return result

# Function to perform sentiment analysis using TextBlob
def analyze_sentiment(input_text):
    blob = TextBlob(input_text)
    polarity = blob.sentiment.polarity
    subjectivity = blob.sentiment.subjectivity
    # Get individual sentiment scores
    sentiment = "Positive" if polarity > -0.5 else "Neutral" if -0.05 <= polarity <= 0.5 else "Negative"
    polarity_sign = "-" if polarity < 0.2 else "+"  # Sign for polarity score
            
    return polarity, subjectivity, sentiment, polarity_sign
st.set_page_config(page_title="Sentiment Analysis",
                   page_icon='🤖',
                   layout='centered',
                   initial_sidebar_state='collapsed')

st.header("Sentiment Analysis 🤖")

input_text = st.text_input("Enter the comment or review")
model_name = st.selectbox("Select Model", ["SamLowe/roberta-base-go_emotions", "lxyuan/distilbert-base-multilingual-cased-sentiments-student", "mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis","michellejieli/emotion_text_classifier"])

submit = st.button("Generate")

## Final response
if submit:
    # Define prompt template
    prompt_template = "What is the sentiment of \"{input_text}\" It is {sentiment}"

    # Create a prompt using the template
    prompt = PromptTemplate(input_variables=["input_text", "sentiment"], template=prompt_template)

    # Analyze sentiment using TextBlob
    polarity, subjectivity, sentiment, polarity_sign= analyze_sentiment(input_text)

    # Generate the formatted prompt
    formatted_prompt = prompt.format(input_text=input_text, sentiment=sentiment)
    #st.write(formatted_prompt)

    # Get response from Hugging Face model
    hugging_face_response = getHuggingFaceResponse(formatted_prompt, model_name)
    sentiment_label = hugging_face_response[0]['label']  # Extracting only the label from the response

	# Initialize an empty list to store processed sentiment scores
    processed_scores = []

# Iterate over each response in the list of dictionaries
    for response in hugging_face_response:
    # Extract the sentiment label from the dictionary
        label = response['label']
        
        # Extract the sentiment score from the dictionary
        score = response['score']
        
        # Determine the sign based on the sentiment label
        if score >= 0.5:
            formatted_score = '+' + str(score)
        else:
            formatted_score = '-' + str(abs(score))
        
        # Append the formatted score to the list
        processed_scores.append(formatted_score)

    import streamlit as st

    # Assuming you have your sentiment value, polarity score, and subjectivity stored in variables

    sentiment_color = "#47531E"  # Base color for sentiment
    positive_color = "#257906"  # Green for positive sentiment
    negative_color = "#E33F37"  # Red for negative sentiment

    st.write(
        f"<h3 style='color: {sentiment_color}; font-family: Source Sans Pro, sans-serif; font-size: 28px; margin-bottom: 10px; margin-top: 50px;'>Sentiment: {sentiment_label}</h3>",
        unsafe_allow_html=True,
    )

    polarity_color = positive_color if polarity_sign == "+" else negative_color

    st.write(
        f"<h3 style='color: {polarity_color}; font-family: Source Sans Pro, sans-serif; font-size: 24px; margin-bottom: 5px;'>Polarity Score: {polarity_sign} {abs(polarity)}</h3>",
        unsafe_allow_html=True,
    )

    st.write(
        f"<h3 style='color: #471105; font-family: Source Sans Pro, sans-serif; font-size: 24px; margin-top: 0px;'>Sentiment Subjectivity: {subjectivity}</h3>",
        unsafe_allow_html=True,
    )

# Display the processed sentiment scores
    #st.write("Processed Sentiment Scores:", processed_scores)
    #st.write("Sentiment :", sentiment_label)

    #st.write("Polarity Score:", polarity_sign + " " + str(abs(polarity)))  # Displaying the polarity score with sign
    #st.write(f"Sentiment Subjectivity: {subjectivity}")
    #st.write(f"Sentiment: {sentiment}")

    # Plotting the polarity and subjectivity
    fig, ax = plt.subplots()
    ax.bar(['Polarity', 'Subjectivity'], [polarity, subjectivity], color=['blue', 'green'])
    ax.set_ylabel('Score')
    ax.set_title('Sentiment Analysis')
    st.pyplot(fig)