import streamlit as st
import matplotlib.pyplot as plt
from transformers import pipeline
from textblob import TextBlob  

from langchain.prompts import PromptTemplate
from langchain.llms import CTransformers
from googletrans import Translator

## Function To get response from Hugging Face model

def getHuggingFaceResponse(input_text, model_name):
    # Load the model from Hugging Face Model Hub
    classifier = pipeline("sentiment-analysis", model=model_name,trust_remote_code=True,use_auth_token=False)
    # Perform sentiment analysis
    result = classifier(input_text)
    return result


def translate_to_english(text):
    translator = Translator()
    translated = translator.translate(text, src='ar', dest='en')
    return translated.text

def analyze_arabic_sentiment(text):
    # Translate Arabic text to English
    english_text = translate_to_english(text)
    
    # Create a TextBlob object for the translated English text
    blob = TextBlob(english_text)
    
    # Analyze sentiment polarity and subjectivity
    polarity = blob.sentiment.polarity
    subjectivity = blob.sentiment.subjectivity
    sentiment = "Positive" if polarity >= 0 else "Negative"
    return polarity, subjectivity, sentiment
    
    

st.set_page_config(page_title="Sentiment Analysis",
                   page_icon='🤖',
                   layout='centered',
                   initial_sidebar_state='collapsed')

st.header("Sentiment Analysis 🤖")

input_text = st.text_input("Enter the comment or review")
model_name = st.selectbox("Select Model", [ "Abdou/arabert-base-algerian","akhooli/xlm-r-large-arabic-sent","core42/jais-13b-chat"])
#Jais,BLOOM,LLaMA2,AraT5,AraBART
submit = st.button("Generate")

## Final response
if submit:
    # Define prompt template
    prompt_template = "What is the sentiment of \"{input_text}\" It is {sentiment}"

    # Create a prompt using the template
    prompt = PromptTemplate(input_variables=["input_text","sentiment"], template=prompt_template)

    # Analyze sentiment using TextBlob
    polarity, subjectivity, sentiment = analyze_arabic_sentiment(input_text)

    # Generate the formatted prompt
    formatted_prompt = prompt.format(input_text=input_text,sentiment=sentiment)
    st.write(formatted_prompt)

    # Get response from Hugging Face model
    hugging_face_response = getHuggingFaceResponse(formatted_prompt, model_name)
    st.write("Hugging Face Response:", hugging_face_response)

    st.write(f"Sentiment Polarity: {polarity}")
    st.write(f"Sentiment Subjectivity: {subjectivity}")
    st.write(f"Sentiment: {sentiment}")

    # Plotting the polarity and subjectivity
    fig, ax = plt.subplots()
    ax.bar(['Polarity', 'Subjectivity'], [polarity, subjectivity], color=['blue', 'green'])
    ax.set_ylabel('Score')
    ax.set_title('Sentiment Analysis')
    st.pyplot(fig)
