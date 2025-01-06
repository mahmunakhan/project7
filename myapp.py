################################################## new update code file of both language combined #################################
########################################### date :29/02/2024 ####################





import streamlit as st
from langchain.prompts import PromptTemplate
from langchain.llms import CTransformers
from textblob import TextBlob
import matplotlib.pyplot as plt
import numpy as np
from nltk.tokenize import word_tokenize
import streamlit as st  
from textblob import TextBlob
import pandas as pd
import altair as alt

#st.title("What Does It Really Mean? Go Beyond Words with Sentiment Analysis")
def page1():
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
        

        st.header("Sentiment Analysis in English Language 😊😞")

        input_text = st.text_input("Enter the comment or review 👇")
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





def page2():
        import streamlit as st
        from langchain.prompts import PromptTemplate
        from langchain.llms import CTransformers
        from transformers import pipeline
        import matplotlib.pyplot as plt
        import altair as alt
        from textblob import TextBlob
        import streamlit.components.v1 as components

        ## Function To get response from LLAma 2 model

        def getLLamaresponse(input_text):
                #llm="lxyuan/distilbert-base-multilingual-cased-sentiments-student"

                # Load the model from Hugging Face Model Hub
                classifier = pipeline("sentiment-analysis", model="CAMeL-Lab/bert-base-arabic-camelbert-da-sentiment")
                # Perform sentiment analysis
                


            
                ## Prompt Template

                template="""
                    What is the sentiment of {input_text} ?
                    

                    """
                
                prompt=PromptTemplate(input_variables=["input_text"],
                                    template=template)
                
                ## Generate the ressponse from the LLama 2 model
                response=classifier(prompt.format(input_text=input_text))
                print(response)
                return response




        

        st.header("Sentiment Analysis in Arabic Language 😊😞")

        input_text=st.text_input("Enter the comment or review👇")



            
        submit=st.button("Generate")

        ## Final response

        if submit:
            #
            template="""
                    What is the sentiment of {input_text} ?
                    

                    """
                
            prompt=PromptTemplate(input_variables=["input_text"],
                                    template=template)
                
            # Generate the formatted prompt
            formatted_prompt = prompt.format(input_text=input_text)
            #st.write(formatted_prompt)

            # Get response from Hugging Face model
            hugging_face_response = getLLamaresponse(formatted_prompt)

            # Initialize an empty list to store processed sentiment scores
            processed_scores = []

            # Iterate over each response in the list of dictionaries
            for response in hugging_face_response:
                # Extract the sentiment label from the dictionary
                label = response['label']

                # Extract the sentiment score from the dictionary
                score = response['score']

                # Determine the sign based on the sentiment label
                if label == "negative":
                    formatted_score = '-' + str(score)
                elif label == "positive":
                    formatted_score = '+' + str(score)
                else:
                    formatted_score = str(score)

                # Append the formatted score to the list of dictionaries
            processed_scores.append({
                "label": label,
                "score": formatted_score
                })

            # Output the processed scores with adjusted font size and each entry on a new line
            for entry in processed_scores:
    	        st.write(f"<p style='font-size: 25px; color: black;'><b style='color: black;'>Label:</b> <span style='color: #B5631F;'>{entry['label']}</span><br><br><b style='color: black;'>Formatted Score:</b> 	<span style='color:#B5631F;'>{entry['score']}</span></p>", unsafe_allow_html=True)
    




st.sidebar.markdown('**LinguaFeel - Multilingual Sentiment Analysis**')

st.sidebar.markdown('''Description:
LinguaFeel is a cutting-edge sentiment analysis application designed to analyze text sentiment across two different languages. With its intuitive interface and advanced algorithms, LinguaFeel empowers users to gain valuable insights into the emotions and opinions expressed in various texts, regardless of language barriers..''')



    
    
    
page_names_to_funcs = {
    "English Language": page1,
    "Arabic Language": page2,
	
    


    
}



selected_page = st.sidebar.selectbox("Select a Language", page_names_to_funcs.keys(),key="3")
page_names_to_funcs[selected_page]()


