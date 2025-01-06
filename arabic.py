
##################################### NEW UPDATE CODE FILE ###########################################
########## DATE:29/02/2024 ########################################## 





import streamlit as st
from langchain.prompts import PromptTemplate
from langchain.llms import CTransformers
from transformers import pipeline


## Function To get response from LLAma 2 model

def getLLmresponse(input_text):
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
        
        ## Generate the ressponse from the LLm 2 model
        response=classifier(prompt.format(input_text=input_text))
        print(response)
        return response




st.set_page_config(page_title="Sentiment Analysis",
                    page_icon='🤖',
                    layout='centered',
                    initial_sidebar_state='collapsed')

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
    hugging_face_response = getLLmresponse(formatted_prompt)

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
    
    