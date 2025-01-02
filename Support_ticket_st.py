def context(context):
    """


Business Context
In today's dynamic business landscape, organizations are increasingly recognizing 
the pivotal role customer feedback plays in shaping the trajectory of their products and services. 
The ability to swiftly and effectively respond to customer input not only fosters enhanced customer 
experiences but also serves as a catalyst for growth, prolonged customer engagement, 
and the nurturing of lifetime value relationships.
 
As a dedicated Product Manager or Product Analyst, staying attuned to the voice of your 
customers is not just a best practice; it's a strategic imperative.
 
While your organization may be inundated with a wealth of customer-generated feedback 
and support tickets, your role entails much more than just processing these inputs. 
To make your efforts in managing customer experience and expectations truly impactful, 
you need a structured approach – a method that allows you to discern the most pressing 
issues, set priorities, and allocate resources judiciously.
 
One of the most effective strategies at your disposal as an organization is to harness 
the power of automated Support Ticket Categorization - done in the modern day using Large
 Language Models and Generative AI.
 
Project Objective
Develop a Generative AI application using a Large Language Model to automate the 
classification and processing of support tickets. The application will aim to predict
 ticket categories, assign priority, suggest estimated resolution times, generate 
 responses based on sentiment analysis, and store the results in a structured DataFrame.

Specific Objective
The goal is to develop an AI-powered system to classify and process support tickets 
using a Large Language Model (LLM). The AI system will perform the following key 
tasks for each support ticket:

1. Classify the Category: Identify the general issue (e.g., technical issues, hardware issues, data recovery, etc.).
2. Assign Tags: Provide descriptive tags for more granular insights (e.g., "data loss," "internet connectivity").
3. Set Priority: Assess the urgency (e.g., high, medium, low).
4. Estimate Time for Resolution (ETA): Provide an approximate time required for resolution (e.g., 2 hrs, 4 hrs).
5. Generate First Response: Automatically craft a reply for the customer.
6. Sentiment Analysis: Analyze the sentiment (positive, negative, neutral).

All the outputs in a structured DataFrame for reporting and analysis.
Foundation Model used: Amazon Bedrock's Titan or Llama from Ollama
"""

import boto3
import json
import streamlit as st

# Define session and Bedrock endpoint
# Establishing a session with AWS using boto3 to interact with Bedrock services
session = boto3.session.Session()

bedrock_endpoint = "https://bedrock.us-east-1.amazonaws.com"  # using us east
bedrock_client = session.client(
    service_name="bedrock",
    region_name="us-east-1",   
    endpoint_url=bedrock_endpoint
)

# Initialize Bedrock runtime client to send inference requests
bedrock_inference = boto3.client(service_name="bedrock-runtime")

# Define the template for generating responses from the AI model
template = """
You are an AI assistant designed to classify and respond to support tickets.
For each ticket, perform the following tasks:
1. Classify the issue into a professional category based on the content of the ticket.
Categories can include but are not limited to: technical issues, hardware issues,
data recovery, software issues, user error, connectivity issues, or other relevant
categories based on the ticket.
2. Assign relevant tags that describe the issue. Tags can include data loss,
internet connectivity, slow performance, security concerns, software crashes,
or anything else related to the issue at hand.
3. Determine the priority based on urgency: high, medium, or low.
4. Suggest an estimated resolution time. For example, 2 hours, 4 hours, 1 day, or
any reasonable estimate based on the issue's complexity.
5. Generate a polite and empathetic first reply that acknowledges the user's concern,
offers assistance, and sets expectations.
6. Analyze sentiment of the ticket: positive, negative, or neutral,
based on the tone and content of the customer.

Support Ticket: {ticket_text}

Your response should be in the following JSON format:
{{
    "category": "<category>",
    "tags": ["<tag1>", "<tag2>"],
    "priority": "<priority>",
    "suggested_eta": "<time>",
    "generated_reply": "<reply>",
    "sentiment": "<sentiment>"
}}
"""

# Function to send the prompt to Bedrock (Titan model)
def send_prompt(prompt_data):
    """
    Sends a formatted prompt to the Bedrock Titan model and retrieves the response.
    
    Arguments:
        prompt_data (str): The ticket text provided by the user, formatted with the predefined template.
    
    Returns:
        dict: Parsed response with relevant categories and answers from Bedrock, or an error message.
    """
    
    # Prepare the body of the request to be sent to Bedrock
    body = json.dumps(
        {
            "inputText": prompt_data,  # The formatted input text with ticket details
            "textGenerationConfig": {
                "temperature": 0.0,  #  temperature set to 0.0 to ensure deterministic results
                "topP": 1.0,         #  topP set to 1.0 for balanced diversity in text generation
                "maxTokenCount": 1000  # Maximum number of tokens (words) the model will generate
            }
        }
    )

    # Define the Bedrock model and request headers
    modelId = "amazon.titan-text-premier-v1:0"  # using Titan model from Bedrock
    contentType = "application/json"
    accept = "application/json"

    try:
        # Invoke the model and retrieve the response from Bedrock
        response = bedrock_inference.invoke_model(
            body=body,
            modelId=modelId,
            accept=accept,
            contentType=contentType
        )

        # Parse the response and handle the output cleanly
        response_data = json.loads(response["body"].read().decode("utf-8"))

        # Extract only the necessary output text and structure
        if 'results' in response_data and len(response_data['results']) > 0:
            result = response_data['results'][0]
            if 'outputText' in result:
                output_text = result['outputText']

                # Parse and clean up the JSON data contained in the outputText
                try:
                    structured_output = json.loads(output_text)  # Convert outputText to JSON format
                    return structured_output  # Return the cleaned output
                except json.JSONDecodeError:
                    return {"error": "Error decoding the model's JSON response."}

        return {"error": "Invalid response format from Bedrock."}

    except Exception as e:
        return {"error": f"Error occurred: {str(e)}"}

# Streamlit app layout
def main():
    """
    Streamlit app that collects a support ticket from the user and processes it using the Bedrock model.
    Displays the structured response (category, tags, priority, etc.) or an error message.
    """
    
    # Basic app setup and title
    st.title("AI Support Ticket Assistant")
    st.write("Enter your support ticket inquiry, and the AI will analyze and respond.")

    # Input text area for user to describe their support ticket
    ticket_text = st.text_area("Enter your support ticket details here:")

    # Button to process the ticket
    if st.button("Analyze Ticket"):
        if ticket_text.strip():  # to check if the user has provided a valid ticket
            st.write("Processing your ticket...")

            # Format the prompt and send it to Bedrock for inference
            prompt = template.format(ticket_text=ticket_text)
            ai_response = send_prompt(prompt)

            if ai_response:
                # Display the AI response in a structured JSON format
                if 'error' in ai_response:
                    st.error(ai_response['error'])
                else:
                    st.subheader("AI Response:")
                    st.json(ai_response)  # Display the structured JSON response from Bedrock
            else:
                st.error("AI response could not be processed. Please try again.")
        else:
            st.warning("Please enter a valid support ticket.")

# Run the Streamlit app
if __name__ == "__main__":
    main()


# streamlit run Support_ticket_st.py

# inquiry: "My laptop refuses to boot, and I have an important meeting in 2 hours. I need urgent assistance to resolve this issue."