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

# Import necessary libraries
import pandas as pd  # For data manipulation and analysis
from langchain_aws import BedrockLLM  # For interacting with the Amazon Bedrock LLM
from langchain.prompts import PromptTemplate  # For defining the input prompt structure
from langchain.chains import LLMChain  # For creating the LLM workflow chain
import json  # For handling JSON responses

# Initialize the Language Model (Amazon Bedrock)
llm = BedrockLLM(
    model_id="amazon.titan-text-lite-v1",  # using titsn
    model_kwargs={"temperature": 0.1, "maxTokenCount": 1000},   
    region_name="us-east-1",  # Specify the AWS region
    endpoint_url="https://bedrock-runtime.us-east-1.amazonaws.com/"   
)

# Define the prompt template
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

# Load the support tickets from a CSV file
file_path = "Support_ticket_text_data.csv"  # Define the path to the input CSV file
support_df = pd.read_csv(file_path)  # Load the CSV file into a DataFrame

# Ensure the CSV contains the expected columns
if "support_tick_id" not in support_df.columns or "support_ticket_text" not in support_df.columns:
    #  if required columns are present; if not, raise an error
    raise ValueError("The input CSV file must contain 'support_tick_id' and 'support_ticket_text' columns.")

# Define LLM chain
prompt = PromptTemplate(input_variables=["ticket_text"], template=template)  # Defining the input prompt structure
chain = LLMChain(llm=llm, prompt=prompt)   

# Initialize an empty list to store results
results = []

# Process each ticket and collect results
for _, row in support_df.iterrows():  # Loop through each row in the DataFrame
    ticket_text = row["support_ticket_text"]  # Extract the support ticket text
    try:
        response = chain.run({"ticket_text": ticket_text})  # Run the LLM to process the ticket
        parsed_response = json.loads(response)  # Parse the JSON output from the LLM
        results.append(parsed_response)  # Append the parsed response to the results list
    except json.JSONDecodeError:
        # Handle cases where the LLM response is not valid JSON
        print(f"Invalid JSON response for ticket ID {row['support_tick_id']}: {response}")
        # Append a default error response
        results.append({
            "category": "error",
            "tags": [],
            "priority": "low",
            "suggested_eta": "N/A",
            "generated_reply": "Error processing ticket.",
            "sentiment": "neutral"
        })
    except Exception as e:
        # Handle any other errors that occur during processing
        print(f"Error processing ticket ID {row['support_tick_id']}: {e}")
        # Append a default error response
        results.append({
            "category": "error",
            "tags": [],
            "priority": "low",
            "suggested_eta": "N/A",
            "generated_reply": "Error processing ticket.",
            "sentiment": "neutral"
        })

# Combining results  
processed_data = pd.DataFrame(results)   
final_df = pd.concat([support_df, processed_data], axis=1)  # Concatenate original and processed data

# Save the output to a CSV file
output_path = "processed_support_tickets.csv"   
final_df.to_csv(output_path, index=False)  # Saving the output


# python support_ticket.py