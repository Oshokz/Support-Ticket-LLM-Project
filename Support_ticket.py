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
import boto3  #  to interact with Bedrock
import json  # For handling JSON data
import pandas as pd  # For loading and processing the dataset

# Define session and Bedrock endpoint to connect to Amazon Bedrock service
session = boto3.session.Session()

# instantiate Bedrock endpoint URL for current region  
bedrock_endpoint = "https://bedrock.us-east-1.amazonaws.com"  # using us east
bedrock_client = session.client(
    service_name="bedrock",
    region_name="us-east-1",  
    endpoint_url=bedrock_endpoint
)

# Initialize the Bedrock runtime client to send inference requests to the model
bedrock_inference = boto3.client(service_name="bedrock-runtime")

# create prompt template  
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

# Function to send the formatted prompt to Bedrock's Titan model
def send_prompt(prompt_data, verbose=False):
    """
    Sends a formatted prompt to the Bedrock Titan model and retrieves the response.
    
    Arguments:
        prompt_data (str): The ticket text, formatted using the template.
        verbose (bool): If True, prints the response details to the terminal.
    
    Returns:
        dict: The model's structured response with category, tags, etc. or an error message.
    """
    
    # Prepare the request body for the model input
    body = json.dumps(
        {
            "inputText": prompt_data,  # The formatted ticket text
            "textGenerationConfig": {
                "temperature": 0.0,  # low temperature to 0 for deterministic output
                "topP": 1.0,         # Set topP to 1.0 to maximize diversity in the output
                "maxTokenCount": 1000  # Limit the response to 1000 tokens (words)
            }
        }
    )

    modelId = "amazon.titan-text-premier-v1:0"  # using Titan model ID from Bedrock
    contentType = "application/json"  # Content type for the request
    accept = "application/json"  # Acceptable response type from the model

    try:
        # Invoke the Bedrock model and get the response
        response = bedrock_inference.invoke_model(
            body=body,
            modelId=modelId,
            accept=accept,
            contentType=contentType
        )

        # Parse the response body into a Python dictionary
        response_data = json.loads(response["body"].read().decode("utf-8"))

        # Check if there is a valid result
        if 'results' in response_data and len(response_data['results']) > 0:
            result = response_data['results'][0]
            if 'outputText' in result:
                output_text = result['outputText']

                # Parse the JSON response within the outputText
                try:
                    structured_output = json.loads(output_text)  # Convert the response to structured JSON
                    if verbose:
                        # Only print the model's response, not the input prompt
                        print(f"\nTicket ID: {prompt_data}\nResponse: {json.dumps(structured_output, indent=4)}\n")
                    return structured_output  # Return the structured output
                except json.JSONDecodeError:
                    return {"error": "Error decoding the model's JSON response."}

        return {"error": "Invalid response format from Bedrock."}

    except Exception as e:
        return {"error": f"Error occurred: {str(e)}"}

# Function to process a dataset containing support tickets
def process_tickets(input_csv, output_csv, verbose=False):
    """
    Processes the dataset of support tickets, classifies them, and generates responses.
    Adds additional columns for category, tags, priority, etc., to the dataset and saves it to a CSV file.
    
    Arguments:
        input_csv (str): Path to the input CSV file with support ticket data.
        output_csv (str): Path to save the processed dataset with new columns.
        verbose (bool): If True, prints the responses to the terminal.
    """
    
    # Load the dataset containing support ticket text
    df = pd.read_csv(input_csv)
    
    # Ensure the dataset contains the necessary columns ('support_tick_id' and 'support_ticket_text')
    if 'support_tick_id' not in df.columns or 'support_ticket_text' not in df.columns:
        print("Error: Required columns ('support_tick_id', 'support_ticket_text') are missing from the dataset.")
        return
    
    # Loop through each row in the dataset, process the ticket, and store the results
    generated_data = []
    for index, row in df.iterrows():
        support_ticket_text = row['support_ticket_text']
        
        # Format the prompt using the support ticket text
        prompt = template.format(ticket_text=support_ticket_text)
        
        # Get the response from the model for this ticket
        ai_response = send_prompt(prompt, verbose=verbose)
        
        # If there was an error in generating the response, handle it
        if 'error' in ai_response:
            generated_data.append({
                'support_tick_id': row['support_tick_id'],
                'category': 'Error',
                'tags': 'Error',
                'priority': 'Error',
                'suggested_eta': 'Error',
                'generated_reply': ai_response['error'],
                'sentiment': 'Error'
            })
        else:
            # If the response is valid, extract the relevant fields and add them to the result
            generated_data.append({
                'support_tick_id': row['support_tick_id'],
                'category': ai_response.get('category', 'Unknown'),
                'tags': ', '.join(ai_response.get('tags', [])),
                'priority': ai_response.get('priority', 'Unknown'),
                'suggested_eta': ai_response.get('suggested_eta', 'Unknown'),
                'generated_reply': ai_response.get('generated_reply', 'No reply generated'),
                'sentiment': ai_response.get('sentiment', 'Unknown')
            })
    
    # Convert the result data into a DataFrame
    result_df = pd.DataFrame(generated_data)
    
    # Save the processed data to a new CSV file
    result_df.to_csv(output_csv, index=False)
    print(f"Processed data saved to {output_csv}")

# Main function to trigger the process
def main():
    """
    Main function to initiate processing of the support tickets.
    Specifies the input and output file paths and starts the processing.
    """
    
    # Define the paths for the input CSV and the output CSV
    input_csv = 'Support_ticket_text_data.csv'  # Path to the input dataset (ensure the correct file path)
    output_csv = 'Processed_Support_Tickets.csv'  # Path where processed data will be saved
    verbose = True  # Set to True to enable verbose output in the terminal
    
    # Start processing the tickets and save the results
    process_tickets(input_csv, output_csv, verbose)

# Run the script when executed directly
if __name__ == "__main__":
    main()


# python Support_ticket.py
#JSON acts as a standard to organize the AI's output (e.g., category, tags, sentiment, etc.) in a structured way, making it easier to parse and process programmatically.
 # JSON allows seamless extraction of structured insights (e.g., category, tags, sentiment) from unstructured text 
 # limitations:he AI's prompt might not be explicitly clear enough to enforce a structured output for every kind of input
 # The AI might fail to follow the JSON format strictly if the inquiry doesn't match the structure expected from