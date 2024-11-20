import os
import openai

# Load the OpenAI API key from the environment variable
openai.api_key = os.getenv('OPENAI_API_KEY')

# Check if the API key was loaded correctly
if openai.api_key:
    print("OpenAI API key loaded successfully!")
    print("API Key:", openai.api_key)  # This will print the API key
else:
    print("OpenAI API key not found. Please set the environment variable 'OPENAI_API_KEY'.")
