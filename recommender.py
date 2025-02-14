import os
import requests
from flask import Flask, request, jsonify

app = Flask(__name__)

# Get Hugging Face API key from environment variables
HF_API_KEY = os.getenv("HF_API_KEY")  # You need to set this environment variable

if not HF_API_KEY:
    raise ValueError("Hugging Face API key is missing. Please set the HF_API_KEY environment variable.")

# Hugging Face Inference API URL
HF_API_URL = "https://api-inference.huggingface.co/models/facebook/bart-large-mnli"

# Define the service and shop categories
service_categories = [
    "Car repair",              
    "Motorcycle repair",         
    "Car parts",                  
    "Motorcycle parts",           
    "Automotive services",       
    "Vehicle customization",     
    "Gas services",             
    "Towing services",          
    "Auto parts store",          
    "Vehicle maintenance",      
]

# Function to call the Hugging Face API for zero-shot classification
def recommend_service(query, top_n=3, threshold=0.2):
    headers = {
        "Authorization": f"Bearer {HF_API_KEY}"
    }

    payload = {
        "inputs": query,
        "parameters": {
            "candidate_labels": service_categories,
            "top_k": top_n
        }
    }

    response = requests.post(HF_API_URL, headers=headers, json=payload)

    if response.status_code == 200:
        result = response.json()
        top_services = result['labels'][:top_n]
        top_scores = result['scores'][:top_n]

        # Filter services that meet the threshold score
        relevant_services = [top_services[i] for i in range(len(top_scores)) if top_scores[i] >= threshold]

        if not relevant_services:
            return "Sorry, we couldn't find any relevant services or shops."
        else:
            return f"We recommend the following service(s)/shop(s): {', '.join(relevant_services)}"
    else:
        return f"Error: {response.status_code} - {response.text}"

@app.route("/recommend", methods=["POST"])
def recommend():
    data = request.get_json()
    
    if "query" not in data:
        return jsonify({"error": "Query parameter missing!"}), 400
    
    query = data["query"]
    recommendation = recommend_service(query)
    
    return jsonify({"recommendation": recommendation})

if __name__ == "__main__":
    app.run(debug=True)
