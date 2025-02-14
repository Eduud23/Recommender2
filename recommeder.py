from flask import Flask, request, jsonify
from transformers import pipeline

app = Flask(__name__)

# Initialize the Hugging Face pre-trained pipeline for zero-shot classification with a smaller model
classifier = pipeline("zero-shot-classification", model="distilbert-base-uncased")

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

# Function to recommend services based on the query
def recommend_service(query, top_n=3, threshold=0.2):  # lowered the threshold
    result = classifier(query, candidate_labels=service_categories)
    
    # Get the top N recommendations based on score
    top_services = result['labels'][:top_n]
    top_scores = result['scores'][:top_n]
    
    # Filter services that meet the threshold score
    relevant_services = [top_services[i] for i in range(len(top_scores)) if top_scores[i] >= threshold]
    
    # If no services meet the threshold, show a message
    if not relevant_services:
        return "Sorry, we couldn't find any relevant services or shops."
    else:
        return f"We recommend the following service(s)/shop(s): {', '.join(relevant_services)}"

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
