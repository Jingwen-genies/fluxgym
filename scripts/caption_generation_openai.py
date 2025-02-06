"""
Generates AI captions using openai's api
"""

import base64
import openai
import os
import json
import time

# load the api key
def load_api_key():
    if os.path.exists("OPENAI_API_KEY"):
        with open("OPENAI_API_KEY", "r") as file:
            return file.read().strip()
    if os.environ.get("OPENAI_API_KEY"):
        return os.environ.get("OPENAI_API_KEY")
    raise Exception("No API key found")


api_key = load_api_key()
# Set up the OpenAI API client
client = openai.OpenAI(api_key=api_key)


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def generate_caption(image_path: str, system_prompt: str) -> str:
    print(f"Debug - Image path: {image_path}")
    
    # Add sleep before making the request
    sleep_time = 20
    time.sleep(sleep_time)
    
    base64_image = encode_image(image_path)
    print(f"Debug - Image encoded successfully")

    
    # Debug: Print API request
    print("Debug - Sending API request to OpenAI...")
    response = client.chat.completions.create(
        model="gpt-4o",  # Note: Fixed model name from "gpt-4o-mini"
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": system_prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        },
                    },
                ],
            }
        ],
    )

    # Debug: Print raw response
    print(f"Debug - Raw API response: {response.choices[0].message.content}")

    # Extract the caption from the response
    try:
        results = response.choices[0].message.content.strip()
        print(f"Debug - Raw response: {results}")

        results = results.replace("'", '"')
        
        # Parse JSON and extract result
        parsed_json = json.loads(results)
        result = parsed_json['result']
        
        print(f"Debug - Extracted result: {result}")
        return result
        
    except Exception as e:
        print(f"Debug - Error processing response: {str(e)}")
        return results  # Return raw response if parsing fails

if __name__ == "__main__":
    import yaml
    with open('prompts/genies-style-wearable.yaml', 'r') as file:
        prompt = yaml.safe_load(file)['prompt']
    print(generate_caption(r"../genies-data/Wearable Concept Art/Layer 1.png", prompt))
