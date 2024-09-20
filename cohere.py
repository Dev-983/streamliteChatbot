import requests
import json

#test
def cohere_chat(mess):
    url = "https://api.cohere.ai/v1/chat"

    payload = json.dumps({
      "model": "command-r-plus",
      "message": mess
    })
    headers = {
      'Authorization': 'Bearer IUIh3KfQvt4zJIaIPJtPLyv1bQGAfa8XnT2srbfE',
      'Content-Type': 'application/json'
    }

    response = requests.request("POST", url, headers=headers, data=payload)
    data = json.loads(response.text)
    # Access the chatbot's main response
    chatbot_response = data["text"]
    return chatbot_response


############################################
def cohere_embed(txt):

    url = "https://api.cohere.ai/v1/embed"

    payload = json.dumps({
      "model": "embed-english-v2.0",
      "texts": txt
    })
    headers = {
      'Authorization': 'Bearer IUIh3KfQvt4zJIaIPJtPLyv1bQGAfa8XnT2srbfE',
      'Content-Type': 'application/json'
    }

    response = requests.request("POST", url, headers=headers, data=payload)

    data = json.loads(response.text)

    # Access the text and embedding
    text = data["texts"][0]
    embedding = data["embeddings"][0]
    return embedding,text


