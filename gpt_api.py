import base64
import requests

# prevent accidential activation of API key
ACTIVATED = False

# OpenAI API Key
# WARNING: this is a secret key borrowed from ANOTHER team without permission. DO NOT SHARE.
def get_api_key():
  with open("openai_api_key.txt", "r") as f:
      api_key = f.read().strip()
  if not ACTIVATED:
    api_key = " "
    print("WARNING: API key not activated. Please activate it in the code.")
  return api_key

# Function to encode the image
def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')

def get_description(image_paths, prompt=None):
  api_key = get_api_key()
  base64_images = [encode_image(image_path) for image_path in image_paths]
  content = [{
    "type": "text",
    "text": prompt if prompt else "What’s in this image?"
  }]
  for base64_image in base64_images:
    content.append({
      "type": "image_url",
      "image_url": {
        "url": f"data:image/jpeg;base64,{base64_image}"
      }
    })

  headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {api_key}"
  }

  payload = {
    "model": "gpt-4-vision-preview",
    "messages": [
      {
        "role": "user",
        "content": content
      }
    ],
    "max_tokens": 300,
  }
  
  response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
  return response.json()["choices"][0]["text"]


if __name__ == "__main__":
  api_key = get_api_key()
  image_path = "rendered.png"
  base64_image = encode_image(image_path)

  headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {api_key}"
  }

  payload = {
    "model": "gpt-4-vision-preview",
    "messages": [
      {
        "role": "user",
        "content": [
          {
            "type": "text",
            "text": "What’s in this image?"
          },
          {
            "type": "image_url",
            "image_url": {
              "url": f"data:image/jpeg;base64,{base64_image}"
            }
          }
        ]
      }
    ],
    "max_tokens": 300
  }

  response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)

  print(response.json())