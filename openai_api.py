import base64
import requests
import os
import json

# prevent accidential activation of API key
KEY_ACTIVATED = True


def get_description(image_paths, prompt=None, high_detail=False):
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
                "url": f"data:image/jpeg;base64,{base64_image}",
                "detail": "high" if high_detail else "low"
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
    print("requesting OpenAI API...")
    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
    return response.json()

# OpenAI API Key
def get_api_key():
    with open("openai_api_key.txt", "r") as f:
        api_key = f.read().strip()
    if not KEY_ACTIVATED:
        api_key = " "
        print("WARNING: API key not ACTIVATED. Please activate it in the code.")
    return api_key


def encode_image(image_path):
    """
    Encode an image file as base64 string.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file {image_path} not found.")
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')



if __name__ == "__main__":
    api_key = get_api_key()
    image_path = "02300_annotated.jpg"
    prompt = "I will provide you with a photo of an area inside a room and highlight some key objects in it with 3D boxes. Please describe the main furniture and decorations in the area, along with their placement, in approximately 150 words. When mentioning objects, use angle brackets to enclose the nouns, such as <01 piano>."
    # description = get_description([image_path], prompt)
    # # save the description to a json file
    # with open("testing_description.json", "w") as f:
    #     json.dump(description, f, indent=4)
    with open("testing_description.json", "r") as f:
        description = json.load(f)
    # print(description)
    print(description["choices"][0]["message"]["content"])