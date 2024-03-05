import base64
import requests
import os
import json
from openai import OpenAI

# prevent accidential activation of API key
KEY_ACTIVATED = True


def mimic_chat_budget(user_content_groups, system_prompt=None, max_additional_attempts=0):
    """
        budget version of mimic_chat(). The first round of conversation is done by GPT-4 model, and the remaining rounds are done by GPT-3.5-turbo model.
        NOTE: need to convert into content groups first using get_content_groups_from_source_groups()
        Args:
            model (str): The name of the model to use.
            user_content_groups (list(list)): A list of groups(list), each group of contents is sent 'simutaneously' to the model to generate a response.
            system_prompt (str): A prompt for the system to keep in mind.
        Returns:
            messages (list): The mimic chat with multi-round conversation.
    """
    client = OpenAI(api_key=get_api_key())
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    for i, content_group in enumerate(user_content_groups):
        if i == 0:
            model = "gpt-4-vision-preview"
        else:
            model = "gpt-3.5-turbo"
            # remove the image urls from the previous rounds
            for message in messages:
                for content_component in reversed(message["content"]):
                    # must reverse the order, otherwise some may be remained.
                    if isinstance(content_component, str):
                        continue
                    if content_component["type"] == "image_url":
                        message["content"].remove(content_component)
        messages.append({"role": "user", "content": content_group})
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=2000,
        )
        response = response.choices[0].message.content.strip()            
        messages.append({"role": "assistant", "content": response})
        print(response)
        if "sorry" in response.lower():
            print(f"Additional attempt(s) left: {max_additional_attempts}")
            if max_additional_attempts > 0:
                return mimic_chat_budget(user_content_groups, system_prompt=system_prompt, max_additional_attempts=max_additional_attempts-1)
            else:
                print("WARNING: Maximum additional attempts reached. The result may not be accurate.")
    return messages
 

def mimic_chat(user_content_groups, model=None, system_prompt=None):
    """
        Engage in a conversation with the model by posing a series of predetermined questions, maintaining the conversational flow irrespective of the model's responses, and continue to ask follow-up questions.
        This implementation uses the OpenAI client API.
        NOTE: need to convert into content groups first using get_content_groups_from_source_groups()
        Args:
            model (str): The name of the model to use.
            user_content_groups (list(list)): A list of groups(list), each group of contents is sent 'simutaneously' to the model to generate a response.
            system_prompt (str): A prompt for the system to keep in mind.
        Returns:
            messages (list): The mimic chat with multi-round conversation.
    """
    client = OpenAI(api_key=get_api_key())
    model = model if model else "gpt-3.5-turbo"
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    for content_group in user_content_groups:
        messages.append({"role": "user", "content": content_group})
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=1000,
        )
        response = response.choices[0].message.content.strip()
        messages.append({"role": "assistant", "content": response})
    return messages

    
def get_content_groups_from_source_groups(source_groups):
    """
        Change the format of the input data to the format required by the OpenAI API.
        Args:
            source_groups (list): A list of source content groups. Each group contains multiple sources. A source could be a plain text message, or the path to an image file. 
        Returns:
            content_groups (list): A list of contents in the format required by the OpenAI API. Each content takes the form of {"type": "text", "text": "What is your name?"} or {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}", "detail": "high"}}
    """ 
    content_groups = []
    for source_group in source_groups:
        content_group = []
        for source in source_group:
            if os.path.exists(source):
                content_group.append(_get_image_content_for_api(source, high_detail=False))
            else:
                content_group.append(_get_text_content_for_api(source))
        content_groups.append(content_group)
    return content_groups
        

def picture_description_by_LLM(image_paths, prompt=None, high_detail=False, save_json_path=None):
    """
        Get the description of a picture using OpenAI API.
        This version uses the API directly without the client library.
        Args:
            image_paths (list): A list of image file paths.
            prompt (str): A prompt for the image description.
            high_detail (bool): Whether to use high detail or low detail image.
        Returns:
            response (dict): A JSON object containing the description.
            NOTE: response["choices"][0]["message"]["content"] contains the actual description.
    """
    content = []
    content.append(_get_text_content_for_api(prompt))
    for image_path in image_paths:
        content.append(_get_image_content_for_api(image_path))
    headers = get_headers()
    payload = get_payload(content)
    print("requesting OpenAI API...")
    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
    response = response.json()
    if save_json_path is not None:
        if not save_json_path.endswith(".json"):
            save_json_path += ".json"
        with open(save_json_path, "w") as f:
            json.dump(response, f, indent=4)
    return response


def get_headers():
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {get_api_key()}"
    }
    return headers

def get_payload(content, model="gpt-4-vision-preview"):
    """
        prepare the payload suitable for the OpenAI API.
        Args:
            content (dict): the content, 
            model (str): The name of the model to use.
        Returns:
            payload (dict): A JSON object containing the payload. 
    """
    payload = {
        "model": model,
        "messages": get_messages_from_single_content(content),
        "max_tokens": 300,
    }
    return payload

def get_messages_from_single_content(content, role="user"):
    return [{"role": role, "content": content}]
    
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


def _get_text_content_for_api(text):
    """
        prepare the text content suitable for the OpenAI API.
    """
    if isinstance(text, str):
        content = {
            "type": "text",
            "text": text
        }
    elif isinstance(text, dict):
        assert "type" in text and "text" in text, "Invalid text content: {}".format(text)
        content = text
    else:
        raise ValueError("Invalid text content: {}".format(text))
    return content


def _get_image_content_for_api(image_path, high_detail=False):
    """
        prepare the image content suitable for the OpenAI API.
    """
    base64_image = encode_image(image_path)
    content = {
        "type": "image_url",
        "image_url": {
            "url": f"data:image/jpeg;base64,{base64_image}",
            "detail": "high" if high_detail else "low"
        }
    }
    return content

if __name__ == "__main__":
    api_key = get_api_key()
    image_path = "02300_annotated.jpg"
    user_message1 = "I will provide you with a photo of an area inside a room and highlight some key objects in it with 3D boxes. Please describe the main furniture and decorations in the area, along with their placement, in approximately 150 words. When mentioning objects, use angle brackets to enclose the nouns, such as <01 piano>."
    # description = picture_description_by_LLM([image_path], user_message1, save_json_path="testing_description.json")
    user_message2 = "Considering these layouts, can you describe the area's intended use and functionality, as well as provide insights into its level of congestion, organization, lighting, and the presence of any storytelling elements?"
    source_groups = [
        [user_message1, image_path],
        [user_message2]
    ]
    content_groups = get_content_groups_from_source_groups(source_groups)
    system_prompt = "You are an expert interior designer, who is very sensitive at room furnitures and their placements."
    conversation = mimic_chat(content_groups, model="gpt-4-vision-preview", system_prompt=system_prompt)
    # conversation = mimic_chat(user_content_groups)
    for message in conversation:
        # skip the image urls
        if message["role"] == "assistant":
            continue
        else:
            for content_component in message["content"]:
                if content_component["type"] == "image_url":
                    del content_component["image_url"]
    print(conversation)
