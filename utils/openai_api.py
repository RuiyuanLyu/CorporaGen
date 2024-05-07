import base64
import requests
import os
import json
import openai
import random
import logging
from openai import OpenAI, AzureOpenAI
import time
# prevent accidential activation of API key
KEY_ACTIVATED = True
global num_calls_api_key
num_calls_api_key = 0
def get_api_key(key_file=None):
    if key_file is None:
        key_file = "api_keys/openai_api_key.txt"
    if not KEY_ACTIVATED:
        print("WARNING: API key not ACTIVATED. Please activate it in the code.")
        return " "
    with open(key_file, "r") as f:
        api_keys = [line.strip() for line in f.readlines()]
        api_keys = [key for key in api_keys if key.strip() and not key.startswith("#")]
    if len(api_keys) == 0:
        print(f"WARNING: No API key found. Please add one in the file: {key_file}.")
        return " "
    # api_key = random.choice(api_keys)
    global num_calls_api_key
    index = num_calls_api_key % len(api_keys)
    api_key = api_keys[index]
    num_calls_api_key += 1
    return api_key

def get_client(model=""):
    # print(f"Using model: {model}")
    if '4' in model and 'vision' in model:
        return AzureOpenAI(api_key=get_api_key("api_keys/gpt_4_vision_pjm.txt"),
                    api_version="2024-02-15-preview",
                    base_url="https://gpt-4-vision-pjm.openai.azure.com/openai/deployments/gpt-4-vision-preview")
    if '35' in model or '3.5' in model:
        return AzureOpenAI(api_key=get_api_key("api_keys/gpt_35_turbo_0125_pjm.txt"),
                    api_version="2024-02-15-preview",
                    base_url="https://gpt-35-turbo-0125-pjm.openai.azure.com/openai/deployments/gpt-35-turbo-1106")
    # if random.random() <= 1: # always use the chatweb API key. The machine cannot connect to the public API.
    #     return OpenAI(api_key=get_api_key("api_keys/chatweb_api_key.txt"), base_url="https://api.chatweb.plus/v1")
    return OpenAI(api_key=get_api_key())

def get_full_response(messages, model="gpt-3.5-turbo", max_tokens=1000, max_tries=3, report_token_usage=False, json_mode=False):
    """
        Get a text response from the OpenAI API. Returns None if max_tries is reached.
        Returns:
            response (str): The text response.
            prompt_tokens (optional, int): The number of tokens used in the prompt.
            completion_tokens (optional, int): The number of tokens used in the completion.
    """
    client = get_client(model)
    try:
        if json_mode:
            response = client.chat.completions.create(
                model=model,
                response_format={"type": "json_object"},
                messages=messages,
                max_tokens=max_tokens
            )
        else:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens
            )
        if response is None:
            raise Exception("No response returned.")
    except Exception as e:
        if isinstance(e, openai.BadRequestError):
            logging.error(f"Error: {e.message}")
            return None, 0, 0 if report_token_usage else None
        print(f"Error: {e}")
        print(f"WARNING: Triggered by messages: {messages}")
        if max_tries > 0:
            time.sleep(10)
            return get_full_response(messages, model=model, max_tokens=max_tokens, max_tries=max_tries-1, report_token_usage=report_token_usage, json_mode=json_mode)
        else:
            return None, 0, 0 if report_token_usage else None
    if report_token_usage:
        prompt_tokens = response.usage.prompt_tokens
        completion_tokens = response.usage.completion_tokens
        return response, prompt_tokens, completion_tokens
    return response

def step_token_length(prompt_length, step, max_steps, max_tokens):
    # to reduce the "max tokens" usage, we gradually increase the token length
    return max_tokens
    # future_length = max_tokens - prompt_length
    # if future_length <= 0:
    #     return 0
    # future_steps = max_steps - step
    # return prompt_length + int(future_length / future_steps)

def mimic_chat_budget(user_content_groups, system_prompt=None, max_additional_attempts=0, num_turns_expensive=1, report_token_usage=False, max_token_length=1000, json_mode=False):
    """
        budget version of mimic_chat(). The first round of conversation is done by GPT-4 model, and the remaining rounds are done by GPT-3.5-turbo model.
        NOTE: need to convert into content groups first using get_content_groups_from_source_groups()
        Args:
            model (str): The name of the model to use.
            user_content_groups (list(list)): A list of groups(list), each group of contents is sent 'simutaneously' to the model to generate a response.
            system_prompt (str): A prompt for the system to keep in mind.
            max_additional_attempts (int): The maximum number of additional attempts to make if the model responds with "Sorry, I don't understand."
            num_turns_expensive (int): The number of rounds of 'expensive' conversation to conduct.
            report_token_usage (bool): Whether to report the total number of tokens used in the entire conversation.
        Returns:
            messages (list): The mimic chat with multi-round conversation.
            token_usage (optional) (dict): A dictionary containing the total number of tokens used in the entire conversation.
    """
    messages = []
    token_usage = {"prompt_tokens": 0, "completion_tokens": 0}
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    prompt_tokens, completion_tokens = 0, 0
    for i, content_group in enumerate(user_content_groups):
        if i < num_turns_expensive:
            model = "gpt-4-vision-preview"
            # model = "gpt-4-turbo-2024-04-09"
        else:
            model = "gpt-3.5-turbo-0125"
            # remove the image urls from the previous rounds
            for message in messages:
                for content_component in reversed(message["content"]):
                    # must reverse the order, otherwise some may be remained.
                    if isinstance(content_component, str):
                        continue
                    if content_component["type"] == "image_url":
                        message["content"].remove(content_component)
        messages.append({"role": "user", "content": content_group})
        # prompt tokens and completion tokens are counted separately for each round of conversation.
        steped_length = step_token_length(prompt_tokens + completion_tokens, i, len(user_content_groups), max_token_length)
        full_response, prompt_tokens, completion_tokens = get_full_response(messages, model=model, max_tokens=steped_length, max_tries=3, report_token_usage=True, json_mode=json_mode)
        token_usage["prompt_tokens"] += prompt_tokens
        token_usage["completion_tokens"] += completion_tokens
        if full_response is None:
            print("WARNING: No response returned. The result may not be accurate.")
            if report_token_usage:
                return messages, token_usage
            return messages
        response = full_response.choices[0].message.content.strip()            
        messages.append({"role": "assistant", "content": response})
        # print(response)
        if "sorry" in response.lower():
            # print(f"Additional attempt(s) left: {max_additional_attempts}")
            if max_additional_attempts > 0:
                return mimic_chat_budget(user_content_groups, system_prompt=system_prompt, max_additional_attempts=max_additional_attempts-1)
            else:
                # print("WARNING: Maximum additional attempts reached. The result may not be accurate.")
                pass
    if report_token_usage:
        return messages, token_usage
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
    model = model if model else "gpt-3.5-turbo"
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    for content_group in user_content_groups:
        messages.append({"role": "user", "content": content_group})
        full_response, prompt_tokens, completion_tokens = get_full_response(messages, model=model, max_tokens=1000, report_token_usage=True)
        if full_response is None:
            print("WARNING: No response returned. The result may not be accurate.")
            return messages
        response = full_response.choices[0].message.content.strip()
        messages.append({"role": "assistant", "content": response})
    return messages

import cv2
import numpy as np
def create_grid(image_paths):
    assert len(image_paths) <= 6, "The number of images should be less than or equal to 6."
    images = [cv2.imread(path) for path in image_paths]
    if any(img is None for img in images):
        raise ValueError("One or more images could not be loaded.")
    if not all(img.shape == images[0].shape for img in images):
        raise ValueError("All images must have the same shape.")
    height, width, _ = images[0].shape
    num_rows, num_cols = (3,2) if height < width else (2,3)
    grid = np.zeros((height*num_rows, width*num_cols, 3), dtype=np.uint8)
    for i in range(num_rows):
        for j in range(num_cols):
            if i*num_cols+j < len(images):
                grid[i*height:(i+1)*height, j*width:(j+1)*width] = images[i*num_cols+j]
    cv2.imwrite('temp.jpg', grid)
    
def check_img_path_exists(image_path):
    if not isinstance(image_path, str):
        return False
    if not os.path.exists(image_path):
        return False
    if not (image_path.endswith(".jpg") or image_path.endswith(".png")):
        return False
    return True

def get_content_groups_from_source_groups(source_groups, high_detail=False):
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
        images = [source for source in source_group if check_img_path_exists(source)]
        merge_mode = False
        if len(images) > 10:
            # so many images, use the high detail API and merge every 4 images into one
            merge_mode = True
        for source in source_group:
            if check_img_path_exists(source):
                if not merge_mode:
                    content_group.append(_get_image_content_for_api(source, high_detail=high_detail))
            else:
                if len(source_group) == 1:
                    content_group = source
                else:
                    if merge_mode:
                        source += "The images are provided in a grid format."
                    content_group.append(_get_text_content_for_api(source))
        if merge_mode:
            images_lists = [images[i:i+6] for i in range(0, len(images), 6)]
            for image_list in images_lists:
                # text first, then images
                create_grid(image_list)
                content_group.append(_get_image_content_for_api("temp.jpg", high_detail=True))
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

import tiktoken
def num_tokens_from_string(string: str, encoding_name: str = "cl100k_base") -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

if __name__ == "__main__":
    text = "Hello, who are you?"
    messages = get_messages_from_single_content(text)
    get_full_response(messages, max_tokens=1000)
    # api_key = get_api_key()
    # image_path = "02300_annotated.jpg"
    # user_message1 = "I will provide you with a photo of an area inside a room and highlight some key objects in it with 3D boxes. Please describe the main furniture and decorations in the area, along with their placement, in approximately 150 words. When mentioning objects, use angle brackets to enclose the nouns, such as <01 piano>."
    # # description = picture_description_by_LLM([image_path], user_message1, save_json_path="testing_description.json")
    # user_message2 = "Considering these layouts, can you describe the area's intended use and functionality, as well as provide insights into its level of congestion, organization, lighting, and the presence of any storytelling elements?"
    # source_groups = [
    #     [user_message1, image_path],
    #     [user_message2]
    # ]
    # content_groups = get_content_groups_from_source_groups(source_groups)
    # system_prompt = "You are an expert interior designer, who is very sensitive at room furnitures and their placements."
    # conversation = mimic_chat(content_groups, model="gpt-4-vision-preview", system_prompt=system_prompt)
    # # conversation = mimic_chat(user_content_groups)
    # for message in conversation:
    #     # skip the image urls
    #     if message["role"] == "assistant":
    #         continue
    #     else:
    #         for content_component in message["content"]:
    #             if content_component["type"] == "image_url":
    #                 del content_component["image_url"]
    # print(conversation)

