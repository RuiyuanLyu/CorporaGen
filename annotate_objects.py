from openai_api import mimic_chat, mimic_chat_budget, get_content_groups_from_source_groups



def annotate_object_image(image_path):
    """
        Uses GPT-4 to annotate an object in an image.
        Returns:
            The discription of the object in the image.
    """
    user_message1 = "Please describe the objects in the box, mainly including the following aspects: appearance (shape, color), material, size (e.g., larger or smaller compared to similar items), condition (e.g., whether a door is open or closed), placement (e.g.,vertical/leaning/slanting/stacked), functionality (compared to similar items), and design features (e.g., whether the chair has armrests/backrest)" 
    user_message2 = "Please omit the plain and ordinary parts of the description, only retaining the unique characteristics of the objects; rewrite and recombine the retained descriptions to make the language flow naturally, without being too rigid."
    system_prompt = "You are an expert interior designer, who is very sensitive at room furnitures and their placements."
    source_groups = [
        [user_message1, image_path],
        [user_message2]
    ]
    content_groups = get_content_groups_from_source_groups(source_groups)
    # conversation = mimic_chat(content_groups, model="gpt-4-vision-preview", system_prompt=system_prompt)
    conversation = mimic_chat_budget(content_groups, system_prompt=system_prompt)
    annotation = []
    for message in conversation:
        if message["role"] == "assistant":
            annotation.append(message["content"])
    return annotation


if __name__ == "__main__":
    image_path = "./example_data/anno_lang/painted_images/068_chair_00232.jpg"
    annotation = annotate_object_image(image_path)
    for i, line in enumerate(annotation):
        print(f"Line {i+1}:")
        print(line)