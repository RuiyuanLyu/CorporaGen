import gradio as gr  # gradio==3.50.2
import os
import json
from object_text_anno import check_annotation
from utils_read import load_json

QUESTIONS = {
    "meta": "这段文字是否在尝试描述框中的物体？",
    "category": "物体类别是否准确？",
    "appearance": "物体的外观（形状颜色）是否准确？",
    "material": "物体的材质是否准确？",
    "size": "物体的尺寸是否准确？",
    "state": "物体的状态（比如灯的开关/门的开关）是否准确？",
    "position": "物体的位置是否准确？",
    "placement": "物体的摆放（比如竖直/斜靠/平躺/堆叠）是否准确？",
    "special_function": "物体的特殊功能是否准确？请注意是否有编造的谎言。",
    "other_features": "物体的特点（比如椅子扶手）是否准确？"
}
KEYS = ["category", "appearance", "material", "size", "state", "position", "placement", "special_function", "other_features"] # Manually sorted and removed "meta"


def get_valid_directories():
    """
    Returns a list of valid directories in the current working directory or subdirectories.
    """
    valid_directories = []
    for dir_path, dir_names, file_names in os.walk("."):
        if "corpora_object" in dir_names and "painted_images" in dir_names:
            valid_directories.append(dir_path)
    return valid_directories

with gr.Blocks() as demo:
    valid_directories = get_valid_directories()
    with gr.Row():
        directory = gr.Dropdown(label="Select a directory", choices=valid_directories)
        object_name = gr.Dropdown(label="Select an object", value="Select an object", allow_custom_value=True, interactive=True)

    original_description = gr.Textbox(label="Original Description", interactive=False)
    warning_text = gr.Textbox(label="Warning", value="", visible=False, interactive=False)
    translated_description = gr.Textbox(label="Translated Description", interactive=False)

    core_question = gr.Radio(label="Question 0", choices=["是", "否", "不要选这个选项"], value="不要选这个选项", info=QUESTIONS["meta"], interactive=True)
    with gr.Row():
        questions_radio = []
        # Questions for the user to answer
        num_questions = 9
        questions = [f"Q{i}" for i in range(num_questions)]
        choices = ["不要选这个选项"]
        for i in range(num_questions):
            r = gr.Radio(label=f"Question {i+1}", choices=choices, value=choices[-1], info=questions[i], interactive=True, visible=False)
            questions_radio.append(r)

    max_textboxes = 20
    with gr.Row():
        with gr.Column():
            image = gr.Image(label="Image", type="filepath")
        with gr.Column():
            textboxes = []
            for i in range(max_textboxes):
                t = gr.Textbox(f"Textbox {i}", visible=False)
                textboxes.append(t)
    
    save_button = gr.Button(value="保存物体标注", visible=False)

    def update_object_name_choices(directory):
        """
        Updates the choices of the object_name input based on the selected directory.
        """
        valid_objects = []
        for json_file in os.listdir(os.path.join(directory, "corpora_object")):
            if json_file.endswith(".json"):
                annotation = load_json(os.path.join(directory, "corpora_object", json_file))
                is_valid, error_message = check_annotation(annotation)
                if is_valid:
                    valid_objects.append(json_file.split(".")[0])
        return gr.Dropdown(label="Select an object", choices=valid_objects, allow_custom_value=True)
    directory.change(fn=update_object_name_choices, inputs=directory, outputs=[object_name])

    def get_description_and_image_path(object_name, directory):
        """
        Returns the description and image path of the given object.
        """
        json_file = os.path.join(directory, "corpora_object", f"{object_name}.json")
        annotation = load_json(json_file)
        warning_text = gr.Textbox(label="Warning", value="", visible=False, interactive=False)
        assert isinstance(annotation, dict), f"Invalid annotation type: {type(annotation)}"
        original_description = annotation["simplified_description"]
        if "modified_description" in annotation:
            translated_description = annotation["modified_description"]
            warning_text = gr.Textbox(label="Warning", value="该物体的描述此前已经被修改过。", visible=True, interactive=False)
        else:
            translated_description = annotation.get("translated_description", '没有可用的中文描述.请在Q0选择"否"。')
        image_path = os.path.join(directory, "painted_images", f"{object_name}.jpg")
        return original_description, translated_description, image_path, warning_text
    object_name.change(fn=get_description_and_image_path, inputs=[object_name, directory], outputs=[original_description, translated_description, image, warning_text])

    def refresh_core_question():
        core_question = gr.Radio(label="Question 0", choices=["是", "否", "不要选这个选项"], value="不要选这个选项", info=QUESTIONS["meta"], interactive=True)
        return core_question
    object_name.change(fn=refresh_core_question, inputs=None, outputs=core_question)

    def refresh_questions(core_question):
        visible = map_choice_to_bool(core_question)
        questions_radio = []
        # Questions for the user to answer
        questions = [QUESTIONS.get(k, "") for k in KEYS]
        choices = ["是", "否", "该物体没有这一属性", "该物体具有这一属性，描述遗漏了", "不要选这个选项"]
        for i in range(len(questions)):
            r = gr.Radio(label=f"Question {i+1}", choices=choices, value=choices[-1], info=questions[i], interactive=True, visible=visible)
            questions_radio.append(r)
        return questions_radio
    object_name.change(fn=refresh_questions, inputs=core_question, outputs=questions_radio)
    core_question.change(fn=refresh_questions, inputs=core_question, outputs=questions_radio)

    def refresh_save_button(core_question):
        visible = core_question in ["是", "否"]
        save_button = gr.Button(value="保存物体标注", visible=visible)
        return save_button
    core_question.change(fn=refresh_save_button, inputs=core_question, outputs=save_button)

    def map_choice_to_bool(choice):
        if choice == "是":
            return True
        elif choice == "否":
            return False
        elif choice == "该物体没有这一属性":
            return True
        elif choice == "该物体具有这一属性，描述遗漏了":
            return False
        elif choice == "不要选这个选项":
            return None # special value if the user is lazy.
        else:
            raise ValueError(f"Invalid choice: {choice}")
        
    def map_choice_to_text(choice):
        if choice == "是":
            return "True"
        elif choice == "否":
            return "False"
        elif choice == "该物体没有这一属性":
            return "Inrelevant"
        elif choice == "该物体具有这一属性，描述遗漏了":
            return "Missing"
        elif choice == "不要选这个选项":
            return None # special value if the user is lazy.
        else:
            raise ValueError(f"Invalid choice: {choice}")

    def parse_description(description):
        """
        Parses the given description. Returns a list of gr.Textbox components with the parsed description.
        """
        parsed = description.strip().split('。')
        parsed = [p.strip(" ") for p in parsed]
        parsed = [p + "。" for p in parsed if p]
        num_textboxes = min(len(parsed), max_textboxes)
        out = []
        for i in range(num_textboxes):
            out.append(gr.Textbox(parsed[i], visible=True, interactive=True))
        for i in range(num_textboxes, max_textboxes):
            out.append(gr.Textbox("", visible=False))
        return out
    translated_description.change(fn=parse_description, inputs=translated_description, outputs=textboxes)

    preview_description = gr.Textbox(label="Preview Annotation", value="", visible=False, interactive=False)
    def update_preview_description(*textboxes):
        """
        Updates the preview annotation based on the user's answers.
        """
        value = ""
        for t in textboxes:
            value += t.strip(" ") + "。"
        value = value.replace("。。", "。")
        value = value.strip("。") + "。"
        return gr.Textbox(label="Preview Annotation", value=value, visible=False, interactive=False)
    for t in textboxes:
        t.change(fn=update_preview_description, inputs=[*textboxes], outputs=preview_description)
    save_button.click(fn=update_preview_description, inputs=[*textboxes], outputs=preview_description)

    def save_annotations(directory, object_name, core_question, preview_description, *questions_radio):
        """
        Saves the annotations to the given directory.
        """
        json_file_path = os.path.join(directory, "corpora_object", f"{object_name}.json")
        annotation = load_json(json_file_path)
        annotation_to_save = {}
        assert isinstance(annotation, dict), f"Invalid annotation type: {type(annotation)}"
        annotation_to_save = annotation.copy()
        annotation_to_save["modified_description"] = preview_description

        accuracy_dict = {} # whether the attribute (key) of the original description is accurate or not
        accuracy_dict["meta"] = map_choice_to_bool(core_question)
        if map_choice_to_bool(core_question):
            for i in range(len(questions_radio)):
                key = KEYS[i]
                value = map_choice_to_text(questions_radio[i])
                if value is None:
                    save_button = gr.Button(value="有问题没回答，保存失败！")
                    return save_button
                accuracy_dict[key] = value
        else:
            for i in range(len(questions_radio)):
                key = KEYS[i]
                accuracy_dict[key] = False
        annotation_to_save["accuracy_dict"] = accuracy_dict
        
        with open(json_file_path, "w", encoding="utf-8") as f:
            json.dump(annotation_to_save, f, ensure_ascii=False, indent=4)
        save_button = gr.Button(value="保存成功！请选择下一物体。")
        return save_button
    save_button.click(fn=save_annotations, inputs=[directory, object_name, core_question, preview_description, *questions_radio], outputs=save_button)


if __name__ == "__main__":
    demo.launch()