import gradio as gr  # gradio==3.50.2
import os
import json
import numpy as np
from object_text_anno import check_annotation_validity, map_choice_to_bool, map_choice_to_text
from utils_read import load_json

DEFAULT_CORPORA_STRING = "corpora_object_XComposer2_crop"
SUPP_CORPORA_STRING = "corpora_object_cogvlm_crop"
QUESTIONS = {
    "meta": "这段文字是否在尝试描述框中的物体？",
    "visual_info_sufficient": "选择的照片提供的信息是否足以对该物体进行描述？（信息不足的例子：遮挡太多/过于模糊）",
    "category": "物体类别是否准确？",
    "appearance": "物体的外观（形状颜色）是否准确？",
    "material": "物体的材质是否准确？",
    "size": "物体的尺寸是否准确？",
    "state": "物体的状态（比如灯的开关/门的开关）是否准确？",
    "position": "物体的位置是否准确？",
    "placement": "物体的摆放（比如竖直/斜靠/平躺/堆叠）是否准确？",
    "special_function": "物体的特殊功能是否准确？请注意是否有编造的内容。",
    "other_features": "物体的其它特点（比如椅子扶手）是否准确？"
}
KEYS = ["category", "appearance", "material", "size", "state", "position", "placement", "special_function", "other_features"] # Manually sorted and removed "meta"
DATA_ROOT = "data"
SUPER_USERNAMES = ["openrobotlab", "lvruiyuan"]

VALID_MODEL_CORPORAS = ["corpora_object_cogvlm_crop", "corpora_object_gpt4v_crop", "corpora_object_gpt4v_paint_highres", "corpora_object_XComposer2_crop"]


with gr.Blocks() as demo:
    user_name = gr.Textbox(label="用户名，您的标注会保存在这里", value="", placeholder="在此输入用户名，首位必须为字母，不要带空格。")
    user_name_is_valid = gr.State(False)

    main_corpora = gr.Dropdown(label="Select a model corpus", choices=VALID_MODEL_CORPORAS, value=DEFAULT_CORPORA_STRING, visible=True)
    with gr.Row():
        directory = gr.Dropdown(label="Select a directory", choices=[], allow_custom_value=True, visible=True)
        previous_user = gr.Dropdown(label="Select which user's annotations to load", choices=[], visible=False, allow_custom_value=True)
        object_name = gr.Dropdown(label="Select an object", value="Select an object", allow_custom_value=True, interactive=True)

    with gr.Row():
        original_description = gr.Textbox(label="Original Description", interactive=False)
        backup_description = gr.Textbox(label="Backup Description", value="", visible=False, interactive=False)
    warning_text = gr.Textbox(label="Warning", value="", visible=False, interactive=False)
    with gr.Row():
        translated_description = gr.Textbox(label="Translated Description", interactive=False)
        backup_translated_description = gr.Textbox(label="Backup Translated Description", value="", visible=False, interactive=False)
    
    with gr.Row():
        core_question = gr.Radio(label="Core Question", choices=["是", "否", "不要选这个选项"], value="不要选这个选项", info=QUESTIONS["meta"], interactive=True)
        core_question2 = gr.Radio(label="Core Question 2", choices=["是", "否", "不要选这个选项"], value="不要选这个选项", info=QUESTIONS["visual_info_sufficient"], interactive=True)
    with gr.Row():
        questions_radio = []
        # Questions for the user to answer
        num_questions = len(KEYS)
        questions = [f"Q{i}" for i in range(num_questions)]
        choices = ["不要选这个选项"]
        for i in range(num_questions):
            r = gr.Radio(label=f"Question {i+1}", choices=choices, value=choices[-1], info=questions[i], interactive=True, visible=False)
            questions_radio.append(r)

    max_textboxes = 20
    with gr.Row():
        with gr.Column():
            image = gr.Image(label="Image", type="filepath")
            aux_image = gr.Image(label="Auxiliary Image", type="filepath")
        with gr.Column():
            textboxes = []
            for i in range(max_textboxes):
                t = gr.Textbox(f"Textbox {i}", visible=False)
                textboxes.append(t)
    
    save_button = gr.Button(value="保存物体标注", visible=False)

    def check_user_name_validity(user_name):
        if len(user_name) == 0 or ' ' in user_name or not user_name[0].isalpha():
            gr.Warning("用户名不合法。请首位必须为字母，并不要带空格。请重新输入。")
            return False
        return True
    user_name.blur(fn=check_user_name_validity, inputs=[user_name], outputs=[user_name_is_valid])

    def update_previous_user_choices(directory, main_corpora, user_name):
        dir_to_view = os.path.join(directory, main_corpora)
        previous_users = [name for name in os.listdir(dir_to_view) if os.path.isdir(os.path.join(dir_to_view, name))]
        previous_users = [name.strip("user_") for name in previous_users]
        if len(previous_users) == 0 or not user_name in SUPER_USERNAMES:
            return gr.Dropdown(label="Select which user's annotations to load", choices=[], value="", visible=False)
        return gr.Dropdown(label="Select which user's annotations to load", choices=previous_users, value="", visible=True)
    directory.change(fn=update_previous_user_choices, inputs=[directory, main_corpora, user_name], outputs=[previous_user])
    user_name.blur(fn=update_previous_user_choices, inputs=[directory, main_corpora, user_name], outputs=[previous_user])

    def update_valid_directories(main_corpora):
        """
        Returns a list of valid directories in the current working directory or subdirectories.
        """
        valid_directories = []
        for dir_path, dir_names, file_names in os.walk(DATA_ROOT):
            if any(name.startswith(main_corpora) for name in dir_names) and "painted_objects" in dir_names:
                valid_directories.append(dir_path)
        return gr.Dropdown(label="Select a directory", choices=valid_directories, value=valid_directories[0], allow_custom_value=True, visible=True)
    user_name.blur(fn=update_valid_directories, inputs=[main_corpora], outputs=[directory])
    main_corpora.change(fn=update_valid_directories, inputs=[main_corpora], outputs=[directory])

    def update_object_name_choices(directory, main_corpora, previous_user):
        """
        Updates the choices of the object_name input based on the selected directory.
        """
        valid_objects = []
        if previous_user:
            dir_to_load = os.path.join(directory, main_corpora, f"user_{previous_user}")
        else:
            dir_to_load = os.path.join(directory, main_corpora)
        for json_file in os.listdir(dir_to_load):
            if json_file.endswith(".json"):
                annotation = load_json(os.path.join(dir_to_load, json_file))
                is_valid, error_message = check_annotation_validity(annotation)
                if is_valid:
                    object_name = json_file.split(".")[0]
                    valid_objects.append(object_name)
        return gr.Dropdown(label="Select an object", choices=valid_objects, allow_custom_value=True)
    directory.change(fn=update_object_name_choices, inputs=[directory, main_corpora, previous_user], outputs=[object_name])
    previous_user.change(fn=update_object_name_choices, inputs=[directory, main_corpora, previous_user], outputs=[object_name])

    def get_description(json_file_path, is_aux=False):
        """
            Args:
                json_file_path: str, the path to the JSON file.
                is_aux: bool, whether the JSON file is an auxiliary file or not.
            Returns:
                original_description: str, the original description of the object.
                translated_description: str, the translated description of the object.
                warning_text: str or None, the warning message if the translated description is not available.
        """
        annotation = load_json(json_file_path)
        assert isinstance(annotation, dict), f"Invalid annotation type: {type(annotation)}"
        if "simplified_description" in annotation:
            original_description = annotation["simplified_description"]
        else:
            original_description = annotation["original_description"]
        if "modified_description" in annotation:
            translated_description = annotation["modified_description"]
            warning_text = "提示：该物体的描述此前已经被修改过。"
            if not is_aux:
                gr.Info("提示：该物体的描述此前已经被修改过。")
        else:
            translated_description = annotation.get("translated_description", '没有可用的中文描述。如果左右都没有，请在Q0中选择“否”。')
            warning_text = None
        return original_description, translated_description, warning_text

    def get_description_and_image_path(object_name, directory, main_corpora, user_name, previous_user):
        """
        Returns the description and image path of the given object.
        """
        json_file = os.path.join(directory, main_corpora, f"{object_name}.json")
        user_json_file = os.path.join(directory, main_corpora, f"user_{user_name}", f"{object_name}.json")
        if os.path.exists(user_json_file):
            json_file = user_json_file
        if previous_user:
            json_file = os.path.join(directory, main_corpora, f"user_{previous_user}", f"{object_name}.json")
        original_description, translated_description, warning_text = get_description(json_file)
        backup_json_file = os.path.join(directory, SUPP_CORPORA_STRING, f"{object_name}.json")
        if os.path.exists(backup_json_file):
            backup_description, backup_translated_description, _ = get_description(backup_json_file, is_aux=True)
            backup_description = gr.Textbox(label="Backup Description", value=backup_description, visible=True, interactive=False)
            backup_translated_description = gr.Textbox(label="Backup Translated Description", value=backup_translated_description, visible=True, interactive=False)
        else:
            backup_description = gr.Textbox(label="Backup Description", value="", visible=False, interactive=False)
            backup_translated_description = gr.Textbox(label="Backup Translated Description", value="", visible=False, interactive=False)
        warning_text = gr.Textbox(label="Warning", value=warning_text, visible=bool(warning_text), interactive=False)
        image_path = os.path.join(directory, "painted_objects", f"{object_name}.jpg")
        aux_image_path = os.path.join(directory, "cropped_objects", f"{object_name}.jpg")
        return original_description, translated_description, backup_description, backup_translated_description, image_path, aux_image_path, warning_text
    object_name.change(fn=get_description_and_image_path, inputs=[object_name, directory, main_corpora, user_name, previous_user], outputs=[original_description, translated_description, backup_description, backup_translated_description, image, aux_image, warning_text])

    def refresh_core_questions():
        core_question = "不要选这个选项"
        core_question2 = "不要选这个选项"
        return core_question, core_question2
    object_name.change(fn=refresh_core_questions, inputs=None, outputs=[core_question, core_question2])

    def refresh_questions(core_question, core_question2):
        visible = map_choice_to_bool(core_question) and map_choice_to_bool(core_question2)
        questions_radio = []
        # Questions for the user to answer
        questions = [QUESTIONS.get(k, "") for k in KEYS]
        choices = ["是", "否", "该物体没有这一属性", "该物体具有这一属性，描述遗漏了", "不要选这个选项"]
        for i in range(len(questions)):
            value = choices[-1] if np.random.rand() < 0.1 else choices[0] # to check the user is not too lazy
            r = gr.Radio(label=f"Question {i+1}", choices=choices, value=value, info=questions[i], interactive=True, visible=visible)
            questions_radio.append(r)
        return questions_radio
    object_name.change(fn=refresh_questions, inputs=[core_question, core_question2], outputs=questions_radio)
    core_question.change(fn=refresh_questions, inputs=[core_question, core_question2], outputs=questions_radio)
    core_question2.change(fn=refresh_questions, inputs=[core_question, core_question2], outputs=questions_radio)

    def refresh_save_button(core_question):
        visible = core_question in ["是", "否"]
        save_button = gr.Button(value="保存物体标注", visible=visible)
        return save_button
    core_question.change(fn=refresh_save_button, inputs=core_question, outputs=save_button)


    def parse_description(description):
        """
        Parses the given description. Returns a list of gr.Textbox components with the parsed description.
        """
        parsed = description.strip().split('。')
        parsed = [p.strip() for p in parsed]
        parsed = [p + "。" for p in parsed if p]
        num_textboxes = min(len(parsed), max_textboxes)
        out = []
        for i in range(num_textboxes):
            out.append(gr.Textbox(parsed[i], show_label=False, visible=True, interactive=True))
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

    def save_annotations(directory, main_corpora, user_name, object_name, core_question, core_question2, preview_description, *questions_radio):
        """
        Saves the annotations to the given directory.
        """
        json_file_path_raw = os.path.join(directory, main_corpora, f"{object_name}.json")
        annotation = load_json(json_file_path_raw)
        annotation_to_save = {}
        assert isinstance(annotation, dict), f"Invalid annotation type: {type(annotation)}"
        annotation_to_save = annotation.copy()
        annotation_to_save["modified_description"] = preview_description

        accuracy_dict = {} # whether the attribute (key) of the original description is accurate or not
        accuracy_dict["meta"] = map_choice_to_bool(core_question)
        accuracy_dict["visual_info_sufficient"] = map_choice_to_bool(core_question2)
        if map_choice_to_bool(core_question) and map_choice_to_bool(core_question2):
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
                accuracy_dict[key] = "False"
        annotation_to_save["accuracy_dict"] = accuracy_dict
        
        json_file_path_to_save = os.path.join(directory, main_corpora, f"user_{user_name}" , f"{object_name}.json")
        os.makedirs(os.path.dirname(json_file_path_to_save), exist_ok=True)
        with open(json_file_path_to_save, "w", encoding="utf-8") as f:
            json.dump(annotation_to_save, f, ensure_ascii=False, indent=4)
        save_button = gr.Button(value="保存成功！请选择下一物体。")
        return save_button
    save_button.click(fn=save_annotations, inputs=[directory, main_corpora, user_name, object_name, core_question, core_question2, preview_description, *questions_radio], outputs=save_button)
demo.queue(concurrency_count=20)

if __name__ == "__main__":
    demo.launch(server_port=7858)