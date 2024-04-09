import gradio as gr  # gradio==3.50.2
import os
import json
import numpy as np
from object_text_anno import check_annotation_validity, map_choice_to_bool, map_choice_to_text
from utils_read import load_json

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
SUPER_USERNAMES = ["openrobotlab", "lvruiyuan", "test"]

VALID_MODEL_CORPORAS = ["corpora_object_cogvlm_crop", "corpora_object_gpt4v_crop", "corpora_object_gpt4v_paint_highdetail", "corpora_object_XComposer2_crop", "corpora_object_InternVL-Chat-V1-2-Plus_crop"]
DEFAULT_SAVE_STRING = "corpora_object"
MAIN_CORPORA_STRING = VALID_MODEL_CORPORAS[0]


################################################################################
## Start defining the interface for the webui
with gr.Blocks() as demo:
    with gr.Row():
        user_name = gr.Textbox(label="用户名", value="", placeholder="在此输入用户名，首位必须为字母，不要带空格。")
        user_name_locked = gr.State(False)
        lock_user_name_btn = gr.Button(value="确认并锁定用户名（刷新网页才能重置用户名）", label="确认用户名")

    with gr.Row():
        directory = gr.Dropdown(label="选择一个场景名", choices=[], allow_custom_value=True, visible=True)
        previous_user = gr.Dropdown(label="Select which user's annotations to load", choices=[], visible=False, allow_custom_value=True)
        object_name = gr.Dropdown(label="选择一个物体", value="Select an object", allow_custom_value=True, interactive=True)
        check_objects_btn = gr.Button(label="Check Missing Objects", value="检查缺失标注的物体")
        supp_corpora = gr.Dropdown(label="选择备用语料来源模型", choices=VALID_MODEL_CORPORAS, value=VALID_MODEL_CORPORAS[-1], visible=True)
        record_corpora_source = gr.Checkbox(label="保存时记录标注来源模型", value=False, visible=False)

    with gr.Row():
        original_description = gr.Textbox(label="待检查/修改的描述", interactive=False, lines=5)
        supp_description = gr.Textbox(label="若主描述答非所问，备用描述", value="", visible=False, interactive=False, lines=5)
    warning_text = gr.Textbox(label="警告", value="", visible=False, interactive=False)
    with gr.Row():
        translated_description = gr.Textbox(label="翻译后的描述", interactive=False, lines=3)
        supp_translated_description = gr.Textbox(label="备用翻译描述", value="", visible=False, interactive=False, lines=3)
    
    with gr.Row():
        core_question = gr.Radio(label=QUESTIONS["meta"], choices=["是", "否", "不要选这个选项"], value="不要选这个选项", interactive=True)
        core_question2 = gr.Radio(label=QUESTIONS["visual_info_sufficient"], choices=["是", "否", "不要选这个选项"], value="不要选这个选项", interactive=True)
        supp_activated = gr.Checkbox(label="左边描述没有在尝试描述框中的物体，勾选以切换到右边的描述。（在取消前，都将使用右边的描述）", value=False, visible=True)

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
            image = gr.Image(label="物体图片", type="filepath")
            aux_image = gr.Image(label="补充图片", type="filepath")
        with gr.Column():
            textboxes = []
            for i in range(max_textboxes):
                t = gr.Textbox(f"拆分后的描述 {i}", visible=False)
                textboxes.append(t)
            save_btn = gr.Button(value="保存物体标注", visible=False)
            next_btn = gr.Button(value="下一个物体", visible=True)

################################################################################
## Start defining the logic for the webui

    def check_user_name_validity(user_name):
        if len(user_name) == 0 or ' ' in user_name or not user_name[0].isalpha():
            gr.Warning("用户名不合法。请首位必须为字母，并不要带空格。请重新输入。")
            return False
        return 
    
    def lock_user_name(user_name, user_name_locked):
        if check_user_name_validity(user_name):
            user_name = user_name.strip()
            user_name = gr.Textbox(label="用户名", value=user_name, interactive=False)
            user_name_locked = True
        return user_name, user_name_locked
    lock_user_name_btn.click(lock_user_name, inputs=[user_name, user_name_locked],
                                outputs=[user_name, user_name_locked])

    def update_record_corpora_source_visibility(user_name):
        visible = user_name in SUPER_USERNAMES
        record_corpora_source = gr.Checkbox(label="保存时记录标注来源模型", value=False, visible=visible)
        return record_corpora_source
    lock_user_name_btn.click(update_record_corpora_source_visibility, inputs=[user_name], outputs=[record_corpora_source])

    def update_previous_user_choices(directory, user_name):
        dir_to_check = os.path.join(directory, DEFAULT_SAVE_STRING)
        os.makedirs(dir_to_check, exist_ok=True)
        previous_users = [name for name in os.listdir(dir_to_check) if os.path.isdir(os.path.join(dir_to_check, name))]
        previous_users = [name.strip("user_") for name in previous_users]
        if len(previous_users) == 0 or not user_name in SUPER_USERNAMES:
            return gr.Dropdown(label="Select which user's annotations to load", choices=[], value="", visible=False)
        return gr.Dropdown(label="Select which user's annotations to load", choices=previous_users, value="", visible=True)
    directory.change(fn=update_previous_user_choices, inputs=[directory, user_name], outputs=[previous_user])

    def update_valid_directories():
        """
        Returns a list of valid directories in the current working directory or subdirectories.
        """
        main_corpora = MAIN_CORPORA_STRING
        valid_directories = []
        for dir_path, dir_names, file_names in os.walk(DATA_ROOT):
            if any(name.startswith(main_corpora) for name in dir_names) and "painted_objects" in dir_names:
                valid_directories.append(dir_path)
        valid_directories.sort()
        return gr.Dropdown(label="Select a directory", choices=valid_directories, value=valid_directories[0], allow_custom_value=True, visible=True)
    lock_user_name_btn.click(fn=update_valid_directories, inputs=[], outputs=[directory])


    def update_supp_corpora_choices(directory, object_name):
        """
        Updates the choices of the supp_corpora input based on the selected directory.
        """
        import copy
        valid_directories = copy.deepcopy(VALID_MODEL_CORPORAS)
        for dir_path in valid_directories:
            if not os.path.exists(os.path.join(directory, dir_path, object_name + ".json")):
                valid_directories.remove(dir_path)
        return gr.Dropdown(label="选择备用语料来源模型", choices=valid_directories, value=valid_directories[-1], visible=True)
    object_name.change(fn=update_supp_corpora_choices, inputs=[directory, object_name], outputs=[supp_corpora])

    def _get_object_names(directory, corpora_source, user_name=None):
        valid_objects = []
        if user_name:
            dir_to_load = os.path.join(directory, DEFAULT_SAVE_STRING, f"user_{user_name}")
        else:
            dir_to_load = os.path.join(directory, corpora_source)
        if not os.path.exists(dir_to_load):
            return valid_objects
        for json_file in os.listdir(dir_to_load):
            if json_file.endswith(".json"):
                annotation = load_json(os.path.join(dir_to_load, json_file))
                is_valid, error_message = check_annotation_validity(annotation)
                if is_valid:
                    object_name = json_file.split(".")[0]
                    valid_objects.append(object_name)
        valid_objects.sort()
        return valid_objects

    def update_object_name_choices(directory, previous_user):
        """
        Updates the choices of the object_name input based on the selected directory.
        """
        # previous_user is [] if the user is not a superuser or if there are no previous annotations.
        main_corpora = MAIN_CORPORA_STRING
        valid_objects = _get_object_names(directory, main_corpora, previous_user)
        return gr.Dropdown(label="Select an object", choices=valid_objects, value=valid_objects[0], allow_custom_value=True)
    directory.change(fn=update_object_name_choices, inputs=[directory, previous_user], outputs=[object_name])
    previous_user.change(fn=update_object_name_choices, inputs=[directory, previous_user], outputs=[object_name])

    def get_next_object_name(object_name, directory, previous_user):
        """
        Returns the next object name to be annotated.
        """
        main_corpora = MAIN_CORPORA_STRING
        valid_objects = _get_object_names(directory, main_corpora, previous_user)
        if len(valid_objects) == 0:
            return "Select an object"
        if object_name not in valid_objects:
            return valid_objects[0]
        index = valid_objects.index(object_name)
        if index == len(valid_objects) - 1:
            gr.Info("已完成最后一个物体的标注。请点击检查用的按钮")
            return valid_objects[0]
        return valid_objects[index+1]
    next_btn.click(fn=get_next_object_name, inputs=[object_name, directory, previous_user], outputs=[object_name])

    def check_objects(directory, supp_corpora, supp_activated, user_name):
        """
        Checks missing objects in the selected directory and returns a list of missing objects.
        """
        ref_objects = _get_object_names(directory, MAIN_CORPORA_STRING)
        if supp_activated:
            user_objects = _get_object_names(directory, supp_corpora, user_name)
        else:
            user_objects = _get_object_names(directory, DEFAULT_SAVE_STRING, user_name)
        user_objects = set(user_objects)
        missing_objects = [obj for obj in ref_objects if obj not in user_objects]
        gr.Info(f"Missing {len(missing_objects)} objects: {missing_objects}")
        return missing_objects
    check_objects_btn.click(fn=check_objects, inputs=[directory, supp_corpora, supp_activated, user_name], outputs=warning_text)

    def _get_description(json_file_path, is_aux=False):
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

    def get_description_and_image_path(object_name, directory, supp_corpora, supp_activated, record_corpora_source, user_name, previous_user):
        """
        Returns the description and image path of the given object.
        """
        json_file = os.path.join(directory, MAIN_CORPORA_STRING, f"{object_name}.json")
        load_string = supp_corpora if record_corpora_source and supp_activated else DEFAULT_SAVE_STRING
        user_json_file = os.path.join(directory, load_string, f"user_{user_name}", f"{object_name}.json")
        if os.path.exists(user_json_file):
            json_file = user_json_file
        if previous_user:
            json_file = os.path.join(directory, load_string, f"user_{previous_user}", f"{object_name}.json")
        original_description, translated_description, warning_text = _get_description(json_file)
        supp_json_file = os.path.join(directory, supp_corpora, f"{object_name}.json")
        if os.path.exists(supp_json_file):
            supp_description, supp_translated_description, _ = _get_description(supp_json_file, is_aux=True)
            supp_description = gr.Textbox(label="Supp Description", value=supp_description, visible=True, interactive=False)
            supp_translated_description = gr.Textbox(label="Supp Translated Description", value=supp_translated_description, visible=True, interactive=False)
        else:
            supp_description = gr.Textbox(label="Supp Description", value="", visible=False, interactive=False)
            supp_translated_description = gr.Textbox(label="Supp Translated Description", value="", visible=False, interactive=False)
        warning_text = gr.Textbox(label="Warning", value=warning_text, visible=bool(warning_text), interactive=False)
        image_path = os.path.join(directory, "repainted_objects", f"{object_name}.jpg")
        if not os.path.exists(image_path):
            image_path = os.path.join(directory, "painted_objects", f"{object_name}.jpg")
        aux_image_path = os.path.join(directory, "cropped_objects", f"{object_name}.jpg")
        return original_description, translated_description, supp_description, supp_translated_description, image_path, aux_image_path, warning_text
    object_name.change(fn=get_description_and_image_path,
                       inputs=[object_name, directory, supp_corpora, record_corpora_source, supp_activated, user_name, previous_user],
                       outputs=[original_description, translated_description, supp_description, supp_translated_description, image, aux_image, warning_text])
    supp_corpora.change(fn=get_description_and_image_path,
                       inputs=[object_name, directory, supp_corpora, record_corpora_source, supp_activated, user_name, previous_user],
                       outputs=[original_description, translated_description, supp_description, supp_translated_description, image, aux_image, warning_text])

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
    core_question.change(fn=refresh_save_button, inputs=core_question, outputs=save_btn)


    def parse_description(desc, supp_desc, supp_activated):
        """
        Parses the given description. Returns a list of gr.Textbox components with the parsed description.
        """
        description = supp_desc if supp_activated else desc
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
    translated_description.change(fn=parse_description, inputs=[translated_description, supp_translated_description, supp_activated], outputs=textboxes)
    supp_description.change(fn=parse_description, inputs=[translated_description, supp_translated_description, supp_activated], outputs=textboxes)
    supp_activated.change(fn=parse_description, inputs=[translated_description, supp_translated_description, supp_activated], outputs=textboxes)

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
    save_btn.click(fn=update_preview_description, inputs=[*textboxes], outputs=preview_description)

    def save_annotations(directory, supp_corpora, supp_activated, record_corpora_source, user_name, object_name, core_question, core_question2, preview_description, *questions_radio):
        """
        Saves the annotations to the given directory.
        """
        corpora_string = supp_corpora if supp_activated else MAIN_CORPORA_STRING
        json_file_path_raw = os.path.join(directory, corpora_string, f"{object_name}.json")
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
        save_corpora_string = DEFAULT_SAVE_STRING
        if record_corpora_source:
            save_corpora_string = supp_corpora if supp_activated else MAIN_CORPORA_STRING
        json_file_path_to_save = os.path.join(directory, save_corpora_string, f"user_{user_name}" , f"{object_name}.json")
        os.makedirs(os.path.dirname(json_file_path_to_save), exist_ok=True)
        with open(json_file_path_to_save, "w", encoding="utf-8") as f:
            json.dump(annotation_to_save, f, ensure_ascii=False, indent=4)
        save_button = gr.Button(value="保存成功！请选择下一个物体。")
        return save_button
    save_btn.click(fn=save_annotations, inputs=[directory, supp_corpora, supp_activated, record_corpora_source, user_name, object_name, core_question, core_question2, preview_description, *questions_radio], outputs=save_btn)
demo.queue(concurrency_count=20)

if __name__ == "__main__":
    demo.launch(server_port=7858)