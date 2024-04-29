import gradio as gr  # gradio==3.50.2
import os
import json
import numpy as np
from object_text_anno import check_annotation_validity, map_choice_to_bool, map_CNchoice_to_ENtext, map_ENtext_to_CNchoice
from utils.utils_read import load_json
from difflib import Differ
from object_text_anno import remove_specific_expressions


QUESTIONS = {
    "meta": "这段文字是否在尝试描述框中的物体？",
    "visual_info_sufficient": "选择的照片提供的信息是否足以对该物体进行描述？（信息不足的例子：遮挡太多/过于模糊）",
    "category": "物体类别是否准确？",
    "appearance": "物体的外观（形状颜色）是否准确？",
    "material": "物体的材质是否准确？",
    "size": "物体的尺寸是否准确？",
    "state": "物体的状态（比如门的开关/桶的空满）是否准确？",
    "position": "物体的位置（比如在桌上/A和B之间）是否准确？",
    "placement": "物体的摆放（比如竖直/斜靠/平躺/堆叠）是否准确？",
    "special_function": "物体的特殊功能是否准确？请注意是否有编造的内容。",
    "other_features": "物体的其它特点（比如椅子扶手）是否准确？"
}
KEYS = ["category", "appearance", "material", "size", "state", "position", "placement", "special_function", "other_features"] # Manually sorted and removed "meta"
DATA_ROOT = "data"
SUPER_USERNAMES = ["openrobotlab", "lvruiyuan", "test"]

VALID_MODEL_CORPORAS = ["corpora_object_cogvlm_crop", "corpora_object_gpt4v_crop", "corpora_object_gpt4v_paint_highdetail", "corpora_object_XComposer2_crop", "corpora_object_InternVL-Chat-V1-2-Plus_crop"]
DEFAULT_SAVE_STR = "corpora_object"
MAIN_CORPORA_STR = VALID_MODEL_CORPORAS[0]


################################################################################
## Start defining the interface for the webui
with gr.Blocks() as demo:
    with gr.Row():
        user_name = gr.Textbox(label="用户名", value="", placeholder="在此输入用户名，首位必须为字母，不要带空格。")
        user_name_locked = gr.State(False)
        lock_user_name_btn = gr.Button(value="确认并锁定用户名（刷新网页才能重置用户名）", label="确认用户名")

    with gr.Row():
        DIRECTORY_GR_STR = "选择一个场景名"
        directory = gr.Dropdown(label=DIRECTORY_GR_STR, choices=[], allow_custom_value=True, visible=True)
        PREVIOUS_USER_GR_STR = "选择要检查标注结果的用户名"
        previous_user = gr.Dropdown(label=PREVIOUS_USER_GR_STR, choices=[], visible=False, allow_custom_value=True)
        OBJECT_NAME_GR_STR = "选择一个物体"
        object_name = gr.Dropdown(label=OBJECT_NAME_GR_STR, value=OBJECT_NAME_GR_STR, allow_custom_value=True, interactive=True)
        CHECK_OBJECTS_BTN_GR_STR = "检查缺失标注的物体"
        check_objects_btn = gr.Button(label=CHECK_OBJECTS_BTN_GR_STR, value=CHECK_OBJECTS_BTN_GR_STR)
        SUPP_CORPORA_GR_STR = "选择备用语料来源模型"
        supp_corpora = gr.Dropdown(label=SUPP_CORPORA_GR_STR, choices=VALID_MODEL_CORPORAS, value=VALID_MODEL_CORPORAS[-1], visible=True)
        DO_RECORD_CORPORA_SOURCE_IN_DIR_NAME_GR_STR = "通过目录名记录标注来源模型"
        do_record_corpora_source_in_dir_name = gr.Checkbox(label=DO_RECORD_CORPORA_SOURCE_IN_DIR_NAME_GR_STR, value=False, visible=False)

    with gr.Row():
        ORIGIONAL_DESCRIPTION_GR_STR = "待检查/修改的描述"
        original_description = gr.Textbox(label=ORIGIONAL_DESCRIPTION_GR_STR, interactive=False, lines=5)
        SUPP_DESCRIPTION_GR_STR = "若主描述答非所问，备用描述"
        supp_description = gr.Textbox(label=SUPP_DESCRIPTION_GR_STR, value="", visible=False, interactive=False, lines=5)
    warning_text = gr.Textbox(label="警告", value="", visible=False, interactive=False)
    with gr.Row():
        TRANSLATED_DESCRIPTION_GR_STR = "翻译后的描述"
        translated_description = gr.Textbox(label=TRANSLATED_DESCRIPTION_GR_STR, interactive=False, lines=3)
        SUPP_TRANSLATED_DESCRIPTION_GR_STR = "备用翻译描述"
        supp_translated_description = gr.Textbox(label=SUPP_TRANSLATED_DESCRIPTION_GR_STR, value="", visible=False, interactive=False, lines=3)
    with gr.Row():
        MODIFIED_DESCRIPTION_GR_STR = "修改后的描述"
        modified_description = gr.Textbox(label=MODIFIED_DESCRIPTION_GR_STR, value="", visible=True, interactive=False, lines=3)
        DELTA_DESCRIPTION_GR_STR = "变化的部分"
        delta_description = gr.HighlightedText(label=DELTA_DESCRIPTION_GR_STR,combine_adjacent=True,show_legend=True,color_map={"-": "red", "+": "green"}, visible=True, interactive=True)
        CHECK_DELTA_BTN_GR_STR = "检查加载的描述和原始描述的变化（如果在下面修改区进行了修改，这里会自动更新，不需要点击）"
    check_delta_btn = gr.Button(value=CHECK_DELTA_BTN_GR_STR, visible=True)
    with gr.Row():
        core_question = gr.Radio(label=QUESTIONS["meta"], choices=["是", "否", "不要选这个选项"], value="不要选这个选项", interactive=True)
        core_question2 = gr.Radio(label=QUESTIONS["visual_info_sufficient"], choices=["是", "否", "不要选这个选项"], value="不要选这个选项", interactive=True)
        SUPP_ACTIVATED_GR_STR = "左边描述没有在尝试描述框中的物体，勾选以切换到右边的描述。（在取消前，都将使用右边的描述）"
        supp_activated = gr.Checkbox(label=SUPP_ACTIVATED_GR_STR, value=False, visible=True)

    with gr.Row():
        accuracy_dict = gr.State({})
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

    def _check_user_name_validity(user_name):
        if len(user_name) == 0 or ' ' in user_name or not user_name[0].isalpha():
            gr.Warning("用户名不合法。请首位必须为字母，并不要带空格。请重新输入。")
            return False
        return True
    
    is_super_user = gr.State(False)
    def lock_user_name(user_name, user_name_locked):
        if _check_user_name_validity(user_name):
            user_name = user_name.strip()
            is_super_user = user_name in SUPER_USERNAMES
            user_name = gr.Textbox(label="用户名", value=user_name, interactive=False)
            user_name_locked = True
            print(f"Super user: {is_super_user}")
        # check_delta_btn = gr.Button(value=CHECK_DELTA_BTN_GR_STR, visible=(user_name in SUPER_USERNAMES))
        # delta_description = gr.HighlightedText(label=DELTA_DESCRIPTION_GR_STR, combine_adjacent=True, show_legend=True,color_map={"-": "red", "+": "green"}, visible=(user_name in SUPER_USERNAMES), interactive=False)
        return user_name, user_name_locked, is_super_user, check_delta_btn, delta_description
    lock_user_name_btn.click(lock_user_name, inputs=[user_name, user_name_locked],
                                outputs=[user_name, user_name_locked, is_super_user, check_delta_btn, delta_description])

    def update_do_record_corpora_source_in_dir_name_visibility(user_name):
        do_record_corpora_source_in_dir_name = gr.Checkbox(label=DO_RECORD_CORPORA_SOURCE_IN_DIR_NAME_GR_STR, value=False, visible=user_name in SUPER_USERNAMES)
        return do_record_corpora_source_in_dir_name
    lock_user_name_btn.click(update_do_record_corpora_source_in_dir_name_visibility, inputs=[user_name], outputs=[do_record_corpora_source_in_dir_name])

    def update_previous_user_choices(directory, user_name, is_super_user):
        dir_to_check = os.path.join(directory, DEFAULT_SAVE_STR)
        if dir_to_check.startswith(DATA_ROOT):
            os.makedirs(dir_to_check, exist_ok=True)
        previous_users = [name for name in os.listdir(dir_to_check) if os.path.isdir(os.path.join(dir_to_check, name))]
        previous_users = [name.replace("user_", "") for name in previous_users]
        if len(previous_users) == 0 or not is_super_user:
            return gr.Dropdown(label=PREVIOUS_USER_GR_STR, choices=[], value="", visible=False)
        return gr.Dropdown(label=PREVIOUS_USER_GR_STR, choices=previous_users, value=previous_users[0], visible=True)
    directory.change(fn=update_previous_user_choices, inputs=[directory, user_name, is_super_user], outputs=[previous_user])

    def update_valid_directories():
        """
        Returns a list of valid directories in the current working directory or subdirectories.
        """
        main_corpora = MAIN_CORPORA_STR
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
        return gr.Dropdown(label=SUPP_CORPORA_GR_STR, choices=valid_directories, value=valid_directories[-1], visible=True)
    object_name.change(fn=update_supp_corpora_choices, inputs=[directory, object_name], outputs=[supp_corpora])

    def _get_object_names(directory, corpora_source, user_name=None):
        valid_objects = []
        if user_name:
            dir_to_load = os.path.join(directory, DEFAULT_SAVE_STR, f"user_{user_name}")
        else:
            dir_to_load = os.path.join(directory, corpora_source)
        if not os.path.exists(dir_to_load):
            gr.Info(f"目录 {dir_to_load} 不存在。没有可用的物体。")
            return valid_objects
        for json_file in os.listdir(dir_to_load):
            if json_file.endswith(".json"):
                annotation = load_json(os.path.join(dir_to_load, json_file))
                is_valid, error_message = check_annotation_validity(annotation)
                if is_valid:
                    object_name = json_file.split(".")[0]
                    valid_objects.append(object_name)
            if len(valid_objects) == 0:
                gr.Info(f"目录 {dir_to_load} 中没有可用的物体。")
        valid_objects.sort()
        return valid_objects

    def update_object_name_choices(directory, previous_user):
        """
        Updates the choices of the object_name input based on the selected directory.
        """
        # previous_user is [] if the user is not a superuser or if there are no previous annotations.
        if previous_user:
            valid_objects = _get_object_names(directory, DEFAULT_SAVE_STR, previous_user)
        else:
            valid_objects = _get_object_names(directory, MAIN_CORPORA_STR)
        if len(valid_objects) == 0:
            value = ''
            gr.Warning("没有可用的物体。请选择其他目录。")
        else:
            value = valid_objects[0]
        return gr.Dropdown(label=OBJECT_NAME_GR_STR, choices=valid_objects, value=value, allow_custom_value=True)
    directory.change(fn=update_object_name_choices, inputs=[directory, previous_user], outputs=[object_name])
    previous_user.change(fn=update_object_name_choices, inputs=[directory, previous_user], outputs=[object_name])

    def get_next_object_name(object_name, directory, previous_user):
        """
        Returns the next object name to be annotated.
        """
        main_corpora = MAIN_CORPORA_STR
        valid_objects = _get_object_names(directory, main_corpora, previous_user)
        if len(valid_objects) == 0:
            return OBJECT_NAME_GR_STR
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
        ref_objects = _get_object_names(directory, MAIN_CORPORA_STR)
        if supp_activated:
            user_objects = _get_object_names(directory, supp_corpora, user_name)
        else:
            user_objects = _get_object_names(directory, DEFAULT_SAVE_STR, user_name)
        user_objects = set(user_objects)
        missing_objects = [obj for obj in ref_objects if obj not in user_objects]
        gr.Info(f"Missing {len(missing_objects)}/{len(ref_objects)} objects: {missing_objects}")
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
                modified_description: str or None, the modified description of the object if it exists.
                warning_text: str or None, the warning message if the translated description is not available.
                accuracy_dict: dict, the accuracy dictionary of the object.
        """
        annotation = load_json(json_file_path)
        assert isinstance(annotation, dict), f"Invalid annotation type: {type(annotation)}"
        if "simplified_description" in annotation:
            original_description = annotation["simplified_description"]
        else:
            original_description = annotation["original_description"]
        translated_description = annotation.get("translated_description", '没有可用的中文描述。如果左右都没有，请在Q0中选择“否”。')
        modified_description = annotation.get("modified_description", None)
        if modified_description:
            warning_text = "提示：该物体的描述此前已经被修改过。"
            if not is_aux:
                gr.Info("提示：该物体的描述此前已经被修改过。")
        else:
            warning_text = None
        accuracy_dict = annotation.get("accuracy_dict", {})
        original_description = remove_specific_expressions(original_description)
        translated_description = remove_specific_expressions(translated_description)
        if modified_description:
            modified_description = remove_specific_expressions(modified_description)
        return original_description, translated_description, modified_description, warning_text, accuracy_dict

    def get_description_answers_image_path(object_name, directory, supp_corpora, supp_activated, do_record_corpora_source_in_dir_name, user_name, previous_user):
        """
        Returns the description, answers, and image path of the given object.
        """
        json_file = os.path.join(directory, MAIN_CORPORA_STR, f"{object_name}.json")
        load_str = supp_corpora if do_record_corpora_source_in_dir_name and supp_activated else DEFAULT_SAVE_STR
        user_json_file = os.path.join(directory, load_str, f"user_{user_name}", f"{object_name}.json")
        if os.path.exists(user_json_file):
            json_file = user_json_file
        if previous_user:
            json_file = os.path.join(directory, load_str, f"user_{previous_user}", f"{object_name}.json")
        original_description, translated_description, modified_description, warning_text, accuracy_dict = _get_description(json_file)
        if modified_description:
            modified_description = gr.Textbox(label=MODIFIED_DESCRIPTION_GR_STR, value=modified_description, visible=True, interactive=False)
        else:
            modified_description = gr.Textbox(label=MODIFIED_DESCRIPTION_GR_STR, value="", visible=False, interactive=False)
        supp_json_file = os.path.join(directory, supp_corpora, f"{object_name}.json")
        if os.path.exists(supp_json_file):
            supp_description, supp_translated_description, _, _, _ = _get_description(supp_json_file, is_aux=True)
            supp_description = gr.Textbox(label=SUPP_DESCRIPTION_GR_STR, value=supp_description, visible=True, interactive=False)
            supp_translated_description = gr.Textbox(label=SUPP_TRANSLATED_DESCRIPTION_GR_STR, value=supp_translated_description, visible=True, interactive=False)
        else:
            supp_description = gr.Textbox(label=SUPP_DESCRIPTION_GR_STR, value="", visible=False, interactive=False)
            supp_translated_description = gr.Textbox(label=SUPP_TRANSLATED_DESCRIPTION_GR_STR, value="", visible=False, interactive=False)
        warning_text = gr.Textbox(label="Warning", value=warning_text, visible=bool(warning_text), interactive=False)
        image_path = os.path.join(directory, "repainted_objects", f"{object_name}.jpg")
        if not os.path.exists(image_path):
            image_path = os.path.join(directory, "painted_objects", f"{object_name}.jpg")
        aux_image_path = os.path.join(directory, "cropped_objects", f"{object_name}.jpg")
        return original_description, translated_description, modified_description, supp_description, supp_translated_description, image_path, aux_image_path, warning_text, accuracy_dict
    object_name.change(fn=get_description_answers_image_path,
                       inputs=[object_name, directory, supp_corpora, do_record_corpora_source_in_dir_name, supp_activated, user_name, previous_user],
                       outputs=[original_description, translated_description, modified_description, supp_description, supp_translated_description, image, aux_image, warning_text, accuracy_dict])
    supp_corpora.change(fn=get_description_answers_image_path,
                       inputs=[object_name, directory, supp_corpora, do_record_corpora_source_in_dir_name, supp_activated, user_name, previous_user],
                       outputs=[original_description, translated_description, modified_description, supp_description, supp_translated_description, image, aux_image, warning_text, accuracy_dict])
    
    preview_description = gr.Textbox(label="Preview Annotation", value="", visible=False, interactive=False)

    def diff_texts(text1, text2):
        d = Differ()
        return [
            (token[2:], token[0] if token[0] != " " else None)
            for token in d.compare(text1, text2)
        ]
    modified_description.change(fn=diff_texts, inputs=[translated_description, modified_description], outputs=delta_description)
    check_delta_btn.click(fn=diff_texts, inputs=[translated_description, modified_description], outputs=delta_description)
    preview_description.change(fn=diff_texts, inputs=[translated_description, preview_description], outputs=delta_description)

    def update_core_questions(accuracy_dict):
        core_question = map_ENtext_to_CNchoice(accuracy_dict.get("meta", None))
        core_question2 = map_ENtext_to_CNchoice(accuracy_dict.get("visual_info_sufficient", None))
        return core_question, core_question2
    object_name.change(fn=update_core_questions, inputs=accuracy_dict, outputs=[core_question, core_question2])

    def update_questions(core_question, core_question2, accuracy_dict):
        visible = map_choice_to_bool(core_question) and map_choice_to_bool(core_question2)
        questions_radio = []
        # Questions for the user to answer
        choices = ["是", "否", "该物体没有这一属性", "该物体具有这一属性，描述遗漏了", "不要选这个选项"]
        for i in range(len(questions)):
            question = QUESTIONS.get(KEYS[i], "")
            if KEYS[i] in accuracy_dict:
                value = map_ENtext_to_CNchoice(accuracy_dict[KEYS[i]])
            else:
                value = choices[-1] if np.random.rand() < 0.1 else choices[0] # to check the user is not too lazy
            r = gr.Radio(label=f"Question {i+1}", choices=choices, value=value, info=question, interactive=True, visible=visible)
            questions_radio.append(r)
        return questions_radio
    object_name.change(fn=update_questions, inputs=[core_question, core_question2, accuracy_dict], outputs=questions_radio)
    core_question.change(fn=update_questions, inputs=[core_question, core_question2, accuracy_dict], outputs=questions_radio)
    core_question2.change(fn=update_questions, inputs=[core_question, core_question2, accuracy_dict], outputs=questions_radio)

    def update_save_button(core_question):
        visible = core_question in ["是", "否"]
        save_button = gr.Button(value="保存物体标注", visible=visible)
        return save_button
    core_question.change(fn=update_save_button, inputs=core_question, outputs=save_btn)
    object_name.change(fn=update_save_button, inputs=core_question, outputs=save_btn)

    def parse_description(desc, modified_des, supp_desc, supp_activated):
        """
        Parses the given description. Returns a list of gr.Textbox components with the parsed description.
        """
        description = supp_desc if supp_activated else desc
        if modified_des:
            description = modified_des
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
    translated_description.change(fn=parse_description, inputs=[translated_description, modified_description, supp_translated_description, supp_activated], outputs=textboxes)
    supp_description.change(fn=parse_description, inputs=[translated_description, modified_description, supp_translated_description, supp_activated], outputs=textboxes)
    supp_activated.change(fn=parse_description, inputs=[translated_description, modified_description, supp_translated_description, supp_activated], outputs=textboxes)

    def update_preview_description(*textboxes):
        """
        Updates the preview annotation based on the user's answers.
        """
        value = ""
        for t in textboxes:
            value += t.strip(" ") + "。"
        value = value.replace("。。", "。")
        value = value.replace("，。", "。")
        value = value.replace("，，", "。")
        value = value.strip("。") + "。"
        return gr.Textbox(label="Preview Annotation", value=value, visible=False, interactive=False)
    for t in textboxes:
        t.change(fn=update_preview_description, inputs=[*textboxes], outputs=preview_description)
    save_btn.click(fn=update_preview_description, inputs=[*textboxes], outputs=preview_description)

    def save_annotations(directory, supp_corpora, supp_activated, do_record_corpora_source_in_dir_name, user_name,  previous_user, object_name, core_question, core_question2, preview_description, *questions_radio):
        """
        Saves the annotations to the given directory.
        """
        corpora_str = supp_corpora if supp_activated else MAIN_CORPORA_STR
        json_file_path_raw = os.path.join(directory, corpora_str, f"{object_name}.json")
        annotation = load_json(json_file_path_raw)
        annotation_to_save = {}
        assert isinstance(annotation, dict), f"Invalid annotation type: {type(annotation)}"
        annotation_to_save = annotation.copy()
        annotation_to_save["modified_description"] = preview_description
        annotation_to_save["corpora_source"] = supp_corpora if supp_activated else MAIN_CORPORA_STR
        accuracy_dict = {} # whether the attribute (key) of the original description is accurate or not
        accuracy_dict["meta"] = map_choice_to_bool(core_question)
        accuracy_dict["visual_info_sufficient"] = map_choice_to_bool(core_question2)
        if map_choice_to_bool(core_question) and map_choice_to_bool(core_question2):
            for i in range(len(questions_radio)):
                key = KEYS[i]
                value = map_CNchoice_to_ENtext(questions_radio[i])
                if value is None:
                    save_button = gr.Button(value="有问题没回答，保存失败！")
                    return save_button
                accuracy_dict[key] = value
        else:
            for i in range(len(questions_radio)):
                key = KEYS[i]
                accuracy_dict[key] = "False"
        annotation_to_save["accuracy_dict"] = accuracy_dict
        save_corpora_str = DEFAULT_SAVE_STR
        if do_record_corpora_source_in_dir_name:
            # the source of the annotation is always recorded now.
            # This is used to compatible with the previous version of the annotation tool, where the source of the annotation is recorded by the directory name.
            save_corpora_str = supp_corpora if supp_activated else MAIN_CORPORA_STR
        is_super_user = user_name in SUPER_USERNAMES
        if previous_user and is_super_user:
            json_file_path_to_save = os.path.join(directory, save_corpora_str, f"user_{user_name}", f"user_{previous_user}", f"{object_name}.json")
        else:
            json_file_path_to_save = os.path.join(directory, save_corpora_str, f"user_{user_name}" , f"{object_name}.json")
        if json_file_path_to_save.startswith(DATA_ROOT):
            os.makedirs(os.path.dirname(json_file_path_to_save), exist_ok=True)
        with open(json_file_path_to_save, "w", encoding="utf-8") as f:
            json.dump(annotation_to_save, f, ensure_ascii=False, indent=4)
        gr.Info(f"物体标注已保存至 {json_file_path_to_save}")
        save_button = gr.Button(value="保存成功！请选择下一个物体。")
        return save_button
    save_btn.click(fn=save_annotations, inputs=[directory, supp_corpora, supp_activated, do_record_corpora_source_in_dir_name, user_name, previous_user, object_name, core_question, core_question2, preview_description, *questions_radio], outputs=save_btn)
demo.queue(concurrency_count=20)

if __name__ == "__main__":
    demo.launch(server_port=7858)