import streamlit as st

from clarifai.client.app import App


@st.cache_resource
def get_visual_classifiers(_auth):
    classifiers = {"model_type_id": "visual-classifier"}
    classifier_models = App(pat=_auth._pat).list_models(
        filter_by=classifiers, only_in_app=False
    )
    return list(classifier_models)

    # new_list = [type('ModelInfo', (), {'id': model.id, 'user_id': model.user_id, 'app_id': model.app_id, 'name': model.name}) for model in classifier_models]
    # return new_list


@st.cache_resource
def get_llms(_auth):
    multimodal_only = {"model_type_id": "multimodal-to-text"}
    llvms = App(pat=_auth._pat).list_models(
        filter_by=multimodal_only, only_in_app=False
    )
    return list(llvms)


def sidebar(st, config):
    #### Page Display Options

    st.subheader("Page Setup")

    st.subheader("Model Selections")
    # Visual Classifier Selection

    community_visual_classifiers = get_visual_classifiers(config.auth)
    community_visual_classifiers_ids_only = [x.id for x in community_visual_classifiers]

    vis_class_model_id = st.selectbox(
        label="Select Image Classification",
        options=community_visual_classifiers_ids_only,
        index=10,  # this should hopefully be the `general-image-recognition`
    )

    selected_vis_clas_model = [
        x for x in community_visual_classifiers if x.id == vis_class_model_id
    ][0]
    config.classifier_user_id = selected_vis_clas_model.user_id
    config.classifier_app_id = selected_vis_clas_model.app_id
    config.classifier_model_id = selected_vis_clas_model.id

    config.max_concepts = st.slider(
        label="Specify max concepts",
        min_value=1,
        max_value=100,
        value=config.max_concepts,
    )

    # LLVM Selection
    st.write("")
    st.write("")

    community_llvms = get_llms(config.auth)
    community_llvms_ids_only = [x.id for x in community_llvms]

    llvm_model_id = st.selectbox(
        label="Select LLVM",
        options=community_llvms_ids_only,
        index=13,  # this should hopefully be the `gpt-4o`
    )

    selected_llvm = [x for x in community_llvms if x.id == llvm_model_id][0]
    config.llvm_model_id = selected_llvm.id
    config.llvm_user_id = selected_llvm.user_id
    config.llvm_app_id = selected_llvm.app_id

    # llvm inference params
    config.llvm_temp = st.slider(
        label="Temperature", min_value=0.0, max_value=1.0, value=config.llvm_temp
    )

    config.llvm_max_tokens = st.number_input(
        label="Max Tokens", value=config.llvm_max_tokens
    )

    config.llvm_top_p = st.slider(
        label="Top P", min_value=0.0, max_value=1.0, value=config.llvm_top_p
    )

    #### Output Display options
    st.divider()

    st.subheader("Output Display Options")

    config.tag_bg_color = st.color_picker(
        label="Tag Background Color (hex)", value=config.tag_bg_color
    )

    config.tag_text_color = st.color_picker(
        label="Tag Text Color (hex)", value=config.tag_text_color
    )
