from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor

from typing import Optional

import streamlit as st
from annotated_text import annotated_text  # https://github.com/tvst/st-annotated-text

from clarifai.client.app import App
from clarifai.client.auth import create_stub
from clarifai.client.auth.helper import ClarifaiAuthHelper
from clarifai.client.input import Inputs
from clarifai.client.model import Model
from clarifai.client.workflow import Workflow

from prompt import Prompt
from sidebar import sidebar
from footer import footer
import json


from clarifai_grpc.grpc.api import (
    resources_pb2,
    service_pb2,
)  # hopefully temporary for the Search component


import base64
from io import BytesIO
from PIL import Image
import requests


# streamlit config
st.set_page_config(layout="wide", initial_sidebar_state="collapsed")


@dataclass
class DemoConfig:
    title: str = "Schwarz Ingredience Extractor"
    logo: str = (
        "https://raw.githubusercontent.com/walkxcode/dashboard-icons/main/svg/lidl.svg"
    )
    logo_width: int = 100
    auth: ClarifaiAuthHelper = ClarifaiAuthHelper.from_env(validate=False)

    max_concepts: int = 20

    classifier_user_id: Optional[str] = None
    classifier_app_id: Optional[str] = None
    classifier_model_id: Optional[str] = None

    llvm_model_id: Optional[str] = None
    llvm_user_id: Optional[str] = None
    llvm_app_id: Optional[str] = None

    llvm_temp: float = 0.0
    llvm_max_tokens: int = 1512
    llvm_top_p: float = 0.8

    system_prompt: str = (
        f"Only respond with a JSON list, no markdown, no name value pairs. Use the following guidance to provide at most {max_concepts} concepts found in the image"
    )

    tag_bg_color: str = "#aabbcc"
    tag_text_color: str = "#2B2D37"


config = DemoConfig()

stub = create_stub(config.auth)
userDataObject = config.auth.get_user_app_id_proto()
app = App(user_id=userDataObject.user_id, app_id=userDataObject.app_id)


####################
####  SIDEBAR   ####
####################

with st.sidebar:
    sidebar(st, config)


####################
####  MAIN PAGE ####
####################

st.image(config.logo, width=config.logo_width)

st.title(config.title)

tab1, tab2 = st.tabs(["Upload by file", "Upload by URL"])

# local file version
with tab1:
    with st.form(key="input-data-file"):
        upload_image_file = st.file_uploader(
            label="Upload an image to classify",
            label_visibility="collapsed",
            type=["jpg", "jpeg", "png", "gif", "webp"],
        )

        submitted_file = st.form_submit_button("Upload")

# image url version
with tab2:
    with st.form(key="input-data-url"):

        # url version
        upload_image_url = st.text_input(
            label="Enter image url:",
            value="https://samples.clarifai.com/metro-north.jpg",
        )

        submitted_url = st.form_submit_button("Upload")

####################
####  RESULTS   ####
####################

results_container = st.container()

####################
####  FOOTER    ####
####################

footer(st)


# Return true if the inputs are valid
def has_valid_image():
    return (submitted_url and upload_image_url != "") or (
        submitted_file and upload_image_file
    )


def predict_workflow_class(image: bytes | str):

    workflow_url = (
        "https://clarifai.com/schwarz/label-understanding/workflows/workflow-ingredient"
    )
    # Creating the workflow
    image_workflow = Workflow(url=workflow_url, pat=config.auth._pat)

    if isinstance(image, str):
        return image_workflow.predict_by_url(url=upload_image_url, input_type="image")
    else:
        return image_workflow.predict_by_bytes(input_bytes=image, input_type="image")


def predict_llvm_class(image: bytes | str):

    # call the workflow
    workflow_pred = predict_workflow_class(image)

    # print(f"************** {workflow_pred.results[0].outputs[1]}")

    foundCropImage = False

    # check if there is a image crop
    try:
        if workflow_pred.results[0].outputs[1].data.regions[0].data.image.base64:
            print("Image crop found")
    except:
        llvm_container.error("No Ingredient Detected")
        return "workflow could not detect an acceptable ingredient label"

    precision = float(
        workflow_pred.results[0].outputs[1].data.regions[0].data.concepts[0].value
    )
    precision_formated = f"{precision:.3f}"

    print(
        f"?????? Concept: {workflow_pred.results[0].outputs[1].data.regions[0].data.concepts[0].value}"
    )
    print(
        f"XXXXX Image Info: {workflow_pred.results[0].outputs[1].data.regions[0].data.image.image_info}"
    )

    image_crop_bytes = (
        workflow_pred.results[0].outputs[1].data.regions[0].data.image.base64
    )

    print("****** Displaying crop image ********")
    image = Image.open(BytesIO(image_crop_bytes))
    # image.show()

    # Display the image in Streamlit
    try:
        llvm_container.image(
            image,
            caption=f"Ingredient Cropped Image - precision: {precision_formated}",
        )
        print("Image displayed in Streamlit.")
    except Exception as e:
        print(f"Error displaying image in Streamlit: {e}")
        return "Error displaying image in Streamlit"

    llvm_class_model = Model(
        pat=config.auth._pat,
        model_id=config.llvm_model_id,
        user_id=config.llvm_user_id,
        app_id=config.llvm_app_id,
    )

    llvm_inference_params = {
        "temperature": config.llvm_temp,
        "max_tokens": config.llvm_max_tokens,
        "top_p": config.llvm_top_p,
    }
    print(f"system prompt: {config.system_prompt}")

    llvm_pred = llvm_class_model.predict(
        inputs=[
            Inputs.get_multimodal_input(
                input_id="",
                image_bytes=image_crop_bytes,
                raw_text=f"{business_prompt}",
            )
        ],
        inference_params=llvm_inference_params,
    )

    llvm_output = llvm_pred.outputs[0].data.text.raw

    llvm_container.success(llvm_output)

    return llvm_output


def predict_vis_search(image: bytes | str):
    if isinstance(image, str):
        res_pvs = stub.PostInputsSearches(
            service_pb2.PostInputsSearchesRequest(
                user_app_id=userDataObject,
                searches=[
                    resources_pb2.Search(
                        query=resources_pb2.Query(
                            ranks=[
                                resources_pb2.Rank(
                                    annotation=resources_pb2.Annotation(
                                        data=resources_pb2.Data(
                                            image=resources_pb2.Image(url=image)
                                        )
                                    )
                                )
                            ]
                        )
                    )
                ],
            ),
            metadata=(("authorization", "Key " + config.auth._pat),),
        )
    else:
        res_pvs = stub.PostInputsSearches(
            service_pb2.PostInputsSearchesRequest(
                user_app_id=userDataObject,
                searches=[
                    resources_pb2.Search(
                        query=resources_pb2.Query(
                            ranks=[
                                resources_pb2.Rank(
                                    annotation=resources_pb2.Annotation(
                                        data=resources_pb2.Data(
                                            image=resources_pb2.Image(base64=image)
                                        )
                                    )
                                )
                            ]
                        )
                    )
                ],
            ),
            metadata=(("authorization", "Key " + config.auth._pat),),
        )

    return res_pvs


def fast_classify_output(vis_class_pred):
    vis_class_tuple_of_tuples = tuple(
        [
            (
                f"{x.name}",
                f"{x.value*100:.0f}%",
                config.tag_bg_color,
                config.tag_text_color,
            )
            for x in vis_class_pred.outputs[0].data.concepts
        ]
    )

    list_with_empty_strings = []
    for item in vis_class_tuple_of_tuples:
        list_with_empty_strings.append(item)
        list_with_empty_strings.append(" ")  # Add an empty string after each item

    # Remove the last empty string as it's not needed
    if list_with_empty_strings[-1] == "":
        list_with_empty_strings.pop()

    # Convert back to a tuple if needed
    vis_class_tuple_of_tuples = tuple(list_with_empty_strings)

    annotated_text(*vis_class_tuple_of_tuples)


def vis_search_output(image: bytes | str):
    vis_class_model = Model(
        pat=config.auth._pat,
        model_id=config.classifier_model_id,
        user_id=config.classifier_user_id,
        app_id=config.classifier_app_id,
    )

    if isinstance(image, str):
        return vis_class_model.predict_by_url(
            url=upload_image_url,
            input_type="image",
            inference_params={"max_concepts": config.max_concepts},
        )
    else:
        return vis_class_model.predict_by_bytes(
            input_bytes=image,
            input_type="image",
            inference_params={"max_concepts": config.max_concepts},
        )


def llvm_classify_output(llvm_pred):
    # wrangling output and cleaning up / converting to labels
    # llvm_output = llvm_pred.outputs[0].data.text.raw

    llvm_output = llvm_pred
    st.success(llvm_output)


def llvm_classify_output2(llvm_pred):

    # wrangling output and cleaning up / converting to labels
    llvm_output = llvm_pred.outputs[0].data.text.raw

    ########################################
    #### <hardcoded debugging section>  ####

    # extra cleanup, in case instructions get ignored
    llvm_output = llvm_output.replace("json", "").strip()
    llvm_output = llvm_output.replace("```", "").strip()

    #### </hardcoded debugging section> ####
    ########################################

    try:
        llvm_tuple_of_tuples = None
        json_concepts = json.loads(llvm_output)
        if len(json_concepts) > 0:
            llvm_tuple_of_tuples = tuple(
                [
                    (x, "", config.tag_bg_color, config.tag_text_color)
                    for x in json_concepts
                ]
            )

            list_with_empty_strings = []
            for item in llvm_tuple_of_tuples:
                list_with_empty_strings.append(item)
                list_with_empty_strings.append(
                    " "
                )  # Add an empty string after each item

            # Remove the last empty string as it's not needed
            if list_with_empty_strings[-1] == "":
                list_with_empty_strings.pop()

            # Convert back to a tuple if needed
            llvm_tuple_of_tuples = tuple(list_with_empty_strings)

            annotated_text(*llvm_tuple_of_tuples)
        else:
            # No answer
            st.write("No concepts found.")

    except Exception as e:
        print(f"Error: {e}")
        print(f"Raw output from LLVM: {llvm_output}")
        if llvm_tuple_of_tuples:
            print(f"Tuple of tuples: {llvm_tuple_of_tuples}")

        raise


def vis_search_output(vis_search_pred):
    hits = vis_search_pred.hits

    end_results = []
    for hit in hits:
        temp_dict = {
            "score": hit.score,
            "im_url": hit.input.data.image.hosted.prefix
            + "/orig/"
            + hit.input.data.image.hosted.suffix,
        }

        end_results.append(temp_dict)

    return end_results


if has_valid_image():

    #### display image

    # figure out which one was submitted
    # TODO: resize the image height using pillow so the image doesn't take up too much space
    if submitted_url:
        results_container.image(upload_image_url, caption=upload_image_url)
        image = upload_image_url
    if submitted_file:
        results_container.image(upload_image_file, caption="")
        image_bytes = upload_image_file.read()
        image = image_bytes

    # business_prompt = Prompt("ingredient-label-extracter").prompt

    business_prompt = "OCR this image. Do not translate to english"

    print(f"Prompt: {business_prompt}")

    executor = ThreadPoolExecutor()
    # llvm_classifier = executor.submit(predict_llvm_class, image)s
    # fast_classifier = executor.submit(predict_vis_class, image)
    vis_searcher = executor.submit(predict_vis_search, image)

    #### Visual Classifier Output
    # results_container.subheader("Fast Classify:")
    # fast_container = results_container.container(height=150, border=False)

    ### LLVM Output
    results_container.subheader(f"Label Extraction with Generative AI:")
    llvm_container = results_container.container()

    #### Visual Search Container
    results_container.subheader(f"Visual Search:")
    vis_search_container = results_container.container()

    # with fast_container, st.spinner():
    #    fast_classify_output(fast_classifier.result())

    with llvm_container, st.spinner():
        # llvm_classify_output(llvm_classifier.result())
        predict_llvm_class(image)

    with vis_search_container, st.spinner():
        with st.expander("Results"):
            end_results = vis_search_output(vis_searcher.result())

            num_columns = 4

            im_urls = [result["im_url"] for result in end_results]
            score = [result["score"] for result in end_results]

            for i in range(int(len(im_urls) / num_columns)):
                cols = st.columns(num_columns)

                # insert rest of the images
                for col_idx in range(0, num_columns + 1):
                    try:
                        # image
                        # cols[col_idx].image(im_urls[(i*num_columns)+col_idx-1])
                        cols[col_idx].image(im_urls[(i * num_columns) + col_idx])

                        # metadata?
                        # cols[col_idx].text(score[(i*num_columns)+col_idx-1])
                        cols[col_idx].text(score[(i * num_columns) + col_idx])
                    except IndexError:
                        continue
