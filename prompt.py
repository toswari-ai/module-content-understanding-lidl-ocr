import os
from typing import Optional

from clarifai_grpc.grpc.api.service_pb2 import GetModelRequest
from clarifai_grpc.channel.clarifai_channel import ClarifaiChannel
from clarifai_grpc.grpc.api import resources_pb2, service_pb2_grpc
from clarifai_grpc.grpc.api.status import status_code_pb2

# Obtain the prompt text from a Prompt Model
# Example Usage:
#   prompt = Prompt('user').prompt
#   print(prompt)
#
class Prompt:
  def __init__(self, model_id: str, user_id: Optional[str] = None, app_id: Optional[str] = None):
    self.user_id = user_id or os.environ.get("CLARIFAI_USER_ID")
    self.app_id = app_id or os.environ.get("CLARIFAI_APP_ID")
    if self.user_id is None or self.app_id is None:
      raise ValueError("app_id and user_id must be provided")
    self.model_id = model_id
    self._prompt = None

  def _get_prompt(self):
    stub = service_pb2_grpc.V2Stub(ClarifaiChannel.get_grpc_channel())

    auth = (("authorization", "Key " + os.environ.get("CLARIFAI_PAT")),)
    user = resources_pb2.UserAppIDSet(user_id=self.user_id, app_id=self.app_id)

    request = GetModelRequest(model_id=self.model_id, user_app_id=user)
    response = stub.GetModel(request, metadata=auth)

    if response.status.code != status_code_pb2.SUCCESS:
      raise Exception(f"Get model request failed, status: {response.status.description}")

    prompt = response.model.model_version.output_info.params.fields[
        "prompt_template"
    ].string_value
    return prompt

  def _remove_template(self, prompt):
    template = "{data.text.raw}"
    return prompt.replace(template, "")

  @property
  def prompt(self):
    if not self._prompt:
      self._prompt = self._remove_template(self._get_prompt())
    return self._prompt
