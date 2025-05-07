"""Main entry point for the app.

This app is generated based on your prompt in Vertex AI Studio using
Google GenAI Python SDK (https://googleapis.github.io/python-genai/) and
Gradio (https://www.gradio.app/).

You can customize the app by editing the code in Cloud Run source code editor.
You can also update the prompt in Vertex AI Studio and redeploy it.
"""

import base64
from google import genai
from google.genai import types
import gradio as gr
import utils


def generate(
    message,
    history: list[gr.ChatMessage],
    request: gr.Request
):
  """Function to call the model based on the request."""

  validate_key_result = utils.validate_key(request)
  if validate_key_result is not None:
    yield validate_key_result
    return

  client = genai.Client(
      vertexai=True,
      project="ai-media25mun-306",
      location="us-central1",
  )
  text1 = types.Part.from_text(text=f"""Analyze the main video provided and detect the genre of the video, than put it in segments with timestamps and titles and put it in a table. These segments and titles will be used such as youtube segments to identify starting points. Also at the end do a whole video summarization. And start directly with the answer without jargon.""")

  model = "gemini-2.5-pro-preview-05-06"
  contents = [
      types.Content(
          role="user",
          parts=[
            text1
          ]
      )
  ]
  for prev_msg in history:
    role = "user" if prev_msg["role"] == "user" else "model"
    parts = utils.get_parts_from_message(prev_msg["content"])
    if parts:
      contents.append(types.Content(role=role, parts=parts))

  if message:
    contents.append(
        types.Content(role="user", parts=utils.get_parts_from_message(message))
    )

  generate_content_config = types.GenerateContentConfig(
      temperature=1,
      top_p=0.9,
      seed=0,
      max_output_tokens=65535,
      response_modalities=["TEXT"],
      safety_settings=[
          types.SafetySetting(
              category="HARM_CATEGORY_HATE_SPEECH",
              threshold="OFF"
          ),
          types.SafetySetting(
              category="HARM_CATEGORY_DANGEROUS_CONTENT",
              threshold="OFF"
          ),
          types.SafetySetting(
              category="HARM_CATEGORY_SEXUALLY_EXPLICIT",
              threshold="OFF"
          ),
          types.SafetySetting(
              category="HARM_CATEGORY_HARASSMENT",
              threshold="OFF"
          )
      ],
  )

  results = []
  for chunk in client.models.generate_content_stream(
      model=model,
      contents=contents,
      config=generate_content_config,
  ):
    if chunk.candidates and chunk.candidates[0] and chunk.candidates[0].content:
      results.extend(
          utils.convert_content_to_gr_type(chunk.candidates[0].content)
      )
      if results:
        yield results


with gr.Blocks(theme=utils.custom_theme) as demo:
  with gr.Row():
    gr.HTML(utils.public_access_warning)
  with gr.Row():
    with gr.Column(scale=1):
      with gr.Row():
        gr.HTML("<h2>Welcome to Falke video Analysis and Summarization GenAI</h2>")
      with gr.Row():
        gr.HTML("""This prototype was built using Gemini 2.5 and requires an upload of maximum 35 mb video which will detect segments of the video, provides titles and give a summary of it
            Upload the video to begin.""")
      with gr.Row():
        gr.HTML(utils.next_steps_html)

    with gr.Column(scale=2, variant="panel"):
      gr.ChatInterface(
          fn=generate,
          title="Video Analysis and Segmentation",
          type="messages",
          editable=True,
          multimodal=True,
      )
  demo.launch(show_error=True)