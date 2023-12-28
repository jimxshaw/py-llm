import gradio as gradio
from dotenv import load_dotenv
import os
from openai import OpenAI

load_dotenv()

openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def ask(text):
  completion = openai_client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[{
      "role": "user",
      "content": text
    }],
    temperature=0
  )

  return completion.choices[0].message.content

with gradio.Blocks() as server:
  with gradio.Tab("LLM Inferencing"):
    llm_input = gradio.Textbox(label="Question:", value="Input question here...", interactive=True)

    ask_button = gradio.Button("Ask")
    llm_output = gradio.Textbox(label="Answer:", interactive=False, value="Output answer here...")

  ask_button.click(ask, inputs=[llm_input], outputs=[llm_output])

server.launch()
