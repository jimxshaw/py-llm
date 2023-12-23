import gradio as gradio

def ask(text):
  return text.upper()

with gradio.Blocks() as server:
  with gradio.Tab("LLM Inferencing"):
    llm_input = gradio.Textbox(label="Question:", value="Input question here...", interactive=True)

    ask_button = gradio.Button("Ask")
    llm_output = gradio.Textbox(label="Answer:", interactive=False, value="Output answer here...")

  ask_button.click(ask, inputs=[llm_input], outputs=[llm_output])

server.launch()
