import gradio as gradio

with gradio.Blocks() as server:
  with gradio.Tab("LLM Inferencing"):
    llm_input = gradio.Textbox(label="Question:", value="Input question here...", interactive=True)
    llm_output = gradio.Textbox(label="Answer:", interactive=False, value="Output answer here...")

server.launch()
