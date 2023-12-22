import gradio as gradio

with gradio.Blocks() as server:
  gradio.Textbox(label="Input", value="Default value...")

server.launch()
