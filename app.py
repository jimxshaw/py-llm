import gradio as gradio
import torch, os
from transformers import AutoTokenizer, AutoModelForCausalLM
from dotenv import load_dotenv

load_dotenv()

llm = os.getenv("LLM")

def ask(text):
  tokenizer = AutoTokenizer.from_pretrained(llm)
  model = AutoModelForCausalLM.from_pretrained(llm, torch_dtype=torch.bfloat16)

  prompt = f'<human>: {text}\n<bot>:'

  inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

  input_length = inputs.input_ids.shape[1]

  outputs = model.generate(**inputs, max_new_tokens=100, return_dict_in_generate=True)

  tokens = outputs.sequences[0, input_length:]

  return tokenizer.decode(tokens)

with gradio.Blocks() as server:
  with gradio.Tab("LLM Inferencing"):
    llm_input = gradio.Textbox(label="Question:", value="Input question here...", interactive=True)

    ask_button = gradio.Button("Ask")
    llm_output = gradio.Textbox(label="Answer:", interactive=False, value="Output answer here...")

  ask_button.click(ask, inputs=[llm_input], outputs=[llm_output])

server.launch()
