import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# Set the HF_HOME environment variable to your desired cache folder
# If you haven't set it yet, you can use the following command:
# setx HF_HOME E:\projects\huggingface_cache

# Activate your virtual environment if you haven't already
# E.g., for Windows:
# E:\projects\auroraai\venv\Scripts\activate

# Load the model and tokenizer
model = AutoModelForCausalLM.from_pretrained("stabilityai/StableBeluga2", torch_dtype=torch.float16, low_cpu_mem_usage=True)
tokenizer = AutoTokenizer.from_pretrained("stabilityai/StableBeluga2", use_fast=False)

# Define the conversation prompt
system_prompt = "### System:\nYou are Stable Beluga, an AI that follows instructions extremely well. Help as much as you can. Remember, be safe, and don't do anything illegal.\n\n"
message = "Write me a poem please"
prompt = f"{system_prompt}### User: {message}\n\n### Assistant:\n"

# Encode the prompt and generate the response
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
output = model.generate(**inputs, do_sample=True, top_p=0.95, top_k=0, max_new_tokens=256)

# Print the decoded response
print(tokenizer.decode(output[0], skip_special_tokens=True))
