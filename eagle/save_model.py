from transformers import AutoTokenizer, AutoModelForCausalLM

# Specify a local directory to save the model and tokenizer
local_model_path = "./llama3.2-3b-instruct-local"

# Download and save the tokenizer
tokenizer = AutoTokenizer.from_pretrained("unsloth/Llama-3.2-3B-Instruct")
tokenizer.save_pretrained(local_model_path)

# Download and save the model
model = AutoModelForCausalLM.from_pretrained("unsloth/Llama-3.2-3B-Instruct")
model.save_pretrained(local_model_path)

print(f"Model and tokenizer saved to {local_model_path}")