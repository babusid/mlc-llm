from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os
import argparse

parser = argparse.ArgumentParser(description="Load a tokenizer from a local model directory and tokenize a sample input.")
parser.add_argument("--model_path",
                    type=str,
                    required=False,
                    default="/Users/sidhartb/Work/mlc-llm/dist/models/Phi-4-mini-instruct",
                    help="Path to the local model directory")
parser.add_argument("--input_text",
                    type=str,
                    required=False,
                    default="<|system|>\nYou are a helpful digital assistant. Please provide safe, ethical and accurate information to the user.<|user|>\nWhat is the weather in Pittsburgh?<|end|>\n<|assistant|>\n",
                    help="Text to tokenize")

args = parser.parse_args()

# Use absolute path to the locally cloned model
model_path = os.path.abspath(args.model_path)

# Verify the model directory exists
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model directory not found: {model_path}")

print(f"Loading Tokenizer from: {model_path}")

# Load the model and cast its weights to float32
tokenizer: AutoTokenizer = AutoTokenizer.from_pretrained(
    model_path,
    torch_dtype=torch.float32,  # Specify the desired output dtype
    local_files_only=True,     # Only use local files, don't try to download
    trust_remote_code=False,
)

print(f"Tokenizer loaded. Tokenizing input: {args.input_text}")
inputs = tokenizer(args.input_text, return_tensors="pt", return_attention_mask=False)
print(inputs)
