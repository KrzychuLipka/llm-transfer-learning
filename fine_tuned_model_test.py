from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base_model = AutoModelForCausalLM.from_pretrained(
    "mistralai/Mistral-7B-Instruct-v0.3", device_map="auto", load_in_4bit=True
)

model = PeftModel.from_pretrained(base_model, "geo-lora")

tokenizer = AutoTokenizer.from_pretrained("geo-lora")

# ....