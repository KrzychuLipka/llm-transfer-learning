# baseline

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import json

# ===============================
# 1. Model bazowy (baseline)
# ===============================
baseline_model_name = "HuggingFaceTB/SmolLM2-135M-Instruct"

tokenizer = AutoTokenizer.from_pretrained(baseline_model_name)
model = AutoModelForCausalLM.from_pretrained(baseline_model_name).to("cuda")

# ===============================
# 2. Prompt baseline
# ===============================
BASELINE_PROMPT = """
You convert structured user’s indoor location phrases into a natural English sentence.

Task:
Generate a concise, natural, and grammatically correct description of the user’s current location.

Rules:
- The sentence MUST start with "You".
- Use only the information provided in the input.
- Do not add any new details or assumptions.
- Ensure that the information from all input phrases is reflected in the sentence.
- Maintain logical spatial flow.
- Avoid repetition.
- Use proper articles and prepositions where needed.

Input:
{input}

Output:
"""

# ===============================
# 3. Funkcja generująca odpowiedź
# ===============================
def generate_baseline(text):
    prompt = BASELINE_PROMPT.format(input=text)
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    outputs = model.generate(
        **inputs,
        max_new_tokens=50,
        temperature=0.3,
        do_sample=False  # deterministyczna generacja dla eksperymentu
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# ===============================
# 4. Wczytanie test setu
# ===============================
test_set_path = "/content/drive/MyDrive/test_set_formatted.jsonl"
results = []

with open(test_set_path) as f:
    for line in f:
        item = json.loads(line)
        # wyciągnięcie input i reference z formatu messages
        input_text = item["messages"][0]["content"]
        reference  = item["messages"][1]["content"]

        # generowanie odpowiedzi baseline
        baseline_output = generate_baseline(input_text)

        # zapis do listy wyników
        results.append({
            "input": input_text,
            "reference": reference,
            "baseline": baseline_output
        })

# ===============================
# 5. Zapis wyników do pliku
# ===============================
with open("/content/drive/MyDrive/baseline_results.jsonl", "w") as f_out:
    for r in results:
        f_out.write(json.dumps(r, ensure_ascii=False) + "\n")

print("Baseline generation finished! Results saved to baseline_results.jsonl")