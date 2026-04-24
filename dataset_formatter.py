import json

input_file = "test_set.jsonl"
output_file = "test_set_formatted.jsonl"

with open(input_file, "r", encoding="utf-8") as f_in, open(
    output_file, "w", encoding="utf-8"
) as f_out:

    for line in f_in:
        item = json.loads(line)

        new_item = {
            "messages": [
                {"role": "user", "content": item["input"]},
                {"role": "assistant", "content": item["output"]},
            ]
        }

        f_out.write(json.dumps(new_item) + "\n")
