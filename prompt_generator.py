import random
from data_repository import (
    floors,
    rooms,
    landmarks,
    landmark_positions,
    user_positions,
)


class PromptGenerator:
    TAG = "PromptGenerator"
    OUTPUT_FILE = "prompt.txt"

    def __init__(
        self,
        dataset_size: int = 1000
    ):
        self.dataset_size = dataset_size

    def generate(self):
        dataset_lines = [self.generate_input_line() for _ in range(self.dataset_size)]

        prompt = (
            "Generate a dataset based on the input lines provided below.\n"
            "Each input element is a single line.\n"
            "Return the result in JSONL format, where each line has the structure:\n"
            '{"input": "...", "output": "..."}.\n\n'
            "OUTPUT REQUIREMENTS:\n"
            "- Generate a concise, natural, and grammatically correct description of the user's current location, starting with 'You'.\n"
            "- Ensure that all information from the input phrases is included.\n"
            "- Maintain proper sentence structure and logical flow by adding necessary articles (the, a, an) and prepositions (in, on, at).\n"
            "- Do not introduce any details, descriptions, or assumptions beyond the given input phrases.\n"
            "- Use only the information explicitly provided.\n"
            "- Avoid repetition.\n"
            "- The output style must strictly follow the expert examples below.\n"
            "- Produce natural, precise, single‑sentence spatial descriptions.\n"
            "- Do not repeat the examples — generate only new records.\n\n"
            "EXPERT EXAMPLES:\n"
            '{"input": "floor: basement level; userPositionInfo: end of spacious laboratory room, next to double roller gate; landmarkInfo: two columns in middle of room, behind double roller gate is exit ramp for vehicles.", '
            '"output": "You are on the basement level, at the end of a spacious laboratory room with two columns in the middle, next to a double roller gate with a vehicle exit ramp behind it."}\n'
            '{"input": "floor: first floor; userPositionInfo: end of southwest corridor", '
            '"output": "You are at the end of the southwest corridor on the first floor."}\n'
            '{"input": "floor: first floor; userPositionInfo: beginning of north-eastern corridor; landmarkInfo: corridor access through fire door.", '
            '"output": "You are on the first floor, at the beginning of the northeastern corridor, which is accessed through a fire door."}\n'
            '{"input": "floor: ground floor; userPositionInfo: end of northeastern corridor, next to entrance to conference room.", '
            '"output": "You are on the ground floor, at the end of the northeastern corridor, next to the entrance to the conference room."}\n'
            '{"input": "floor: first floor; userPositionInfo: in small central hall, next to spiral staircase", '
            '"output": "You are in a small central hall on the first floor, close to the spiral staircase."}\n'
            '{"input": "floor: ground floor; userPositionInfo: in corridor, before toilets entrance, next to building’s entrance hall.", '
            '"output": "You are in the corridor before the entrance to the toilets, next to the building’s entrance hall."}\n\n'
            "INPUT DATA:\n" + "\n".join(dataset_lines)
        )
        
        with open(self.OUTPUT_FILE, "w", encoding="utf-8") as file: 
            file.write(prompt) 
        
        print(f"Prompt saved to: {self.OUTPUT_FILE}") 

    def generate_input_line(self) -> str:
        floor = random.choice(floors)
        room = random.choice(rooms)
        landmark = random.choice(landmarks)
        user_position = random.choice(user_positions)
        landmark_position = random.choice(landmark_positions)
        include_landmark_info = random.choice([True, False])

        parts: list[str] = [
            f"floor: {floor};",
            f"userPositionInfo: {user_position} {room};",
        ]

        if include_landmark_info:
            parts.append(f"landmarkInfo: {landmark} {landmark_position};")

        return " ".join(parts)


if __name__ == "__main__":
    generator = PromptGenerator()
    generator.generate()
