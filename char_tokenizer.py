import torch

class TinyArithmeticCharTokenizer:
    def __init__(self):
        self.all_chars = "0123456789+-*/=."
        self.special_tokens = [" ", "<pad>", "<eos>", "<unk>"]
        self.vocab_list = list(self.all_chars) + self.special_tokens
        
        self.char_to_int = {char: i for i, char in enumerate(self.vocab_list)}
        self.int_to_char = {i: char for i, char in enumerate(self.vocab_list)}
        self.vocab_size = len(self.vocab_list)

    def encode(self, input_string: str) -> list[int]:
        """Converts raw text into a list of integer token IDs."""
        return [self.char_to_int.get(char, self.char_to_int["<unk>"]) for char in input_string]

    def decode(self, token_ids: list[int]) -> str:
        """Converts a list of integer token IDs back into readable text."""
        decoded_chars = []
        for token_id in token_ids:
            char = self.int_to_char.get(token_id)
            if char is not None and char not in ["<eos>", "<pad>", "<unk>", " "]:
                decoded_chars.append(char)
            elif char == "<eos>":
                break  # Stop decoding once we hit the end of the text generation
        return "".join(decoded_chars)

