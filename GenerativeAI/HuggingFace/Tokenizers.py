from transformers import AutoTokenizer

text = "Initiating GenAI by splitting text input into tokens and again converting them back to text"
print("Original text:", text)

tokens = text.split()
print("Basic split tokens:", tokens)
print("\n")

# BERT base uncased https://huggingface.co/google-bert/bert-base-uncased
print("Bert Base uncased https://huggingface.co/google-bert/bert-base-uncased")
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

print(f"Original text: '{text}'")
tokens = tokenizer.tokenize(text)
print("Tokens from BERT tokenizer:", tokens)

input_ids = tokenizer.encode(text)
print("Input IDs from BERT tokenizer:", input_ids)

output_tokens = tokenizer.decode(input_ids)
print("Output from decode of BERT tokenizer:", output_tokens)
