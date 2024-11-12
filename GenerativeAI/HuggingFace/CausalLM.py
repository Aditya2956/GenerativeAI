from transformers import GPT2Tokenizer, GPT2LMHeadModel
from transformers import pipeline

tokenizer = GPT2Tokenizer.from_pretrained('distilgpt2')
model = GPT2LMHeadModel.from_pretrained('distilgpt2')

#print(model)
#print("\n")

text_generator = pipeline('text-generation', model='distilgpt2')

prompt = "Once global warming started"
print(f"Input prompt: '{prompt}'")

print("Generating text...")
output = text_generator(prompt, max_length=50, num_return_sequences=1)

print("\nGenerated Text:")
print(output[0]['generated_text'])