from transformers import T5Tokenizer, T5ForConditionalGeneration

tokenizer = T5Tokenizer.from_pretrained('t5-small')
model = T5ForConditionalGeneration.from_pretrained('t5-small')

#print(model)
#print("\n")

text = """
Distil models are trained using a process called distillation, 
where a smaller, lighter model (the student) learns to mimic the behavior of a larger, 
pre-trained model (the teacher). During training, 
the student model tries to replicate the teacher model's output probabilities, capturing its knowledge and performance.
This involves using a combination of the teacher's soft labels and the ground truth labels.
Distillation helps reduce model size and computational requirements while maintaining similar performance levels.
The process leverages techniques like knowledge distillation and careful optimization.
"""

input_text = "Summarize in ten words: " + text
print(f"Prefixed input text (for summarization): '{input_text}'")

input_ids = tokenizer.encode(input_text, return_tensors='pt', max_length=512, truncation=True)
#print(f"Tokenized input IDs: {input_ids}")

summary_ids = model.generate(input_ids, max_length=50, num_beams=5, early_stopping=True)


summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
print("\nSummary:")
print(summary)
print("\n")


text = "Translate English to French: I am an engineer"
print(f"Original text for translation: '{text}'")

input_ids = tokenizer.encode(text, return_tensors='pt')
#print(f"Tokenized input IDs: {input_ids}")

translated_ids = model.generate(input_ids, max_length=40, num_beams=5, early_stopping=True)

translation = tokenizer.decode(translated_ids[0], skip_special_tokens=True)
print("\nTranslation generated:")
print(translation)