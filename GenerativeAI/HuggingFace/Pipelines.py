from transformers import pipeline


print("Distilbert sentiment analysis https://huggingface.co/lxyuan/distilbert-base-multilingual-cased-sentiments-student")
classifier = pipeline("sentiment-analysis", model="lxyuan/distilbert-base-multilingual-cased-sentiments-student", return_all_scores=True)

text = "Generative AI is an amazing technology"
print(f"Text for sentiment analysis: '{text}'")

result = classifier(text)
print("Sentiment analysis result (all scores):", result)
print("\n")


print("Zero Shot classification https://huggingface.co/valhalla/distilbart-mnli-12-1")
classifier = pipeline("zero-shot-classification", model="valhalla/distilbart-mnli-12-1")

text = "Is global warming going to affect future higher education"
candidate_labels = ["danger", "safe", "PhD", "College", "winter"]
print(f"Text for zero-shot classification: '{text}'")
print(f"Candidate labels: {candidate_labels}")

result = classifier(text, candidate_labels=candidate_labels)
print("Zero-shot classification result:", result)
print("\n")  # For clarity in output


print("NER")
ner = pipeline("ner", grouped_entities=True)

text = "My MBA was from Indian School of Business, I went for my exchange to ESADE Business School in Spain"
print(f"Text for NER: '{text}'")
entities = ner(text)

print("Named Entities found in the text:")
for entity in entities:
    print(f"Entity: {entity['word']}, Type: {entity['entity_group']}")