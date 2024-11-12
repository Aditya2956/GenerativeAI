from transformers import BertTokenizer, BertForMaskedLM
from transformers import pipeline

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForMaskedLM.from_pretrained('bert-base-uncased')


#print(model)
#print("\n")

mlm_pipeline = pipeline('fill-mask', model='bert-base-uncased')

text = "The polls predict [MASK] will win the US election."
print(f"Input sentence with masked token: '{text}'")

results = mlm_pipeline(text)

print("\nPredictions for the masked token:")
for result in results:
    print(f"Predicted word: {result['token_str']}, Confidence: {result['score']:.4f}, Text: {result['sequence']}")