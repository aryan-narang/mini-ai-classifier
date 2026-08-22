from transformers import pipeline

classifier = pipeline(
    "sentiment-analysis",
    model="distilbert/distilbert-base-uncased-finetuned-sst-2-english"
)

text = input("Enter a sentence: ")

result = classifier(text)[0]

label = result["label"]
confidence = result["score"] * 100

print(f"\nSentiment: {label}")
print(f"Confidence: {confidence:.2f}%")