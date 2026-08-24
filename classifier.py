from transformers import pipeline

classifier = pipeline(
    "sentiment-analysis",
    model="distilbert/distilbert-base-uncased-finetuned-sst-2-english"
)
while True:
    text = input("Enter a sentence: ")

    if text.lower() == "exit":
        print("Exiting the program.")
        break
    result = classifier(text)[0]

    label = result["label"]
    confidence = result["score"] * 100

    print(f"\nSentiment: {label}")
    print(f"Confidence: {confidence:.2f}%")    
