from transformers import pipeline

classifier = pipeline(
    "sentiment-analysis",
    model="cardiffnlp/twitter-roberta-base-sentiment-latest"
)
while True:
    text = input("Enter a sentence: ")

    if text.lower() == "exit":
        print("Exiting the program.")
        break
    results = classifier(text, top_k=3)
    best_result = max(results, key=lambda result: result["score"])
    print("\nSentiment probabilities:")
    for result in results:
        label = result["label"]
        score = result["score"] * 100

        print(f"{label.capitalize()}: {score:.2f}%")
    
    print("\nFINAL RESULT:")

    if best_result["score"] >= 0.7:
        print(f"SENTIMENT: {best_result['label'].capitalize()} (Confidence: {best_result['score'] * 100:.2f}%)")
    else:
        print("SENTIMENT: Uncertain (Confidence below threshold)")

