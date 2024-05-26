import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk import pos_tag
from nltk.tokenize import word_tokenize
import nltk


def nltk_speach_tag(sentence):
    nltk.download("punkt")
    nltk.download("averaged_perceptron_tagger")
    nltk.download("stopwords")

    # Tokenize the sentence
    tokens = word_tokenize(sentence)

    # Filter out stopwords and punctuation
    stop_words = set(stopwords.words("english"))
    filtered_tokens = [
        word for word in tokens if word.lower() not in stop_words and word.isalnum()
    ]

    # Perform Part-of-Speech tagging
    tagged_tokens = pos_tag(filtered_tokens)

    # Extract nouns and proper nouns
    salient_tokens = [
        token
        for token, pos in tagged_tokens
        if pos in ["NN", "NNP", "NNS", "NNPS", "ADJ", "JJ", "FW"]
    ]
    salient_tokens = list(set(salient_tokens))

    # Re-add commas or periods relative to the original sentence

    comma_period_indices = [i for i, char in enumerate(sentence) if char in [",", "."]]
    salient_tokens_indices = [sentence.index(token) for token in salient_tokens]

    # Add commas or periods between words if there was one in the original sentence
    out = ""
    for i, index in enumerate(salient_tokens_indices):
        out += salient_tokens[i]
        distance_between_next = (
            salient_tokens_indices[i + 1] - index
            if i + 1 < len(salient_tokens_indices)
            else None
        )

        puncuated = False
        if not distance_between_next:
            puncuated = True
        else:
            for i in range(index, index + distance_between_next):
                if i in comma_period_indices:
                    puncuated = True
                    break

        if not puncuated:
            # IF the previous word was an adjective, and current is a noun, add a space
            if (
                i > 0
                and tagged_tokens[i - 1][1] in ["JJ", "ADJ"]
                and tagged_tokens[i][1] in ["NN", "NNP", "NNS", "NNPS"]
            ):
                out += " "
            else:
                out += ", "
        else:
            out += ". "

    # Add the last token
    out += sentence[-1]

    # Print the salient tokens
    return out.strip().strip(",").strip(".").strip()


def extract_keywords(text: str) -> str:
    tokenizer = AutoTokenizer.from_pretrained("yanekyuk/bert-keyword-extractor")
    model = AutoModelForTokenClassification.from_pretrained(
        "yanekyuk/bert-keyword-extractor"
    )
    """Return keywords from text using a BERT model trained for keyword extraction as 
    a comma-separated string."""
    print(f"Extracting keywords from text: {text}")

    for char in ["\n", "\t", "\r"]:
        text = text.replace(char, " ")

    sentences = text.split(".")
    result = ""

    for sentence in sentences:
        print(f"Extracting keywords from sentence: {sentence}")
        inputs = tokenizer(sentence, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            logits = model(**inputs).logits

        predicted_token_class_ids = logits.argmax(dim=-1)

        predicted_keywords = []
        for token_id, token in zip(
            predicted_token_class_ids[0],
            tokenizer.convert_ids_to_tokens(inputs["input_ids"][0]),
        ):
            if token_id == 1:
                predicted_keywords.append(token)

        print(f"Extracted keywords: {predicted_keywords}")
        result += ", ".join(predicted_keywords) + ", "

    print(f"All Keywords: {result}")
    return result
