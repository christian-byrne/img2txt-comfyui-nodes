#!pip install transformers[sentencepiece]
# from transformers import pipeline
# text = "Angela Merkel is a politician in Germany and leader of the CDU"
# hypothesis_template = "This text is about {}"
# classes_verbalized = ["politics", "economy", "entertainment", "environment"]
# zeroshot_classifier = pipeline("zero-shot-classification", model="MoritzLaurer/deberta-v3-large-zeroshot-v2.0")  # change the model identifier here
# output = zeroshot_classifier(text, classes_verbalized, hypothesis_template=hypothesis_template, multi_label=False)
# print(output)