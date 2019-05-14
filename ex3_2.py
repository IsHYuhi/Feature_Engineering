import pandas as pd
import json

with open('./yelp_dataset/review.json') as f:
    js = []
    for i in range(10):
        js.append(json.loads(f.readline()))
review_df = pd.DataFrame(js)

#Spacy
import spacy

nlp = spacy.load('en')

#spaCyの言語モデルを使いテキストからPandas Seriesを作成する
doc_df = review_df['text'].apply(nlp)

print("Spacy")
#細かい品詞タグは.pos_ , 荒い品詞タグは .tag_
for doc in doc_df[4]:
    print([doc.text, doc.pos_, doc.tag_])

#spaCyは基本的な名詞句も .noun_chunks で提供する
print([chunk for chunk in doc_df[4].noun_chunks])


print("textBlobライブラリでも同じことが可能")

from textblob import TextBlob
import nltk
nltk.download('averaged_perceptron_tagger')
nltk.download('punkt')
nltk.download('brown')

blob_df = review_df['text'].apply(TextBlob)

print(blob_df[4].tags)
print()
print([np for np in blob_df[4].noun_phrases])