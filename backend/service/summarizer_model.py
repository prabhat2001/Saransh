# %%
import sys
!{sys.executable} -m pip install spacy

# %%
from sklearn.feature_extraction.text import TfidfVectorizer
from spacy.lang.en import English
import numpy as np

# %%
target_url = "https://firebasestorage.googleapis.com/v0/b/saransh-36252.appspot.com/o/text%2Fnew.txt?alt=media&token=dd2854fc-4356-4df4-a648-cd311dacc99b"

import requests

response = requests.get(target_url)
data = response.text

print(data)

# %%
nlp = English()
nlp.add_pipe('sentencizer')

# %%
text_corpus = data

# %%
def summarizer(text, tokenizer, max_sent_in_summary=3):
    document = nlp(text_corpus.replace("\n", ""))
    sentences = [sent.text.strip() for sent in document.sents]
    sentence_organizer = {k:v for v,k in enumerate(sentences)}
    tf_idf_vectorizer = TfidfVectorizer(min_df=2,  max_features=None, 
                                        strip_accents='unicode', 
                                        analyzer='word',
                                        token_pattern=r'\w{1,}',
                                        ngram_range=(1, 3), 
                                        use_idf=1,smooth_idf=1,
                                        sublinear_tf=1,
                                        stop_words = 'english')
    
    tf_idf_vectorizer.fit(sentences)
    sentence_vectors = tf_idf_vectorizer.transform(sentences)
    sentence_scores = np.array(sentence_vectors.sum(axis=1)).ravel()
    N = max_sent_in_summary
    top_n_sentences = [sentences[ind] for ind in np.argsort(sentence_scores, axis=0)[::-1][:N]]
    mapped_top_n_sentences = [(sentence,sentence_organizer[sentence]) for sentence in top_n_sentences]
    mapped_top_n_sentences = sorted(mapped_top_n_sentences, key = lambda x: x[1])
    ordered_scored_sentences = [element[0] for element in mapped_top_n_sentences]
    summary = " ".join(ordered_scored_sentences)
    return summary

# %%
print("Summarizer Result: \n", summarizer(text=text_corpus, tokenizer=nlp, max_sent_in_summary=3))


