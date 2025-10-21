import nltk
import numpy as np
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
from sklearn.metrics.pairwise import cosine_similarity
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.spice.spice import Spice
from transformers import BertTokenizer, BertModel
import torch

# nltk.download("punkt") # downloaded
# nltk.download("punkt_tab") # downloaded
# nltk.download('wordnet') # downloaded

# BLEU Score Class
class BleuMetric:
    def __init__(self, n_gram=4):
        self.n_gram = n_gram
        self.smoothing_function = SmoothingFunction().method1

    def compute(self, references, candidate):
        scores = []
        for n in range(1, self.n_gram + 1):
            score = sentence_bleu(
                references,
                candidate,
                weights=[1.0 / n] * n,
                smoothing_function=self.smoothing_function,
            )
            scores.append(score)
        return np.mean(scores)


# METEOR Score Class
class MeteorMetric:
    def __init__(self):
        pass

    def compute(self, references, candidate):
        return meteor_score(references, candidate)


# ROUGE Score Class
class RougeMetric:
    def __init__(self):
        self.rouge_scorer = rouge_scorer.RougeScorer(
            ["rouge1", "rouge2", "rougeL"], use_stemmer=True
        )

    def compute(self, references, candidate):
        scores = {"rouge1": [], "rouge2": [], "rougeL": []}
        for ref in references:
            score = self.rouge_scorer.score(ref, candidate)
            for key in scores:
                scores[key].append(score[key].fmeasure)
        return {key: np.mean(val) for key, val in scores.items()}


# CIDEr Score Class
class CiderMetric:
    def __init__(self):
        self.cider = Cider()

    def compute(self, references, candidate):
        # references and candidate are list of strings
        score, scores = self.cider.compute_score({0: references}, {0: candidate})
        # print(score)
        # print(scores)
        return score


# SPICE Score Class
class SpiceMetric:
    def __init__(self):
        self.spice = Spice()

    def compute(self, references, candidate):
        score, _ = self.spice.compute_score({0: references}, {0: candidate})
        return score


# Example of usage:
if __name__ == "__main__":
    references = [
        "A person riding a bike down a road",
        "A cyclist is riding a bike down the street",
    ]
    candidate = "A person is cycling on a road"

    tokenized_references = [nltk.word_tokenize(ref.lower()) for ref in references]
    tokenized_candidate = nltk.word_tokenize(candidate.lower())
    print(tokenized_candidate)
    bleu1 = BleuMetric(n_gram=1)
    bleu2 = BleuMetric(n_gram=2)
    bleu3 = BleuMetric(n_gram=3)
    bleu4 = BleuMetric(n_gram=4)
    meteor = MeteorMetric()
    rouge = RougeMetric()
    cider = CiderMetric()
    spice = SpiceMetric()

    # print(f"BLEU-1 Score: {bleu1.compute(tokenized_references, tokenized_candidate)}")
    # print(f"BLEU-2 Score: {bleu2.compute(tokenized_references, tokenized_candidate)}")
    # print(f"BLEU-3 Score: {bleu3.compute(tokenized_references, tokenized_candidate)}")
    # print(f"BLEU-4 Score: {bleu4.compute(tokenized_references, tokenized_candidate)}")
    # print(f"METEOR Score: {meteor.compute(tokenized_references, tokenized_candidate)}")
    # print(f"ROUGE Score: {rouge.compute(references, candidate)}")
    print(f"CIDEr Score: {cider.compute(references, [candidate])}")
    # print(f"SPICE Score: {spice.compute(references, [candidate])}")
