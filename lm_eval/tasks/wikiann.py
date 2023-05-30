"""
SuperGLUE: A Stickier Benchmark for General-Purpose Language Understanding Systems
https://w4ngatang.github.io/static/papers/superglue.pdf

WikiANN (sometimes called PAN-X) is a multilingual named entity recognition dataset 
consisting of Wikipedia articles annotated with LOC (location), PER (person), and 
ORG (organisation) tags in the IOB2 format. This version corresponds to the balanced 
train, dev, and test splits of Rahimi et al. (2019), which supports 176 of the 282 
languages from the original WikiANN corpus.

Homepage: https://github.com/afshinrahimi/mmner

"""

from lm_eval.api.task import PromptSourceTask

_CITATION = """
@inproceedings{rahimi-etal-2019-massively,
    title = "Massively Multilingual Transfer for {NER}",
    author = "Rahimi, Afshin  and
      Li, Yuan  and
      Cohn, Trevor",
    booktitle = "Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics",
    month = jul,
    year = "2019",
    address = "Florence, Italy",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/P19-1015",
    pages = "151--164",
}
"""


class Wikiann(PromptSourceTask):
    VERSION = 0
    DATASET_PATH = "wikiann"
    DATASET_NAME = "fr"

    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return True

    def training_docs(self):
        if self.has_training_docs():
            return self.dataset["train"]

    def validation_docs(self):
        if self.has_validation_docs():
            return self.dataset["validation"]

    def test_docs(self):
        if self.has_test_docs():
            return self.dataset["test"]

    def max_generation_length(self):
        return 512
