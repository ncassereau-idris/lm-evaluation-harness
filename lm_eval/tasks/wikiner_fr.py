"""
WikiNER_fr: Wikipedia sentences in French annotated for named entities (person, place, 
organisation).
https://metatext.io/datasets/wikiner

Created by Nothman et al. at 2013, the WikiNER Dataset contains 7,200 manually-labelled 
Wikipedia articles across nine languages: English, German, French, Polish, Italian, 
Spanish,Dutch, Portuguese and Russian. This is the French portion of the dataset.

Homepage: https://github.com/afshinrahimi/mmner

"""

from lm_eval.api.task import PromptSourceTask

_CITATION = """
@article{NOTHMAN2013151,
    author = {Joel Nothman and Nicky Ringland and Will Radford and Tara Murphy and James R. Curran},
    title = {Learning multilingual named entity recognition from Wikipedia},
    year = {2013},
    journal = {Artificial Intelligence},
    volume = {194},
    pages = {151--175},
    issn = {0004-3702},
    doi = {https://doi.org/10.1016/j.artint.2012.03.006},
    url = {https://www.sciencedirect.com/science/article/pii/S0004370212000276}
}
"""


class WikiNER_fr(PromptSourceTask):
    VERSION = 0
    DATASET_PATH = "Jean-Baptiste/wikiner_fr"
    DATASET_NAME = ""

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
        return 64
