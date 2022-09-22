"""
An overview of the BIOASQ large-scale biomedical semantic indexing and question answering competition
https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-015-0564-6

BIOASQ Task 1b was created by using benchmark datasets containing 29 development
and 282 test English questions, along with gold standard (reference) answers,
prepared by a team of biomedical experts from around Europe and asking participants
to automatically produce answers.

Homepage: http://participants-area.bioasq.org/datasets/
"""
from lm_eval.api.task import PromptSourceTask


_CITATION = """
Tsatsaronis, G., Balikas, G., Malakasiotis, P. et al.
An overview of the BIOASQ large-scale biomedical semantic indexing and question answering competition.
BMC Bioinformatics 16, 138 (2015).
https://doi.org/10.1186/s12859-015-0564-6
"""


class BioAsqBase(PromptSourceTask):
    VERSION = 0
    DATASET_PATH = "lm_eval/datasets/biomedical/bigbio/biodatasets/bioasq_task_b"
    DATASET_NAME = None
    SPLIT = None

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


class BioAsqQA(BioAsqBase):
    DATASET_NAME = "bioasq_blurb_bigbio_qa"
