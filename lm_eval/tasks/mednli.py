"""
MedNLI - A Natural Language Inference Dataset For The Clinical Domain

MedNLI is a dataset annotated by doctors, performing a natural language inference
task (NLI), grounded in the medical history of patients.

Homepage: https://physionet.org/content/mednli/1.0.0/
"""
from lm_eval.api.task import PromptSourceTask


_CITATION = """
Shivade, C. (2019).
MedNLI - A Natural Language Inference Dataset For The Clinical Domain (version 1.0.0).
PhysioNet. https://doi.org/10.13026/C2RS98.
"""


class MedNliBase(PromptSourceTask):
    VERSION = 0
    DATASET_PATH = "lm_eval/datasets/biomedical/bigbio/biodatasets/mednli"
    DATASET_NAME = None

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


class MedNliTE(MedNliBase):
    DATASET_NAME = "mednli_bigbio_te"
