"""
GAD Gene-Disease Associations

Gene-disease associations curated from genetic association studies.

Homepage: https://maayanlab.cloud/Harmonizome/dataset/GAD+Gene-Disease+Associations
"""
from lm_eval.api.task import PromptSourceTask


_CITATION = """
Becker, KG et al. (2004)
The genetic association database.
Nat Genet. 36:431-2.
"""


class GadBase(PromptSourceTask):
    VERSION = 0
    DATASET_PATH = "lm_eval/datasets/biomedical/bigbio/biodatasets/gad"
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


class GadTEXT(GadBase):
    DATASET_NAME = "gad_blurb_bigbio_text"
