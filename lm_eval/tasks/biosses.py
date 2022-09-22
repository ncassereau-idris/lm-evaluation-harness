"""
BIOSSES : Biomedical Semantic Similarity Estimation System
https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5870675/

The BIOSSES data set comprises total 100 sentence pairs all of which were selected
from the "TAC2 Biomedical Summarization Track Training Data Set".

Homepage: https://tabilab.cmpe.boun.edu.tr/BIOSSES/DataSet.html
"""
from lm_eval.api.task import PromptSourceTask


_CITATION = """
Sogancioglu, G., Öztürk, H., & Özgür, A. (2017).
BIOSSES: a semantic sentence similarity estimation system for the biomedical domain.
Bioinformatics (Oxford, England), 33(14), i49–i58.
https://doi.org/10.1093/bioinformatics/btx238
"""


class BiossesBase(PromptSourceTask):
    VERSION = 0
    DATASET_PATH = "lm_eval/datasets/biomedical/bigbio/biodatasets/biosses"
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


class BiossesPAIRS(BiossesBase):
    DATASET_NAME = "biosses_bigbio_pairs"
