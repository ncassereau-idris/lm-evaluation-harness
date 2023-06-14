"""
WARNING : THIS VERSION OF THE DATASET IS MODIFIED IN FORMAT AND CONTENT
FROM THE ORIGINAL DATASET AVAILABLE HERE. NESTED ENTITIES HAVE BEEN 
REMOVED AND THIS DATASET ONLY RETAINS THE LARGEST OF NESTED ENTITIES. 
OVERALL, THIS CORRESPONDS TO 80% OF THE ENTITIES ANNOTATED IN THE ORIGINAL
DATASET. 

The QUAERO French Medical Corpus has been initially developed as a resource
for named entity recognition and normalization [1]. It was then improved
 with the purpose of creating a gold standard set of normalized entities 
for French biomedical text, that was used in the CLEF eHealth evaluation
lab [2][3].

"""
from lm_eval.api.task import PromptSourceTask


_CITATION = """
@article{QuaeroFrenchMed,
  title={The QUAERO French medical corpus: A ressource for medical entity recognition and normalization},
  author={N{\'e}v{\'e}ol, Aur{\'e}lie and Grouin, Cyril and Leixa, Jeremy and Rosset, Sophie and Zweigenbaum, Pierre},
  journal={Proc of BioTextMining Work},
  pages={24--30},
  year={2014}
}
"""


class QuaeroFrenchMed(PromptSourceTask):
    VERSION = 0
    DATASET_PATH = "meczifho/QuaeroFrenchMed"
    DATASET_NAME = ""

    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return False

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
        return 32
