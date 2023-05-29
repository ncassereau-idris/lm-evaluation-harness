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
import numpy as np
import sklearn
import transformers.data.metrics.squad_metrics as squad_metrics
from typing import Callable, List, Mapping, Optional, Tuple, Union
from functools import partial
from lm_eval.api.metric import (
    bits_per_byte,
    bleu,
    mean,
    rouge,
    sari,
    weighted_perplexity,
)

from lm_eval.api.task import PromptSourceTask
from lm_eval.metrics import fuzzy_list_comparison


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
        return 512

    def aggregation(self) -> Mapping[str, Callable]:
        out = {}
        for metric in self.prompt_template.metadata.metrics:
            if metric == "fuzzy_list_p":
                out["fuzzy_list_p"] = partial(fuzzy_list_comparison.score, fuzzy_list_comparison.precision)
            elif metric == "fuzzy_list_r":
                out["fuzzy_list_r"] = partial(fuzzy_list_comparison.score, fuzzy_list_comparison.recall)
            elif metric == "fuzzy_list_f":
                out["fuzzy_list_f"] = partial(fuzzy_list_comparison.fscore, fuzzy_list_comparison.precision, fuzzy_list_comparison.recall)
        return out



