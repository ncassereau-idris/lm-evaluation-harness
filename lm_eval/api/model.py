import abc
import hashlib
import json
import os
import torch
import torch.nn.functional as F
from sqlitedict import SqliteDict
from tqdm import tqdm
from typing import Iterable, List, Optional, Tuple, Union
from transformers import BatchEncoding
import accelerate

from lm_eval.api import utils


class LM(abc.ABC):
    def __init__(self):
        self.cache_hook = CacheHook(None)

    @abc.abstractmethod
    def loglikelihood(
        self, requests: List[Tuple[str, str]]
    ) -> List[Tuple[float, bool]]:
        """Compute log-likelihood of generating a continuation from a context.
        Downstream tasks should attempt to use loglikelihood instead of other
        LM calls whenever possible.

        :param requests: list
            A list of pairs (context, continuation)
            context: str
                Context string. Implementations of LM must be able to handle an
                empty context string.
            continuation: str
                The continuation over which log likelihood will be calculated. If
                there is a word boundary, the space should be in the continuation.
                For example, context="hello" continuation=" world" is correct.
        :return: list
            A list of pairs (logprob, isgreedy)
            logprob: float
                The log probability of `continuation`
            isgreedy:
                Whether `continuation` would be generated by greedy sampling from `context`
        """
        pass

    @abc.abstractmethod
    def loglikelihood_rolling(self, requests: List[Tuple[str, str]]) -> List[float]:
        """Compute full log-likelihood of a string, with no truncation, for perplexity computation
        - We will use the full max context length of the model.
        - For inputs that exceed the max context length, we divide the tokenized string into chunks of up to
        the max context length.
        - IMPORTANT: Each document's loglikelihood/perplexity is computed *separately*, unlike other implementations
          which may simply concatenate multiple documents together.
        - IMPORTANT: We maximize the amount of context for each prediction. Specifically, for inputs that we break into
          multiple chunks, the last input will still a full-sized context.
          Example:
            Input tokens: [ 0 1 2 3 4 5 6 7 8 9 ]
            Prefix: EOT
            Max context length: 4
            Resulting input/prediction pairs:

                INPUT:  EOT   0   1   2
                PRED:     0   1   2   3

                INPUT:    3   4   5   6
                PRED:     4   5   6   7

                INPUT:    5   6   7   8
                PRED:             8   9

          Observe that:
            1. Each token is predicted exactly once
            2. For the last pair, we provide the full context, but only score the last two tokens

        :param requests: list
            A list of strings
            string: str
                String for which we are computing per-token loglikelihood
        :return:
            A list of logprobs on the `continuation`.
        """
        pass

    @abc.abstractmethod
    def greedy_until(self, requests: List[Tuple[str, dict]]) -> List[str]:
        """Generate greedily until a stopping sequence or max generation length.

        :param requests: list
            A list of pairs (context, args)
            context: str
                Context string
            args: dict
                A dictionary of generation arguments in the form:
                {
                    stop_sequences: str,
                    max_generation_length: int,
                    num_fewshot: int
                }
        :return: list
            A list of strings continuation
            continuation: str
                The generated continuation.
        """
        pass

    def set_cache_hook(self, cache_hook: "CacheHook"):
        self.cache_hook = cache_hook


TokenSequence = Union[List[int], torch.LongTensor, torch.Tensor, BatchEncoding]


class TokenLM(LM):
    """A language model that assumes inputs, and possibly outputs, are
    tokenized text as opposed to language model APIs that only support
    string-based input and output systems.
    """

    @abc.abstractmethod
    def tok_encode(self, string: str):
        pass

    @abc.abstractmethod
    def tok_decode(self, tokens: Iterable[int]) -> List[str]:
        pass

    @property
    @abc.abstractmethod
    def eot_token(self) -> str:
        pass

    @property
    @abc.abstractmethod
    def eot_token_id(self) -> int:
        pass

    @property
    @abc.abstractmethod
    def max_gen_toks(self) -> int:
        """The maximum number of tokens to generate - not including context."""
        pass

    @property
    @abc.abstractmethod
    def max_length(self) -> int:
        """The maximum sequence length of the model."""
        pass

    @property
    @abc.abstractmethod
    def batch_size(self) -> int:
        pass

    @property
    @abc.abstractmethod
    def device(self) -> Union[int, str, torch.device]:
        pass

    def loglikelihood(
        self, context_inputs, target_inputs
    ) -> List[Tuple[float, bool]]:
        return self._loglikelihood_tokens(context_inputs, target_inputs)

    def loglikelihood_rolling(self, requests: List[Tuple[str, str]]) -> List[float]:
        # TODO: Implement caching once we've confirmed the perplexity implementation
        # TODO: Automatic batch size detection for vectorization
        loglikelihoods = []
        for (string,) in tqdm(requests):
            rolling_token_windows = list(
                map(
                    utils.make_disjoint_window,
                    utils.get_rolling_token_windows(
                        token_list=self.tok_encode(string),
                        prefix_token=self.eot_token_id,
                        max_seq_len=self.max_length,
                        context_len=1,
                    ),
                )
            )
            rolling_token_windows = [(None,) + x for x in rolling_token_windows]
            # TODO: Extract out this call so it only gets called once and
            # also somehow figure out partial caching for that.
            string_nll = self._loglikelihood_tokens(
                rolling_token_windows, disable_tqdm=True
            )
            # Discard `is_greedy`
            string_nll = [x[0] for x in string_nll]
            string_nll = sum(string_nll)
            loglikelihoods.append(string_nll)
        return loglikelihoods

    def _loglikelihood_tokens(
        self,
        context_inputs, # :List[Tuple[Tuple[str, str], TokenSequence, TokenSequence]],
        target_inputs,
        disable_tqdm: Optional[bool] = False,
    ) -> List[Tuple[float, bool]]:
        """Helper method for computing log-likelihood of generating a
        continuation from a context that have both been tokenized/encoded.

        :param requests: list
            A list of pairs ((context, continuation), context_enc, continuation_enc)
            context: str
                Context string. Implementations of LM must be able to handle an
                empty context string.
            continuation: str
                The continuation over which log likelihood will be calculated. If
                there is a word boundary, the space should be in the continuation.
                For example, context="hello" continuation=" world" is correct.
            context_enc: TokenSequence
                The tokenized/encoded context.
            continuation_enc: TokenSequence
                The tokenized/encoded continuation.
        :param disable_tqdm: Optional[bool]
            Whether to disable `tqdm` progress bar.
        :return: list
            A list of pairs (logprob, isgreedy)
            logprob: float
                The log probability of `continuation`
            isgreedy:
                Whether `continuation` would be generated by greedy sampling from `context`
        """
        # print(context_inputs['input_ids'].device)
        batched_input_ids = torch.cat([
            context_inputs['input_ids'], target_inputs['input_ids']
        ], dim=1)
        batched_attention_masks = torch.cat([
            context_inputs['attention_mask'], target_inputs['attention_mask']
        ], dim=1)
        batched_inputs = BatchEncoding({
            'input_ids': batched_input_ids,
            'attention_mask': batched_attention_masks
        })
        # print(batched_inputs['input_ids'].device)
        log_softmaxes = F.log_softmax(
            self._model_call(batched_inputs), dim=-1
        )  # [batch, padding_length, vocab]
        # multi_logits = accelerate. 
        print(f"{'='*80}")
        print(f"Multi Logits Shape: {log_softmaxes.shape}")
        print(f"Context Tokens Shape: {context_inputs['input_ids'].shape}")
        print(f"{'='*40}")

        logprobs_results = []
        exact_match_results = []
        for i, log_softmax in enumerate(log_softmaxes):
            print("log_softmax:", log_softmax)
            print(f'Logits Shape: {log_softmax.shape}')
            target_logits = log_softmax[context_inputs['input_ids'].shape[1] : ]
            target_tokens = target_inputs['input_ids'][i].unsqueeze(0)
            print(f"Target Logits Shape: {target_logits.shape}")
            print(f"Target Tokens Shape: {target_tokens.shape}")
            greedy_tokens = target_logits.argmax(dim=-1)
            exact_match = (greedy_tokens == target_tokens).all().unsqueeze(0).to(torch.bool)
            target_logits = target_logits.unsqueeze(0)  # shape: [1, seq_len, vocab]
            print(f"Unsqueezed Target Logits Shape: {target_logits.shape}")
            logprob_per_token = torch.gather(target_logits, 2, target_tokens.unsqueeze(-1)).squeeze(-1)
            print(f"Cont tokens: {context_inputs}")
            print(f"Target tokens: {target_tokens}")
            print(f"Logprob Per Token: {logprob_per_token}")
            logprobs = logprob_per_token.sum().unsqueeze(0)
            logprobs_results.append(logprobs)
            exact_match_results.append(exact_match)
        print(f"{'='*80}")
        return torch.cat(logprobs_results, dim=0), torch.cat(exact_match_results, dim=0)
        #     for (cache_key, _, _), logits, input, input_len, cont_tokens in zip(
        #         chunk, multi_logits, inputs, input_lens, cont_tokens_list
        #     ):
        #         # Slice to original seq length
        #         cont_len = len(cont_tokens)
        #         # [1, seq, vocab]
        #         logits = logits[input_len - cont_len : input_len].unsqueeze(0)
        #         # Check if per-token argmax is exactly equal to continuation
        #         greedy_tokens = logits.argmax(dim=-1)
        #         # [1, seq]
        #         cont_tokens = torch.tensor(cont_tokens, dtype=torch.long).unsqueeze(0)
        #         max_equal = (greedy_tokens == cont_tokens).all()
        #         # Obtain logprobs at the corresponding continuation token indices
        #         # last_token_slice = logits[:, -1, :].squeeze(0).tolist()
        #         # [1, seq]
        #         logits = torch.gather(logits, 2, cont_tokens.unsqueeze(-1)).squeeze(-1)
        #         # Answer: (log prob, is-exact-match)
        #         answer = (float(logits.sum()), bool(max_equal))
        #         # Partial caching
        #         if cache_key is not None:
        #             self.cache_hook.add_partial("loglikelihood", cache_key, answer)
        #         results.append(answer)
 

    @abc.abstractmethod
    def _model_call(
        self, inputs: TokenSequence, labels: Optional[TokenSequence] = None
    ) -> TokenSequence:
        """
        :param inputs: TokenSequence
            A list of strings or torch tensor of shape [batch, sequence]
            the size of sequence may vary from call to call
        :param labels: Optional[TokenSequence]
            A list of strings or torch tensor of shape [batch, sequence]
            useful for sequence-to-sequence language models.
        :return: A list of ints or torch tensor of shape
            [batch, sequence, vocab] with the logits returned from the model
        """
        pass

    @abc.abstractmethod
    def _model_generate(
        self, inputs: TokenSequence, max_tokens: int, stop: Optional[List[str]] = None
    ) -> Union[TokenSequence, List[str]]:
        """
        :param inputs: TokenSequence
            A list of strings/ints or torch tensor of shape [batch, sequence]
            the size of sequence may vary from call to call
        :param max_tokens: int
            The maximum number of tokens to generate.
        :param stop: Optional[List[str]]
            A list of stopping sequences. If provided, the generation will stop
            when any string sequence in the list is encountered.
        :return: A list of ints/strings or a torch tensor of shape
            [batch, sequence, vocab] with continuation tokens/string of the inputs.
        """
        pass


def hash_args(attr, args):
    data = json.dumps([attr] + list(args))
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


class CachingLM:
    def __init__(self, lm: LM, cache_db: str):
        """LM wrapper that returns cached results if they exist, and uses the underlying LM if not.

        :param lm: LM
            Underlying LM
        :param cache_db: str
            Path to cache db
        """
        self.lm = lm
        if os.path.dirname(cache_db):
            os.makedirs(os.path.dirname(cache_db), exist_ok=True)
        self.cache_db = cache_db
        self.dbdict = SqliteDict(cache_db, autocommit=True)
        # Add hook to lm
        lm.set_cache_hook(self.get_cache_hook())

    def __getattr__(self, attr):
        def fn(requests):
            res = []
            remaining_reqs = []

            # Figure out which ones are cached and which ones are new
            for req in requests:
                hsh = hash_args(attr, req)
                if hsh in self.dbdict:
                    ob = self.dbdict[hsh]

                    assert ob is not None
                    res.append(ob)
                else:
                    res.append(None)
                    remaining_reqs.append(req)

            # Actually run the LM on the requests that do not have cached results
            rem_res = getattr(self.lm, attr)(remaining_reqs)

            # Stick the new ones back into the list and also cache any of the new ones
            resptr = 0
            for req, r in zip(remaining_reqs, rem_res):
                while res[resptr] is not None:
                    resptr += 1

                res[resptr] = r
                # Caching
                hsh = hash_args(attr, req)
                self.dbdict[hsh] = r
            self.dbdict.commit()
            return res

        return fn

    def get_cache_hook(self):
        return CacheHook(self)


class CacheHook:
    def __init__(self, cachinglm: CachingLM):
        if cachinglm is None:
            self.dbdict = None
            return
        self.dbdict = cachinglm.dbdict

    def add_partial(self, attr, req, res):
        if self.dbdict is None:
            return
        hsh = hash_args(attr, req)
        self.dbdict[hsh] = res
