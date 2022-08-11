from requests import request
import transformers
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Any, Optional, List, Tuple


REQUEST_RETURN_LENGTHS = {
    "loglikelihood": 2,
    "greedy_until": 1,
    "loglikelihood_rolling": None,
}


class RequestDataset(Dataset):
    def __init__(
        self,
        request_type: str,
        requests: List[Tuple[str, 'Request']],
        model 
    ):
        # self.requests: List[(request_type, list[Request])]
        assert len({r.request_type for r in requests}) == 1 and request_type == requests[0].request_type
        self.request_type = request_type  
        self.requests = requests
        self.model = model

    def __len__(self):
        return len(self.requests)

    def __getitem__(self, idx):
        if isinstance(self.requests[idx], Request):
            request = self.requests[idx]
        else:
            request = self.requests[idx][1]

        request_data = {
            'unique_request_id': torch.tensor([request.unique_request_id]), 
            'doc_id': torch.tensor([request.doc_id]), 
        }

        if self.request_type == "loglikelihood":
            context, target = request.args
            # NOTE: Padding-side should be agnostic for Seq2Seq models.
            self.model.tokenizer.padding_side = "right"
            context_inputs = self.model.tok_encode(context)
            target_inputs = self.model.tok_encode(target)
            decoder_inputs = context + target
            decoder_input_tokens = self.model.tok_encode(decoder_inputs)
            decoder_inputs = {
                k: v[:-1][-self.model.max_length:]
                for k, v in decoder_input_tokens.items()
            }
            return {
                **request_data,
                'context_inputs': context_inputs,
                'target_inputs': target_inputs,
                'decoder_inputs': decoder_inputs,
            }
        elif self.request_type == "greedy_until":
            # NOTE: Padding-side should be agnostic for Seq2Seq models.
            context, args = request.args
            self.model.tokenizer.padding_side = "left"
            context_inputs_tokenized = self.model.tok_encode(context)
            context_inputs = {}
            for k, v in context_inputs_tokenized.items():
                context_inputs[k] = v[self.model.user_defined_max_generation_length - self.model.max_length:]
            # TODO: Find a better way to handle stop sequences for 0-shot.
            if args['stop_sequences'] is None or args['num_fewshot'] == 0:
                stop_sequences = [self.model.eot_token]
            else:
                stop_sequences = args['stop_sequences'] + [self.model.eot_token]
            
            if args['max_generation_length'] is None:
                max_tokens = self.model.user_defined_max_generation_length
            else:
                max_tokens = args['max_generation_length']
            return {
                **request_data,
                'context_inputs': context_inputs,
                'stop_sequences': self.model.tok_encode(stop_sequences)['input_ids'],
                'num_fewshot': torch.tensor([args['num_fewshot']]),
                'max_generation_length': torch.tensor([max_tokens]),
            }


class RequestCollator:
    def __init__(self, request_type: str, *args, **kwargs):
        self.collator = transformers.data.DataCollatorWithPadding(*args, **kwargs)
        self.request_type = request_type

    def __call__(self, samples):
        unique_request_id_batch = [sample['unique_request_id'] for sample in samples]
        doc_id_batch = [sample['doc_id'] for sample in samples]

        context_tokens = [sample['context_inputs'] for sample in samples]
        context_batch = self.collator(context_tokens)
        
        request_batch = {
            'unique_request_id': torch.cat(unique_request_id_batch, dim=0),
            'doc_id': torch.cat(doc_id_batch, dim=0),
            'context_inputs': context_batch,
        }
        if self.request_type == "loglikelihood":
            target_tokens = [sample['target_inputs'] for sample in samples]
            target_batch = self.collator(target_tokens)
            decoder_input_tokens = [sample['decoder_inputs'] for sample in samples]
            decoder_tokens_batch = self.collator(decoder_input_tokens)
            return {
                **request_batch,
                'target_inputs': target_batch,
                'decoder_inputs': decoder_tokens_batch
            }
        elif self.request_type == "greedy_until":
            return {
                **request_batch,
                'stop_sequences': torch.tensor(samples[0]['stop_sequences']),
                'num_fewshot': torch.tensor(samples[0]['num_fewshot']),
                'max_generation_length': torch.tensor(samples[0]['max_generation_length']),
            }
        raise NotImplementedError("Request type {} not implemented".format(self.request_type))


class Request:
    def __init__(
        self, 
        request_type: str,
        args: Optional[Any] = None,
        index: Optional[int] = None,
        unique_request_id: int = None,
        doc_id: int = None
    ):
        if request_type not in REQUEST_RETURN_LENGTHS.keys():
            raise NotImplementedError(
                "The request type {} is not implemented!".format(request_type)
            )
        self.request_type = request_type
        self.args = args
        self.index = index
        self.unique_request_id = unique_request_id
        self.doc_id = doc_id

    def __iter__(self):
        if REQUEST_RETURN_LENGTHS[self.request_type] is None:
            raise IndexError("This request type does not return multiple arguments!")
        for return_index in range(REQUEST_RETURN_LENGTHS[self.request_type]):
            yield Request(self.request_type, self.args, return_index, self.unique_request_id, self.doc_id)

    def __getitem__(self, i: int):
        if REQUEST_RETURN_LENGTHS[self.request_type] is None:
            raise IndexError("This request type does not return multiple arguments!")
        return Request(self.request_type, self.args, i, self.unique_request_id, self.doc_id)

    def __eq__(self, other: "Request"):
        return (
            self.request_type == other.request_type
            and self.args == other.args
            and self.index == other.index
            and self.unique_request_id == other.unique_request_id
            and self.doc_id == other.doc_id
        )

    def __repr__(self):
        return f"Req_{self.request_type}{self.args}[{self.index}]_{self.doc_id}\n"


class RequestFactory:
    def __getattr__(self, attr):
        def fn(*args):
            return Request(attr, args)

        return fn


rf = RequestFactory()
