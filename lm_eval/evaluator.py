import collections
import itertools
import json
import logging
import numpy as np
import torch
from tqdm import tqdm
from typing import List, Optional, Tuple

from torch.utils.data import Dataset, DataLoader
from accelerate import Accelerator
from dataclasses import dataclass


import lm_eval.models
import lm_eval.tasks
import lm_eval.api.metric
import lm_eval.api.model
from lm_eval.api.utils import set_seed
from lm_eval.api.task import Task
from lm_eval.api.request import Request, RequestDataset, RequestCollator


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def setup_example_logger(output_path):
    """Sets up a logger that will save each example and prediction."""
    example_logger = logging.getLogger("examples")
    filename = f"./outputs/examples-{output_path}.jsonl"
    formatter = logging.Formatter("%(message)s")
    handler = logging.FileHandler(filename)
    handler.setFormatter(formatter)
    example_logger.addHandler(handler)
    example_logger.setLevel(logging.INFO)


def cli_evaluate(
    *,
    model_api_name: str,
    model_args: str,
    task_name: str,
    template_names: List[str],
    num_fewshot: Optional[int] = 0,
    batch_size: Optional[int] = None,
    device: Optional[str] = None,
    use_cache: Optional[bool] = False,
    bootstrap_iters: Optional[int] = 100000,
    limit: Optional[int] = None,
    seed: Optional[int] = 1234,
    accelerator: Optional[Accelerator] = None,
) -> dict:
    """Evaluate a model from an api on a given task with multiple possible prompt
     formats. This is effectively a wrapper around `evaluate` for command-line
     interface (CLI) like usage; only primitive type arguments.

     :param model_api_name: str
         Name of the language model api to use - see:
             `lm_eval.models.list_model_apis`.
    :param model_args: Optional[str]
         String arguments for the model api - see:
             `lm_eval.api.model.get_model_from_args_string`
     :param task_name: str
         The task name of the task to evaluate the model on.
     :param template_names: List[str]
         List of template names for the specified `task_name` to evaluate under.
     :param num_fewshot: int
         Number of examples in few-shot context.
     :param batch_size: int, optional
         Batch size for model.
     :param device: str, optional
         PyTorch device (e.g. "cpu" or "cuda:0") for running models.
     :param use_cache: bool
         Whether or not to use a cache for language model results.
     :param bootstrap_iters:
         Number of iterations for bootstrap statistics.
     :param limit: int, optional
         Limit the number of examples per task (only use this for testing).
     :param seed: int
         Random seed.
     :return
         Dictionary of results
    """
    tasks = lm_eval.tasks.get_task_list(task_name, template_names)
    model = lm_eval.models.get_model_from_args_string(model_api_name, model_args)

    if use_cache:
        cache_args = model_args.replace("=", "-").replace(",", "_").replace("/", "-")
        # TODO: Make `cache_location` path configurable thru an environment var.
        cache_location = f"lm_cache/{model_api_name}_{cache_args}.db"
        model = lm_eval.api.model.CachingLM(model, cache_location)

    results = evaluate(
        model=model,
        tasks=tasks,
        num_fewshot=num_fewshot,
        batch_size=batch_size,
        limit=limit,
        bootstrap_iters=bootstrap_iters,
        seed=seed,
        accelerator=accelerator,
    )

    # Add info about the model and few shot config.
    results["config"] = {
        "model": model_api_name,
        "model_args": model_args,
        "num_fewshot": num_fewshot,
        "batch_size": batch_size,
        "device": device,
        "use_cache": use_cache,
        "limit": limit,
        "bootstrap_iters": bootstrap_iters,
    }
    return results


def evaluate(
    *,
    model: lm_eval.api.model.LM,
    tasks: List[Task],
    num_fewshot: Optional[int] = 0,
    batch_size: Optional[int] = None,
    bootstrap_iters: Optional[int] = 100000,
    limit: Optional[int] = None,
    seed: Optional[int] = 1234,
    accelerator: Optional[Accelerator] = Accelerator(),
) -> dict:
    """Instantiate and evaluate a model on a list of tasks.

    :param model: lm_eval.api.model.LM
        Language Model
    :param tasks: List[Task]
        List of tasks to evaluate `model` on.
    :param num_fewshot: int
        Number of examples in few-shot context
    :param bootstrap_iters:
        Number of iterations for bootstrap statistics
    :param limit: int, optional
        Limit the number of examples per task (only use this for testing)
    :param rng: np.random.Generator
        Random number generator for shuffling documents and sampling few-shot examples.
        Default: np.random.default_rng()
    :return
        Dictionary of results
    """
    set_seed(seed)
    rng = np.random.default_rng(seed)

    # TODO: Completely refactor this entire function to not be a huge mess, ideally breaking it down into smaller pieces
    task_dict = {}
    for task in tasks:
        if task.has_validation_docs() is False and task.has_test_docs() is False:
            logger.info(
                f"Ignoring Task: {lm_eval.tasks.get_registry_name_from_task(task)} has no validation or test docs"
            )
            continue
        # Create unique keys for each task-template pair.
        task_name = lm_eval.tasks.get_registry_name_from_task(task)
        template_name = task.prompt_template.name if task.prompt_template else None
        key = lm_eval.tasks._get_task_template_key(task_name, template_name)
        task_dict[key] = task

    results = collections.defaultdict(dict)
    versions = collections.defaultdict(dict)
    requests = collections.defaultdict(list)
    requests_origin = collections.defaultdict(list)

    # TODO: We need unit tests & sanity checks or something to ensure that the return of `validation_docs` is stable
    docs = {}

    # Build contexts and collect language model requests.
    for task_template_key, task in task_dict.items():
        task_docs = task.evaluation_docs()

        logger.info(f"\n» Assigning unique IDs to '{task_template_key}' docs")
        task_docs = task_docs.map(
            lambda ex, idx: {**ex, "doc_id": idx}, with_indices=True
        )

        logger.info(f"» Filtering invalid docs from '{task_template_key}'")
        task_docs = task_docs.filter(lambda d: not task.invalid_doc_for_prompt(d))
        task_docs = task_docs.shuffle(generator=rng)

        logger.info(f"» Constructing '{task_template_key}' contexts and requests")
        pbar_limit = len(task_docs) if not limit else np.minimum(limit, len(task_docs))

        for doc_id, doc in enumerate(
            tqdm(itertools.islice(task_docs, 0, limit), total=pbar_limit)
        ):
            # assert doc_id == doc['doc_id']
            docs[(task_template_key, doc_id)] = doc
            ctx, fewshotex_logging_info = task.fewshot_context(
                doc=doc,
                num_fewshot=num_fewshot,
                rng=rng,
            )
            fewshotex_logging_info["doc_id"] = doc["doc_id"]
            args = {"num_fewshot": num_fewshot}
            reqs = task.construct_requests(doc, ctx, args)
            if not isinstance(reqs, (list, tuple)):
                reqs = [reqs]
            for i, req in enumerate(reqs):
                req.doc_id = doc_id
                req.unique_request_id = len(requests_origin[req.request_type])
                requests[req.request_type].append(req)
                # i: Index in requests for a single task instance
                # doc_id: Unique id that we can get back to a doc using `docs`
                request_return_index = req.index
                requests_origin[req.request_type].append(
                    (
                        i,
                        task_template_key,
                        doc,
                        doc_id,
                        fewshotex_logging_info,
                        request_return_index,
                    )
                )
        # Store the task version.
        versions[task_template_key] = task.VERSION

    requests_datasets = {
        request_type: RequestDataset(request_type, request_list, model)
        for request_type, request_list in requests.items()
    }

    requests_data_loaders = {
        request_type: DataLoader(
            dataset,
            collate_fn=RequestCollator(
                request_type=request_type,
                tokenizer=model.tokenizer,
                # TODO: probably should set padding_side here
                padding=True,
            ),
            batch_size=batch_size,
            shuffle=False,
        )
        for request_type, dataset in requests_datasets.items()
    }

    model.model = accelerator.prepare_model(model.model)
    requests_data_loaders = {
        request_type: accelerator.prepare_data_loader(dataloader)
        for request_type, dataloader in requests_data_loaders.items()
    }

    # All responses for each (task, doc)
    process_response_queue = collections.defaultdict(list)
    # Execute each type of request
    # TODO: Add k-datasets/dataloaders for each request type.
    #       for data_loader in [dataloaders]:
    for request_type, requests_data_loader in requests_data_loaders.items():
        samples_seen = 0
        for step, request_batch in enumerate(requests_data_loader):
            # TODO: Right now, this code runs multiple separate LM requests for
            # multiple Requests differing only in index. We could implement some
            # kind of caching, but that would be more of a band-aid solution. We
            # could also implement some kind of auto-grouping here; they should
            # end up next to each other.
            logger.info(f"\n» Running all `{request_type}` requests")
            # TODO: Make sure all requests are the same type for the given `request_type`
            if request_type == "loglikelihood_rolling":
                responses = model.loglikelihood_rolling(
                    request_batch["target_inputs"],
                )
            elif request_type == "loglikelihood":
                responses = model.loglikelihood(
                    request_batch["context_inputs"],
                    request_batch["target_inputs"],
                    request_batch["decoder_inputs"],
                )
            elif request_type == "greedy_until":
                responses = model.greedy_until(
                    request_batch["context_inputs"],
                    request_batch["stop_sequences"],
                    request_batch["max_generation_length"],
                )
            else:
                raise NotImplementedError(
                    f"Request type '{request_type}' not implemented"
                )

            if accelerator.use_distributed:
                request_batch["doc_id"] = accelerator.gather(
                    request_batch["doc_id"].squeeze()
                )
                request_batch["unique_request_id"] = accelerator.gather(
                    request_batch["unique_request_id"].squeeze()
                )
                if request_type == "loglikelihood":
                    responses = accelerator.gather(responses[0]), accelerator.gather(
                        responses[1]
                    )
                    if step == len(requests_data_loader) - 1:
                        # Last batch needs to be truncated on distributed systems as it contains additional samples
                        responses = responses[
                            : len(requests_data_loader.dataset) - samples_seen
                        ]
                    else:
                        # Otherwise we add the number of samples seen
                        samples_seen += len(responses)
                elif request_type == "greedy_until":
                    # TODO: Gather completions from greedy_until
                    responses = accelerator.gather(responses)
                    responses = model.tok_decode(responses.tolist())

            if request_type == "greedy_until":
                split_responses = []
                for response in responses:
                    for stop_sequence in request_batch["stop_sequences"]:
                        stop_sequence = stop_sequence.tolist()
                        stop_sequence_string = model.tokenizer.decode(stop_sequence)
                        response = response.split(stop_sequence_string)[0]
                    # TODO: Partial caching
                    # self.cache_hook.add_partial("greedy_until", (context, until), response)
                    split_responses.append(response)
                responses = split_responses
                # accelerator.print('GREEDY UNTIL RESPONSES', responses)
            if (
                request_type == "loglikelihood"
            ):  # TODO: This may effect the `loglikelihood_rolling` responses.
                responses = list(zip(*responses))  # Tuple
            for unique_request_id, doc_id, response in zip(
                request_batch["unique_request_id"], request_batch["doc_id"], responses
            ):
                (
                    i,
                    task_template_key,
                    doc,
                    origin_doc_id,
                    fewshotex_logging_info,
                    request_return_index,
                ) = requests_origin[request_type][unique_request_id]

                assert doc_id == origin_doc_id
                response = (
                    response[request_return_index]
                    if request_return_index is not None
                    else response
                )
                if isinstance(response, torch.Tensor):
                    response = response.cpu()
                process_response_queue[(task_template_key, int(doc_id))].append(
                    (i, response, fewshotex_logging_info, unique_request_id)
                )
    # Unpack results and sort back in order and return control to Task
    vals = collections.defaultdict(list)
    example_logger = logging.getLogger("examples")
    for (task_template_key, doc_id), per_doc_requests in process_response_queue.items():
        # accelerator.print(f'per_doc_reequest:', per_doc_requests)
        # print(f'per_doc_reequest:', per_doc_requests[0])
        request_batch["unique_request_id"] = set()
        filtered_per_doc_requests = []
        for per_doc_request in per_doc_requests:
            if per_doc_request[-1].item() not in request_batch["unique_request_id"]:
                filtered_per_doc_requests.append(per_doc_request)
                request_batch["unique_request_id"].add(per_doc_request[-1].item())
        per_doc_requests = filtered_per_doc_requests
        per_doc_requests.sort(key=lambda x: x[0])

        per_doc_results = [x[1] for x in per_doc_requests]
        # accelerator.print(f'filtered per_doc_results:', per_doc_results)
        fewshot_logging_info = [x[2] for x in per_doc_requests][0]

        task = task_dict[task_template_key]
        doc = docs[(task_template_key, doc_id)]

        output = task.process_results(doc, per_doc_results)
        if task.save_examples:
            metrics, example = output
            example.update(fewshot_logging_info)
            example.update(task.get_logging_info())
            if accelerator.is_main_process:
                example_logger.info(json.dumps(example))
        else:
            metrics = output
            example = fewshot_logging_info
            example.update(task.get_logging_info())
            if accelerator.is_main_process:
                example_logger.info(json.dumps(example))

        for metric, value in metrics.items():
            vals[(task_template_key, metric)].append(value)

    # Aggregate results
    metric_results = []
    for (task_template_key, metric), items in vals.items():
        task_name, prompt_name = lm_eval.tasks._split_task_template_key(
            task_template_key
        )
        results[task_template_key]["task_name"] = task_name
        results[task_template_key]["prompt_name"] = prompt_name
        task = task_dict[task_template_key]
        results[task_template_key][metric] = task.aggregation()[metric](items)

        _metric_results = {
            "task_name": task_name,
            "prompt_name": prompt_name,
            metric: task.aggregation()[metric](items),
            **task.get_logging_info(),
        }
        # NOTE: bleu, chrf, ter seem to be really expensive to bootstrap
        # so we run them less iterations.
        # TODO: Find an efficient work around.
        stderr = lm_eval.api.metric.stderr_for_metric(
            metric=task.aggregation()[metric],
            bootstrap_iters=min(bootstrap_iters, 1000)
            if metric in ["bleu", "chrf", "ter"]
            else bootstrap_iters,
        )
        if stderr is not None:
            results[task_template_key][metric + "_stderr"] = stderr(items)
            _metric_results[metric + "_stderr"] = stderr(items)
        metric_results.append(_metric_results)
    return {
        # List of results that tracks the averages per model and prompt.
        "results": metric_results,
        "versions": dict(versions),
        # List of all prompt x doc examples with additional information in it.
        # Original results used for generating the table when running this file.
        "table_results": dict(results),
    }


def make_table(results: dict) -> str:
    """Returns a markdown table from an evaluation results `dict`.

    Args:
        results: A dict of results as found in the `"table_results"` key
            of the dictionary returned by `evaluate`.
    """
    from pytablewriter import MarkdownTableWriter

    md_writer = MarkdownTableWriter()
    md_writer.headers = ["Task", "Prompt", "Version", "Metric", "Value", "", "Stderr"]

    values = []
    for k, result_dict in results["table_results"].items():
        version = results["versions"][k]
        for m, v in result_dict.items():
            if m.endswith("_stderr"):
                continue
            if "_name" in m:
                continue
            if m + "_stderr" in result_dict:
                se = result_dict[m + "_stderr"]
                values.append(
                    [
                        result_dict["task_name"],
                        result_dict["prompt_name"],
                        version,
                        m,
                        "%.4f" % v,
                        "±",
                        "%.4f" % se,
                    ]
                )
            else:
                values.append(
                    [
                        result_dict["task_name"],
                        result_dict["prompt_name"],
                        version,
                        m,
                        "%.4f" % v,
                        "",
                        "",
                    ]
                )
            version = ""
    md_writer.value_matrix = values
    return md_writer.dumps()
