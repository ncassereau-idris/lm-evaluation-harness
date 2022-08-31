import lm_eval
import json
import argparse
import logging
from accelerate import Accelerator

argparser = argparse.ArgumentParser()
argparser.add_argument("--model", type=str, choices=["hf-causal", "hf-seq2seq"])
argparser.add_argument("--branch", type=str, default="add-data-parallel")
argparser.add_argument("--limit", type=int, default=-1)
argparser.add_argument("--task_type", type=str, choices=["gen", "cls", "ppl"])
args = argparser.parse_args()


def test_llh_tasks():
    tasks = lm_eval.get_task_list(
        "sst", template_names=["happy or mad", "review", "said"]
    )
    tasks += lm_eval.get_task_list(
        "mnli",
        template_names=["GPT-3 style", "MNLI crowdsource", "always/sometimes/never"],
    )
    tasks += lm_eval.get_task_list(
        "rte", template_names=["imply", "imply separated", "mean"]
    )
    tasks += lm_eval.get_task_list(
        "mrpc", template_names=["want to know", "equivalent", "replace"]
    )
    tasks += lm_eval.get_task_list(
        "wic",
        template_names=[
            "GPT-3-prompt",
            "GPT-3-prompt-with-label",
            "affirmation_true_or_false",
        ],
    )
    tasks += lm_eval.get_task_list(
        "axg",
        template_names=[
            "GPT-3 style",
            "MNLI crowdsource",
            "based on the previous passage",
        ],
    )
    return tasks


def test_loglikehood_rolling():
    tasks = lm_eval.get_task_list(
        "flores_101_ppl",
        template_names=[
            "translate-this-zul-cat",
            "translate-this-fra-eng",
            "translate-this-fra-eng",
        ],
    )
    return tasks


def test_greedy_tasks():
    tasks = lm_eval.get_task_list("mrpc", template_names=["generate_sentence"])
    return tasks


if args.limit < 0:
    limit = None
else:
    limit = args.limit

task_map = {'gen': test_greedy_tasks,
            'ppl': test_loglikehood_rolling,
            'cls': test_llh_tasks}

if __name__ == "__main__":
    accelerator = Accelerator()
    if args.model == "hf-seq2seq":
        pretrained_model = "google/t5-small-lm-adapt"
    elif args.model == "hf-causal":
        pretrained_model = "EleutherAI/gpt-neo-1.3B"
    
    model = lm_eval.get_model(args.model, pretrained=pretrained_model, user_defined_max_generation_length=50, accelerator=accelerator)
    tasks = task_map[args.task_type]()
  


    if args.branch == "master":
        results = lm_eval.evaluate(model=model, tasks=tasks, limit=limit,)
        with open(
            f"{args.task_type}_{pretrained_model.split('/')[-1]}_results_batch_size={model.batch_size}-branch={args.branch}.json", "w"
        ) as f:
            json.dump(results, f, indent=2)
    else:
        for batch_size in [50, 1]:
            example_logger = logging.getLogger("examples")
            filename = f"{args.task_type}_{pretrained_model.split('/')[-1]}_batch_size={batch_size}_examples.jsonl"
            formatter = logging.Formatter("%(message)s")
            handler = logging.FileHandler(filename)
            handler.setFormatter(formatter)
            example_logger.addHandler(handler)
            example_logger.setLevel(logging.INFO)
            results = lm_eval.evaluate(
                model=model, tasks=tasks, batch_size=batch_size, accelerator=accelerator, limit=limit,
            )
            if accelerator.is_main_process:
                with open(
                    f"{args.task_type}_{pretrained_model.split('/')[-1]}_results_batch_size={batch_size}-branch={args.branch}.json", "w"
                ) as f:
                    json.dump(results, f, indent=2)
