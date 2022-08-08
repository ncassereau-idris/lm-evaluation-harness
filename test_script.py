import lm_eval
import json
import sys
import accelerate


if __name__ == "__main__":
    # model = lm_eval.get_model("hf-causal", pretrained='gpt2', device='cuda', batch_size=10)
    # model = lm_eval.get_model("hf-causal", pretrained='gpt2', device='cuda')
    model = lm_eval.get_model("hf-seq2seq", pretrained='google/t5-small-lm-adapt', device='cuda')
    tasks = lm_eval.get_task_list(
        'sst',
        template_names=['happy or mad', 'review', 'said']
    )
    tasks += lm_eval.get_task_list(
        'mnli',
        template_names=['GPT-3 style', 'MNLI crowdsource', 'always/sometimes/never']
    )
    tasks += lm_eval.get_task_list(
        'rte',
        template_names=['imply', 'imply separated', 'mean']
    )
    tasks += lm_eval.get_task_list(
        'mrpc',
        template_names=['want to know', 'equivalent', 'replace']
    )
    tasks += lm_eval.get_task_list(
        'wic',
        template_names=['GPT-3-prompt', 'GPT-3-prompt-with-label', 'affirmation_true_or_false']
    )
    tasks += lm_eval.get_task_list(
        'axg',
        template_names=['GPT-3 style', 'MNLI crowdsource', 'based on the previous passage']
    )
    if sys.argv[1] == 'master':
        results = lm_eval.evaluate(model=model, tasks=tasks)
        with open(f'results_batch_size={model.batch_size}-branch={sys.argv[1]}.json', 'w') as f:
            json.dump(results, f, indent=2)
    else:
        for batch_size in [80, 1]:
            results = lm_eval.evaluate(
                model=model, tasks=tasks, batch_size=batch_size)
            with open(f'results_batch_size={batch_size}-branch={sys.argv[1]}.json', 'w') as f:
                json.dump(results, f, indent=2)
