import argparse
import datetime
import json
import re
import statistics
from typing import Dict, List

from fastNLP import logger
from tqdm import tqdm


def build_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_file', type=str, default='testset_hr.jsonl')
    parser.add_argument('--num_return_sequences', type=int, default=1)
    parser.add_argument('--test_k', type=int, default=0)
    parser.add_argument('--log_suffix', type=str, default='batch/')
    parser.add_argument('--save_json', type=str, default='')
    return parser


def parse_pipe_separated(text: str) -> List[str]:
    return [s.strip() for s in text.split('|') if s.strip()]


def normalize_pred_line(line: str) -> List[str]:
    """
    Convert one prediction line into a token list.

    Supported formats:
    1. plain answer line: "78 33 128 90"
    2. tagged answer: "<answer>78 33 128 90</answer>"
    3. delimiters like "|" or "," are also tolerated
    """
    line = line.strip()

    # Prefer content inside <answer>...</answer> if present
    match = re.search(r'<answer>(.*?)</answer>', line, flags=re.IGNORECASE | re.DOTALL)
    if match is not None:
        line = match.group(1).strip()

    # Normalize common delimiters to spaces
    line = re.sub(r'[|,\t]+', ' ', line)
    line = re.sub(r'\s+', ' ', line).strip()

    if not line:
        return []
    return line.split(' ')


def normalize_step_category(step_text: str) -> str:
    """
    Map a step label to one of:
    - step1
    - step2
    - step3
    - step4+

    Robust to labels such as:
    "step1", "Step 1", "1", "step 2", "third", etc.
    """
    s = str(step_text).strip().lower()

    num_match = re.search(r'(\d+)', s)
    if num_match:
        step_id = int(num_match.group(1))
        if step_id == 1:
            return 'step1'
        elif step_id == 2:
            return 'step2'
        elif step_id == 3:
            return 'step3'
        else:
            return 'step4+'

    if 'first' in s:
        return 'step1'
    if 'second' in s:
        return 'step2'
    if 'third' in s:
        return 'step3'

    return 'step4+'


def safe_rate(correct: int, total: int) -> float:
    return correct / total if total > 0 else 0.0


def mean_std(values: List[float]) -> Dict[str, float]:
    if len(values) == 0:
        return {'mean': 0.0, 'std': 0.0}
    if len(values) == 1:
        return {'mean': values[0], 'std': 0.0}
    return {
        'mean': statistics.mean(values) * 100,
        'std': statistics.stdev(values) * 100,  # sample std
    }


def evaluate_one_run(
    data_list: List[dict],
    pred_lines: List[str],
    run_idx: int,
) -> Dict[str, float]:
    if len(pred_lines) < len(data_list):
        data_list = data_list[:len(pred_lines)]

    micro_sum = 0.0
    macro_sum = 0.0

    step_correct = {'step1': 0, 'step2': 0, 'step3': 0, 'step4+': 0}
    step_total = {'step1': 0, 'step2': 0, 'step3': 0, 'step4+': 0}

    for idx, data in enumerate(data_list):
        gt_answers = parse_pipe_separated(data['markdown_answer'])
        gt_steps = parse_pipe_separated(data['markdown_answer_step'])

        if len(gt_answers) == 0:
            raise ValueError(f'Example {idx} has empty markdown_answer.')

        if len(gt_answers) != len(gt_steps):
            raise ValueError(
                f'Example {idx}: len(markdown_answer)={len(gt_answers)} '
                f'!= len(markdown_answer_step)={len(gt_steps)}'
            )

        pred_answers = normalize_pred_line(pred_lines[idx])

        # Macro acc:
        # 1 if the whole sequence matches exactly, else 0
        exact_match = (pred_answers == gt_answers)
        macro_sum += float(exact_match)

        # Micro acc:
        # fraction of correctly predicted items among the ground-truth items
        correct_cnt = 0
        for j, gt in enumerate(gt_answers):
            pred_j = pred_answers[j] if j < len(pred_answers) else ''
            if pred_j == gt:
                correct_cnt += 1

        instance_micro = correct_cnt / len(gt_answers)
        micro_sum += instance_micro

        # Step-wise correctness
        for j, (gt, step_text) in enumerate(zip(gt_answers, gt_steps)):
            pred_j = pred_answers[j] if j < len(pred_answers) else ''
            cat = normalize_step_category(step_text)
            step_total[cat] += 1
            if pred_j == gt:
                step_correct[cat] += 1

    num_instances = len(data_list)
    result = {
        'micro_acc': micro_sum / num_instances,
        'macro_acc': macro_sum / num_instances,
        'step1_correct_rate': safe_rate(step_correct['step1'], step_total['step1']),
        'step2_correct_rate': safe_rate(step_correct['step2'], step_total['step2']),
        'step3_correct_rate': safe_rate(step_correct['step3'], step_total['step3']),
        'step4+_correct_rate': safe_rate(step_correct['step4+'], step_total['step4+']),
        'step1_correct': step_correct['step1'],
        'step1_total': step_total['step1'],
        'step2_correct': step_correct['step2'],
        'step2_total': step_total['step2'],
        'step3_correct': step_correct['step3'],
        'step3_total': step_total['step3'],
        'step4+_correct': step_correct['step4+'],
        'step4+_total': step_total['step4+'],
    }
    return result


def main():
    parser = build_argparser()
    args = parser.parse_args()

    logger.info(f'Arguments: {vars(args)}')

    log_suffix = args.log_suffix
    if not log_suffix.endswith('_'):
        log_suffix += '_'

    # Load dataset
    data_list = []
    with open(args.test_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data_list.append(json.loads(line))

    if args.test_k > 0:
        data_list = data_list[:args.test_k]

    # Load predictions for each try
    answer_list = []
    for idx in range(args.num_return_sequences):
        log_file = f'{log_suffix}run_{idx + 1}.log'
        if not os.path.exists(log_file):
            # raise FileNotFoundError(f'Cannot find prediction file: {log_file}')
            logger.warning(f'Cannot find prediction file: {log_file}')
            continue
        with open(log_file, 'r', encoding='utf-8') as f:
            answer_list.append(f.readlines())

    logger.info(f'Start evaluating {min(len(data_list), len(answer_list[0]))} examples')

    # Evaluate each try
    run_results = []
    for seq_idx in range(len(answer_list)):
        start_time = datetime.datetime.now()

        run_result = evaluate_one_run(
            data_list=data_list,
            pred_lines=answer_list[seq_idx],
            run_idx=seq_idx,
        )
        run_results.append(run_result)

        end_time = datetime.datetime.now()
        logger.info(
            f'Run {seq_idx + 1} | '
            f'micro_acc={run_result["micro_acc"]:.6f}, '
            f'macro_acc={run_result["macro_acc"]:.6f}, '
            f'step1_correct_rate={run_result["step1_correct_rate"]:.6f}, '
            f'step2_correct_rate={run_result["step2_correct_rate"]:.6f}, '
            f'step3_correct_rate={run_result["step3_correct_rate"]:.6f}, '
            f'step4+_correct_rate={run_result["step4+_correct_rate"]:.6f}'
        )
        logger.info('=' * 60)

    # Aggregate mean/std across tries
    micro_list = [r['micro_acc'] for r in run_results]
    macro_list = [r['macro_acc'] for r in run_results]
    step1_list = [r['step1_correct_rate'] for r in run_results]
    step2_list = [r['step2_correct_rate'] for r in run_results]
    step3_list = [r['step3_correct_rate'] for r in run_results]
    step4_list = [r['step4+_correct_rate'] for r in run_results]

    summary = {
        'num_runs': args.num_return_sequences,
        'num_examples': len(data_list),
        'micro_acc': mean_std(micro_list),
        'macro_acc': mean_std(macro_list),
        'step1_correct_rate': mean_std(step1_list),
        'step2_correct_rate': mean_std(step2_list),
        'step3_correct_rate': mean_std(step3_list),
        'step4+_correct_rate': mean_std(step4_list),
        'per_run': run_results,
    }

    logger.info('Final summary:')
    logger.info(json.dumps(summary, indent=2, ensure_ascii=False))

    if args.save_json:
        with open(args.save_json, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        logger.info(f'Saved summary to: {args.save_json}')


if __name__ == '__main__':
    import os
    main()
