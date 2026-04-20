import argparse
import re
import os

from PIL import Image
from pathlib import Path
import datetime
import json
from typing import Optional, List

import torch
from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration
from fastNLP import logger
from peft import PeftModel
from tqdm import tqdm

from instruction_template import (GENERAL_TASK_DESCRIPTION, TEXT_ONLY_INPUT_DESCRIPTION,
                                  IMAGE_ONLY_INPUT_DESCRIPTION, HYBRID_INPUT_DESCRIPTION,
                                  OUTPUT_FORMAT_DESCRIPTION, TASK_DESCRIPTION)


def extract_answer(output_text: str) -> Optional[str]:
    """
    Extract the content inside the last <answer>...</answer> block.

    Returns:
        - stripped answer string if found
        - None if no valid answer block exists
    """
    pattern = r"<answer>\s*(.*?)\s*</answer>"
    matches = re.findall(pattern, output_text, flags=re.DOTALL | re.IGNORECASE)

    if matches:
        return matches[-1].strip()
    return None


def extract_answer_tokens(output_text: str) -> Optional[List[str]]:
    """
    Extract the answer and split it into tokens by whitespace.

    Example:
        '<answer>12 + 7 35</answer>' -> ['12', '+', '7', '35']
    """
    answer = extract_answer(output_text)
    if answer is None:
        return None
    return answer.split()


argument_parser = argparse.ArgumentParser()
argument_parser.add_argument('--test_file', type=str, default='dataset_hr.jsonl')
argument_parser.add_argument('--model_name', type=str, default='Qwen/Qwen3.5-9B')
argument_parser.add_argument('--adapter_dir', type=str, default=None)
argument_parser.add_argument('--modality', type=str, required=True,
                             choices=['text', 'image', 'hybrid'])
argument_parser.add_argument('--max_new_tokens', type=int, default=16384)
argument_parser.add_argument('--thinking', action='store_true', default=False)
argument_parser.add_argument('--num_return_sequences', type=int, default=1)
argument_parser.add_argument('--test_k', type=int, default=0)
argument_parser.add_argument('--log_suffix', type=str, default='batch/')
args = argument_parser.parse_args()

logger.info(f'Arguments: {vars(args)}')

MAX_NEW_TOKENS = args.max_new_tokens
model_name = args.model_name
num_return_sequences = args.num_return_sequences
test_k = args.test_k
test_file = args.test_file
log_suffix = args.log_suffix
if not log_suffix.endswith('_'):
    log_suffix += '_'

# Load model + processor
model = Qwen3_5ForConditionalGeneration.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto",
    # Optional on supported setups:
    # attn_implementation="flash_attention_2",
)
if args.adapter_dir is not None and args.adapter_dir not in ['None', 'none']:
    adapter_dir = args.adapter_dir
    # load LoRA adapter
    model = PeftModel.from_pretrained(model, adapter_dir)
    model = model.merge_and_unload()
    logger.info(f'Adapter Load From: {adapter_dir}')

processor = AutoProcessor.from_pretrained(model_name)

data_list = []
with open(test_file, 'r', encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            data_list.append(json.loads(line))

total_correct = 0.
total_count = 0
all_correct_chains = 0

instruction = GENERAL_TASK_DESCRIPTION
if args.modality == 'text':
    instruction += TEXT_ONLY_INPUT_DESCRIPTION
elif args.modality == 'image':
    instruction += IMAGE_ONLY_INPUT_DESCRIPTION
elif args.modality == 'hybrid':
    instruction += HYBRID_INPUT_DESCRIPTION
else:
    raise NotImplementedError
instruction += OUTPUT_FORMAT_DESCRIPTION + TASK_DESCRIPTION

if test_k > 0:
    data_list = data_list[:test_k]
logger.info(f'Start Evaluating {len(data_list)} Examples')

for idx, data in enumerate(tqdm(data_list)):
    image_path = str(os.path.abspath(Path(data['img_blank'])))
    textual_markdown = data['markdown_table']

    content_list = [{"type": "text", "text": instruction}]
    if args.modality in ['text', 'hybrid']:
        content_list.append({"type": "text", "text": textual_markdown})
    if args.modality in ['image', 'hybrid']:
        content_list.append({"type": "image", "image": Image.open(image_path).convert("RGB")})
        logger.info(f'Image path: {image_path}')

    messages = [
        {
            "role": "user",
            "content": content_list,
        }
    ]

    # Build model inputs
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
        enable_thinking=args.thinking,
    )
    input_text = processor.decode(inputs["input_ids"][0], skip_special_tokens=False)
    logger.info(f'Start for Example {idx + 1}')
    logger.info(f'Original Input: {input_text}')
    logger.info(f'Input length: {len(inputs["input_ids"][0])}')
    inputs = inputs.to(model.device)
    inputs.pop("token_type_ids", None)  # safe for some transformer builds

    start_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    logger.info(f"Start time for example {idx + 1}: {start_time}")
    with torch.inference_mode():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,
            num_return_sequences=num_return_sequences
        )
    end_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    logger.info(f"End time for example {idx + 1}: {end_time}")

    instance_correct = 0
    for seq_idx in range(num_return_sequences):
        # Trim the prompt part and decode
        gen_only = generated_ids[seq_idx, inputs.input_ids.shape[-1]:]
        text = processor.batch_decode([gen_only], skip_special_tokens=False, clean_up_tokenization_spaces=False)[0]
        lengths = (gen_only != 248044).long().sum().item()
        logger.info(f'-' * 20)
        logger.info(f'Output for Example/OutputIDX {idx + 1}/{seq_idx + 1}:')
        logger.info(text)
        logger.info(f'Sequence Length for Example/OutputIDX {idx + 1}/{seq_idx + 1}: {lengths}')

        ground_truth_answer = [s.strip() for s in data['markdown_answer'].split('|') if len(s.strip()) > 0]
        # ground_truth_answer = [int(s) for s in ground_truth_answer]
        logger.info(f'GT Answer: {" ".join([str(i) for i in ground_truth_answer])}')

        pred_answer = extract_answer_tokens(text)
        if pred_answer is not None:
            logger.info(f'Pred Answer for Example/OutputIDX {idx + 1}/{seq_idx + 1}: '
                        f'{" ".join([str(i) for i in pred_answer])}')
            with open(f'{log_suffix}run_{seq_idx + 1}.log', 'a') as f:
                f.write(f'{" ".join([str(i) for i in pred_answer])}\n')
        else:
            logger.info(f'Pred Answer for Example/OutputIDX {idx + 1}/{seq_idx + 1}: '
                        f'No Answer')
            with open(f'{log_suffix}run_{seq_idx + 1}.log', 'a') as f:
                f.write(f'No Answer\n')

        if pred_answer is not None:
            for pred, gt in zip(pred_answer, ground_truth_answer):
                if pred == gt:
                    instance_correct += 1. / len(ground_truth_answer)
            if (all([p == g for p, g in zip(pred_answer, ground_truth_answer)])
                    and len(pred_answer) == len(ground_truth_answer)):
                all_correct_chains += 1

        total_correct += instance_correct / num_return_sequences
        total_count += 1 / num_return_sequences

    logger.info(f"Start time for example {idx + 1}: {start_time}")
    logger.info(f"End time for example {idx + 1}: {end_time}")
    logger.info(f'Current Instance correct: {instance_correct}')
    logger.info(f'Total correct: {total_correct / total_count}')
    logger.info(f'Number of All Correct Chains: {all_correct_chains}')
    logger.info(f"=" * 20)

