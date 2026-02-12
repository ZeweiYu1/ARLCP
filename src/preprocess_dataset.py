import argparse
import os
from typing import Dict, List, Optional, Any
import pandas as pd
import json
import random

def make_map_fn(split: str, data_source: str = ""):
    """Create a mapping function to process dataset examples.

    Args:
        split: Dataset split name ('train' or 'test')

    Returns:
        Function that processes individual dataset examples
    """
    def process_fn(example: Dict[str, Any], idx: int) -> Optional[Dict[str, Any]]:
        if 'problem' in example:
            question = example['problem']
        elif 'question' in example:
            question = example['question']

        answer = example['answer']
        # print(data_source)
        data = {
            "data_source": data_source,
            "prompt": [{
                "role": "user",
                "content": question
            }],
            "ability": "math",
            "reward_model": {
                "style": "rule",
                "ground_truth": str(answer)
            },
            "extra_info": {
                'split': split,
                'index': idx
            }
        }
        return data
    return process_fn


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process datasets for DeepScaler training')
    parser.add_argument('--input_path', default='./data/train/deepscaler.json')
    parser.add_argument('--output_path', default='./data/train/preprocessed_data/deepscaler.parquet')
    parser.add_argument('--repeat', type=int, default=1)
    args = parser.parse_args()

    print('-'*50)
    print("Input Path:", args.input_path)
    print("Repeat:", args.repeat)
    data_source = args.input_path.split('/')[-1].removesuffix(".json").strip()
    print(data_source)
    split = args.input_path.split('/')[-2]
    local_dir = args.output_path.rsplit('/', 1)[0]
    os.makedirs(local_dir, exist_ok=True)

    dataset = json.load(open(args.input_path, 'r')) * args.repeat


    data: List[Dict[str, Any]] = []
    process_fn = make_map_fn(split, data_source)
    for idx, example in enumerate(dataset):
        processed_example = process_fn(example, idx)
        if processed_example is not None:
            data.append(processed_example)

    print("random data size:", len(data))

    train_df = pd.DataFrame(data)

    # print(train_df.iloc[0]['prompt'])
    # exit(0)
    train_df.to_parquet(args.output_path)
    print("Save to:", args.output_path)