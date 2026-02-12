# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections import defaultdict

import torch

import verl
from verl.workers.reward_manager.rm import _get_deepscaler_rule_base_reward, extract_answer
from verl import DataProto
from verl.utils.reward_score import default_compute_score
import re
import statistics

def count_keywords_counter(text):
    keywords = ["wait", "alternatively", "hold on", "another thought", "verify", "think again", "but", "however", "alternative", "check", "double-check", "oh", "hmm"]

    # Build a regular expression pattern
    pattern = r'\b(' + '|'.join(re.escape(keyword) for keyword in keywords) + r')\b'

    # Find all matches and convert them to lowercase
    matches = [match.lower() for match in re.findall(pattern, text, re.IGNORECASE)]

    result = len(matches)

    return result

class NaiveRewardManager:
    """The reward manager."""

    def __init__(self, tokenizer, num_examine, compute_score=None, reward_fn_key="data_source") -> None:
        """
        Initialize the NaiveRewardManager instance.

        Args:
            tokenizer: The tokenizer used to decode token IDs into text.
            num_examine: The number of batches of decoded responses to print to the console for debugging purpose.
            compute_score: A function to compute the reward score. If None, `default_compute_score` will be used.
            reward_fn_key: The key used to access the data source in the non-tensor batch data. Defaults to "data_source".
        """
        self.tokenizer = tokenizer  # Store the tokenizer for decoding token IDs
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.compute_score = compute_score or default_compute_score
        self.reward_fn_key = reward_fn_key  # Store the key for accessing the data source

    def __call__(self, data: DataProto, return_dict=False):
        """We will expand this function gradually based on the available datasets"""

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if "rm_scores" in data.batch.keys():
            if return_dict:
                return {"reward_tensor": data.batch["rm_scores"]}
            else:
                return data.batch["rm_scores"]

        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info = defaultdict(list)

        already_print_data_sources = {}
        # Add reflection penalty
        id2correctRefcounts = defaultdict(list)  # store reflection tokens list of correct responses
        id2correctLens = defaultdict(list)       # store response tokens list of correct responses
        # id2allRefcounts = defaultdict(list) 
        for i in range(len(data)):
            data_item = data[i]
            prompt_ids = data_item.batch["prompts"]

            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch["responses"]
            valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)

            ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]

            refcount = count_keywords_counter(response_str)
            # if prompt_str not in id2allRefcounts:
            #     id2allRefcounts[prompt_str] = []
            # id2allRefcounts[prompt_str].append(refcount)

            pred = extract_answer(response_str)
            if _get_deepscaler_rule_base_reward(pred, ground_truth):
                # record the response token and reflection token of correct response
                id2correctRefcounts[prompt_str].append(refcount)
                id2correctLens[prompt_str].append(int(valid_response_length))

        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem

            prompt_ids = data_item.batch["prompts"]

            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch["responses"]
            valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)

            ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]
            data_source = data_item.non_tensor_batch[self.reward_fn_key]
            extra_info = data_item.non_tensor_batch.get("extra_info", None)

            refcounts = id2correctRefcounts[prompt_str]


            mean_refcount = 0.0
            std_refcount = 1.0
            if len(refcounts) >= 1:
                mean_refcount = statistics.mean(refcounts)
                std_refcount = statistics.stdev(refcounts) if len(refcounts) > 1 else 1.0

            cor_lengths = id2correctLens[prompt_str]

            mean_corlen = 0.0
            std_corlen = 1.0
            if len(cor_lengths) >= 1:
                mean_corlen = statistics.mean(cor_lengths)
                std_corlen = statistics.stdev(cor_lengths) if len(cor_lengths) > 1 else 1.0

            # all_mean_refcount = statistics.mean(id2allRefcounts[prompt_str]) if id2allRefcounts[prompt_str] else 0

            extra_info['information'] = {
                'solution_len': int(valid_response_length),
                'mean_refcount':mean_refcount,
                'std_refcount':std_refcount,
                'mean_corlen':mean_corlen,
                'std_corlen':std_corlen
                # 'all_mean_refcount': all_mean_refcount
            }

            # print(extra_info['information'])
            score = self.compute_score(
                data_source=data_source,
                solution_str=response_str,
                ground_truth=ground_truth,
                extra_info=extra_info,
            )

            if isinstance(score, dict):
                reward = score["score"]
                # Store the information including original reward
                for key, value in score.items():
                    reward_extra_info[key].append(value)
            else:
                reward = score

            reward_tensor[i, valid_response_length - 1] = reward

            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print("[prompt]", prompt_str)
                print("[response]", response_str)
                print("[ground_truth]", ground_truth)
                if isinstance(score, dict):
                    for key, value in score.items():
                        print(f"[{key}]", value)
                else:
                    print("[score]", score)

        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
            }
        else:
            return reward_tensor