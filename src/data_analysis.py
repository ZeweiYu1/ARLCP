import json
import re
import os
import argparse
from transformers import AutoTokenizer

def count_keywords_counter(text):
    """
    Count the occurrences of specific keywords in the text.
    """
    keywords = [
        "wait", "alternatively", "hold on", "another thought",
        "verify", "think again", "but", "however", 
        "alternative", "check", "double-check", "oh", "hmm"
    ]

    # Build regex pattern using word boundaries (\b)
    pattern = r'\b(' + '|'.join(re.escape(keyword) for keyword in keywords) + r')\b'

    # Find all matches and convert to lower case
    matches = [match.lower() for match in re.findall(pattern, text, re.IGNORECASE)]

    return len(matches)

def analyze_jsonl_core(file_path, tokenizer):
    """
    Core analysis logic: Reads JSONL, calculates tokens using the tokenizer (if needed),
    and compiles statistics for keywords and accuracy.
    """
    stats = {}
    print(f"Reading data from: {file_path}")

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_number, line in enumerate(f, 1):
                try:
                    line = line.strip()
                    if not line:
                        continue

                    data = json.loads(line)

                    # 1. Extract data_source
                    data_source = data.get('data_source', 'unknown')
                    if not isinstance(data_source, str): 
                        data_source = 'unknown'

                    # Initialize statistics for this category if new
                    if data_source not in stats:
                        stats[data_source] = {
                            'total_score': 0.0, 
                            'total_tokens': 0, 
                            'total_keywords': 0, 
                            'count': 0
                        }

                    # 2. Extract score
                    score = data.get('acc')
                    if score is None or not isinstance(score, (int, float)):
                        # Skip record if score is invalid
                        continue

                    # 3. Extract output and calculate metrics
                    output_text = data.get('output')
                    tokens_count = 0
                    keywords_count = 0

                    if isinstance(output_text, str):
                        # Use 'solution_len' if available, otherwise calculate using tokenizer
                        tokens_count = data.get('solution_len', 0)
                        if tokens_count == 0 and tokenizer:
                            # Note: add_special_tokens=False typically excludes BOS/EOS tokens
                            tokens_count = len(tokenizer(output_text, add_special_tokens=False)['input_ids'])

                        # Count keywords
                        keywords_count = count_keywords_counter(output_text)

                    # 4. Accumulate stats
                    stats[data_source]['total_score'] += score
                    stats[data_source]['total_tokens'] += tokens_count
                    stats[data_source]['total_keywords'] += keywords_count
                    stats[data_source]['count'] += 1

                except json.JSONDecodeError:
                    print(f"[Warn] Line {line_number}: JSON decode error")
                except Exception as e:
                    print(f"[Warn] Line {line_number}: {e}")

        # === Print Results ===
        print_results(stats, file_path)

    except FileNotFoundError:
        print(f"Error: File not found at '{file_path}'")
    except Exception as e:
        print(f"Unexpected error: {e}")

def print_results(stats, file_path):
    """
    Format and print the statistics results in a table.
    """
    print(f"\n{'='*20} Analysis Result {'='*20}")
    print(f"Source File: {os.path.basename(file_path)}")

    # Define table header
    header = f"{'Data Source':<20} {'ACC':<10} {'Avg Tokens':<15} {'Avg Keywords':<15} {'Count':<8}"
    print("-" * len(header))
    print(header)
    print("-" * len(header))

    # Sort and print by data_source
    for ds in sorted(stats.keys()):
        s = stats[ds]
        if s['count'] > 0:
            avg_acc = s['total_score'] / s['count']
            avg_tokens = s['total_tokens'] / s['count']
            avg_keywords = s['total_keywords'] / s['count']
            print(f"{ds:<20} {avg_acc:<10.4f} {avg_tokens:<15.2f} {avg_keywords:<15.2f} {s['count']:<8}")

    # Calculate and print Overall statistics
    total_valid = sum(s['count'] for s in stats.values())
    if total_valid > 0:
        t_score = sum(s['total_score'] for s in stats.values())
        t_tokens = sum(s['total_tokens'] for s in stats.values())
        t_keywords = sum(s['total_keywords'] for s in stats.values())

        overall_acc = t_score / total_valid
        overall_tokens = t_tokens / total_valid
        overall_keywords = t_keywords / total_valid

        print("-" * len(header))
        print(f"{'Overall':<20} {overall_acc:<10.4f} {overall_tokens:<15.2f} {overall_keywords:<15.2f} {total_valid:<8}")

    print(f"\n✅ Done.")

def main():
    # Define command-line arguments
    parser = argparse.ArgumentParser(description="Analyze JSONL for reasoning keywords and metrics.")

    # Set arguments; default values are kept from the original request
    parser.add_argument(
        "--model_path", 
        type=str, 
        default="/input2/yzw/models/DeepSeek-R1-Distill-Qwen-1.5B",
        help="Path to the model or tokenizer (default: DeepSeek-R1-Distill-Qwen-1.5B)"
    )

    parser.add_argument(
        "--data_path", 
        type=str, 
        default="/input2/yzw/efficient_reasoning/ARLCP_data/7b/230.jsonl",
        help="Path to the JSONL data file to analyze"
    )

    args = parser.parse_args()

    # Print current configuration
    print(f"--- Configuration ---")
    print(f"Model Path: {args.model_path}")
    print(f"Data Path : {args.data_path}")
    print(f"---------------------")

    # Load Tokenizer
    print("Loading tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        return

    # Run analysis
    analyze_jsonl_core(args.data_path, tokenizer)

if __name__ == "__main__":
    main()