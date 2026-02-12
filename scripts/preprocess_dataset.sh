
python src/preprocess_dataset.py --input_path ./data/train/deepscaler.json --output_path ./data/train/preprocessed_data/deepscaler.parquet

python src/preprocess_dataset.py --input_path ./data/test/gsm8k.json --output_path ./data/test/preprocessed_data/gsm8k.parquet

python src/preprocess_dataset.py --input_path ./data/test/math.json --output_path ./data/test/preprocessed_data/math.parquet

python src/preprocess_dataset.py --input_path ./data/test/amc23.json --output_path ./data/test/preprocessed_data/amc23*16.parquet --repeat 16

python src/preprocess_dataset.py --input_path ./data/test/aime24.json --output_path ./data/test/preprocessed_data/aime24*16.parquet --repeat 16

python src/preprocess_dataset.py --input_path ./data/test/aime25.json --output_path ./data/test/preprocessed_data/aime25*16.parquet --repeat 16

python src/preprocess_dataset.py --input_path ./data/test/mmlu.json --output_path ./data/test/preprocessed_data/mmlu.parquet
