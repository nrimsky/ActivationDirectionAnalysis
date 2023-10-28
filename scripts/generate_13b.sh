if [ ! -d "venv" ]; then
    echo "Creating venv"
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
else
    source venv/bin/activate
fi

python generate_vectors.py --layers $(seq 0 39) --model_size "13b" --data_path datasets/agreeableness.json --dataset_name agreeableness
python analyze_vectors.py --dataset_name agreeableness

python generate_vectors.py --layers $(seq 0 39) --model_size "13b" --data_path datasets/conscientiousness.json --dataset_name conscientiousness
python analyze_vectors.py --dataset_name conscientiousness

python generate_vectors.py --layers $(seq 0 39) --model_size "13b" --data_path datasets/extraversion.json --dataset_name extraversion
python analyze_vectors.py --dataset_name extraversion

python generate_vectors.py --layers $(seq 0 39) --model_size "13b" --data_path datasets/powerseeking.json --dataset_name powerseeking
python analyze_vectors.py --dataset_name powerseeking

python generate_vectors.py --layers $(seq 0 39) --model_size "13b" --data_path datasets/sycophancy.json --dataset_name sycophancy
python analyze_vectors.py --dataset_name sycophancy