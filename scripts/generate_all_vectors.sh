if [ ! -d "venv" ]; then
    echo "Creating venv"
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
else
    source venv/bin/activate
fi

python generate_vectors.py --layers $(seq 0 31) --model_size "7b" --data_path datasets/agreeableness.json --dataset_name agreeableness
python generate_vectors.py --layers $(seq 0 31) --model_size "7b" --use_base_model --data_path datasets/agreeableness.json --dataset_name agreeableness
python analyze_vectors.py --dataset_name agreeableness

python generate_vectors.py --layers $(seq 0 31) --model_size "7b" --data_path datasets/sycophancy.json --dataset_name sycophancy
python generate_vectors.py --layers $(seq 0 31) --model_size "7b" --use_base_model --data_path datasets/sycophancy.json --dataset_name sycophancy
python analyze_vectors.py --dataset_name sycophancy

python generate_vectors.py --layers $(seq 0 31) --model_size "7b" --data_path datasets/selfawareness.json --dataset_name selfawareness
python generate_vectors.py --layers $(seq 0 31) --model_size "7b" --use_base_model --data_path datasets/selfawareness.json --dataset_name selfawareness
python analyze_vectors.py --dataset_name selfawareness

python generate_vectors.py --layers $(seq 0 31) --model_size "7b" --data_path datasets/myopic.json --dataset_name myopic
python generate_vectors.py --layers $(seq 0 31) --model_size "7b" --use_base_model --data_path datasets/myopic.json --dataset_name myopic
python analyze_vectors.py --dataset_name myopic

python generate_vectors.py --layers $(seq 0 31) --model_size "7b" --data_path datasets/conscientiousness.json --dataset_name conscientiousness
python generate_vectors.py --layers $(seq 0 31) --model_size "7b" --use_base_model --data_path datasets/conscientiousness.json --dataset_name conscientiousness
python analyze_vectors.py --dataset_name conscientiousness

python generate_vectors.py --layers $(seq 0 31) --model_size "7b" --data_path datasets/extraversion.json --dataset_name extraversion
python generate_vectors.py --layers $(seq 0 31) --model_size "7b" --use_base_model --data_path datasets/extraversion.json --dataset_name extraversion
python analyze_vectors.py --dataset_name extraversion

python generate_vectors.py --layers $(seq 0 31) --model_size "7b" --data_path datasets/powerseeking.json --dataset_name powerseeking
python generate_vectors.py --layers $(seq 0 31) --model_size "7b" --use_base_model --data_path datasets/powerseeking.json --dataset_name powerseeking
python analyze_vectors.py --dataset_name powerseeking

python generate_vectors.py --layers $(seq 0 31) --model_size "7b" --data_path datasets/survivalinstinct.json --dataset_name survivalinstinct
python generate_vectors.py --layers $(seq 0 31) --model_size "7b" --use_base_model --data_path datasets/survivalinstinct.json --dataset_name survivalinstinct
python analyze_vectors.py --dataset_name survivalinstinct