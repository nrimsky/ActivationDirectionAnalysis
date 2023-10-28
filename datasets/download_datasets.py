import requests
import json
from random import sample

urls = {
    "sycophancy": "https://huggingface.co/datasets/Anthropic/model-written-evals/raw/main/sycophancy/sycophancy_on_political_typology_quiz.jsonl",
    "powerseeking": "https://huggingface.co/datasets/Anthropic/model-written-evals/raw/main/advanced-ai-risk/human_generated_evals/power-seeking-inclination.jsonl",
    "agreeableness": "https://huggingface.co/datasets/Anthropic/model-written-evals/raw/main/persona/agreeableness.jsonl",
    "extraversion": "https://huggingface.co/datasets/Anthropic/model-written-evals/raw/main/persona/extraversion.jsonl",
    "conscientiousness": "https://huggingface.co/datasets/Anthropic/model-written-evals/raw/main/persona/conscientiousness.jsonl",
}
N_SAMPLES = 1000


def get_data(n_samples, urls):
    for name, url in urls.items():
        r = requests.get(url).text
        raw_data = [json.loads(l) for l in r.split("\n") if l != ""]
        data = []
        for d in raw_data:
            am = d["answer_matching_behavior"].strip().replace(")", "")
            anm = d["answer_not_matching_behavior"].strip().replace(")", "")
            question = d["question"].replace("Question:", "").replace("Answer:", "").strip()
            data.append(
                {
                    "answer_matching_behavior": am,
                    "answer_not_matching_behavior": anm,
                    "question": question,
                }
            )
        if len(data) > n_samples:
            data = sample(data, n_samples)
        with open(f"{name}.json", "w") as f:
            json.dump(data, f)


if __name__ == "__main__":
    get_data(N_SAMPLES, urls)
