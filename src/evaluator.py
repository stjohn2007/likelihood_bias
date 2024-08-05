import time
import os

import openai
import numpy as np
from scipy.special import softmax
import torch
from dotenv import load_dotenv

from src.utils.llama2_util import load_llama2

load_dotenv()

openai.organization = os.environ.get("OPENAI_ORGANIZATION")
openai.api_key = os.environ.get("OPENAI_API_KEY")


evaluator_type2info = {
    "gpt_35_turbo": (("gpt_35_turbo", "geval.csv"), ("llama2_13b", "include_input.csv")),  # change here
    "llama2_13b": (("llama2_13b", "geval.csv"), ("llama2_13b", "include_input.csv")),
}

easy_name2model_name = {
    "gpt_35_turbo": "gpt-3.5-turbo-instruct",
    "llama2_13b": "meta-llama/Llama-2-13b-hf",
}


def evaluator_openai_geval_score(
    base_prompt: str,
    example_prompt: "list[tuple[str]]",
    query_prompt: str,
    scores=[1, 2, 3, 4, 5],
    model="gpt-3.5-turbo-instruct",
    **kwargs,
):
    score_weights = []
    prompt = (
        base_prompt
        + "\n"
        + "\n".join([user + "\n" + assistant for user, assistant in example_prompt])
        + "\n"
        + query_prompt
    )
    kwargs = {
        "engine": model,
        "temperature": 0,
        "top_p": 0,
        "max_tokens": 3,
        "logprobs": 5,
    }
    try:
        r = openai.Completion.create(prompt=prompt, **kwargs)
    except openai.error.RateLimitError as e:
        print("RateLimitError")
        time.sleep(60)
        r = openai.Completion.create(prompt=prompt, **kwargs)
    except openai.error.Timeout as e:
        print("Timeout")
        time.sleep(60)
        r = openai.Completion.create(prompt=prompt, **kwargs)
    logprobs = r["choices"][0]["logprobs"]["top_logprobs"][0]
    for score in scores:
        if str(score) in logprobs:
            score_weights.append(logprobs[str(score)])
        else:
            score_weights.append(np.NINF)
    score_weights = np.array(score_weights)
    score_weights = softmax(score_weights)
    geval_score = np.sum(score_weights * scores)
    return geval_score


class evaluator_llama2_geval_score:
    def __init__(self, model, **kwargs):
        model, tokenizer = load_llama2(model)
        self.model = model
        self.tokenizer = tokenizer

    def __call__(
        self,
        base_prompt: str,
        example_prompt: "list[tuple[str]]",
        query_prompt: str,
        scores=[1, 2, 3, 4, 5],
        **kwargs,
    ):
        # give prompts to llama2 and parse the response
        prompt = (
            base_prompt
            + "\n"
            + "\n".join([user + "\n" + assistant for user, assistant in example_prompt])
            + "\n"
            + query_prompt
        )
        score_ids = [self.tokenizer.encode(str(score))[-1] for score in scores]
        inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda")
        inputs_len = inputs.input_ids.shape[1]
        self.model.eval()

        score_weights = []
        with torch.no_grad():
            outputs = self.model(inputs.input_ids)

            for score_id in score_ids:
                score_weights.append(outputs.logits[0, inputs_len - 1, score_id].item())
        score_weights = np.array(score_weights)
        score_weights = softmax(score_weights)
        scores = np.array(scores)

        geval_score = float(np.sum(score_weights * scores))
        return geval_score
