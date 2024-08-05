from dotenv import load_dotenv

import torch

from src.utils.llama2_util import load_llama2

load_dotenv()


class llama2_ll_score_include_input:
    def __init__(self, model, **kwargs):
        model, tokenizer = load_llama2(model)
        self.model = model
        self.tokenizer = tokenizer

    def __call__(self, input_, output, task, **kwargs) -> int:
        def create_prompt(input_, output, task) -> "tuple[str, int]":
            if task == "gec":
                prompt = f"Please modify the following English text to make it grammatically correct:\n\n{input_}\n\n"
            elif task == "data2text":
                prompt = f"Please generate a description of the following xml data:\n\n{input_}\n\n"
            else:
                raise ValueError("Invalid task type")
            prompt_token_len = len(self.tokenizer.encode(prompt))
            return f"{prompt}{output}", prompt_token_len

        prompt, head_len = create_prompt(input_, output, task)
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt")
        labels = -100 * torch.ones_like(input_ids)
        labels[0, head_len:] = input_ids[0, head_len:]
        input_ids = input_ids.to("cuda")
        labels = labels.to("cuda")
        self.model.eval()

        with torch.no_grad():
            outputs = self.model(input_ids, labels=labels)
            loss = outputs.loss
            loss = loss.detach().cpu().numpy()

        return -loss
