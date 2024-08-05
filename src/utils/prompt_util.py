from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.data_util import rescale_human_score


# Set the average to 0, and rescale the scores to [-1, 1]
def rescale_scores(data):
    # Step 1: Calculate mean, max and min of the data
    data_mean = sum(data) / len(data)
    data_max = max(data)
    data_min = min(data)

    # Step 2: Subtract the mean from the data
    zero_mean_data = [x - data_mean for x in data]

    # Step 3: Rescale the data to the range [-1, 1]
    scale = max(abs(data_max - data_mean), abs(data_min - data_mean))
    rescaled_data = [x / scale for x in zero_mean_data]
    return rescaled_data


class PromptMaker:
    def __init__(self, prompt_path):
        self.prompt_path = Path(prompt_path)
        self.base_prompt = self._load_base_prompt()
        self.input_prompt, self.output_prompt = self._load_template()
        self.examples = self._load_examples()

    def _load_base_prompt(self):
        with open(self.prompt_path / "base.txt", "r") as f:
            base_prompt = f.read()
        return base_prompt

    def _load_template(self):
        with open(self.prompt_path / "input.txt", "r") as f:
            input_prompt = f.read()

        with open(self.prompt_path / "output.txt", "r") as f:
            output_prompt = f.read()
        return input_prompt, output_prompt

    def make_prompt(self, source, output):
        query = (
            self.input_prompt.replace("{{input}}", source).replace("{{output}}", output)
            + "\n"
            + self.output_prompt.replace("{{score}}", "")
        )
        return self.base_prompt + "\n\n\n" + self.examples + "\n" + query

    def make_prompt_with_score(self, source, output, score):
        query = (
            self.input_prompt.replace("{{input}}", source).replace("{{output}}", output)
            + "\n"
            + self.output_prompt.replace("{{score}}", str(score))
        )
        return self.base_prompt + "\n\n\n" + self.examples + "\n" + query


class BaselinePromptMaker(PromptMaker):
    def __init__(self, prompt_path, example_path):
        self.prompt_path = Path(prompt_path)
        self.base_prompt = self._load_base_prompt()
        self.input_prompt, self.output_prompt = self._load_template()
        self.examples = self._load_examples(example_path)

    def _load_examples(self, example_path):
        example_data = pd.read_csv(example_path)
        examples = []
        for _, row in example_data.iterrows():
            score = round(row["score"])
            input_prompt = self.input_prompt.replace("{{input}}", row["source"]).replace("{{output}}", row["output"])
            output_prompt = self.output_prompt.replace("{{score}}", str(score))
            examples.append((input_prompt, output_prompt))

        return examples

    def make_system_prompt(self):
        return self.base_prompt

    def make_example_prompt(self):
        return self.examples

    def make_query_prompt(self, source, output):
        return (
            self.input_prompt.replace("{{input}}", source).replace("{{output}}", output)
            + "\n"
            + self.output_prompt.replace("{{score}}", "")
        )

    def make_output_prompt(self):
        return self.output_prompt


class DebiasPromptMaker(PromptMaker):
    def __init__(
        self,
        prompt_path,
        eval_train_path,
        likelihood_train_path,
        task_name,
    ):
        eval_data = pd.read_csv(eval_train_path)
        self.eval_data = eval_data
        likelihood_data = pd.read_csv(likelihood_train_path)
        self.likelihood_data = likelihood_data
        self.task_name = task_name

        self.prompt_path = Path(prompt_path)
        self.base_prompt = self._load_base_prompt()
        self.input_prompt, self.output_prompt = self._load_template()
        self.examples = self._load_examples()

    def _calc_abs_score(self):
        DS_list = self.likelihood_data["likelihood_score"].to_list()
        rescaled_DS = rescale_scores(DS_list)
        ES_list = self.eval_data["evaluator_score"].to_list()
        rescaled_ES = rescale_scores(ES_list)
        HS_list = self.eval_data["human_score"].to_list()
        rescaled_HS = rescale_scores(HS_list)
        score_diff = [ES - HS for ES, HS in zip(rescaled_ES, rescaled_HS)]
        rescaled_score_diff = rescale_scores(score_diff)
        bias_score = [DS + score_diff for DS, score_diff in zip(rescaled_DS, rescaled_score_diff)]
        bias_score = rescale_scores(bias_score)
        return bias_score

    def _load_examples(self):
        bias_score = self._calc_abs_score()
        bias_score = np.array([abs(score) for score in bias_score])
        example_ids = bias_score.argsort()[-8:].tolist()
        examples = []
        for example_id in example_ids:
            example_source, example_output, example_score = self.eval_data.loc[example_id][
                ["source", "output", "human_score"]
            ]
            example_score = rescale_human_score(example_score, self.task_name)
            example_score = round(example_score)
            examples.append(
                (
                    self.input_prompt.replace("{{input}}", example_source).replace("{{output}}", example_output),
                    self.output_prompt.replace("{{score}}", str(example_score)),
                )
            )
        return examples

    def make_system_prompt(self):
        return self.base_prompt

    def make_query_prompt(self, source, output):
        return (
            self.input_prompt.replace("{{input}}", source).replace("{{output}}", output)
            + "\n"
            + self.output_prompt.replace("{{score}}", "")
        )

    def make_example_prompt(self):
        return self.examples
