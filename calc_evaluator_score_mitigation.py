import argparse
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from src.evaluator import (
    evaluator_openai_geval_score,
    evaluator_llama2_geval_score,
    evaluator_type2info,
    easy_name2model_name,
)
from src.utils.prompt_util import DebiasPromptMaker


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--evaluator_type", type=str, required=True)
    parser.add_argument("-d", "--data_name", type=str, required=True)
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    data_dir = Path("./mitigation_result/data/") / args.data_name
    prompt_dir = Path("./prompt") / args.data_name
    output_dir_name = "mitigated"

    eval_name, eval_method = evaluator_type2info[args.evaluator_type][0]
    likelihood_name, likelihood_method = evaluator_type2info[args.evaluator_type][1]
    result_dir = Path("./mitigation_result/result/") / "evaluator" / output_dir_name / args.data_name / f"{eval_name}-{likelihood_name}"
    result_dir.mkdir(parents=True, exist_ok=True)

    # set evaluator function
    if eval_name == "gpt_35_turbo":
        evaluator_func = evaluator_openai_geval_score
    else:
        evaluator_func = evaluator_llama2_geval_score(easy_name2model_name[eval_name])

    aspect_list = [aspect.name for aspect in data_dir.glob("*") if aspect.is_dir()]
    print(f"Processing {args.data_name}...")
    for aspect_name in aspect_list:
        print(f"Processing {aspect_name}...")
        # if the output path already exists, skip
        output_path = result_dir / aspect_name
        output_file = eval_method
        if (output_path / output_file).exists():
            print(f"{output_path / f'{output_file}'} already exists")

        # set prompt maker
        # use baseline result as eval train
        eval_train_path = (
            Path("./mitigation_result/result/evaluator/baseline")
            / args.data_name
            / eval_name
            / aspect_name
            / "train"
            / eval_method
        )
        likelihood_train_path = (
            Path("./mitigation_result/result/likelihood")
            / args.data_name
            / likelihood_name
            / "train"
            / likelihood_method
        )
        prompt_maker = DebiasPromptMaker(
            prompt_path=prompt_dir / aspect_name,
            eval_train_path=eval_train_path,
            likelihood_train_path=likelihood_train_path,
            task_name=args.data_name,
        )

        result = pd.DataFrame(columns=["source", "output", "human_score", "evaluator_score"])

        # load data
        score_data = pd.read_csv(data_dir / aspect_name / "test_score.csv")
        with open(data_dir / aspect_name / "score_range.txt", "r") as f:
            score_range = [int(score.strip()) for score in f.readlines()]

        # evaluate
        for idx, row in tqdm(score_data.iterrows(), total=len(score_data)):
            source, output, human_score = row["source"], row["output"], row["score"]
            # make prompt
            base_prompt = prompt_maker.make_system_prompt()
            query_prompt = prompt_maker.make_query_prompt(source, output)
            example_prompt = prompt_maker.make_example_prompt()
            evaluator_score = evaluator_func(
                base_prompt=base_prompt,
                example_prompt=example_prompt,
                query_prompt=query_prompt,
                scores=score_range,
                model=easy_name2model_name[eval_name],
            )
            result.loc[idx] = [source, output, human_score, evaluator_score]

        # save
        output_path.mkdir(parents=True, exist_ok=True)
        result.to_csv(output_path / output_file, index=False)


if __name__ == "__main__":
    main()
