import argparse
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from src.evaluator import (
    evaluator_openai_geval_score,
    evaluator_llama2_geval_score,
    evaluator_type2info,
    easy_name2model_name
)
from src.utils.prompt_util import BaselinePromptMaker


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--evaluator_type", type=str, required=True)
    parser.add_argument("-d", "--data_name", type=str, required=True)
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    base_dir = Path("./baseline_result")
    data_dir = Path("./data") / args.data_name
    prompt_dir = Path("./prompt") / args.data_name

    eval_name, eval_method = evaluator_type2info[args.evaluator_type][0]
    result_dir = base_dir / "evaluator" / args.data_name / eval_name
    result_dir.mkdir(parents=True, exist_ok=True)

    # set evaluator function
    if eval_name == "gpt_35_turbo":
        evaluator_func = evaluator_openai_geval_score
    else:
        evaluator_func = evaluator_llama2_geval_score(easy_name2model_name[eval_name])

    aspect_list = [aspect.name for aspect in data_dir.glob("*")]
    print(f"Processing {args.data_name}...")
    for aspect_name in aspect_list:
        print(f"Processing {aspect_name}...")
        # if the output path already exists, skip
        output_path = result_dir / aspect_name
        if (output_path / f"{eval_method}").exists():
            print(f"{output_path / f'{eval_method}'} already exists")
            print(f"Skip {aspect_name}")
            continue

        # set prompt maker
        prompt_maker = BaselinePromptMaker(
            prompt_path=prompt_dir / aspect_name,
            example_path=data_dir / aspect_name / "example.csv",
        )
        result = pd.DataFrame(
            columns=["source", "output", "human_score", "evaluator_score"]
        )

        # load data
        score_data = pd.read_csv(data_dir / aspect_name / "score.csv")
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
        result.to_csv(output_path / eval_method, index=False)


if __name__ == "__main__":
    main()
