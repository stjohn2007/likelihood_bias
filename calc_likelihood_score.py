import argparse
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from src.evaluator import evaluator_type2info, easy_name2model_name
from src.likelihood import llama2_ll_score_include_input


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

    likelihood_name, likelihood_method = evaluator_type2info[args.evaluator_type][1]
    result_dir = base_dir / "likelihood" / args.data_name / likelihood_name
    result_dir.mkdir(parents=True, exist_ok=True)

    likelihood_func = llama2_ll_score_include_input(model=easy_name2model_name[likelihood_name])

    aspect_list = [aspect.name for aspect in data_dir.glob("*")]
    print(f"Processing {args.data_name}...")
    # if the output path already exists, skip
    if (result_dir / f"{likelihood_method}").exists():
        print(f"{result_dir / f'{likelihood_method}'} already exists")
        return

    result = pd.DataFrame(
        columns=["source", "output", "likelihood_score"]
    )

    # load data
    score_data = pd.read_csv(data_dir / aspect_list[0] / "score.csv")

    # calc likelihood score
    for idx, row in tqdm(score_data.iterrows(), total=len(score_data)):
        source, output = row["source"], row["output"]
        likelihood_score = likelihood_func(
            input_=source, output=output, task=args.data_name, model_name=easy_name2model_name[likelihood_name]
        )
        result.loc[idx] = [source, output, likelihood_score]

    # save
    result_dir.mkdir(parents=True, exist_ok=True)
    result.to_csv(result_dir / likelihood_method, index=False)


if __name__ == "__main__":
    main()
