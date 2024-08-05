import argparse
from pathlib import Path
import os

import pandas as pd
import numpy as np
from scipy.stats import spearmanr

from src.evaluator import evaluator_type2info


def rescale_score(score_list):
    max_score = max(score_list)
    min_score = min(score_list)
    score_list = np.array(score_list)
    return ((score_list - min_score) / (max_score - min_score)).tolist()


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--evaluator_type", type=str, required=True)
    parser.add_argument("-d", "--data_name", type=str, required=True)
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    base_dir = Path("./mitigation_result") / "result"

    eval_name, eval_method = evaluator_type2info[args.evaluator_type][0]
    likelihood_name, likelihood_method = evaluator_type2info[args.evaluator_type][1]

    evaluator_dir = base_dir / "evaluator" / "mitigated" / args.data_name / f"{eval_name}-{likelihood_name}"
    likelihood_dir = base_dir / "likelihood" / args.data_name / likelihood_name

    # load data
    aspect_names = [path.name for path in evaluator_dir.glob("*") if path.is_dir()]
    likelihood_score = pd.read_csv(likelihood_dir / "test" / likelihood_method)["likelihood_score"].tolist()
    result = pd.DataFrame(
        columns=[
            "BiasScore",
            "EvaluationPerformance",
        ]
    )

    average_evaluator_score = [0] * len(likelihood_score)
    average_human_score = [0] * len(likelihood_score)

    for aspect in aspect_names:
        file_name = eval_method
        evaluator_path = evaluator_dir / aspect / file_name
        evaluator_data = pd.read_csv(evaluator_path)
        evaluator_score = evaluator_data["evaluator_score"].tolist()
        human_score = evaluator_data["human_score"].tolist()
        unfairness_score = [
            eval_score - hum_score
            for eval_score, hum_score in zip(rescale_score(evaluator_score), rescale_score(human_score))
        ]
        average_evaluator_score = [
            eval_score + tmp_eval_score for eval_score, tmp_eval_score in zip(evaluator_score, average_evaluator_score)
        ]
        average_human_score = [
            hum_score + tmp_hum_score for hum_score, tmp_hum_score in zip(human_score, average_human_score)
        ]
        line = [
            spearmanr(likelihood_score, unfairness_score)[0],
            spearmanr(evaluator_score, human_score)[0],
        ]
        result.loc[aspect] = line

    # Calc total (micro average)
    average_evaluator_score = [tmp_eval_score / len(aspect_names) for tmp_eval_score in average_evaluator_score]
    average_human_score = [tmp_hum_score / len(aspect_names) for tmp_hum_score in average_human_score]
    average_unfairness_score = [
        eval_score - hum_score
        for eval_score, hum_score in zip(rescale_score(average_evaluator_score), rescale_score(average_human_score))
    ]
    line = [
        spearmanr(likelihood_score, average_unfairness_score)[0],
        spearmanr(average_evaluator_score, average_human_score)[0],
    ]
    result.loc["average"] = line

    output_file = "cor.csv"
    output_path = (
        Path("./mitigation_result")
        / "cor"
        / "after"
        / args.data_name
        / f"{eval_name}-{likelihood_name}"
        / output_file
    )
    os.makedirs(output_path.parent, exist_ok=True)
    result.to_csv(output_path, index=True)


if __name__ == "__main__":
    main()
