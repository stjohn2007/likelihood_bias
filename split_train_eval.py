from pathlib import Path
import shutil

import pandas as pd
from sklearn.model_selection import train_test_split

from src.evaluator import evaluator_type2info

RANDOM_STATE = 42

def main():
    ratio = 0.2
    data_dir = Path("./data")
    baseline_result_dir = Path("./baseline_result")
    mitigation_result_dir = Path("./mitigation_result")
    data_name_list = ["gec", "data2text"]
    eval_model_list = ["llama2_13b", "gpt_35_turbo"]

    for data_name in data_name_list:
        data_path = data_dir / data_name
        aspect_list = [a.name for a in data_path.iterdir() if a.is_dir()]
        for aspect in aspect_list:
            # split and save base data
            score_path = data_path / aspect / "score.csv"
            score_df = pd.read_csv(score_path)
            train_score, test_score = train_test_split(score_df, test_size=ratio, random_state=RANDOM_STATE)
            base_save_dir = mitigation_result_dir / "data" / data_path.name / aspect
            base_save_dir.mkdir(parents=True, exist_ok=True)
            train_score.to_csv(base_save_dir / "train_score.csv", index=False)
            test_score.to_csv(base_save_dir / "test_score.csv", index=False)
            # copy score_range.txt
            shutil.copy(data_path / aspect / "score_range.txt", base_save_dir)

            for eval_model in eval_model_list:
                eval_info, likelihood_info = evaluator_type2info[eval_model]
                eval_name, eval_method = eval_info
                likelihood_name, likelihood_method = likelihood_info

                # load evaluator score
                evaluator_path = baseline_result_dir / "evaluator" / data_path.name / eval_name / aspect / eval_method
                if not evaluator_path.exists():
                    print(f"{evaluator_path} does not exist. Skip.")
                    continue
                evaluator_df = pd.read_csv(evaluator_path)

                # load likelihood score
                likelihood_path = baseline_result_dir / "likelihood" / data_path.name / likelihood_name / likelihood_method
                if not likelihood_path.exists():
                    print(f"{likelihood_path} does not exist. Skip.")
                    continue
                likelihood_df = pd.read_csv(likelihood_path)

                # split evaluator/likelihood score
                train_evaluator_score = evaluator_df.iloc[train_score.index]
                test_evaluator_score = evaluator_df.iloc[test_score.index]
                train_likelihood_score = likelihood_df.iloc[train_score.index]
                test_likelihood_score = likelihood_df.iloc[test_score.index]

                # save evaluator/likelihood score
                evaluator_save_dir = mitigation_result_dir / "result" / "evaluator" / "baseline" / data_path.name / eval_name / aspect
                (evaluator_save_dir / "train").mkdir(parents=True, exist_ok=True)
                (evaluator_save_dir / "test").mkdir(parents=True, exist_ok=True)
                train_evaluator_score.to_csv(evaluator_save_dir / "train" / eval_method, index=False)
                test_evaluator_score.to_csv(evaluator_save_dir / "test" / eval_method, index=False)
                likelihood_save_dir = mitigation_result_dir / "result" / "likelihood" / data_path.name / likelihood_name
                (likelihood_save_dir / "train").mkdir(parents=True, exist_ok=True)
                (likelihood_save_dir / "test").mkdir(parents=True, exist_ok=True)
                train_likelihood_score.to_csv(likelihood_save_dir / "train" / likelihood_method, index=False)
                test_likelihood_score.to_csv(likelihood_save_dir / "test" / likelihood_method, index=False)
    return


if __name__ == "__main__":
    main()
