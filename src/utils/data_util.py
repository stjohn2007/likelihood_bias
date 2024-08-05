from pathlib import Path
import json
import random
import xml.etree.ElementTree as ET

import pandas as pd


RANDOM_STATE = 42
random.seed(RANDOM_STATE)


def rescale_human_score(score: float, task: str):
    if task == "data2text":
        if score < 15:
            score = 1
        elif score < 40:
            score = 2
        elif score < 65:
            score = 3
        elif score < 90:
            score = 4
        else:
            score = 5

    return score


def format_gec(data_path: str, save_path: str, shot_num: int):
    aspects = [("grammar", "ave_g"), ("fluency", "ave_f")]
    score_range = [0, 1, 2, 3, 4]
    data_path = Path(data_path)
    save_path = Path(save_path)
    data = pd.read_csv(data_path)
    data = data.rename(columns={"grammer": "grammar"})
    for aspect, ave_aspect in aspects:
        new_data = pd.DataFrame(columns=["source", "output", "score", "detail_score"])
        new_data["source"] = data["source"]
        new_data["output"] = data["output"]
        new_data["score"] = data[ave_aspect]
        new_data["detail_score"] = data[aspect]

        few_shot_data = new_data.sample(n=shot_num, random_state=RANDOM_STATE)
        rest_data = new_data.drop(few_shot_data.index)
        few_shot_data.reset_index(drop=True, inplace=True)
        rest_data.reset_index(drop=True, inplace=True)

        (save_path / aspect).mkdir(parents=True, exist_ok=True)
        rest_data.to_csv(save_path / aspect / "score.csv", index=False)
        few_shot_data.to_csv(save_path / aspect / "example.csv", index=False)

        with open(save_path / aspect / "score_range.txt", "w") as f:
            f.write("\n".join([str(score) for score in score_range]))


def format_data2text(data_path: str, save_path: str, shot_num: int):
    aspects = [
        ("Correctness", [1, 2, 3, 4, 5]),
        ("DataCoverage", [1, 2, 3, 4, 5]),
        ("Fluency", [1, 2, 3, 4, 5]),
        ("Relevance", [1, 2, 3, 4, 5]),
        ("TextStructure", [1, 2, 3, 4, 5]),
    ]

    save_path = Path(save_path)
    ipt_path = Path(data_path) / "rdf-to-text-generation-test-data-without-refs-en.xml"

    # load inputs
    inputs = []
    tree = ET.parse(ipt_path)
    root = tree.getroot()

    for entry in root.findall(".//entry"):
        modified_tripleset = entry.find("modifiedtripleset")
        if modified_tripleset is not None:
            new_entry = ET.Element("entry", attrib=entry.attrib)
            new_entry.append(modified_tripleset)
            inputs.append(ET.tostring(new_entry, encoding="unicode"))

    # load outputs and get scores
    hyp_dir_path = Path(data_path) / "challenge-2020" / "submissions" / "rdf2text" / "en"
    eval_dir_path = Path(data_path) / "challenge-2020" / "evaluation" / "human-evaluation" / "results" / "en"

    for aspect, score_range in aspects:
        new_data = pd.DataFrame(columns=["source", "output", "score", "id", "model_name"])
        for hyp_model_path in hyp_dir_path.iterdir():
            if not hyp_model_path.is_dir():
                continue
            model_name = hyp_model_path.name
            hyp_path = hyp_model_path / "primary.en"
            hyp_list = [line.strip() for line in open(hyp_path)]
            eval_path = eval_dir_path / model_name / "primary.json"
            eval_dict = json.load(open(eval_path))
            for ref_id in eval_dict.keys():
                ipt = inputs[int(ref_id) - 1]
                hyp = hyp_list[int(ref_id) - 1]
                eval_ = eval_dict[ref_id]
                if ipt.strip() == "" or hyp.strip() == "" or eval_ is None or len(eval_) == 0:
                    continue
                score = 0
                for value in eval_.values():
                    score += value[aspect]
                score /= len(eval_)
                new_data.loc[len(new_data)] = [ipt, hyp, score, ref_id, model_name]

        few_shot_data = new_data.sample(n=shot_num, random_state=RANDOM_STATE)
        rest_data = new_data.drop(few_shot_data.index)
        few_shot_data.reset_index(drop=True, inplace=True)
        rest_data.reset_index(drop=True, inplace=True)

        # rescale the score of examples
        for i in range(len(few_shot_data)):
            original_score = few_shot_data.loc[i, "score"]
            score = rescale_human_score(original_score, "data2text")
            few_shot_data.loc[i, "score"] = score

        (save_path / aspect).mkdir(parents=True, exist_ok=True)
        rest_data.to_csv(save_path / aspect / "score.csv", index=False)
        few_shot_data.to_csv(save_path / aspect / "example.csv", index=False)
        with open(save_path / aspect / "score_range.txt", "w") as f:
            f.write("\n".join([str(score) for score in score_range]))
