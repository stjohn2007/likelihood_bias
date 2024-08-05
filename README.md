# Code for paper "Likelihood-based Mitigation of Evaluation Bias in Large Language Models"

![](https://img.shields.io/badge/Made_with-python-blue.svg)
[![arXiv](https://img.shields.io/badge/arXiv-2307.11729-b31b1b.svg)](https://arxiv.org/pdf/2402.15987)
[![LICENSE](https://img.shields.io/badge/License-Apache--2.0-green.svg)](./LICENSE)

This is the official code for our ACL2024 paper, "Likelihood-based Mitigation of Evaluation Bias in Large Language Models".

## Create Python environment

```bash
conda create -n likelihood_bias python=3.9
conda activate likelihood_bias
pip install -r requirements.txt
```

## Put API KEY of OPENAI

Please copy `.env.sample` as `.env`, and put your API key on the file.

## Prepare Data

### Data2Text

```bash
cd data_orig/data2text
git clone git@github.com:WebNLG/challenge-2020.git
```

Please download input data from [here](https://gitlab.com/shimorina/webnlg-dataset/-/blob/master/release_v3.0/en/test/rdf-to-text-generation-test-data-without-refs-en.xml) and put it on `data_orig/data2text/rdf-to-text-generation-test-data-without-refs-en.xml`.

### GEC

Please download dataset from [here](https://github.com/tmu-nlp/TMU-GFM-Dataset/blob/main/tmu-gfm-dataset.csv) and put it on `data_orig/gec/tmu-gfm-dataset.csv`.

## Format Data

```bash
python format_data.py
```

This script format data and extract few-shot examples at random.
Formatted data and few-shot examples will be saved in `data`.

## Measuring likelihood bias

```bash
# Calc Score_m
python calc_evaluator_score_baseline.py -e gpt_35_turbo -d data2text
python calc_evaluator_score_baseline.py -e gpt_35_turbo -d gec
python calc_evaluator_score_baseline.py -e llama2_13b -d data2text
python calc_evaluator_score_baseline.py -e llama2_13b -d gec

# Calc likelihood score
python calc_likelihood_score.py -e llama2_13b -d data2text
python calc_likelihood_score.py -e llama2_13b -d gec
```

Due to the design of implementation, the result (BiasScore and Evaluation performance) can be calculated after mitigating likelihood bias.

## Mitigating likelihood bias

Before calculating scores we should split the data into training and evaluation data.

```bash
python split_train_eval.py
```

Then, you can run the following script and get scores under less likelihood bias.

```bash
python calc_evaluator_score_mitigation.py -e gpt_35_turbo -d data2text
python calc_evaluator_score_mitigation.py -e gpt_35_turbo -d gec
python calc_evaluator_score_mitigation.py -e llama2_13b -d data2text
python calc_evaluator_score_mitigation.py -e llama2_13b -d gec
```

Finally, you can get the BiasScore and Evaluation performance for before and after mitigation.

```bash
model_list=("gpt_35_turbo" "llama2_13b")
task_list=("data2text", "gec")
for model in "${model_list[@]}"; do
    for task in "${task_list[@]}"; do
        python calc_cor_before_mitigation.py -e $model -d $task
        python calc_cor_after_mitigation.py -e $model -d $task
    done
done
```

The results will be saved in `results/before/` and `result/after/`.

## Citation

```
@misc{ohi2024likelihoodbasedmitigationevaluationbias,
      title={Likelihood-based Mitigation of Evaluation Bias in Large Language Models}, 
      author={Masanari Ohi and Masahiro Kaneko and Ryuto Koike and Mengsay Loem and Naoaki Okazaki},
      year={2024},
      eprint={2402.15987},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2402.15987}, 
}
```