# Shopping MMLU
This is the repository for 'Shopping MMLU: A Massive Multi-Task Online Shopping Benchmark for Large Language Models', which is accepted by **NeurIPS 2024 Datasets and Benchmarks Track** and used for [Amazon KDD Cup 2024](https://www.aicrowd.com/challenges/amazon-kdd-cup-2024-multi-task-online-shopping-challenge-for-llms). Shopping MMLU is a massive multi-task benchmark for LLMs on online shopping, covering four major shopping skills, **shopping concept understanding**, **shopping knowledge reasoning**, **user behavior alignment**, and **multi-lingual abilities**.

<img width="1604" alt="image" src="https://github.com/user-attachments/assets/38b8784e-34cb-4add-81f8-538eb91ee1e0">

You can find more detailed information about the dataset in the following links:
- The paper and supplementary materials [here](https://arxiv.org/pdf/2410.20745).
- The workshop of our KDD Cup Challenge and winning solutions [here](https://amazon-kddcup24.github.io).
- The HuggingFace Leaderboard [here](https://huggingface.co/spaces/KL4805/shopping_mmlu_leaderboard).

## Repo Organization
```
.
├── data:                You will need to create this folder and put the evaluation data in it.
├── skill_wise_eval:     This folder contains code for evaluating a skill as a whole.
├── task_wise_eval:      This folder contains code for evaluating a single task.
└── README.md
```

## Data
### Where to download?
The zipfile `data.zip` contains all data in Shopping MMLU. Create a new folder `data`, and unzip the zipfile in it.

### Data formats
We have five different types of tasks, **multiple choice**, **retrieval**, **ranking**, **named entity recognition**, and **generation**.

Files for multiple choice questions are organized in `.csv` formats with three columns.
- `question`: The question of the multiple choice.
- `choices`: The possible choices (4 in total) of this multiple choice.
- `answer`: The answer (within 0, 1, 2, and 3), indicating that the correct answer is `choices[answer]`.

Files for other types of tasks are organized in `.json` formats with two fields, `input_field` and `target_field`.

## Running evaluations


### Setup
First let's set up the Habana docker image

```
docker run -d -it --runtime=habana --name shopping -v ~/ShoppingMMLU:/shopping -e HABANA_VISIBLE_DEVICES=all -e OMPI_MCA_btl_vader_single_copy_mechanism=none --cap-add=sys_nice --ipc=host --net=host -e HF_HOME=/data/huggingface vault.habana.ai/gaudi-docker/1.21.1/ubuntu24.04/habanalabs/pytorch-installer-2.6.0:latest /bin/bash

docker exec -it shopping /bin/bash
```

Then inside your docker container you will want to go to /shopping and install the requirements:

```
cd /shopping
pip install -r requirements.txt
```

Then you will need to set up the data for the testing.
```
mkdir data
unzip data.zip -d data/
```

Note that the requirements.txt file uses the correct versions of the libraries for PyTorch 2.6.0. Always use the pytorch version associated with your docker image, e.g. if running Synapse v1.19.0 you'd want to be running Pytorch 2.5.1 instead (and then would need to adjust the versions of transformers, sentence_transformers etc you are using). Also, Docker synapse versions are generally backwards compatible across a reasonable range of versions: the host OS Synapse version can be a bit ahead of the docker image and everything will still work, but be wary of having a Docker image that is a Synapse version ahead of the host OS, that will often lead to trouble.

### Evaluation on a Single Task
Suppose you want to evaluate `Vicuna-7B-v1.5` model on the `multiple_choice` task of `asin_compatibility`, you can do the following steps.
```
cd task_wise_eval/
python3 hf_multi_choice.py --test_subject asin_compatibility --model_name vicuna2
# The 'model_name' argument should be set according to 'utils.py'.
```
Other tasks in other task types involve similar processes.
### Evaluation on a Skill as a whole
Suppose you want to evaluate `Vicuna-7B-v1.5` model on the skill of `skill1_concept`, you can do the following steps.
```
cd skill_wise_eval/
python3 hf_skill_inference.py --model_name vicuna2 --filename skill1_concept --output_filename <your_filename>
# After inference, the output file will be saved at `skill_inference_results/skill1_concept/vicuna2_<your_filename>.json`.
python3 skill_evaluation.py --data_filename skill1_concept --output_filename vicuna2_<your_filename>
# After evaluation, the metrics will be saved at `skill_metrics/skill1_concept/vicuna2_<your_filename>_metrics.json`.
```
Other skills involve similar processes.

### Dependencies
Our evaluation code is based on HuggingFace `transformers` with the following dependencies.
```
transformers==4.49.0
pandas==2.0.3
evaluate==0.4.1
sentence_transformers==3.2.0
rouge_score==0.1.2
accelerate==0.34.2
neural-compressor[pt]==3.3.1
sacrebleu==2.4.1
sacrebleu[jp]
```

## Code Conversion
To convert code from regular PyTorch to run on Gaudi chips you need to make the following changes.

1. As documented here: [GPU Migration Toolkit](https://docs.habana.ai/en/latest/PyTorch/PyTorch_Model_Porting/GPU_Migration_Toolkit/GPU_Migration_Toolkit.html) the GPU Migration Toolkit makes it easy to adapt existing code to run on Habana. In this case the [first commit] (https://github.com/KL4805/ShoppingMMLU/commit/1e73692024d9eb4e362103152631c1559c8bad18) adapted all of the code to at least run on Gaudi chips.

2. To add support for FP8 quantization, you will need the [Intel Neural Compressor] (https://github.com/intel/neural-compressor). You can see the changes necessary to add support for a single file in [this commit] (https://github.com/KL4805/ShoppingMMLU/commit/1b1befe4206ab661f2b86493cffe6e05ee8dea67). There are three functions necessary for this: convert, prepare, and finalize_quantization. prepare() converts a model to track what quantization is necessary, finalize_quantization() saves the results of that model being run on some sample data to a quantization file, and convert uses the existing quantization file (calculated by the prepare function, stored in the finalize_quantization() function) in an actual model to get actual results.

3. Finally, adding support for models too large to fit onto a single card requires the Deepspeed tool to be included to run across multiple cards- also if you want to speed up computation by using multiple cards compute power. As you can see from the [the main commit](https://github.com/KL4805/ShoppingMMLU/commit/4a430c7e7ab82a35d6b3aea26b0660f1d95df12d) there needs to be a lot of boilerplate code. Most of this is unrelated to the actual multi-node running. The problem is that because of the Deepspeed architecture (which runs separate Python processes- running identical code- for each Gaudi we are trying to run on) we don't want to run some things (most importantly, I/O tasks like downloading models, printing out results, etc.) on every process, so we have to suppress anything that doesn't run on the 0th logical card from doing those tasks. That requires some ugly code- e.g. we can't rely on the standard Huggingface library to deal with a model, we have to pull that code into our code so we can ensure that it is done exactly once.

There are various other commits cleaning up bugs or adding convenance features like a requirements.txt or docker files, etc. Do be sure to check the full gaudi_main branch to get all of those bug fixes and convenance, but these three commits each demonstrate the core of the work to add that feature.

## Reference
```
@article{jin2024shopping,
  title={Shopping MMLU: A Massive Multi-Task Online Shopping Benchmark for Large Language Models},
  author={Jin, Yilun and Li, Zheng and Zhang, Chenwei and Cao, Tianyu and Gao, Yifan and Jayarao, Pratik and Li, Mao and Liu, Xin and Sarkhel, Ritesh and Tang, Xianfeng and others},
  journal={arXiv preprint arXiv:2410.20745},
  year={2024}
}
```
