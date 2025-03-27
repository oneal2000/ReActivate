# ReActivate

Code for our paper: Activation Reweighter

This repository contains the code, datasets models used in our work"Activation Re-weighting".

## Reproduce Results

In the In the following GitHub repository, we demonstrate how to test the performance of ReActivate on various QA datasets. Specifically, follow these steps to run ReActivate:

* **Run the Data Augmentation Module** : Transformes documents into a data-augmented dataset.
* **Warm up IA3** (optional) : Use the latter part of the dataset to generate base IA3 weights.
* **Generate Parametric Representations of Documents** :Train additional IA3 parameters.
* **Inference** :Merge the parametric representations of relevant documents, insert them into the LLM, and use the updated LLM for inference.

All the prompts used in the experiment are displayed in the `all_prompt.md` file.

### Install Environment

```
conda create -n prag python=3.10.4
conda activate prag
pip install torch==1.13.1
pip install -r requirements.txt
```

Please change the `ROOT_DIR` variable in `src/root_dir_path.py` to the folder address where you store ReActivate.

### Self-Augmentation

You can directly use the pre-augmented data file `data_aug.tar.gz`. To extract it, run the command `tar -xzvf data_aug.tar.gz` in your terminal.

If you want to perform data augmentation yourself, please process it as follows.

#### Prepare BM25 for retrieval

1. Download the Wikipedia dump from the [DPR repository](https://github.com/facebookresearch/DPR/blob/main/dpr/data/download_data.py#L32) using the following command

```bash
mkdir -p data/dpr
wget -O data/dpr/psgs_w100.tsv.gz https://dl.fbaipublicfiles.com/dpr/wikipedia_split/psgs_w100.tsv.gz
pushd data/dpr
gzip -d psgs_w100.tsv.gz
popd
```

2. Use Elasticsearch to index the Wikipedia dump

```bash
cd data
wget -O elasticsearch-8.15.0.tar.gz https://artifacts.elastic.co/downloads/elasticsearch/elasticsearch-8.15.0-linux-x86_64.tar.gz  # download Elasticsearch
tar zxvf elasticsearch-8.15.0.tar.gz
rm elasticsearch-8.15.0.tar.gz 
cd elasticsearch-8.15.0
nohup bin/elasticsearch &  # run Elasticsearch in background
cd ../..
python prep_elastic.py --data_path data/dpr/psgs_w100.tsv --index_name wiki  # build index
```

#### Download dataset

For 2WikiMultihopQA:

Download the [2WikiMultihopQA](https://www.dropbox.com/s/ms2m13252h6xubs/data_ids_april7.zip?e=1) dataset from its repository [https://www.dropbox.com/s/ms2m13252h6xubs/data_ids_april7.zip?e=1](https://www.dropbox.com/s/ms2m13252h6xubs/data_ids_april7.zip?e=1). Unzip it and move the folder to `data/2wikimultihopqa`.

For HotpotQA:

```bash
mkdir -p data/hotpotqa
wget -P data/hotpotqa/ http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json
```

For PopQA:

Download the [PopQA](https://github.com/AlexTMallen/adaptive-retrieval?tab=readme-ov-file#popqa) dataset from its repository [https://github.com/AlexTMallen/adaptive-retrieval/blob/main/data/popQA.tsv](https://github.com/AlexTMallen/adaptive-retrieval/blob/main/data/popQA.tsv), and put the file `popQA.tsv` into folder `data/popqa`.

```bash
mkdir -p data/popqa
wget -P data/popqa https://github.com/AlexTMallen/adaptive-retrieval/blob/main/data/popQA.tsv
```

For ComplexWebQuestions:

Download the [ComplexWebQuestions](https://www.tau-nlp.sites.tau.ac.il/compwebq) dataset from its repository [https://www.dropbox.com/scl/fo/nqujvpg2gc4y0ozkw3wgr/AOzjVEsdUhv2Fx2pamfJlSw?rlkey=746t7xehfqxf1zr867nxiq8aq&amp;e=1](https://www.dropbox.com/scl/fo/nqujvpg2gc4y0ozkw3wgr/AOzjVEsdUhv2Fx2pamfJlSw?rlkey=746t7xehfqxf1zr867nxiq8aq&e=1), and put the file `ComplexWebQuestions_dev.json` into folder `data/complexwebquestions`.

#### Data Augmentation:

```bash
python src/augment.py \
    --model_name llama3.2-1b-instruct \
    --dataset 2wikimultihopqa \
    --data_path data/2wikimultihopqa/ \
    --sample 300  \
    --topk 3
```

| **Parameter** | **Example/Options**                                                   |
| ------------------- | --------------------------------------------------------------------------- |
| `model_name`      | `llama3.2-1b-instruct`, `qwen2.5-1.5b-instruct`, `llama3-8b-instruct` |
| `dataset`         | `2wikimultihopqa`, `hotpotqa`, `popqa`, `complexwebquestions`       |
| `data_path`       | folder to the saved data, such as `data/2wikimultihopqa`                  |
| `sample`          | Number of questions to run                                                  |
| `topk`            | retrieval number                                                            |

The results of data augmentation will be stored in the file `data_aug/{dataset}/{data_type}.json`.

If you want to apply data augmentation to a new dataset, the default data format for the augmented data is JSON. Each element in the array should include both a 'question' and an 'answer,' as shown in the example below.

```json
[
    {
        "question": "string",
        "answer": "string or list[string]",
    }
]
```

At this point, the input parameter `dataset` refers to the name of the dataset you’ve set, and `data_path` is the path to the JSON file mentioned above. The last filename in `data_path` will be treated as the `data_type`. The output file will be saved in `data_aug/{your_dataset_name}/{data_type}.json`.

### Document Parameterizing

By calling the `src/encode.py` file, you will generate a parameterized representation of the documents (IA3) for the given dataset. The parameters for this file are as follows:

| **Parameter**                                                      | **Example/Options**                                                                     |
| ------------------------------------------------------------------------ | --------------------------------------------------------------------------------------------- |
| `model_name`                                                           | `llama3.2-1b-instruct`, `qwen2.5-1.5b-instruct`, `llama3-8b-instruct`                   |
| `dataset`                                                              | `2wikimultihopqa`, `hotpotqa`, `popqa`, `complexwebquestions`                         |
| `data_type`                                                            | Not set means using the entire dataset, otherwise, specify a particular data type             |
| `with_cot`                                                             | If included, generate a CoT                                                                   |
| `sample`                                                               | Number of questions to run                                                                    |
| `augment_model`                                                        | Model used for data augmentation. If not set, the current model will be used for augmentation |
| `per_device_train_batch_size`, `num_train_epochs`, `learning_rate` | Training parameters                                                                           |
| `IA3`                                                                  | if included, choose the IA3 format parameterized documents.                                   |
| `warm_up`                                                              | if included, choose the base IA3 parameters generated during the warm-up process.             |

When running for the first time with a specific IA3 parameter and "warm_up" is not choosed, an initial random parameter, `base_weight` will be created. All subsequent training will start from this base_weight. If "warm_up" is choosed, all subsequent training will start from the warmed up parameters.

All generated parameters are stored in the `offline` and `offline_warmup` folders.
The specific location of the parameter files is as follows:

```plain
offline/
├── {model_name}/
│   └── IA3/
│       ├── base_weight/
│       └── {dataset}/
│           └── lr={learning_rate}_epoch={num_train_epochs}/
│               └── aug_model={augment_model}/
│                   └── {data_type}/
│                       └── data_{did}/
│                           └── passage_{pid}/
|                               └── parameters
offline_warmup/
├── {model_name}/
│   └── IA3/
│       ├── base_weight/
│       └── {dataset}/
│           └── lr={learning_rate}_epoch={num_train_epochs}/
│               └── aug_model={augment_model}/
│                   └── {data_type}/
│                       └── data_{did}/
│                           └── passage_{pid}/
|                               └── parameters
```

The running parameters of the main experiments in the paper are listed in the `configs` folder.

### Generate

By calling the `src/inference.py` file, you will generate a parameterized representation of the documents (IA3) for the given dataset. The parameters for this file are as follows:

| **Parameter**                                                      | **Example/Options**                                                                     |
| ------------------------------------------------------------------------ | --------------------------------------------------------------------------------------------- |
| `model_name`                                                           | `llama3.2-1b-instruct`, `qwen2.5-1.5b-instruct`, `llama3-8b-instruct`                   |
| `dataset`                                                              | `2wikimultihopqa`, `hotpotqa`, `popqa`, `complexwebquestions`                         |
| `data_type`                                                            | Not set means using the entire dataset, otherwise, specify a particular data type             |
| `with_cot`                                                             | If included, generate a CoT                                                                   |
| `sample`                                                               | Number of questions to run                                                                    |
| `augment_model`                                                        | Model used for data augmentation. If not set, the current model will be used for augmentation |
| `per_device_train_batch_size`, `num_train_epochs`, `learning_rate` | Training parameters                                                                           |
| `IA3`                                                                  | if included, choose the IA3 format parameterized documents.                                   |
| `max_new_tokens`                                                       | Number of generate tokens                                                                     |
| `inference_method`                                                     | "icl" is naive RAG, "prag" is our method, and "combine" is using both methods together        |
| `doc_num`                                                              | Number of documents used in RAG.                                                              |
| `warm_up`                                                              | if included, use the parameterized documents trained from warmed-up base weights.             |

All generated results are stored in the `output` folder. The specific location of the parameter files is as follows:

```plain
output/
├── {model_name}doc_num={doc_num}(_warm_up if warm_up)/
│   └── IA3/
│       └── {dataset}/
│           └── lr={learning_rate}_epoch={num_train_epochs}/
│               └── aug_model={augment_model}/
│                   └── {inference_method}/
│                       └── {data_type}/
│                           ├── config.json
│                           ├── predict.json
│                           └── result.txt
```

Also, the running parameters of the main experiments in the paper are listed in the `configs` folder.

## Warm up IA3

After calling `python src/get_warmup_data.py`, the initialization training data for finetuning will be generated from the **latter** part of the dataset. The data generation code ensures that there is no data leakage.

Then, the following code will be used to train and generate base IA3 weights:

```bash
# the training used 600 data points 
python src/warmup_lora.py \
    --model_name llama3.2-1b-instruct \
    --per_device_train_batch_size 1 \
    --num_train_epochs 1 \
    --learning_rate 3e-4  \
    --block_size 3000 \
    --lora_rank 2 \
    --lora_alpha 32 \
    --IA3 \
    --with_cot 

# the training used 2000 data points  
python src/warmup_lora.py \
    --model_name llama3.2-1b-instruct \
    --per_device_train_batch_size 1 \
    --num_train_epochs 1 \
    --learning_rate 3e-4  \
    --block_size 3000  \
    --lora_rank 2 \
    --IA3  \
    --lora_alpha 32 \
```

## Experimental Results and corresponding Startup Scripts

The following table presents the **experimental results**, where each number corresponds to a **startup script**. Clicking on the number allows you to view the corresponding script.

The two data points in the first row respectively represent the data from the PRAG paper and the data obtained from reproduction.

In rows two through six, the three data points from left to right respectively represent the selection of 3, 2, and 1 documents.

* The second row combines the IA3 method with in-context-learning (without warm-up).
* The third row combines the IA3 method with in-context-learning (with warm-up).
* The fourth row only uses the IA3 method (without warm-up).
* The fifth row only uses the IA3 method (with warm-up).
* The sixth row only uses in-context-learning method.

On the left side of the table, parameter selections are marked. Each set of three numbers in the table corresponds to three experimental results based on the choices of `doc_num=3/2/1`.

If the `warm_up` parameter is selected, it is necessary to perform **warm-up** before proceeding with encoding and inference.

### 2WQA

|                                                       | Compare                                                                        | Bridge                                                                         | Inference                                                                      | Composition                                                                    |
| ----------------------------------------------------- | ------------------------------------------------------------------------------ | ------------------------------------------------------------------------------ | ------------------------------------------------------------------------------ | ------------------------------------------------------------------------------ |
| **LLaMA-1B**                                    | 0.5046/0.4915                                                                  | 0.4595/0.4667                                                                  | 0.2399/0.2472                                                                  | 0.1357/0.1259                                                                  |
| inference_method=combine warm_up=False doc_num=3/2/1  | [0.3849/0.3961/0.3627](configs/2wikimultihopqa_llama3.2-1b-instruct.sh)           | [0.2552/0.2276/0.2488](configs/2wikimultihopqa_llama3.2-1b-instruct.sh)           | [0.2292/0.2267/0.2342](configs/2wikimultihopqa_llama3.2-1b-instruct.sh)           | [0.0894/0.1059/0.0989](configs/2wikimultihopqa_llama3.2-1b-instruct.sh)           |
| inference_method=combine warm_up=True doc_num=3/2/1   | [0.4934/0.4944/0.4960](configs/2wikimultihopqa_llama3.2-1b-instruct.sh)           | [0.4140/0.3531/0.4127](configs/2wikimultihopqa_llama3.2-1b-instruct.sh)           | [0.2196/0.2294/0.2213](configs/2wikimultihopqa_llama3.2-1b-instruct.sh)           | [0.1367/0.1001/0.1209](configs/2wikimultihopqa_llama3.2-1b-instruct.sh)           |
| inference_method=prag warm_up=False doc_num=3/2/1     | **[0.4378/0.4389/0.4389](configs/2wikimultihopqa_llama3.2-1b-instruct.sh)** | **[0.2183/0.2216/0.2249](configs/2wikimultihopqa_llama3.2-1b-instruct.sh)** | **[0.1672/0.1630/0.1700](configs/2wikimultihopqa_llama3.2-1b-instruct.sh)** | **[0.0650/0.0656/0.0644](configs/2wikimultihopqa_llama3.2-1b-instruct.sh)** |
| inference_method=prag warm_up=True doc_num=3/2/1      | [0.4604/0.4604/0.4593](configs/2wikimultihopqa_llama3.2-1b-instruct.sh)           | [0.4332/0.4332/0.4298](configs/2wikimultihopqa_llama3.2-1b-instruct.sh)           | [0.1555/0.1588/0.1573](configs/2wikimultihopqa_llama3.2-1b-instruct.sh)           | [0.0826/0.0821/0.0761](configs/2wikimultihopqa_llama3.2-1b-instruct.sh)           |
| inference_method=icl doc_num=3/2/1                   | [0.3849/0.4274/0.3872](configs/2wikimultihopqa_llama3.2-1b-instruct.sh)           | [0.2552/0.2243/0.2521](configs/2wikimultihopqa_llama3.2-1b-instruct.sh)           | [0.2292/0.23/0.2413](configs/2wikimultihopqa_llama3.2-1b-instruct.sh)             | [0.0894/0.1118/0.0976](configs/2wikimultihopqa_llama3.2-1b-instruct.sh)           |
| **Qwen-1.5B**                                   | 0.4053/0.4050                                                                  | 0.4420/0.4400                                                                  | 0.1705/0.1630                                                                  | 0.1154/0.0817                                                                  |
| inference_method=combine warm_up=False doc_num=3/2/1 | [0.3984/0.3704/0.4302](configs/2wikimultihopqa_llama3-8b-instruct.sh)             | [0.3717/0.3775/0.4140](configs/2wikimultihopqa_llama3-8b-instruct.sh)             | [0.1234/0.1300/0.1500](configs/2wikimultihopqa_llama3-8b-instruct.sh)             | [0.0560/0.0497/0.0703](configs/2wikimultihopqa_llama3-8b-instruct.sh)             |
| inference_method=combine warm_up=True doc_num=3/2/1  | [0.3959/0.4188/0.3975](configs/2wikimultihopqa_llama3-8b-instruct.sh)             | [0.3052/0.2637/0.2267](configs/2wikimultihopqa_llama3-8b-instruct.sh)             | [0.1997/0.2199/0.2327](configs/2wikimultihopqa_llama3-8b-instruct.sh)             | [0.1366/0.1216/0.1106](configs/2wikimultihopqa_llama3-8b-instruct.sh)             |
| inference_method=prag warm_up=False doc_num=3/2/1    | **[0.4547/0.4589/0.4542](configs/2wikimultihopqa_llama3-8b-instruct.sh)**   | **[0.4249/0.4249/0.4216](configs/2wikimultihopqa_llama3-8b-instruct.sh)**   | [0.1745/0.1765/0.1790](configs/2wikimultihopqa_llama3-8b-instruct.sh)             | [0.0670/0.0670/0.0668](configs/2wikimultihopqa_llama3-8b-instruct.sh)             |
| inference_method=prag warm_up=True doc_num=3/2/1     | [0.4112/0.4096/0.4112](configs/2wikimultihopqa_llama3-8b-instruct.sh)             | [0.4418/0.4459/0.4363](configs/2wikimultihopqa_llama3-8b-instruct.sh)             | [0.1702/0.1707/0.1750](configs/2wikimultihopqa_llama3-8b-instruct.sh)             | [0.1073/0.1047/0.1063](configs/2wikimultihopqa_llama3-8b-instruct.sh)             |
| inference_method=icl doc_num=3/2/1                   | [0.3875/0.3664/0.4249](configs/2wikimultihopqa_llama3-8b-instruct.sh)             | [0.3884/0.3954/0.4108](configs/2wikimultihopqa_llama3-8b-instruct.sh)             | [0.1187/0.1280/0.1410](configs/2wikimultihopqa_llama3-8b-instruct.sh)             | [0.0568/0.0511/0.0744](configs/2wikimultihopqa_llama3-8b-instruct.sh)             |
| **LLaMA-8B**                                    |                                                                                |                                                                                |                                                                                |                                                                                |
| inference_method=prag warm_up=False doc_num=3/1      | [0.4925/0.4991](configs/2wikimultihopqa_llama3-8b-instruct.sh)                    | [0.5344/0.5210](configs/2wikimultihopqa_llama3-8b-instruct.sh)                    | [0.2522/0.2531](configs/2wikimultihopqa_llama3-8b-instruct.sh)                    | [0.1439/0.1465](configs/2wikimultihopqa_llama3-8b-instruct.sh)                    |
| inference_method=prag warm_up=True doc_num=3/1       | [0.5532/unfinished](configs/2wikimultihopqa_llama3-8b-instruct.sh)                | [0.5414/0.5506](configs/2wikimultihopqa_llama3-8b-instruct.sh)                    | [0.2495/unfinished](configs/2wikimultihopqa_llama3-8b-instruct.sh)                | [0.1928/0.1961](configs/2wikimultihopqa_llama3-8b-instruct.sh)                    |
| inference_method=icl doc_num=3/1                     | [0.5843/0.5215](configs/2wikimultihopqa_llama3-8b-instruct.sh)                    | [0.4794/0.5105](configs/2wikimultihopqa_llama3-8b-instruct.sh)                    | [0.1833/0.2104](configs/2wikimultihopqa_llama3-8b-instruct.sh)                    | [0.0991/0.1312](configs/2wikimultihopqa_llama3-8b-instruct.sh)                    |

### HotpotQA

|                                                       | Bridge                                                                  | Compare                                                                 |
| ----------------------------------------------------- | ----------------------------------------------------------------------- | ----------------------------------------------------------------------- |
| **LLaMA-1B**                                    | 0.2282/                                                                 | 0.4271/                                                                 |
| inference_method=combine warm_up=False doc_num=3/2/1 | [0.2141/0.2231/0.2042](configs/hotpotqa_llama3.2-1b-instruct.sh)           | [0.414/0.4351/0.4491](configs/hotpotqa_llama3.2-1b-instruct.sh)            |
| inference_method=combine warm_up=True doc_num=3/2/1  | [0.2222/0.198/0.2088](configs/hotpotqa_llama3.2-1b-instruct.sh)            | [0.3931/0.4024/0.3954](configs/hotpotqa_llama3.2-1b-instruct.sh)           |
| inference_method=prag warm_up=False doc_num=3/2/1    | **[0.1294/0.1294/0.1294](configs/hotpotqa_llama3.2-1b-instruct.sh)** | **[0.3799/0.3854/0.3888](configs/hotpotqa_llama3.2-1b-instruct.sh)** |
| inference_method=prag warm_up=True doc_num=3/2/1     | [0.1312/0.1296/0.1345](configs/hotpotqa_llama3.2-1b-instruct.sh)           | [0.3838/0.3808/0.3771](configs/hotpotqa_llama3.2-1b-instruct.sh)           |
| inference_method=icl doc_num=3/2/1                   | [0.2221/0.2256/0.2119](configs/hotpotqa_llama3.2-1b-instruct.sh)           | [0.4183/0.422/0.4484](configs/hotpotqa_llama3.2-1b-instruct.sh)            |
| **Qwen-1.5B**                                   | 0.2383/                                                                 | 0.5037/                                                                 |
| inference_method=combine warm_up=False doc_num=3/2/1 | [0.1641/0.1613/0.1416](configs/hotpotqa_qwen2.5-1.5b-instruct.sh)          | [0.3551/0.3049/0.2765](configs/hotpotqa_qwen2.5-1.5b-instruct.sh)          |
| inference_method=combine warm_up=True doc_num=3/2/1  | [0.2357/0.2247/0.2211](configs/hotpotqa_qwen2.5-1.5b-instruct.sh)          | [0.4045/0.3998/0.3722](configs/hotpotqa_qwen2.5-1.5b-instruct.sh)          |
| inference_method=prag warm_up=False doc_num=3/2/1    | [0.1250/0.1280/0.1280](configs/hotpotqa_qwen2.5-1.5b-instruct.sh)             | [0.4007/0.3905/0.3862](configs/hotpotqa_qwen2.5-1.5b-instruct.sh)          |
| inference_method=prag warm_up=True doc_num=3/2/1     | [0.1416/0.1496/0.1479](configs/hotpotqa_qwen2.5-1.5b-instruct.sh)          | [0.5132/0.5171/0.5064](configs/hotpotqa_qwen2.5-1.5b-instruct.sh)          |
| inference_method=icl doc_num=3/2/1                   | [0.1619/0.1549/0.1386](configs/hotpotqa_qwen2.5-1.5b-instruct.sh)          | [0.3713/0.3139/0.2832](configs/hotpotqa_qwen2.5-1.5b-instruct.sh)          |
| **LLaMA-8B**                                    |                                                                         |                                                                         |
| inference_method=prag warm_up=False doc_num=3/1      | [0.1621/0.1641](configs/hotpotqa_llama3-8b-instruct.sh)                    | [unfinished/0.5027](configs/hotpotqa_llama3-8b-instruct.sh)                |
| inference_method=prag warm_up=True doc_num=3/1       | [0.2925/0.3019](configs/hotpotqa_llama3-8b-instruct.sh)                    | [0.5950/0.6002](configs/hotpotqa_llama3-8b-instruct.sh)                    |
| inference_method=icl doc_num=3/1                     | [0.1823/0.1959](configs/hotpotqa_llama3-8b-instruct.sh)                    | [0.3493/0.4272](configs/hotpotqa_llama3-8b-instruct.sh)                    |

### PopQA

|                                                       | PopQA                                                                |
| ----------------------------------------------------- | -------------------------------------------------------------------- |
| **LLaMA-1B**                                    | 0.2961/                                                              |
| inference_method=combine warm_up=False doc_num=3/2/1 | [0.1803/0.1487/0.1751](configs/popqa_llama3.2-1b-instruct.sh)           |
| inference_method=combine warm_up=True doc_num=3/2/1  | [0.3721/0.3837/0.3576](configs/popqa_llama3.2-1b-instruct.sh)           |
| inference_method=prag warm_up=False doc_num=3/2/1    | **[0.0249/0.0263/0.0241](configs/popqa_llama3.2-1b-instruct.sh)** |
| inference_method=prag warm_up=True doc_num=3/2/1     | [0.1906/0.1939/0.1872](configs/popqa_llama3.2-1b-instruct.sh)           |
| inference_method=icl doc_num=3/2/1                   | [0.1637/0.1364/0.1637](configs/popqa_llama3.2-1b-instruct.sh)           |
| **Qwen-1.5B**                                   | 0.2261/                                                              |
| inference_method=combine warm_up=False doc_num=3/2/1 | [0.1031/0.0958/0.1041](configs/popqa_qwen2.5-1.5b-instruct.sh)          |
| inference_method=combine warm_up=True doc_num=3/2/1  | [0.3197/0.3112/0.3053](configs/popqa_qwen2.5-1.5b-instruct.sh)          |
| inference_method=prag warm_up=False doc_num=3/2/1    | [0.0268/0.0255/0.0249](configs/popqa_qwen2.5-1.5b-instruct.sh)          |
| inference_method=prag warm_up=True doc_num=3/2/1     | [0.1229/0.1229/0.1207](configs/popqa_qwen2.5-1.5b-instruct.sh)          |
| inference_method=icl doc_num=3/2/1                   | [0.0999/0.0930/0.0976](configs/popqa_qwen2.5-1.5b-instruct.sh)           |
| **LLaMA-8B**                                    |                                                                      |
| inference_method=prag warm_up=False doc_num=3/1      | [0.0964/0.1169](configs/popqa_llama3-8b-instruct.sh)                    |
| inference_method=prag warm_up=True doc_num=3/1       | [0.2884/0.3024](configs/popqa_llama3-8b-instruct.sh)                    |
| inference_method=icl doc_num=3/1                     | [0.1613/0.1776](configs/popqa_llama3-8b-instruct.sh)                    |

### CWQ

|                                                       | CWQ                                                                               |
| ----------------------------------------------------- | --------------------------------------------------------------------------------- |
| **LLaMA-1B**                                    | 0.4101/0.3842                                                                     |
| inference_method=combine warm_up=False doc_num=3/2/1 | [0.3732/0.3952/0.3967](configs/complexwebquestions_llama3.2-1b-instruct.sh)          |
| inference_method=combine warm_up=True doc_num=3/2/1   | [0.4269/0.4270/0.4329](configs/complexwebquestions_llama3.2-1b-instruct.sh)           |
| inference_method=prag warm_up=False doc_num=3/2/1    | **[0.363/0.3642/0.3636](configs/complexwebquestions_llama3.2-1b-instruct.sh)** |
| inference_method=prag warm_up=True doc_num=3/2/1     | [0.3811/0.3828/0.3828](configs/complexwebquestions_llama3.2-1b-instruct.sh)          |
| inference_method=icl doc_num=3/2/1                   | [0.3750/0.3903/0.3840](configs/complexwebquestions_llama3.2-1b-instruct.sh)            |
| **Qwen-1.5B**                                   | 0.3495/0.3209/                                                                    |
| inference_method=combine warm_up=False doc_num=3/2/1 | [0.2830/0.2977/0.2747](configs/complexwebquestions_qwen2.5-1.5b-instruct.sh)          |
| inference_method=combine warm_up=True doc_num=3/2/1  | [0.4120/0.4254/0.4228](configs/complexwebquestions_qwen2.5-1.5b-instruct.sh)          |
| inference_method=prag warm_up=False doc_num=3/2/1    | [0.2676/0.2689/0.2695](configs/complexwebquestions_qwen2.5-1.5b-instruct.sh)         |
| inference_method=prag warm_up=True doc_num=3/2/1     | [0.4574/0.4563/0.4513](configs/complexwebquestions_qwen2.5-1.5b-instruct.sh)         |
| inference_method=icl doc_num=3/2/1                   | [0.2823/0.2905/0.2710](configs/complexwebquestions_qwen2.5-1.5b-instruct.sh)          |
| **LLaMA-8B**                                    |                                                                                   |
| inference_method=prag warm_up=False doc_num=3/1      | [0.4315/0.4310](configs/complexwebquestions_llama3-8b-instruct.sh)                   |
| inference_method=prag warm_up=True doc_num=3/1        | [0.5594/0.5597](configs/complexwebquestions_llama3-8b-instruct.sh)                   |
| inference_method=icl doc_num=3/1                     | [0.3545/0.3723](configs/complexwebquestions_llama3-8b-instruct.sh)                   |
