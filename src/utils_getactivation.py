import os
import re
import json
import torch
import string
import numpy as np
from collections import Counter
from typing import List, Union
from transformers import AutoModelForCausalLM, AutoTokenizer
from torchvision.models.feature_extraction import create_feature_extractor
from root_dir_path import ROOT_DIR
from prompt_template import get_prompt

DATA_ROOT_DIR = os.path.join(ROOT_DIR, "data_aug")

class BaseDataset:
    @classmethod
    def normalize_answer(cls, s):
        def remove_articles(text):
            return re.sub(r'\b(a|an|the)\b', ' ', text)
        def white_space_fix(text):
            return ' '.join(text.split())
        def remove_punc(text):
            exclude = set(string.punctuation)
            return ''.join(ch for ch in text if ch not in exclude)
        def lower(text):
            return text.lower()
        return white_space_fix(remove_articles(remove_punc(lower(s))))

    @classmethod
    def exact_match_score(
        cls,
        prediction: str,
        ground_truth: Union[str, List[str]],
        ground_truth_id: Union[str, List[str]] = None
    ):
        ground_truths = {ground_truth} if isinstance(ground_truth, str) else set(ground_truth)
        if ground_truth_id and isinstance(ground_truth_id, str):
            ground_truths.update(cls.get_all_alias(ground_truth_id))

        correct = np.max([int(cls.normalize_answer(prediction) == cls.normalize_answer(gt)) for gt in ground_truths])
        return {'correct': correct, 'incorrect': 1 - correct}

    @classmethod
    def f1_score(
        cls,
        prediction: str,
        ground_truth: Union[str, List[str]],
        ground_truth_id: Union[str, List[str]] = None
    ):
        ground_truths = {ground_truth} if isinstance(ground_truth, str) else set(ground_truth)
        if ground_truth_id and isinstance(ground_truth_id, str):
            ground_truths.update(cls.get_all_alias(ground_truth_id))
            
        final_metric = {'f1': 0, 'precision': 0, 'recall': 0}
        for ground_truth in ground_truths:
            normalized_prediction = cls.normalize_answer(prediction)
            normalized_ground_truth = cls.normalize_answer(ground_truth)
            if normalized_prediction in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_ground_truth:
                continue
            if normalized_ground_truth in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_ground_truth:
                continue
            prediction_tokens = normalized_prediction.split()
            ground_truth_tokens = normalized_ground_truth.split()
            common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
            num_same = sum(common.values())
            if num_same == 0:
                continue

            precision = 1.0 * num_same / len(prediction_tokens)
            recall = 1.0 * num_same / len(ground_truth_tokens)
            f1 = (2 * precision * recall) / (precision + recall)
            for k in ['f1', 'precision', 'recall']:
                final_metric[k] = max(eval(k), final_metric[k])
        return final_metric


def load_data(data_name, data_type, model_name):
    solve_dataset = []
    input_dir = os.path.join(DATA_ROOT_DIR, data_name, model_name)
    files = [f for f in os.listdir(input_dir)]


    if len(files) > 1: # more types in dataset
        if data_type == "total": # merge all types to total
            all_data = {}
            for filename in files:
                with open(os.path.join(input_dir, filename), "r") as fin:
                    all_data[filename] = json.load(fin)
            total_data = []
            idx = {filename: 0 for filename in files}
            for data in all_data["total.json"]:
                typ = data["type"] + ".json"
                if idx[typ] == len(all_data[typ]):
                    break 
                aim_data = all_data[typ][idx[typ]]
                assert aim_data["question"] == data["question"]
                idx[typ] += 1
                total_data.append(aim_data)
            return [["total.json", total_data]]
        for filename in files:
            if filename != "total.json":
                with open(os.path.join(input_dir, filename), "r") as fin:
                    solve_dataset.append((filename, json.load(fin)))
        if data_type is None:
            return solve_dataset
        else:
            data_type = data_type + ".json"
            if data_type not in [v[0] for v in solve_dataset]:
                raise ValueError(f"Invalid {data_type} in Dataset {data_name}")
            tmp = []
            for filename, dataset in solve_dataset:
                if filename == data_type:
                    tmp.append((filename, dataset))
            return tmp
    else:
        with open(os.path.join(input_dir, "total.json"), "r") as fin:
            solve_dataset.append(("total.json", json.load(fin)))
        return solve_dataset
    

def get_model_path(model_name):
    if model_name == "llama3-8b-instruct": 
        return "/liuzyai04/thuir/LLM/Meta-Llama-3-8B-Instruct"
    elif model_name == "qwen2.5-1.5b-instruct":
        return "/liuzyai04/thuir/tyc/LLM/Qwen2.5-1.5B-Instruct"
    elif model_name == "llama3.2-1b-instruct":
        return "/liuzyai04/thuir/tyc/LLM/Llama-3.2-1B-Instruct"
    else:
        return model_name


def get_model(model_name, max_new_tokens=20):
    model_path = get_model_path(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, 
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
        device_map="auto", 
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    generation_config = dict(
        num_beams=1, 
        do_sample=False,
        max_new_tokens=max_new_tokens,
        return_dict_in_generate=True,
        pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0,
    )
    return model, tokenizer, generation_config

# -------------------------------- for augmentation ----------------------------------------

def model_generate(prompt, model, tokenizer, generation_config):
    messages = [{
        'role': 'user', 
        'content': prompt,
    }]
    input_ids = tokenizer.apply_chat_template(
        messages, 
        add_generation_prompt=True
    )
    input_len = len(input_ids)
    input_ids = torch.tensor(input_ids).unsqueeze(0).to(model.device)
    output = model.generate(
        input_ids, 
        attention_mask = torch.ones(input_ids.shape).to(model.device),
        **generation_config
    )
    output = output.sequences[0][input_len:]
    text = tokenizer.decode(output, skip_special_tokens=True)
    return text

# ------------------------------------------------------------------------------------

def read_complete(filepath):
    try:
        with open(filepath, "r") as fin:
            data = json.load(fin)
        return data, len(data)
    except:
        return [], 0

    
def evaluate(pred, ground_truth, with_cot=False):
    if not with_cot:
        pred = pred.strip()
        stop_list = [".", "\n", ","]
        for stop in stop_list:
            end_pos = pred.find(stop)
            if end_pos != -1:
                pred = pred[:end_pos].strip()
    else:
        if "the answer is" in pred:
            pred = pred[pred.find("the answer is") + len("the answer is"):]
        pred = pred.strip()
        stop_list = [".", "\n", ","]
        for stop in stop_list:
            end_pos = pred.find(stop)
            if end_pos != -1:
                pred = pred[:end_pos].strip() 

    em = BaseDataset.exact_match_score(
        prediction=pred,
        ground_truth=ground_truth,
    )["correct"]
    f1_score = BaseDataset.f1_score(
        prediction=pred,
        ground_truth=ground_truth,
    )
    f1, prec, recall = f1_score["f1"], f1_score["precision"], f1_score["recall"]
    return {
        "eval_predict": pred,
        "em": str(em),
        "f1": str(f1),
        "prec": str(prec),
        "recall": str(recall),
    }


def predict(model, tokenizer, generation_config, question, with_cot, passages = None):
    # for name, module in model.named_modules():
    #     print(name, module)
    activation = {}
    def get_activation(name):
        def hook(model, input, output):
            # activation[name] = output.detach()  # 保存中间层输出
            activation[f'{name}_input'] = input[0].detach().squeeze(0)# 输入通常是元组的第一个元素
            # activation[f'{name}_output'] = output.detach().squeeze(0)
        return hook
    # output_dict={x:f"{x}_output" for x in model.state_dict()}
    # output_dict={"base_model.model.model.layers.15.mlp.up_proj.base_layer.weight":"1"}
    # feature_extractor = create_feature_extractor(
    #     model, 
    #     output_dict
    # )
    # model.model.model.layers[0].self_attn.register_forward_hook(get_activation('layer0_self_attn_output'))
    for i in range(16):
        model.model.model.layers[i].mlp.gate_proj.register_forward_hook(get_activation(f'layer{i}_mlp_gate_proj'))
        model.model.model.layers[i].mlp.up_proj.register_forward_hook(get_activation(f'layer{i}_mlp_up_proj'))
        model.model.model.layers[i].mlp.down_proj.register_forward_hook(get_activation(f'layer{i}_mlp_down_proj'))
    #model.model.model.layers[0].mlp.gate_proj.register_forward_hook(get_activation(f'layer{0}_mlp_gate_proj_output'))
    # target=model.model.model.layers[0].mlp.gate_proj
    model.eval()
    input_ids = get_prompt(
        tokenizer, 
        question, 
        passages = passages, 
        with_cot = with_cot)
    input_len = len(input_ids)
    # print(input_len)
    input_ids = torch.tensor(input_ids).unsqueeze(0).to(model.device)
    with torch.no_grad():
        # features = feature_extractor(input_ids)
        # test=target(input_ids)
        # print(test)
        output = model(input_ids)
        # print(activation)
        # print(output)
    # text = tokenizer.decode(output, skip_special_tokens=True)
    # print("=================================")
    # print (activation.keys())
    return activation