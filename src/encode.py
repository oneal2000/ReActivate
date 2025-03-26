import os
import gc
import time
import argparse
import torch
from tqdm import tqdm
from peft import TaskType, get_peft_model, LoraConfig, PeftModel,IA3Config
from torch.utils.data import Dataset
from transformers import DefaultDataCollator
from typing import Dict, List

import prompt_template
from root_dir_path import ROOT_DIR
from utils import get_model, load_data

import numpy as np
import random

seed = 42 
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
cnt=0

class TrainingData(Dataset):
    ignored_id = -100

    def __init__(self, prompt_ids, tokenizer, max_length=3000):
        self.max_length = max_length
        self.dataset = []
        pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        for input_ids in prompt_ids:
            labels = input_ids.copy()
            if len(input_ids) > max_length:
                input_ids = input_ids[:max_length]
                labels = labels[:max_length]
            attention_mask = [1] * len(input_ids) + [0] * (max_length - len(input_ids))
            input_ids += [pad_token_id] * (max_length - len(input_ids))
            labels += [self.ignored_id] * (max_length - len(labels))
            self.dataset.append({
                "input_ids": input_ids,
                "labels": labels,
                "attention_mask": attention_mask,
            })
        self.total_len = len(self.dataset)
    
    def __len__(self):
        return self.total_len
    
    def __getitem__(self, idx) -> Dict[str, list]:
        return self.dataset[idx]


class TrainingDataCollator(DefaultDataCollator):
    def __init__(self, tokenizer, device):
        super().__init__()
        self.tokenizer = tokenizer
        self.device = device
    
    def __call__(self, examples: List[Dict[str, list]]) -> Dict[str, torch.Tensor]:
        input_ids, labels, attention_mask = tuple(
            map(lambda x: [example[x] for example in examples], ["input_ids", "labels", "attention_mask"])
        )
        return {
            "input_ids": torch.tensor(input_ids).to(self.device),
            "labels": torch.tensor(labels).to(self.device),
            "attention_mask": torch.tensor(attention_mask).to(self.device),
        }
    

def get_train_data(aug_model, augments, tokenizer, args):
    from prompt_template import get_prompt
    prompt_ids = []
    for aug in augments:
        psg = aug["passage"]
        rew = aug[f"{aug_model}_rewrite"]
        qas = aug[f"{aug_model}_qa"]
        qpa_cnt = (len(qas) + 1) // 2
        for qid, qa in enumerate(qas):
            if qid < qpa_cnt:
                for ppp in [psg, rew]:
                    prompt_ids.append(get_prompt(tokenizer, qa["question"], 
                                                    [ppp], 
                                                    qa["answer"] if not args.with_cot else qa["full_answer"], 
                                                    with_cot=args.with_cot))
            else:
                prompt_ids.append(get_prompt(tokenizer, qa["question"], 
                                                None, 
                                                qa["answer"] if not args.with_cot else qa["full_answer"], 
                                                with_cot=args.with_cot))
    return prompt_ids


def train(question, augments, args, model, tokenizer, 
          init_adapter_path, save_path):
    global cnt
    cnt+=1
    prompt_ids = get_train_data(args.augment_model, augments, tokenizer, args)
    train_data = TrainingData(prompt_ids, tokenizer)
    train_dataloader = torch.utils.data.DataLoader(
        train_data,
        batch_size=args.per_device_train_batch_size,
        collate_fn=TrainingDataCollator(tokenizer, model.device),
        shuffle=False,
    )
    model = PeftModel.from_pretrained(model, init_adapter_path, is_trainable=True)
    model.is_parallelizable = True
    model.model_parallel = True
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.AdamW(model_parameters, lr=args.learning_rate)
    # time.sleep(20)
    for epoch in range(args.num_train_epochs):
        for step, batch in enumerate(train_dataloader):
            # print(step)
            optimizer.zero_grad()
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
    os.makedirs(save_path, exist_ok=True)
    model.save_pretrained(save_path)
    model = model.unload()
    torch.cuda.empty_cache()
    gc.collect()
    return model


def main(args):
    data_list = load_data(args.dataset, args.data_type, args.augment_model)
    model, tokenizer, _generation_config = get_model(args.model_name)
    # time.sleep(20)
    if args.with_cot:
        prompt_template.get_fewshot(args.dataset)
    if args.IA3:
        if args.warm_up:
            init_adapter_path=os.path.join(
            ROOT_DIR,
            "warmup",
            "IA3_base_weight",
            args.model_name,
            "IA3",
            "cot" if args.with_cot else "direct"
            )
        else:
            init_adapter_path = os.path.join(
            ROOT_DIR, 
            "offline", 
            args.model_name, 
            "IA3",
            "base_weight",
        )
    else:
        init_adapter_path = os.path.join(
            ROOT_DIR, 
            "offline", 
            args.model_name, 
            f"rank={args.lora_rank}_alpha={args.lora_alpha}",
            "base_weight",
        )
    if not os.path.exists(os.path.join(init_adapter_path, "adapter_model.safetensors")):
        if args.IA3:
            print("No IA3 base weight, creating...")
        else:
            print("No LoRA base weight, creating...")
        if not args.IA3:
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                target_modules=['down_proj', 'gate_proj', 'up_proj'],
                inference_mode=False,
                r=args.lora_rank,
                lora_alpha=args.lora_alpha,
                lora_dropout=0, # !!!
            )
        else:
            peft_config = IA3Config(
                task_type=TaskType.CAUSAL_LM,
                target_modules=['down_proj', 'gate_proj', 'up_proj'],
                feedforward_modules=['down_proj', 'gate_proj', 'up_proj'],
                inference_mode=False,
                # r=args.lora_rank,
                # lora_alpha=args.lora_alpha,
                # lora_dropout=0, # !!!
            )
        model = get_peft_model(model, peft_config)
        model.is_parallelizable = True
        model.model_parallel = True
        print(f'Save LoRA base weight to {init_adapter_path}')
        os.makedirs(init_adapter_path, exist_ok=True)
        model.save_pretrained(init_adapter_path)
        time.sleep(2)
        assert os.path.exists(os.path.join(init_adapter_path, "adapter_model.safetensors")) 

    cot_name = "cot" if args.with_cot else "direct"
    for filename, fulldata in data_list:
        filename = filename.split('.')[0] 
        print(f"### Solving {filename} ###")
        if args.IA3:
            output_dir = os.path.join(
                ROOT_DIR, 
                "offline_warmup" if args.warm_up else "offline", 
                args.model_name, 
                "IA3",
                args.dataset,
                f"lr={args.learning_rate}_epoch={args.num_train_epochs}_{cot_name}",
                f"aug_model={args.augment_model}",
                filename,
            )
        else:
            output_dir = os.path.join(
            ROOT_DIR, 
            "offline_warmup" if args.warm_up else "offline", 
            args.model_name, 
            f"rank={args.lora_rank}_alpha={args.lora_alpha}",
            args.dataset,
            f"lr={args.learning_rate}_epoch={args.num_train_epochs}_{cot_name}",
            f"aug_model={args.augment_model}",
            filename,
        )
        os.makedirs(output_dir, exist_ok=True)
        fulldata = fulldata if args.sample == -1 else fulldata[:args.sample]
        for did, data in tqdm(enumerate(fulldata), total=len(fulldata)):
            augment = data["augment"]
            for pid in range(len(augment)):
                save_path = os.path.join(output_dir, f"data_{did}", f"passage_{pid}")
                if os.path.exists(os.path.join(save_path, "adapter_model.safetensors")):
                    continue
                model = train(data["question"], [augment[pid]], args, model, tokenizer, 
                            init_adapter_path, save_path)
                

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--data_type", type=str)
    parser.add_argument("--with_cot", action="store_true")
    parser.add_argument("--sample", type=int, default=-1) # -1 means all
    parser.add_argument("--augment_model", type=str, default=None)
    # Train
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    # LoRA
    parser.add_argument("--lora_rank", type=int, default=None)
    parser.add_argument("--lora_alpha", type=int, default=None)
    parser.add_argument("--IA3", action="store_true", help="Enable IA3")
    parser.add_argument("--warm_up", action="store_true", help="Use Warmup")
    args = parser.parse_args()
    assert (args.lora_rank and args.lora_alpha) or args.IA3, "No config for LoRA or IA3"
    if args.augment_model is None:
        args.augment_model = args.model_name
    print(args)
    main(args)