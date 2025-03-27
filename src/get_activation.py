import os
import gc
import json
import argparse
import string
import torch
from tqdm import tqdm
from peft import PeftModel
import pdb;
import prompt_template
from root_dir_path import ROOT_DIR
from utils_getactivation import get_model, evaluate, predict, load_data, read_complete

def main(args):
    doc_sig=""
    warmup_sig=""
    if args.doc_num!=3:
        doc_sig="doc_num="+str(args.doc_num)
    if args.warm_up:
        warmup_sig="_warmup"
    data_list = load_data(args.dataset, args.data_type, args.augment_model)
    model, tokenizer, generation_config = get_model(
        args.model_name,
        max_new_tokens = args.max_new_tokens,
    )
    if args.with_cot:
        prompt_template.get_fewshot(args.dataset)
    
    cot_name = "cot" if args.with_cot else "direct"
    if args.IA3:
        load_adapter_path = os.path.join(
        ROOT_DIR, 
        f"offline{warmup_sig}", 
        args.model_name, 
        "IA3",
        args.dataset,
        f"lr={args.learning_rate}_epoch={args.num_train_epochs}_{cot_name}",
        f"aug_model={args.augment_model}",
    )
        output_root_dir = os.path.join(
        ROOT_DIR, 
        "new_output",
        args.model_name, 
        f"IA3{doc_sig}{warmup_sig}",
        args.dataset,
        f"lr={args.learning_rate}_epoch={args.num_train_epochs}_{cot_name}",
        f"aug_model={args.augment_model}",
        args.inference_method, 
    )
    else:
        load_adapter_path = os.path.join(
            ROOT_DIR, 
            f"offline{warmup_sig}",
            args.model_name, 
            f"rank={args.lora_rank}_alpha={args.lora_alpha}",
            args.dataset,
            f"lr={args.learning_rate}_epoch={args.num_train_epochs}_{cot_name}",
            f"aug_model={args.augment_model}",
        )
        output_root_dir = os.path.join(
            ROOT_DIR, 
            "new_output",
            args.model_name, 
            f"rank={args.lora_rank}_alpha={args.lora_alpha}{doc_sig}{warmup_sig}",
            args.dataset,
            f"lr={args.learning_rate}_epoch={args.num_train_epochs}_{cot_name}",
            f"aug_model={args.augment_model}",
            args.inference_method, 
        )
    device = torch.device("cuda:3")
    # gate_input = torch.empty(0, 2048).to(device)
    # gate_output = torch.empty(0, 2048).to(device)
    # up_input = torch.empty(0, 2048).to(device)
    # up_output = torch.empty(0, 2048).to(device)
    # down_input = torch.empty(0, 8192).to(device)
    # down_output = torch.empty(0, 8192).to(device)
    all_pairs={}
    for filename, fulldata in data_list:
        filename = filename.split(".")[0]
        print(f"### Solving {filename} ###")
        output_dir = os.path.join(output_root_dir, filename)
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, "config.json"), "w") as fout:
            json.dump(vars(args), fout, indent=4)

        predict_file = os.path.join(output_dir, "predict.json")
        ret, start_with = read_complete(predict_file)
        start_with=0#调试
        fulldata = fulldata[start_with:] if args.sample == -1 else fulldata[start_with:args.sample]
        for test_id, data in tqdm(enumerate(fulldata), total=len(fulldata)):
            test_id = test_id + start_with
            # assert test_id == len(ret), f"test_id {test_id} != len(ret) {len(ret)}"

            question = data["question"]
            passages = data["passages"]
            answer = data["answer"]
            passages=passages[:args.doc_num]
            def get_pred(model, psgs):
                
                features = predict(model, tokenizer, generation_config, 
                                        question, with_cot=args.with_cot, 
                                        passages=psgs)
                # pred = {
                #     "test_id": test_id, 
                #     "question": question, 
                #     "answer": answer, 
                #     "text": text,
                # }
                # pred.update(evaluate(text, answer, args.with_cot))
                return features

            if args.inference_method == "icl":
                ret.append(get_pred(model, psgs=passages))
            else:
                for pid in range(len(passages)):
                    adapter_path = os.path.join(load_adapter_path, filename, f"data_{test_id}", f"passage_{pid}")
                    if pid == 0:
                        model, tokenizer, generation_config = get_model(
                            args.model_name,
                            max_new_tokens = args.max_new_tokens,
                        )
                        model = PeftModel.from_pretrained(
                            model, 
                            adapter_path,
                            adapter_name = "0", 
                            is_trainable = False
                        )
                    else:
                        model.load_adapter(adapter_path, adapter_name = str(pid)) 
                if args.IA3:
                        model.add_weighted_adapter(
                        adapters = [str(i) for i in range(len(passages))], 
                        weights = [1/args.doc_num]* len(passages) if args.IA3 else [1] * len(passages),
                        adapter_name = "merge", 
                    )
                else:
                    model.add_weighted_adapter(
                        adapters = [str(i) for i in range(len(passages))], 
                        weights = [1] * len(passages),
                        adapter_name = "merge", 
                        combination_type = "cat",
                    )

                model.set_adapter("merge")
                # if test_id==0:
                # # if test_id==5 or test_id==6 or test_id==0 or test_id==1 or test_id==2:
                #     # print("Adapters merged:", [str(i) for i in range(len(passages))])
                #     # for name in model.state_dict(): 
                #     #     print(name) # 直接索引某一层的name来输出该层的参数 print(model.state_dict()['1.weight'])
                #     for name,parameters in model.state_dict().items():
                #         if "proj" in name :
                #             print (name,parameters.shape)

                
                all_features=get_pred(model, psgs=None if args.inference_method == "prag" else passages)
                gate_tmp_input=[v.to(device) for k,v in all_features.items() if "gate" in k]
                up_tmp_input=[v.to(device) for k,v in all_features.items() if "up" in k]
                down_tmp_input=[v.to(device) for k,v in all_features.items() if "down" in k]
                gate_ia3=[v.to(device) for k,v in model.state_dict().items() if ("gate" in k and "merge" in k)]
                up_ia3=[v.to(device) for k,v in model.state_dict().items() if ("up" in k and "merge" in k)]
                down_ia3=[v.to(device) for k,v in model.state_dict().items() if ("down" in k and "merge" in k)]

                gate_tmp_output=[gate_tmp_input[i]*gate_ia3[i] for i in range(len(gate_tmp_input))]
                down_tmp_output=[down_tmp_input[i]*down_ia3[i] for i in range(len(down_tmp_input))]
                up_tmp_output=[up_tmp_input[i]*up_ia3[i] for i in range(len(up_tmp_input))]
                pairs=[]
                for i in range(len(gate_tmp_input)):
                    tmp={}
                    tmp["layer_name"]=f"layer{i}_mlp_gate_proj"
                    tmp["before_IA3"]=gate_tmp_input[i].tolist()
                    tmp["after_IA3"]=gate_tmp_output[i].tolist()
                    tmp["IA3_Vector"]=gate_ia3[i].tolist()
                    pairs.append(tmp)
                    tmp={}
                    tmp["layer_name"]=f"layer{i}_mlp_up_proj"
                    tmp["before_IA3"]=up_tmp_input[i].tolist()
                    tmp["after_IA3"]=up_tmp_output[i].tolist()
                    tmp["IA3_Vector"]=up_ia3[i].tolist()
                    pairs.append(tmp)
                    tmp={}
                    tmp["layer_name"]=f"layer{i}_mlp_down_proj"
                    tmp["before_IA3"]=down_tmp_input[i].tolist()
                    tmp["after_IA3"]=down_tmp_output[i].tolist()
                    tmp["IA3_Vector"]=down_ia3[i].tolist()
                    pairs.append(tmp)
                all_pairs[test_id]=pairs
                output_dir = os.path.join(output_root_dir, "activation.json")
                with open (output_dir,"w") as fout:
                    json.dump(all_pairs, fout, indent=4)
                # gate_tmp_output=torch.cat(gate_tmp_output,dim=0)
                # up_tmp_output=torch.cat(up_tmp_output,dim=0)
                # down_tmp_output=torch.cat(down_tmp_output,dim=0)
                # print(gate_tmp_output.shape)
                # print(up_tmp_output.shape)
                # print(down_tmp_output.shape)
                # gate_output=torch.cat([gate_output,gate_tmp_output],dim=0)
                # up_output=torch.cat([up_output,up_tmp_output],dim=0)
                # down_output=torch.cat([down_output,down_tmp_output],dim=0)

                # gate_tmp_input=torch.cat(gate_tmp_input,dim=0)
                # up_tmp_input=torch.cat(up_tmp_input,dim=0)
                # down_tmp_input=torch.cat(down_tmp_input,dim=0)

                # gate_input=torch.cat([gate_input,gate_tmp_input],dim=0)
                # up_input=torch.cat([up_input,up_tmp_input],dim=0)
                # down_input=torch.cat([down_input,down_tmp_input],dim=0)


                model.delete_adapter("merge")
                model = model.unload()
                torch.cuda.empty_cache()
                gc.collect()

        # all_pairs["gate_input"]=gate_input.tolist()
        # all_pairs["gate_output"]=gate_output.tolist()
        # all_pairs["up_input"]=up_input.tolist()
        # all_pairs["up_output"]=up_output.tolist()
        # all_pairs["down_input"]=down_input.tolist()
        # all_pairs["down_output"]=down_output.tolist()
        # with open (f"get_{args.dataset}_activate_new.txt","w") as fout:
        #     json.dump(all_pairs, fout, indent=4)
        # return
        #     with open(predict_file, "w") as fout:
        #         json.dump(ret, fout, indent=4)

        # ##### Evaluating #####
        # metrics = ["em", "f1", "prec", "recall"]
        # ret_str = ""
        # for met in metrics:
        #     acc = sum(float(d[met]) for d in ret) / len(ret)
        #     acc = round(acc, 4)
        #     ret_str += f"{met}\t{acc}\n"
        # ret_str += "\n" + json.dumps(vars(args), indent=4)
        # with open(os.path.join(output_dir, "result.txt"), "w") as fout:
        #     fout.write(ret_str)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--max_new_tokens", type=int, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--data_type", type=str)
    parser.add_argument("--with_cot", action="store_true")
    parser.add_argument("--sample", type=int, default=-1) # -1 means all
    parser.add_argument("--augment_model", type=str, default=None)  
    parser.add_argument("--num_train_epochs", type=int, required=True)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--inference_method", type=str, required=True, choices=["icl", "prag", "combine"])
    # LoRA
    parser.add_argument("--lora_rank", type=int)
    parser.add_argument("--lora_alpha", type=int)
    parser.add_argument("--doc_num", type=int,default=3)
    parser.add_argument("--IA3", action="store_true", help="Enable IA3")
    parser.add_argument("--warm_up", action="store_true", help="Use Warmup")
    args = parser.parse_args()
    assert (args.lora_rank and args.lora_alpha) or args.IA3, "No config for LoRA or IA3"
    assert (args.doc_num>=0) and (args.doc_num<=3),"Illegal document number"
    if args.augment_model is None:
        args.augment_model = args.model_name
    print(args)
    main(args)