# python -m eagle.application.webui --ea-model-path /data1/zzh/huggingface/hub/models--yuhuili--EAGLE-mixtral-instruct-8x7B/snapshots/f2e9cd1e1efaf0dec41c2da1b1fae4327727871d --base-model-path /data1/zzh/huggingface/hub/models--mistralai--Mixtral-8x7B-Instruct-v0.1/snapshots/41bd4c9e7e4fb318ca40e721131d4933966c2cc1 --model-type mixtral --total-token -1
import torch
import sys
sys.path.append("/home/zzh/codes/SpecMoE/EagleTest/EAGLE")
from eagle.model.ea_model import EaModel
from fastchat.model import get_conversation_template
import torch.autograd.profiler as profiler
from torch.profiler import profile, record_function, ProfilerActivity
from datasets import load_dataset
from transformers import DataCollatorForLanguageModeling, AutoTokenizer
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
import numpy as np

draft_token_number = 10

model = EaModel.from_pretrained(
    base_model_path="/data1/zzh/huggingface/hub/models--mistralai--Mixtral-8x7B-Instruct-v0.1/snapshots/41bd4c9e7e4fb318ca40e721131d4933966c2cc1",
    ea_model_path="/data1/zzh/huggingface/hub/models--yuhuili--EAGLE-mixtral-instruct-8x7B/snapshots/f2e9cd1e1efaf0dec41c2da1b1fae4327727871d",
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
    device_map="auto",
    use_eagle3=False,
    total_token=draft_token_number
)
model.eval()

tokenizer = model.get_tokenizer()
# tokenizer = AutoTokenizer.from_pretrained("/data1/zzh/huggingface/hub/models--mistralai--Mixtral-8x7B-Instruct-v0.1/snapshots/41bd4c9e7e4fb318ca40e721131d4933966c2cc1", use_fast=False)
seq_len = 128
tokenizer.pad_token = tokenizer.eos_token
dataset = load_dataset("cnn_dailymail", "1.0.0", split="test[0:1000]")
num_eval_steps = 200
for i in tqdm(range(num_eval_steps)):
# for i in range(num_eval_steps):
    sequence = dataset[i]["article"]
    inputs_id = tokenizer(sequence, return_tensors='pt',max_length=seq_len,padding=True,truncation=True)
    input_ids = inputs_id['input_ids'][..., :128]
    attention_mask = inputs_id['attention_mask'][..., :128]
    output_ids=model.eagenerate(input_ids,temperature=0.5,max_new_tokens=128)

acc_length_list = model.get_accept_length_logger()
# print(acc_length_list)
print(np.mean(acc_length_list))
savefile = "/home/zzh/codes/SpecMoE/EagleTest/EAGLE/mytest/log.txt"
with open(savefile, "a") as f:
    f.write(f"draft token: {draft_token_number}, mean accept length: {np.mean(acc_length_list)}\n")


# your_message="Hello, do you know where is Nanjing University Xianlin Campus?"
# conv = get_conversation_template("mixtral")
# conv.append_message(conv.roles[0], your_message)
# conv.append_message(conv.roles[1], None)
# prompt = conv.get_prompt()
# input_ids=model.tokenizer([prompt]).input_ids
# input_ids = torch.as_tensor(input_ids).cuda()

# # with profile(
# #   activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], 
# #   on_trace_ready=torch.profiler.tensorboard_trace_handler('/home/zzh/codes/SpecMoE/EagleTest/EAGLE/logs'), 
# #   profile_memory = True,
# # ):
# output_ids=model.eagenerate(input_ids,temperature=0.5,max_new_tokens=10)
# output=model.tokenizer.decode(output_ids[0])
# print(output)