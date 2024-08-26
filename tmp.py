# import transformers
# import torch
#
# model_id = "meta-llama/Meta-Llama-3.1-8B"
#
# pipeline = transformers.pipeline(
#     "text-generation", model=model_id, model_kwargs={"torch_dtype": torch.bfloat16}, device_map="auto"
# )
#
# pipeline("Hey how are you doing today?")
import torch

model = torch.load('/home/youssef/.llama/checkpoints/Meta-Llama3.1-8B/consolidated.00.pth')
