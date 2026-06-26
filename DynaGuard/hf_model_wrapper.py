# import json
# import os
# import time

# import torch
# from tqdm import tqdm
# from transformers import AutoModelForCausalLM, AutoTokenizer

# from constants import (
#     COT_OPENING_QWEN,
#     GUARDREASONER_COT_OPENING,
#     GUARDREASONER_LABEL_OPENING,
#     LABEL_OPENING,
#     LLAMAGUARD_LABEL_OPENING,
#     NEG_LABEL,
#     NEMOGUARD_LABEL_OPENING,
#     POS_LABEL,
#     SHIELDGEMMA_LABEL_OPENING,
#     WILDGUARD_LABEL_OPENING,
# )


# class ComplianceProjectError(ValueError):
#     pass


# class ModelWrapper:
#     def get_message_template(self, system_content=None, user_content=None, assistant_content=None):
#         message = []
#         if system_content is not None:
#             message.append({"role": "system", "content": system_content})
#         if user_content is not None:
#             message.append({"role": "user", "content": user_content})
#         if assistant_content is not None:
#             message.append({"role": "assistant", "content": assistant_content})
#         if not message:
#             raise ComplianceProjectError("No content provided for any role.")
#         return message

#     def apply_chat_template(self, system_content=None, user_content=None, assistant_content=None, enable_thinking=None):
#         if assistant_content is None:
#             assistant_content = COT_OPENING_QWEN + "\n" if enable_thinking else LABEL_OPENING + "\n"
#         return self.get_message_template(system_content, user_content, assistant_content)


# class LocalModelWrapper(ModelWrapper):
#     def __init__(self, model_name, temperature=0.6, top_k=20, top_p=0.95, min_p=0, max_new_tokens=1000, custom_name=None):
#         self.model_name = custom_name or model_name
#         self.temperature = temperature
#         self.top_k = top_k
#         self.top_p = top_p
#         self.min_p = min_p
#         self.max_new_tokens = max_new_tokens
#         if "nemoguard" in model_name:
#             self.tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
#         else:
#             self.tokenizer = AutoTokenizer.from_pretrained(model_name)
#         self.tokenizer.pad_token_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id

#     def apply_chat_template(self, system_content, user_content=None, assistant_content=None, enable_thinking=True):
#         if assistant_content is not None:
#             assert "wildguard" not in self.model_name.lower(), (
#                 f"Gave assistant_content of {assistant_content} to model {self.model_name} but this type of model can only take a system prompt and that is it."
#             )
#             message = self.get_message_template(system_content, user_content, assistant_content)
#             try:
#                 prompt = self.tokenizer.apply_chat_template(message, tokenize=False, continue_final_message=True)
#             except ValueError as e:
#                 if "continue_final_message is set" in str(e):
#                     prompt = self.tokenizer.apply_chat_template(message, tokenize=False, continue_final_message=False)
#                     if "<|im_end|>\n" in prompt[-11:]:
#                         prompt = prompt[:-11]
#                 else:
#                     raise ComplianceProjectError(f"Error applying chat template: {e}")
#         else:
#             if "qwen3" in self.model_name.lower() or "dynaguard" in self.model_name.lower():
#                 if enable_thinking:
#                     message = self.get_message_template(system_content, user_content)
#                     prompt = self.tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True, enable_thinking=True)
#                     prompt = prompt + f"\n{COT_OPENING_QWEN}"
#                 else:
#                     message = self.get_message_template(system_content, user_content, assistant_content=f"{LABEL_OPENING}\n")
#                     prompt = self.tokenizer.apply_chat_template(message, tokenize=False, continue_final_message=True, enable_thinking=False)
#             elif "guardreasoner" in self.model_name.lower():
#                 assistant_content = GUARDREASONER_COT_OPENING if enable_thinking else GUARDREASONER_LABEL_OPENING
#                 message = self.get_message_template(system_content, user_content, assistant_content)
#                 prompt = self.tokenizer.apply_chat_template(message, tokenize=False, continue_final_message=True)
#             elif "wildguard" in self.model_name.lower():
#                 if enable_thinking:
#                     prompt = f"<s><|user|>\n[INST] {system_content} [/INST]\n<|assistant|>"
#                 else:
#                     prompt = f"<s><|user|>\n[INST] {system_content} [/INST]\n<|assistant|>{WILDGUARD_LABEL_OPENING}"
#             elif "llama-guard" in self.model_name.lower():
#                 if enable_thinking:
#                     prompt = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>{system_content}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
#                 else:
#                     prompt = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>{system_content}<|eot_id|><|start_header_id|>assistant<|end_header_id|>{LLAMAGUARD_LABEL_OPENING}"
#             elif "nemoguard" in self.model_name.lower():
#                 if enable_thinking:
#                     prompt = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>{system_content}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
#                 else:
#                     prompt = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>{system_content}<|eot_id|><|start_header_id|>assistant<|end_header_id|>{NEMOGUARD_LABEL_OPENING}"
#             elif "shieldgemma" in self.model_name.lower():
#                 if enable_thinking:
#                     prompt = f"<bos>{system_content}"
#                 else:
#                     prompt = f"<bos>{system_content}{SHIELDGEMMA_LABEL_OPENING}"
#             elif "mistral" in self.model_name.lower():
#                 assistant_content = COT_OPENING_QWEN + "\n" if enable_thinking else LABEL_OPENING + "\n"
#                 sys_user_combined = f"{system_content}\n\n{user_content}"
#                 message = self.get_message_template(user_content=sys_user_combined, assistant_content=assistant_content)
#                 prompt = self.tokenizer.apply_chat_template(message, tokenize=False, continue_final_message=True)
#             else:
#                 assistant_content = COT_OPENING_QWEN + "\n" if enable_thinking else LABEL_OPENING + "\n"
#                 message = self.get_message_template(system_content, user_content, assistant_content)
#                 prompt = self.tokenizer.apply_chat_template(message, tokenize=False, continue_final_message=True)
#         return prompt


# class HfModelWrapper(LocalModelWrapper):
#     def __init__(self, model_name, temperature=0.6, top_k=20, top_p=0.95, min_p=0, max_new_tokens=1000, custom_name=None, batch_size=8):
#         super().__init__(model_name, temperature, top_k, top_p, min_p, max_new_tokens, custom_name)
#         self.model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype=torch.bfloat16).eval()
#         self.batch_size = batch_size
#         self.tokenizer.padding_side = "left"

#     def get_prediction_probs(self, messages, strict=False, pos_label=POS_LABEL, neg_label=NEG_LABEL):
#         batch_size = self.batch_size
#         pos_token_id = self.tokenizer.encode(pos_label, add_special_tokens=False)[0]
#         neg_token_id = self.tokenizer.encode(neg_label, add_special_tokens=False)[0]

#         prob_pairs = []
#         logit_pairs = []
#         for start in tqdm(range(0, len(messages), batch_size), desc="HF prediction batches"):
#             batch_messages = messages[start : start + batch_size]
#             inputs = self.tokenizer(batch_messages, return_tensors="pt", padding=True).to(self.model.device)
#             with torch.no_grad():
#                 logits = self.model(**inputs).logits

#             # Tokenizer is configured for left padding, so the final real token
#             # for every sequence is always at the last time step in the batch.
#             prediction_logits = logits[:, -1, :]
#             prediction_probs = torch.nn.functional.softmax(prediction_logits, dim=-1)

#             for row_idx in range(prediction_logits.shape[0]):
#                 row_logits = prediction_logits[row_idx]
#                 row_probs = prediction_probs[row_idx]
#                 prob_pairs.append((row_probs[pos_token_id].item(), row_probs[neg_token_id].item()))
#                 logit_pairs.append((row_logits[pos_token_id].item(), row_logits[neg_token_id].item()))

#         return prob_pairs, logit_pairs

#     def get_responses(self, messages, temperature=None, top_k=None, top_p=None, logit_bias_dict=None, stream_output_path=None, stream_sample_id=None, stream_metadata=None):
#         if logit_bias_dict is not None:
#             token = list(logit_bias_dict.keys())[0]
#             bias = float(logit_bias_dict[token])
#             token_ids = self.tokenizer.encode(token, add_special_tokens=False)
#             logit_bias_dict = {tuple(token_ids): bias}

#         outputs = [None] * len(messages)
#         completed_indices = set()
#         if stream_output_path is not None and os.path.exists(stream_output_path):
#             with open(stream_output_path, "r", encoding="utf-8") as f:
#                 for line in f:
#                     line = line.strip()
#                     if not line:
#                         continue
#                     try:
#                         record = json.loads(line)
#                     except json.JSONDecodeError:
#                         # Ignore a partial final line if the process was interrupted mid-write.
#                         continue
#                     if record.get("sample_round") != stream_sample_id:
#                         continue
#                     index = record.get("index")
#                     if isinstance(index, int) and 0 <= index < len(messages):
#                         outputs[index] = record.get("output")
#                         completed_indices.add(index)

#         batch_size = self.batch_size
#         pending_indices = [i for i in range(len(messages)) if i not in completed_indices]
#         if pending_indices:
#             print(f"HF resume: {len(completed_indices)}/{len(messages)} already done for round {stream_sample_id}.")
#         for start in tqdm(range(0, len(pending_indices), batch_size), desc="HF generation batches"):
#             batch_indices = pending_indices[start : start + batch_size]
#             batch_messages = [messages[i] for i in batch_indices]
#             inputs = self.tokenizer(batch_messages, return_tensors="pt", padding=True).to(self.model.device)
#             with torch.no_grad():
#                 output_content = self.model.generate(
#                     **inputs,
#                     max_new_tokens=self.max_new_tokens,
#                     num_return_sequences=1,
#                     temperature=temperature or self.temperature,
#                     top_k=top_k or self.top_k,
#                     top_p=top_p or self.top_p,
#                     min_p=self.min_p,
#                     pad_token_id=self.tokenizer.pad_token_id,
#                     sequence_bias=logit_bias_dict,
#                     renormalize_logits=True,
#                 )

#             prompt_len = inputs.input_ids.shape[-1]
#             new_token_ids = output_content[:, prompt_len:]
#             batch_outputs = self.tokenizer.batch_decode(new_token_ids, skip_special_tokens=True)
#             for idx, output_text in zip(batch_indices, batch_outputs):
#                 outputs[idx] = output_text

#             if stream_output_path is not None:
#                 os.makedirs(os.path.dirname(stream_output_path), exist_ok=True)
#                 with open(stream_output_path, "a", encoding="utf-8") as f:
#                     for idx, output_text in zip(batch_indices, batch_outputs):
#                         record = {
#                             "sample_round": stream_sample_id,
#                             "index": idx,
#                             "output": output_text,
#                         }
#                         if stream_metadata is not None:
#                             record["metadata"] = stream_metadata[idx]
#                         f.write(json.dumps(record, ensure_ascii=False) + "\n")
#         return outputs



# import json
# import os
# import time

# import torch
# from tqdm import tqdm
# from transformers import AutoModelForCausalLM, AutoTokenizer

# from constants import (
#     COT_OPENING_QWEN,
#     GUARDREASONER_COT_OPENING,
#     GUARDREASONER_LABEL_OPENING,
#     LABEL_OPENING,
#     LLAMAGUARD_LABEL_OPENING,
#     NEG_LABEL,
#     NEMOGUARD_LABEL_OPENING,
#     POS_LABEL,
#     SHIELDGEMMA_LABEL_OPENING,
#     WILDGUARD_LABEL_OPENING,
# )


# class ComplianceProjectError(ValueError):
#     pass


# class ModelWrapper:
#     def get_message_template(self, system_content=None, user_content=None, assistant_content=None):
#         message = []
#         if system_content is not None:
#             message.append({"role": "system", "content": system_content})
#         if user_content is not None:
#             message.append({"role": "user", "content": user_content})
#         if assistant_content is not None:
#             message.append({"role": "assistant", "content": assistant_content})
#         if not message:
#             raise ComplianceProjectError("No content provided for any role.")
#         return message

#     def apply_chat_template(self, system_content=None, user_content=None, assistant_content=None, enable_thinking=None):
#         if assistant_content is None:
#             assistant_content = COT_OPENING_QWEN + "\n" if enable_thinking else LABEL_OPENING + "\n"
#         return self.get_message_template(system_content, user_content, assistant_content)


# class LocalModelWrapper(ModelWrapper):
#     def __init__(self, model_name, temperature=0.6, top_k=20, top_p=0.95, min_p=0, max_new_tokens=1000, custom_name=None):
#         self.model_name = custom_name or model_name
#         self.temperature = temperature
#         self.top_k = top_k
#         self.top_p = top_p
#         self.min_p = min_p
#         self.max_new_tokens = max_new_tokens
#         if "nemoguard" in model_name:
#             self.tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
#         else:
#             self.tokenizer = AutoTokenizer.from_pretrained(model_name)
#         self.tokenizer.pad_token_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id

#     def apply_chat_template(self, system_content, user_content=None, assistant_content=None, enable_thinking=True):
#         if assistant_content is not None:
#             assert "wildguard" not in self.model_name.lower(), (
#                 f"Gave assistant_content of {assistant_content} to model {self.model_name} but this type of model can only take a system prompt and that is it."
#             )
#             message = self.get_message_template(system_content, user_content, assistant_content)
#             try:
#                 prompt = self.tokenizer.apply_chat_template(message, tokenize=False, continue_final_message=True)
#             except ValueError as e:
#                 if "continue_final_message is set" in str(e):
#                     prompt = self.tokenizer.apply_chat_template(message, tokenize=False, continue_final_message=False)
#                     if "<|im_end|>\n" in prompt[-11:]:
#                         prompt = prompt[:-11]
#                 else:
#                     raise ComplianceProjectError(f"Error applying chat template: {e}")
#         else:
#             if "qwen3" in self.model_name.lower() or "dynaguard" in self.model_name.lower():
#                 if enable_thinking:
#                     message = self.get_message_template(system_content, user_content)
#                     prompt = self.tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True, enable_thinking=True)
#                     prompt = prompt + f"\n{COT_OPENING_QWEN}"
#                 else:
#                     message = self.get_message_template(system_content, user_content, assistant_content=f"{LABEL_OPENING}\n")
#                     prompt = self.tokenizer.apply_chat_template(message, tokenize=False, continue_final_message=True, enable_thinking=False)
#             elif "guardreasoner" in self.model_name.lower():
#                 assistant_content = GUARDREASONER_COT_OPENING if enable_thinking else GUARDREASONER_LABEL_OPENING
#                 message = self.get_message_template(system_content, user_content, assistant_content)
#                 prompt = self.tokenizer.apply_chat_template(message, tokenize=False, continue_final_message=True)
#             elif "wildguard" in self.model_name.lower():
#                 if enable_thinking:
#                     prompt = f"<s><|user|>\n[INST] {system_content} [/INST]\n<|assistant|>"
#                 else:
#                     prompt = f"<s><|user|>\n[INST] {system_content} [/INST]\n<|assistant|>{WILDGUARD_LABEL_OPENING}"
#             elif "llama-guard" in self.model_name.lower():
#                 if enable_thinking:
#                     prompt = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>{system_content}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
#                 else:
#                     prompt = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>{system_content}<|eot_id|><|start_header_id|>assistant<|end_header_id|>{LLAMAGUARD_LABEL_OPENING}"
#             elif "nemoguard" in self.model_name.lower():
#                 if enable_thinking:
#                     prompt = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>{system_content}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
#                 else:
#                     prompt = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>{system_content}<|eot_id|><|start_header_id|>assistant<|end_header_id|>{NEMOGUARD_LABEL_OPENING}"
#             elif "shieldgemma" in self.model_name.lower():
#                 if enable_thinking:
#                     prompt = f"<bos>{system_content}"
#                 else:
#                     prompt = f"<bos>{system_content}{SHIELDGEMMA_LABEL_OPENING}"
#             elif "mistral" in self.model_name.lower():
#                 assistant_content = COT_OPENING_QWEN + "\n" if enable_thinking else LABEL_OPENING + "\n"
#                 sys_user_combined = f"{system_content}\n\n{user_content}"
#                 message = self.get_message_template(user_content=sys_user_combined, assistant_content=assistant_content)
#                 prompt = self.tokenizer.apply_chat_template(message, tokenize=False, continue_final_message=True)
#             else:
#                 assistant_content = COT_OPENING_QWEN + "\n" if enable_thinking else LABEL_OPENING + "\n"
#                 message = self.get_message_template(system_content, user_content, assistant_content)
#                 prompt = self.tokenizer.apply_chat_template(message, tokenize=False, continue_final_message=True)
#         return prompt


# class HfModelWrapper(LocalModelWrapper):
#     def __init__(self, model_name, temperature=0.6, top_k=20, top_p=0.95, min_p=0, max_new_tokens=1000, custom_name=None, batch_size=8):
#         super().__init__(model_name, temperature, top_k, top_p, min_p, max_new_tokens, custom_name)
#         self.model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype=torch.bfloat16).eval()
#         self.batch_size = batch_size
#         self.tokenizer.padding_side = "left"

#     def get_hidden_state_features(self, messages, layer_idx=-1):
#         """
#         Return last-token hidden-state features for each message.

#         This mirrors the safe-route pattern:
#         - run the causal LM with ``output_hidden_states=True``
#         - select ``hidden_states[layer_idx]``
#         - take the final token representation ``[:, -1, :]``

#         Because the tokenizer is left-padded, the last time step is the last
#         real token for every sequence in the batch.
#         """
#         batch_size = self.batch_size
#         features = []

#         for start in tqdm(range(0, len(messages), batch_size), desc="HF hidden-state batches"):
#             batch_messages = messages[start : start + batch_size]
#             inputs = self.tokenizer(batch_messages, return_tensors="pt", padding=True).to(self.model.device)
#             with torch.no_grad():
#                 outputs = self.model(
#                     **inputs,
#                     output_hidden_states=True,
#                 )

#             hidden = outputs.hidden_states[layer_idx]
#             batch_features = hidden[:, -1, :]
#             features.append(batch_features)

#         if not features:
#             return torch.empty((0, self.model.config.hidden_size), device=self.model.device, dtype=torch.bfloat16)
#         return torch.cat(features, dim=0)

#     def get_prediction_probs(self, messages, strict=False, pos_label=POS_LABEL, neg_label=NEG_LABEL):
#         batch_size = self.batch_size
#         pos_token_id = self.tokenizer.encode(pos_label, add_special_tokens=False)[0]
#         neg_token_id = self.tokenizer.encode(neg_label, add_special_tokens=False)[0]

#         prob_pairs = []
#         logit_pairs = []
#         for start in tqdm(range(0, len(messages), batch_size), desc="HF prediction batches"):
#             batch_messages = messages[start : start + batch_size]
#             inputs = self.tokenizer(batch_messages, return_tensors="pt", padding=True).to(self.model.device)
#             with torch.no_grad():
#                 logits = self.model(**inputs).logits

#             # Tokenizer is configured for left padding, so the final real token
#             # for every sequence is always at the last time step in the batch.
#             prediction_logits = logits[:, -1, :]
#             prediction_probs = torch.nn.functional.softmax(prediction_logits, dim=-1)

#             for row_idx in range(prediction_logits.shape[0]):
#                 row_logits = prediction_logits[row_idx]
#                 row_probs = prediction_probs[row_idx]
#                 prob_pairs.append((row_probs[pos_token_id].item(), row_probs[neg_token_id].item()))
#                 logit_pairs.append((row_logits[pos_token_id].item(), row_logits[neg_token_id].item()))

#         return prob_pairs, logit_pairs

#     def get_responses(self, messages, temperature=None, top_k=None, top_p=None, logit_bias_dict=None, stream_output_path=None, stream_sample_id=None, stream_metadata=None):
#         if logit_bias_dict is not None:
#             token = list(logit_bias_dict.keys())[0]
#             bias = float(logit_bias_dict[token])
#             token_ids = self.tokenizer.encode(token, add_special_tokens=False)
#             logit_bias_dict = {tuple(token_ids): bias}

#         outputs = [None] * len(messages)
#         completed_indices = set()
#         if stream_output_path is not None and os.path.exists(stream_output_path):
#             with open(stream_output_path, "r", encoding="utf-8") as f:
#                 for line in f:
#                     line = line.strip()
#                     if not line:
#                         continue
#                     try:
#                         record = json.loads(line)
#                     except json.JSONDecodeError:
#                         # Ignore a partial final line if the process was interrupted mid-write.
#                         continue
#                     if record.get("sample_round") != stream_sample_id:
#                         continue
#                     index = record.get("index")
#                     if isinstance(index, int) and 0 <= index < len(messages):
#                         outputs[index] = record.get("output")
#                         completed_indices.add(index)

#         batch_size = self.batch_size
#         pending_indices = [i for i in range(len(messages)) if i not in completed_indices]
#         if pending_indices:
#             print(f"HF resume: {len(completed_indices)}/{len(messages)} already done for round {stream_sample_id}.")
#         for start in tqdm(range(0, len(pending_indices), batch_size), desc="HF generation batches"):
#             batch_indices = pending_indices[start : start + batch_size]
#             batch_messages = [messages[i] for i in batch_indices]
#             inputs = self.tokenizer(batch_messages, return_tensors="pt", padding=True).to(self.model.device)
#             with torch.no_grad():
#                 output_content = self.model.generate(
#                     **inputs,
#                     max_new_tokens=self.max_new_tokens,
#                     num_return_sequences=1,
#                     temperature=temperature or self.temperature,
#                     top_k=top_k or self.top_k,
#                     top_p=top_p or self.top_p,
#                     min_p=self.min_p,
#                     pad_token_id=self.tokenizer.pad_token_id,
#                     sequence_bias=logit_bias_dict,
#                     renormalize_logits=True,
#                 )

#             prompt_len = inputs.input_ids.shape[-1]
#             new_token_ids = output_content[:, prompt_len:]
#             batch_outputs = self.tokenizer.batch_decode(new_token_ids, skip_special_tokens=True)
#             for idx, output_text in zip(batch_indices, batch_outputs):
#                 outputs[idx] = output_text

#             if stream_output_path is not None:
#                 os.makedirs(os.path.dirname(stream_output_path), exist_ok=True)
#                 with open(stream_output_path, "a", encoding="utf-8") as f:
#                     for idx, output_text in zip(batch_indices, batch_outputs):
#                         record = {
#                             "sample_round": stream_sample_id,
#                             "index": idx,
#                             "output": output_text,
#                         }
#                         if stream_metadata is not None:
#                             record["metadata"] = stream_metadata[idx]
#                         f.write(json.dumps(record, ensure_ascii=False) + "\n")
#         return outputs







import json
import os
import time

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from constants import (
    COT_OPENING_QWEN,
    GUARDREASONER_COT_OPENING,
    GUARDREASONER_LABEL_OPENING,
    LABEL_OPENING,
    LLAMAGUARD_LABEL_OPENING,
    NEG_LABEL,
    NEMOGUARD_LABEL_OPENING,
    POS_LABEL,
    SHIELDGEMMA_LABEL_OPENING,
    WILDGUARD_LABEL_OPENING,
)


class ComplianceProjectError(ValueError):
    pass


class ModelWrapper:
    def get_message_template(self, system_content=None, user_content=None, assistant_content=None):
        message = []
        if system_content is not None:
            message.append({"role": "system", "content": system_content})
        if user_content is not None:
            message.append({"role": "user", "content": user_content})
        if assistant_content is not None:
            message.append({"role": "assistant", "content": assistant_content})
        if not message:
            raise ComplianceProjectError("No content provided for any role.")
        return message

    def apply_chat_template(self, system_content=None, user_content=None, assistant_content=None, enable_thinking=None):
        if assistant_content is None:
            assistant_content = COT_OPENING_QWEN + "\n" if enable_thinking else LABEL_OPENING + "\n"
        return self.get_message_template(system_content, user_content, assistant_content)


class LocalModelWrapper(ModelWrapper):
    def __init__(self, model_name, temperature=0.6, top_k=20, top_p=0.95, min_p=0, max_new_tokens=1000, custom_name=None):
        self.model_name = custom_name or model_name
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.min_p = min_p
        self.max_new_tokens = max_new_tokens
        if "nemoguard" in model_name:
            self.tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id

    def apply_chat_template(self, system_content, user_content=None, assistant_content=None, enable_thinking=True):
        if assistant_content is not None:
            assert "wildguard" not in self.model_name.lower(), (
                f"Gave assistant_content of {assistant_content} to model {self.model_name} but this type of model can only take a system prompt and that is it."
            )
            message = self.get_message_template(system_content, user_content, assistant_content)
            try:
                prompt = self.tokenizer.apply_chat_template(message, tokenize=False, continue_final_message=True)
            except ValueError as e:
                if "continue_final_message is set" in str(e):
                    prompt = self.tokenizer.apply_chat_template(message, tokenize=False, continue_final_message=False)
                    if "<|im_end|>\n" in prompt[-11:]:
                        prompt = prompt[:-11]
                else:
                    raise ComplianceProjectError(f"Error applying chat template: {e}")
        else:
            if "qwen3" in self.model_name.lower() or "dynaguard" in self.model_name.lower():
                if enable_thinking:
                    message = self.get_message_template(system_content, user_content)
                    prompt = self.tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True, enable_thinking=True)
                    prompt = prompt + f"\n{COT_OPENING_QWEN}"
                else:
                    message = self.get_message_template(system_content, user_content, assistant_content=f"{LABEL_OPENING}\n")
                    prompt = self.tokenizer.apply_chat_template(message, tokenize=False, continue_final_message=True, enable_thinking=False)
            elif "guardreasoner" in self.model_name.lower():
                assistant_content = GUARDREASONER_COT_OPENING if enable_thinking else GUARDREASONER_LABEL_OPENING
                message = self.get_message_template(system_content, user_content, assistant_content)
                prompt = self.tokenizer.apply_chat_template(message, tokenize=False, continue_final_message=True)
            elif "wildguard" in self.model_name.lower():
                if enable_thinking:
                    prompt = f"<s><|user|>\n[INST] {system_content} [/INST]\n<|assistant|>"
                else:
                    prompt = f"<s><|user|>\n[INST] {system_content} [/INST]\n<|assistant|>{WILDGUARD_LABEL_OPENING}"
            elif "llama-guard" in self.model_name.lower():
                if enable_thinking:
                    prompt = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>{system_content}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
                else:
                    prompt = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>{system_content}<|eot_id|><|start_header_id|>assistant<|end_header_id|>{LLAMAGUARD_LABEL_OPENING}"
            elif "nemoguard" in self.model_name.lower():
                if enable_thinking:
                    prompt = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>{system_content}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
                else:
                    prompt = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>{system_content}<|eot_id|><|start_header_id|>assistant<|end_header_id|>{NEMOGUARD_LABEL_OPENING}"
            elif "shieldgemma" in self.model_name.lower():
                if enable_thinking:
                    prompt = f"<bos>{system_content}"
                else:
                    prompt = f"<bos>{system_content}{SHIELDGEMMA_LABEL_OPENING}"
            elif "mistral" in self.model_name.lower():
                assistant_content = COT_OPENING_QWEN + "\n" if enable_thinking else LABEL_OPENING + "\n"
                sys_user_combined = f"{system_content}\n\n{user_content}"
                message = self.get_message_template(user_content=sys_user_combined, assistant_content=assistant_content)
                prompt = self.tokenizer.apply_chat_template(message, tokenize=False, continue_final_message=True)
            else:
                assistant_content = COT_OPENING_QWEN + "\n" if enable_thinking else LABEL_OPENING + "\n"
                message = self.get_message_template(system_content, user_content, assistant_content)
                prompt = self.tokenizer.apply_chat_template(message, tokenize=False, continue_final_message=True)
        return prompt


class HfModelWrapper(LocalModelWrapper):
    def __init__(self, model_name, temperature=0.6, top_k=20, top_p=0.95, min_p=0, max_new_tokens=1000, custom_name=None, batch_size=8):
        super().__init__(model_name, temperature, top_k, top_p, min_p, max_new_tokens, custom_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype=torch.bfloat16).eval()
        self.batch_size = batch_size
        self.tokenizer.padding_side = "left"

    def get_hidden_state_features(self, messages, layer_idx=-1):
        """
        Return last-token hidden-state features for each message.

        This mirrors the safe-route pattern:
        - run the causal LM with ``output_hidden_states=True``
        - select ``hidden_states[layer_idx]``
        - take the final token representation ``[:, -1, :]``

        Because the tokenizer is left-padded, the last time step is the last
        real token for every sequence in the batch.
        """
        batch_size = self.batch_size
        features = []

        for start in tqdm(range(0, len(messages), batch_size), desc="HF hidden-state batches"):
            batch_messages = messages[start : start + batch_size]
            inputs = self.tokenizer(batch_messages, return_tensors="pt", padding=True).to(self.model.device)
            with torch.no_grad():
                outputs = self.model(
                    **inputs,
                    output_hidden_states=True,
                )

            hidden = outputs.hidden_states[layer_idx]
            batch_features = hidden[:, -1, :]
            features.append(batch_features)

        if not features:
            return torch.empty((0, self.model.config.hidden_size), device=self.model.device, dtype=torch.bfloat16)
        return torch.cat(features, dim=0)

    def get_prediction_probs(self, messages, strict=False, pos_label=POS_LABEL, neg_label=NEG_LABEL):
        batch_size = self.batch_size
        pos_token_id = self.tokenizer.encode(pos_label, add_special_tokens=False)[0]
        neg_token_id = self.tokenizer.encode(neg_label, add_special_tokens=False)[0]

        prob_pairs = []
        logit_pairs = []
        for start in tqdm(range(0, len(messages), batch_size), desc="HF prediction batches"):
            batch_messages = messages[start : start + batch_size]
            inputs = self.tokenizer(batch_messages, return_tensors="pt", padding=True).to(self.model.device)
            with torch.no_grad():
                logits = self.model(**inputs).logits

            # Tokenizer is configured for left padding, so the final real token
            # for every sequence is always at the last time step in the batch.
            prediction_logits = logits[:, -1, :]
            prediction_probs = torch.nn.functional.softmax(prediction_logits, dim=-1)

            for row_idx in range(prediction_logits.shape[0]):
                row_logits = prediction_logits[row_idx]
                row_probs = prediction_probs[row_idx]
                prob_pairs.append((row_probs[pos_token_id].item(), row_probs[neg_token_id].item()))
                logit_pairs.append((row_logits[pos_token_id].item(), row_logits[neg_token_id].item()))

        return prob_pairs, logit_pairs

    def get_responses(self, messages, temperature=None, top_k=None, top_p=None, logit_bias_dict=None, stream_output_path=None, stream_sample_id=None, stream_metadata=None):
        if logit_bias_dict is not None:
            token = list(logit_bias_dict.keys())[0]
            bias = float(logit_bias_dict[token])
            token_ids = self.tokenizer.encode(token, add_special_tokens=False)
            logit_bias_dict = {tuple(token_ids): bias}

        outputs = [None] * len(messages)
        completed_indices = set()
        if stream_output_path is not None and os.path.exists(stream_output_path):
            with open(stream_output_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        # Ignore a partial final line if the process was interrupted mid-write.
                        continue
                    if record.get("sample_round") != stream_sample_id:
                        continue
                    index = record.get("index")
                    if isinstance(index, int) and 0 <= index < len(messages):
                        outputs[index] = record.get("output")
                        completed_indices.add(index)

        batch_size = self.batch_size
        pending_indices = [i for i in range(len(messages)) if i not in completed_indices]
        if pending_indices:
            print(f"HF resume: {len(completed_indices)}/{len(messages)} already done for round {stream_sample_id}.")
        for start in tqdm(range(0, len(pending_indices), batch_size), desc="HF generation batches"):
            batch_indices = pending_indices[start : start + batch_size]
            batch_messages = [messages[i] for i in batch_indices]
            inputs = self.tokenizer(batch_messages, return_tensors="pt", padding=True).to(self.model.device)
            with torch.no_grad():
                output_content = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    num_return_sequences=1,
                    temperature=temperature or self.temperature,
                    top_k=top_k or self.top_k,
                    top_p=top_p or self.top_p,
                    min_p=self.min_p,
                    pad_token_id=self.tokenizer.pad_token_id,
                    sequence_bias=logit_bias_dict,
                    renormalize_logits=True,
                )

            prompt_len = inputs.input_ids.shape[-1]
            new_token_ids = output_content[:, prompt_len:]
            batch_outputs = self.tokenizer.batch_decode(new_token_ids, skip_special_tokens=True)
            for idx, output_text in zip(batch_indices, batch_outputs):
                outputs[idx] = output_text

            if stream_output_path is not None:
                os.makedirs(os.path.dirname(stream_output_path), exist_ok=True)
                with open(stream_output_path, "a", encoding="utf-8") as f:
                    for idx, output_text in zip(batch_indices, batch_outputs):
                        record = {
                            "sample_round": stream_sample_id,
                            "index": idx,
                            "output": output_text,
                        }
                        if stream_metadata is not None:
                            record["metadata"] = stream_metadata[idx]
                        f.write(json.dumps(record, ensure_ascii=False) + "\n")
        return outputs
