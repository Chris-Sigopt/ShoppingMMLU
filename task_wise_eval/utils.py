from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import habana_frameworks.torch.core as htcore

from huggingface_hub import list_repo_files, snapshot_download
from transformers import modeling_utils
import tempfile
import os
import json
from pathlib import Path
from transformers.utils import is_offline_mode

def get_repo_root(model_name_or_path, local_rank=-1, token=None):
    """
    Downloads the specified model checkpoint and returns the repository where it was downloaded.
    """
    if Path(model_name_or_path).is_dir():
        # If it is a local model, no need to download anything
        return model_name_or_path
    else:
        # Checks if online or not
        if is_offline_mode():
            if local_rank == 0:
                print("Offline mode: forcing local_files_only=True")

        # Only download PyTorch weights by default
        if any(
            ".safetensors" in filename for filename in list_repo_files(model_name_or_path, token=token)
        ):  # Some models like Falcon-180b are in only safetensors format
            allow_patterns = ["*.safetensors"]
        elif any(".bin" in filename for filename in list_repo_files(model_name_or_path, token=token)):
            allow_patterns = ["*.bin"]
        else:
            raise TypeError("Only PyTorch models are supported")

        # Download only on first process
        if local_rank in [-1, 0]:
            cache_dir = snapshot_download(
                model_name_or_path,
                local_files_only=is_offline_mode(),
                cache_dir=os.getenv("TRANSFORMERS_CACHE", None),
                allow_patterns=allow_patterns,
                max_workers=16,
                token=token,
            )
            if local_rank == -1:
                # If there is only one process, then the method is finished
                return cache_dir

        # Make all processes wait so that other processes can get the checkpoint directly from cache
        if torch.distributed.is_initialized():
            torch.distributed.barrier()

        return snapshot_download(
            model_name_or_path,
            local_files_only=is_offline_mode(),
            cache_dir=os.getenv("TRANSFORMERS_CACHE", None),
            allow_patterns=allow_patterns,
            token=token,
        )

def get_checkpoint_files(model_name_or_path, local_rank, token=None):
    cached_repo_dir = get_repo_root(model_name_or_path, local_rank=local_rank, token=token)

    # Extensions: .bin | .safetensors | .pt
    # Creates a list of paths from all downloaded files in cache dir

    if any(file.suffix == ".bin" for file in Path(cached_repo_dir).rglob("*")):
        (name, ext) = os.path.splitext(modeling_utils.WEIGHTS_NAME)
    elif any(file.suffix == ".safetensors" for file in Path(cached_repo_dir).rglob("*")):
        (name, ext) = os.path.splitext(modeling_utils.SAFE_WEIGHTS_NAME)
    else:
        (name, ext) = ("*", ".pt")

    file_list = [
        str(entry)
        for entry in Path(cached_repo_dir).rglob("*")
        if (entry.is_file() and entry.name.startswith(name) and entry.name.endswith(ext))
    ]

    return file_list



def write_checkpoints_json(model_name_or_path, local_rank, f, token=None):
    """
    Dumps metadata into a JSON file for DeepSpeed-inference.
    """
    checkpoint_files = get_checkpoint_files(model_name_or_path, local_rank, token)
    data = {"type": "ds_model", "checkpoints": checkpoint_files, "version": 1.0}
    json.dump(data, f)
    f.flush()

def get_ds_injection_policy(model_path):
    policy = {}
    if model_path:
        if "llama" in model_path:
            from transformers.models.llama.modeling_llama import LlamaDecoderLayer
            policy = {LlamaDecoderLayer: ("self_attn.o_proj", "mlp.down_proj")}

        elif "mistral" in model_type:
            from transformers.models.mistral.modeling_mistral import MistralDecoderLayer
            policy = {MistralDecoderLayer: ("self_attn.o_proj", "mlp.down_proj")}

        elif "bloom" in model_path:
            from transformers.models.bloom.modeling_bloom import BloomBlock
            policy = {BloomBlock: ("self_attention.dense", "mlp.dense_4h_to_h")}

        elif "opt" in model_path:
            from transformers.models.opt.modeling_opt import OPTDecoderLayer
            policy = {OPTDecoderLayer: ("self_attn.out_proj", ".fc2")}

        elif "gpt2" in model_path:
            from transformers.models.gpt2.modeling_gpt2 import GPT2MLP
            policy = {GPT2MLP: ("attn.c_proj", "mlp.c_proj")}

        elif "gptj" in model_path:
            from transformers.models.gptj.modeling_gptj import GPTJBlock
            policy = {GPTJBlock: ("attn.out_proj", "mlp.fc_out")}

        elif "gpt_neox" in model_path:
            from transformers.models.gpt_neox.modeling_gpt_neox import GPTNeoXLayer

            policy = {GPTNeoXLayer: ("attention.dense", "mlp.dense_4h_to_h")}
    return policy

def setup_distributed_model(model_path):
    import deepspeed
    # List of model types that need max position embeddings capped at 8192
    deepspeed.init_distributed(dist_backend="hccl")
    with deepspeed.OnDevice(dtype=torch.bfloat16, device="meta"):
        if 'mixtral' in model_path:
            model = AutoModelForCausalLM.from_pretrained(model_path, device_map='auto', torch_dtype='auto', trust_remote_code=True)
        elif 'gemma' in model_path:
            model = AutoModelForCausalLM.from_pretrained(model_path, device_map='auto', torch_dtype=torch.bfloat16, trust_remote_code=True)
        else:
            model = AutoModelForCausalLM.from_pretrained(model_path, device_map='auto', torch_dtype=torch.bfloat16, trust_remote_code=True)
    checkpoints_json = tempfile.NamedTemporaryFile(suffix=".json", mode="+w")
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    write_checkpoints_json(
            model_path,
            local_rank,
            checkpoints_json,
            token=None,
        )
    ds_inference_kwargs = {"dtype": torch.bfloat16}
    ds_inference_kwargs["tensor_parallel"] = {"tp_size": 8}
    ds_inference_kwargs["enable_cuda_graph"] = False
    ds_inference_kwargs["injection_policy"] = get_ds_injection_policy(model_path)
    ds_inference_kwargs["checkpoint"] = checkpoints_json.name
    model = deepspeed.init_inference(model, **ds_inference_kwargs)
    model = model.module

    return model

def setup_quantization(model, quant_config):
    try:
        from neural_compressor.torch.quantization import FP8Config, convert, prepare
    except ImportError:
        raise ImportError(
            "Module neural_compressor is missing. Please use a newer Synapse version to use quantization."
        )

    config = FP8Config.from_json_file(quant_config)
    if config.measure:
        model = prepare(model, config)
    if config.quantize:
        model = convert(model, config)

    return model

def finalize_quantization(model, quant_config):
    try:
        from neural_compressor.torch.quantization import FP8Config, finalize_calibration
    except ImportError:
        raise ImportError(
            "Module neural_compressor is missing. Please use a newer Synapse version to use quantization."
        )
    config = FP8Config.from_json_file(quant_config)
    if config.measure:
        finalize_calibration(model)

def load_tokenizer_and_model(model_name, quant_config, use_deepspeed=False):
    if model_name == 'llama':
        model_path = '../llama1'
    if model_name == 'llama2-7b':
        model_path = 'meta-llama/Llama-2-7b-hf'
    if model_name == 'llama2-13b':
        model_path = 'meta-llama/Llama-2-13b-hf'
    if model_name == 'llama2-70b':
        model_path = 'meta-llama/Llama-2-70b-hf'
    if model_name == 'llama2-7b-chat':
        model_path = 'meta-llama/Llama-2-7b-chat-hf'
    if model_name == 'llama2-13b-chat':
        model_path = 'meta-llama/Llama-2-13b-chat-hf'
    if model_name == 'llama2-70b-chat':
        model_path = 'meta-llama/Llama-2-70b-chat-hf'
    if model_name == 'llama3-70b':
        model_path = 'meta-llama/Llama-3.1-70B-Instruct'
    if model_name == 'alpaca':
        model_path = '../alpaca'
    if model_name == 'vicuna1':
        model_path = 'lmsys/vicuna-7b-v1.3'
    if model_name == 'vicuna2':
        model_path = 'lmsys/vicuna-7b-v1.5'
    if model_name == 'vicuna2-13b':
        model_path = 'lmsys/vicuna-13b-v1.5'
    if model_name == 'yi6b':
        model_path = '01-ai/Yi-6B'
    if model_name == 'zephyr':
        model_path = 'HuggingFaceH4/zephyr-7b-beta'
    if model_name == 'mistral':
        model_path = 'mistralai/Mistral-7B-v0.1'
    if model_name == 'falcon7b':
        model_path = 'tiiuae/falcon-7b'
    if model_name == 'qwen0.5b':
        model_path = 'Qwen/Qwen1.5-0.5B'
    if model_name == 'qwen1.8b':
        model_path = 'Qwen/Qwen1.5-1.8B'
    if model_name == 'qwen4b':
        model_path = 'Qwen/Qwen1.5-4B'
    if model_name == 'qwen7b':
        model_path = 'Qwen/Qwen1.5-7B'
    if model_name == 'qwen14b':
        model_path = 'Qwen/Qwen1.5-14B'
    if model_name == 'qwen72b':
        model_path = 'Qwen/Qwen1.5-72B'
    if model_name == 'qwen4b-chat':
        model_path = 'Qwen/Qwen1.5-4B-Chat'
    if model_name == 'qwen7b-chat':
        model_path = 'Qwen/Qwen1.5-7B-Chat'
    if model_name == 'qwen14b-chat':
        model_path = 'Qwen/Qwen1.5-14B-Chat'
    if model_name == 'mistral-instruct':
        model_path = 'mistralai/Mistral-7B-Instruct-v0.2'
    if model_name == 'mixtral-8x7b':
        model_path = "mistralai/Mixtral-8x7B-v0.1"
    if model_name == 'phi2':
        model_path = 'microsoft/phi-2'
    if model_name == 'gemma2b':
        model_path = 'google/gemma-2b'
    if model_name == 'gemma2b-it':
        model_path = 'google/gemma-2b-it'
    if model_name == 'gemma7b':
        model_path = 'google/gemma-7b'
    if model_name == 'gemma7b-it':
        model_path = 'google/gemma-7b-it'
    if model_name == 'ecellm-m':
        model_path = 'NingLab/eCeLLM-M'
    if model_name == 'ecellm-s':
        model_path = 'NingLab/eCeLLM-S'
    if model_name == 'llama3-8b':
        model_path = 'meta-llama/Meta-Llama-3-8B'
    if model_name == 'llama3-8b-instruct':
        model_path = 'meta-llama/Meta-Llama-3-8B-Instruct'
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if use_deepspeed:
        model = setup_distributed_model(model_path)
    else:
        if 'mixtral' in model_name:
            model = AutoModelForCausalLM.from_pretrained(model_path, device_map='auto', torch_dtype='auto', trust_remote_code=True).to("hpu")
        elif 'gemma' in model_name:
            model = AutoModelForCausalLM.from_pretrained(model_path, device_map='auto', torch_dtype=torch.float16, trust_remote_code=True).to("hpu")
        else:
            model = AutoModelForCausalLM.from_pretrained(model_path, device_map='auto', torch_dtype=torch.float16, trust_remote_code=True).to("hpu")
    if quant_config != "":
        model = setup_quantization(model, quant_config)
    model = torch.compile(model,backend="hpu_backend")
    return tokenizer, model

def format_subject(subject):
    l = subject.split("_")
    s = ""
    for entry in l:
        if entry == 'pt':
            entry = 'product type'
        s += " " + entry
    return s

def format_example(df, idx, is_multi_choice=False, args=None):
    if not is_multi_choice:
        prompt = df.iloc[idx, 0]
        answer = df.iloc[idx, 1]

        return prompt
    else:
        if not args.use_letter_choices:
            if 'review_rating_prediction' not in args.test_subject:
                choices = ['0', '1', '2', '3']
            else:
                choices = ['1', '2', '3', '4', '5']
        else:
            choices = ['A', 'B', 'C', 'D']
        prompt = df.iloc[idx, 0]
        k = 4
        if 'review_rating_prediction' not in args.test_subject:
            candidates = eval(df.iloc[idx, 1])
            for j in range(k):
                if args.use_letter_choices:
                    prompt += "\n({}) {}".format(choices[j], candidates[j])
                else:
                    prompt += "\n{}. {}".format(choices[j], candidates[j])
        if args.use_letter_choices:
            prompt += '\n\nPlease answer the question with a single letter indicating the choice. '
        prompt += "\nAnswer: "
        return prompt

def gen_system_prompt(args, is_multiple_choice=False):
    if not is_multiple_choice:
        if args.use_task_specific_prompt:
            prompt = 'You are required to perform the task of %s. Please follow the given instructions.\n\n'%format_subject(args.test_subject)
        else:
            prompt = 'You are a helpful online shopping assistant. Please answer the following question about online shopping and follow the given instructions.\n\n'
        return prompt
    else:
        if args.use_task_specific_prompt:
            if args.test_subject != 'review_rating_prediction':
                prompt = 'The following is a multiple choice question about {}.\n\n'.format(format_subject(args.test_subject))
            else:
                prompt = 'The following is a review rating prediction question.\n\n'
        else:
            prompt = 'You are a helpful online shopping assistant. Please answer the following question about online shopping and follow the given instructions.\n\n'
        return prompt