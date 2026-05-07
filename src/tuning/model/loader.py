from typing import TYPE_CHECKING, Optional, Tuple
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers.integrations import is_deepspeed_zero3_enabled
from transformers.utils.versions import require_version
from torch import nn
# from trl import AutoModelForCausalLMWithValueHead

import torch
from copy import deepcopy

from src.tuning.model.adapter import init_adapter

import logging
logger = logging.getLogger(__name__)


def _init_fv_projections_if_needed(model, training_args):
    # FVG training path expects `model.fv_projections` to exist.
    if not getattr(training_args, "fv_pr", False):
        return
    if hasattr(model, "fv_projections"):
        return
    if not (hasattr(model, "config") and hasattr(model.config, "hidden_size") and hasattr(model.config, "num_attention_heads")):
        logger.warning("Skipping fv_projections init: missing hidden_size/num_attention_heads in model config.")
        return

    dim = model.config.hidden_size // model.config.num_attention_heads
    model.fv_projections = nn.ModuleList()
    for i in range(10):
        model.fv_projections.append(nn.Linear(dim, dim))
        model.fv_projections[i].apply(init_diagonal)
    logger.info("Initialized fv_projections for non-LLaMA model compatibility.")


def load_model_and_tokenizer(
    model_args,
    training_args,
):

    config_kwargs = {
        "trust_remote_code": True,
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision
    }

    config = AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        use_auth_token=True if model_args.use_auth_token else None,
        **config_kwargs
    )
    model_type = getattr(config, "model_type", "").lower()
    model_name = model_args.model_name_or_path.lower()
    is_llama_model = model_type == "llama" or "llama" in model_name
    is_mistral_model = model_type == "mistral" or "mistral" in model_name

    if is_llama_model or is_mistral_model:
        config.bos_token_id = 1
        config.eos_token_id = 2
        config.pad_token_id = 1

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        use_fast = model_args.use_fast_tokenizer,
        use_auth_token = True if model_args.use_auth_token else None,
        **config_kwargs
    )

    if is_llama_model or is_mistral_model:
        tokenizer.bos_token_id = 1
        tokenizer.eos_token_id = 2
        tokenizer.pad_token_id = 1
    else:
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = 'left'

    if model_args.local_model == "llama" and is_llama_model:
        from src.tuning.model.llama import LlamaForCausalLM
        model = LlamaForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            config=config,
            low_cpu_mem_usage=(not is_deepspeed_zero3_enabled()),
            **config_kwargs
        )
    else:
        if model_args.local_model == "llama" and not is_llama_model:
            logger.warning(
                "model_args.local_model is set to 'llama' but config model_type is '%s'; "
                "falling back to AutoModelForCausalLM for compatibility.",
                model_type or "unknown",
            )
        model = AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            config=config,
            low_cpu_mem_usage=(not is_deepspeed_zero3_enabled()),
            **config_kwargs
        )

    _init_fv_projections_if_needed(model, training_args)

    logger.info("Init adapter !!")
    model = init_adapter(model, model_args, training_args, is_trainable=training_args.do_train)

    print(len(tokenizer))
    if is_llama_model or is_mistral_model:
        model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=32)
    input_embeddings = model.get_input_embeddings() if hasattr(model, "get_input_embeddings") else None
    if input_embeddings is not None and hasattr(input_embeddings, "weight"):
        print(input_embeddings.weight.shape)
    else:
        logger.warning("Could not resolve input embedding weight shape for model debug print.")
    # if model_args.local_model == "llama":

    #     for i in range(10):
    #         model.fv_projections[i].apply(init_diagonal)

    #     for name, param in model.named_parameters():
    #         if name.find("fv_projection") != -1:
    #             param.requires_grad = True

    return model, tokenizer

# def load_model_and_tokenizer(
#     model_args,
#     training_args,
# ):
#     r"""
#     Loads pretrained model and tokenizer.

#     Support both training and inference.
#     """


#     config_kwargs = {
#         "trust_remote_code": True,
#         "cache_dir": model_args.cache_dir,
#         "revision": model_args.model_revision,
#         "token": model_args.hf_hub_token
#     }

#     tokenizer = AutoTokenizer.from_pretrained(
#         model_args.model_name_or_path,
#         use_fast=model_args.use_fast_tokenizer,
#         split_special_tokens=model_args.split_special_tokens,
#         padding_side="right",
#         **config_kwargs
#     )

#     config = AutoConfig.from_pretrained(model_args.model_name_or_path, **config_kwargs)

#     model = None

#     model = AutoModelForCausalLM.from_pretrained(
#         model_args.model_name_or_path,
#         config=config,
#         low_cpu_mem_usage=(not is_deepspeed_zero3_enabled()),
#         **config_kwargs
#     )

#     patch_model(model, tokenizer, model_args, is_trainable)
#     register_autoclass(config, model, tokenizer)

#     model = init_adapter(model, model_args, finetuning_args, is_trainable)


#     if not is_trainable:
#         model.requires_grad_(False)
#         model.eval()
#     else:
#         model.train()


#     trainable_params, all_param = count_parameters(model)
#     logger.info("trainable params: {:d} || all params: {:d} || trainable%: {:.4f}".format(
#         trainable_params, all_param, 100 * trainable_params / all_param
#     ))

#     if not is_trainable:
#         logger.info("This IS expected that the trainable params is 0 if you are using model for inference only.")

#     return model, tokenizer


def init_diagonal(m):
    if type(m) == nn.Linear:
        with torch.no_grad():
            # 获取权重的形状
            shape = m.weight.size()
            # 初始化为零
            m.weight.data = torch.zeros(shape).to(m.weight.device).to(m.weight.dtype)
            # 填充对角线元素为某个值，比如1. 可以根据需要调整
            m.weight.data.fill_diagonal_(1.0)
            # 如果有偏置项，也可以将它们初始化为0或其他值
            if m.bias is not None:
                nn.init.constant_(m.bias.data, 0)
