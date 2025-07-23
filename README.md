用于处理 不同平台Kontext 报错的lora转化


报错示例
  File "/root/autodl-tmp/ComfyUI/custom_nodes/ComfyUI-nunchaku/nodes/lora/flux.py", line 70, in load_lora
    sd = to_diffusers(lora_path)
         ^^^^^^^^^^^^^^^^^^^^^^^

  File "/root/autodl-tmp/conda/envs/py312/lib/python3.12/site-packages/nunchaku/lora/flux/diffusers_converter.py", line 72, in to_diffusers
    new_tensors, alphas = FluxLoraLoaderMixin.lora_state_dict(tensors, return_alphas=True)
                          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

  File "/root/autodl-tmp/conda/envs/py312/lib/python3.12/site-packages/huggingface_hub/utils/_validators.py", line 114, in _inner_fn
    return fn(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^

  File "/root/autodl-tmp/conda/envs/py312/lib/python3.12/site-packages/diffusers/loaders/lora_pipeline.py", line 1706, in lora_state_dict
    state_dict = _convert_kohya_flux_lora_to_diffusers(state_dict)
                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

  File "/root/autodl-tmp/conda/envs/py312/lib/python3.12/site-packages/diffusers/loaders/lora_conversion_utils.py", line 882, in _convert_kohya_flux_lora_to_diffusers
    return _convert_sd_scripts_to_ai_toolkit(state_dict)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

  File "/root/autodl-tmp/conda/envs/py312/lib/python3.12/site-packages/diffusers/loaders/lora_conversion_utils.py", line 611, in _convert_sd_scripts_to_ai_toolkit
    assign_remaining_weights(

  File "/root/autodl-tmp/conda/envs/py312/lib/python3.12/site-packages/diffusers/loaders/lora_conversion_utils.py", line 537, in assign_remaining_weights
    value = source.pop(source_key)
            ^^^^^^^^^^^^^^^^^^^^^^
