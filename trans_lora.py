import gradio as gr
import os
from safetensors.torch import load_file, save_file  # 正确导入
import safetensors.torch
from safetensors import safe_open
import torch
import tempfile,shutil
from collections import defaultdict

def convert_lora_from_path(input_path):

    output_path = input_path.replace(".safetensors", "-fix.safetensors")

    lora_conversion_algorithm(input_path,output_path)

    return f"转换完成，保存为：{output_path}"

def lora_conversion_algorithm(input_path,output_path):

    tensors = {}
    with safe_open(input_path, framework="pt", device="cpu") as f:
        for k in f.keys():
            tensors[k] = f.get_tensor(k)
    output  = ''

    print("First 20 tensor keys in input:")
    for i, key in enumerate(tensors.keys()):
        print(f"  {key}")
        if i >= 19:
            break

    converted = defaultdict(dict)
    final_layer_keys = []

    for key, tensor in tensors.items():
        group = key.rsplit('.', 1)[0]
        converted[group][key] = tensor
        if "final_layer" in key:
            final_layer_keys.append((group, key))

    print("\n发现的final_layer键：")
    for group, key in final_layer_keys:
        print(f"  {key}")

    has_final_layer_linear = any("final_layer_linear" in key for _, key in final_layer_keys)
    has_final_layer_adaLN = any("final_layer_adaLN_modulation_1" in key for _, key in final_layer_keys)

    print(f"\n检查final_layer权重:")
    print(f" 有final_layer.linear: {has_final_layer_linear}")
    print(f" 有final_layer.adaLN_modulation.1: {has_final_layer_adaLN}")

    # Patch missing adaLN weights
    if has_final_layer_linear and not has_final_layer_adaLN:
        print("\n检测到缺少final_layer.adaLN_modulation.1权重")
        print("自动创建虚拟adaLN权重来避免加载错误...")

        linear_down_key = "lora_unet_final_layer_linear.lora_down.weight"
        linear_up_key = "lora_unet_final_layer_linear.lora_up.weight"

        linear_down_tensor = tensors.get(linear_down_key)
        linear_up_tensor = tensors.get(linear_up_key)

        if linear_down_tensor is not None and linear_up_tensor is not None:
            dummy_down = torch.zeros_like(linear_down_tensor)
            dummy_up = torch.zeros_like(linear_up_tensor)

            dummy_base = "lora_unet_final_layer_adaLN_modulation_1"
            converted[dummy_base][f"{dummy_base}.lora_down.weight"] = dummy_down
            converted[dummy_base][f"{dummy_base}.lora_up.weight"] = dummy_up

            print(f"  添加了虚拟权重: {dummy_base}")
            print(f"  虚拟权重形状: down={dummy_down.shape}, up={dummy_up.shape}")
        else:
            print("  错误：无法获取linear权重的形状")

    final_sd = {}
    for weights in converted.values():
        for k, v in weights.items():
            final_sd[k] = v

    safetensors.torch.save_file(final_sd, output_path)
    print('修复成功')
    return final_sd


with gr.Blocks() as demo:
    gr.Markdown("## LoRA 转换工具（输入本地 .safetensors 路径）")
    file_path_input = gr.Textbox(label="本地 LoRA 文件路径（例如：C:\\Models\\example.safetensors）")
    output_text = gr.Textbox(label="处理日志")

    convert_btn = gr.Button("开始转换")
    convert_btn.click(fn=convert_lora_from_path, inputs=file_path_input, outputs=[output_text])

demo.launch()
