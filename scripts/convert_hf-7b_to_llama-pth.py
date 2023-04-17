import argparse
import os
import shutil

from tqdm import tqdm
import torch
from transformers import LlamaConfig, LlamaForCausalLM, LlamaTokenizer


def convert_model(input_model_path, output_model_path, model_size):
    if model_size != '7B':
        raise ValueError(f"Unsupported model size: {model_size}")
    
    # Load model
    hf_config = LlamaConfig.from_pretrained(input_model_path)
    hf_model = LlamaForCausalLM.from_pretrained(input_model_path, config=hf_config, torch_dtype=torch.float16)

    n_layers = hf_config.num_hidden_layers
    n_heads = hf_config.num_attention_heads
    dim = hf_config.hidden_size

    # permute for sliced rotary
    def permute(w):
        return w.view(n_heads, 2, dim // n_heads // 2, dim).transpose(1, 2).reshape(dim, dim)
    
    # Convert huggingface model to pytorch
    hf_state_dict = hf_model.state_dict()

    pth_state_dict = {}
    # possibly dangerous: remove last token embed
    pth_state_dict["tok_embeddings.weight"] = hf_state_dict["model.embed_tokens.weight"][:-1, :]
    pth_state_dict["norm.weight"] = hf_state_dict["model.norm.weight"]
    pth_state_dict["output.weight"] = hf_state_dict["lm_head.weight"][:-1, :]

    for layer_i in tqdm(range(n_layers)):
        pth_state_dict.update({
            f"layers.{layer_i}.attention.wq.weight": permute(
                hf_state_dict[f"model.layers.{layer_i}.self_attn.q_proj.weight"]
            ),
            f"layers.{layer_i}.attention.wk.weight": permute(
                hf_state_dict[f"model.layers.{layer_i}.self_attn.k_proj.weight"]
            ),
            f"layers.{layer_i}.attention.wv.weight": hf_state_dict[f"model.layers.{layer_i}.self_attn.v_proj.weight"],
            f"layers.{layer_i}.attention.wo.weight": hf_state_dict[f"model.layers.{layer_i}.self_attn.o_proj.weight"],
            f"layers.{layer_i}.feed_forward.w1.weight": hf_state_dict[f"model.layers.{layer_i}.mlp.gate_proj.weight"],
            f"layers.{layer_i}.feed_forward.w2.weight": hf_state_dict[f"model.layers.{layer_i}.mlp.down_proj.weight"],
            f"layers.{layer_i}.feed_forward.w3.weight": hf_state_dict[f"model.layers.{layer_i}.mlp.up_proj.weight"],
            f"layers.{layer_i}.attention_norm.weight": hf_state_dict[f"model.layers.{layer_i}.input_layernorm.weight"],
            f"layers.{layer_i}.ffn_norm.weight": hf_state_dict[f"model.layers.{layer_i}.post_attention_layernorm.weight"],
        })

    for layer_i in range(n_layers):
        pth_state_dict[f"layers.{layer_i}.attention.inner_attention.rope.freqs"] = hf_state_dict[f"model.layers.{layer_i}.self_attn.rotary_emb.inv_freq"]

    # Save model
    torch.save(pth_state_dict, output_model_path)
    

def convert_tokenizer(input_tokenizer_path, output_tokenizer_path):
    shutil.copyfile(os.path.join(input_tokenizer_path, "tokenizer.model"), os.path.join(output_tokenizer_path, "tokenizer.model"))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--hf_model_path",
        type=str,
        help="Path to the HuggingFace model",
    )
    parser.add_argument(
        "--hf_tokenizer_path",
        type=str,
        help="Path to the HuggingFace tokenizer",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="tmp",
        help="Path to the output directory",
    )
    
    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    convert_model(args.hf_model_path, os.path.join(args.output_dir, "consolidated.00.pth"), "7B")

    convert_tokenizer(args.hf_tokenizer_path, args.output_dir)

