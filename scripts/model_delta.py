import argparse

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

# reference: https://github.com/lm-sys/FastChat/blob/main/fastchat/model

def make_delta(base_model_path, target_model_path, delta_path):
    print(f"Loading the base model from {base_model_path}")
    base = AutoModelForCausalLM.from_pretrained(
        base_model_path, low_cpu_mem_usage=True)

    print(f"Loading the target model from {target_model_path}")
    target = AutoModelForCausalLM.from_pretrained(target_model_path, low_cpu_mem_usage=True)

    '''
    Problem Shooting:
    RuntimeError: The size of tensor a (32001) must match the size of tensor b (32000) at non-singleton dimension 0
    model.embed_tokens.weight
    因为aplaca的词表加了一维
    所以需要对原始llama的的输入输出层进行resize    
    代码来自于alpaca的smart_tokenizer_and_embedding_resize
    '''
    # special treatment to the input and output embedding of the base model
    base_vocab_size = base.config.vocab_size
    target_vocab_size = target.config.vocab_size
    if base_vocab_size != target_vocab_size:
        print("Resizing the input and output embedding of the base model")
        num_new_tokens = target_vocab_size - base_vocab_size
        print(f"Base model vocab size: {base_vocab_size}")
        print(f"Number of new tokens: {num_new_tokens}")
        base.resize_token_embeddings(base_vocab_size + num_new_tokens)

        input_embeddings = base.get_input_embeddings().weight.data
        output_embeddings = base.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


    print("Calculating the delta")
    for name, param in tqdm(target.state_dict().items(), desc="Calculating delta"):
        # print(name)
        assert name in base.state_dict()
        param.data -= base.state_dict()[name]

    print(f"Saving the delta to {delta_path}")

    target.save_pretrained(delta_path)


def apply_delta(base_model_path, target_model_path, delta_path):
    print(f"Loading the base model from {base_model_path}")
    base = AutoModelForCausalLM.from_pretrained(
        base_model_path, low_cpu_mem_usage=True)

    print(f"Loading the delta from {delta_path}")
    delta = AutoModelForCausalLM.from_pretrained(delta_path, low_cpu_mem_usage=True)

    # special treatment to the input and output embedding of the base model
    base_vocab_size = base.config.vocab_size
    delta_vocab_size = delta.config.vocab_size
    if base_vocab_size != delta_vocab_size:
        print("Resizing the input and output embedding of the base model")
        num_new_tokens = delta_vocab_size - base_vocab_size
        print(f"Base model vocab size: {base_vocab_size}")
        print(f"Number of new tokens: {num_new_tokens}")
        base.resize_token_embeddings(base_vocab_size + num_new_tokens)

        input_embeddings = base.get_input_embeddings().weight.data
        output_embeddings = base.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg

    print("Applying the delta")
    for name, param in tqdm(base.state_dict().items(), desc="Applying delta"):
        # print(name)
        assert name in delta.state_dict()
        param.data += delta.state_dict()[name]

    print(f"Saving the target model to {target_model_path}")
    base.save_pretrained(target_model_path)


def check_model_equality(model1_path, model2_path):
    print("Loading model1 from", model1_path)
    model1 = AutoModelForCausalLM.from_pretrained(model1_path, low_cpu_mem_usage=True)
    print("Loading model2 from", model2_path)
    model2 = AutoModelForCausalLM.from_pretrained(model2_path, low_cpu_mem_usage=True)

    for name, param in model1.state_dict().items():
        print(name)
        assert name in model2.state_dict()
        assert torch.allclose(param, model2.state_dict()[name])
    
    print("All parameters are equal")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model-path", type=str, required=True)
    parser.add_argument("--target-model-path", type=str, required=True)
    parser.add_argument("--delta-path", type=str)
    parser.add_argument("--mode", type=str, required=True, choices=["make_delta", "apply_delta", "check_model_equality"])

    args = parser.parse_args()

    if args.mode == "make_delta":
        make_delta(args.base_model_path, args.target_model_path, args.delta_path)
    elif args.mode == "apply_delta":
        apply_delta(args.base_model_path, args.target_model_path, args.delta_path)
    elif args.mode == "check_model_equality":
        check_model_equality(args.base_model_path, args.target_model_path)
