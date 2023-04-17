# 模型推理

## 运行环境
建议使用Python 3.8及以上版本。

主要依赖库如下：
* `transformers` >= 4.28.0
* `sentencepiece` >= 0.1.97
* `bitsandbytes` >= 0.37.0

开启int8优化后，模型占用GPU显存约为9G。如果没有足够的显存，可以进行模型量化后使用CPU运行。

## 获取模型
为了符合LLaMA的使用规范，我们发布Lawyer LLaMA的增量权重（Delta Weights）。把增量权重加到原始LLaMA的权重上，即可获得Lawyer LLaMA的模型权重。请按以下步骤操作：

1. 通过[官方途径](https://github.com/facebookresearch/llama)获取LLaMA原始模型。

2. 使用transformers提供的脚本[convert_llama_weights_to_hf.py](https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/convert_llama_weights_to_hf.py)，将原版LLaMA模型转换为HuggingFace格式。将原版LLaMA的`tokenizer.model`放在`--input_dir`指定的目录，其余文件放在`${input_dir}/${model_size}`下。执行以下命令后，`--output_dir`中将存放转换好的HF版权重。

```bash
python src/transformers/models/llama/convert_llama_weights_to_hf.py \
    --input_dir /path/to/downloaded/llama/weights --model_size 7B --output_dir /output/path
```

3. 使用我们提供的脚本`model_delta.py`，将增量权重加到原始LLaMA权重上。`base-model-path`为上一步得到的HF版LLaMA权重路径，`--delta-path`为增量权重路径，`--target-model-path`为输出的Lawyer LLaMA模型路径。

```bash
python model_delta.py \
    --base-model-path /path/to/original-llama/weight \
    --target-model-path /path/to/laywer-llama/weight \
    --delta-path /path/to/delta/weight \
    --mode apply_delta
```

4. 可以使用以下Python代码运行模型进行推理。
   
```python
from transformers import LlamaForCausalLM, LlamaTokenizer

tokenizer = LlamaTokenizer.from_pretrained(tokenizer_path)
model = LlamaForCausalLM.from_pretrained(model_path, device_map="auto", load_in_8bit=True)

input_text = "什么是夫妻共同财产？"

input_text += "\n\n"
input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to("cuda")
outputs = model.generate(input_ids, max_new_tokens=1000)
output_text = str(tokenizer.decode(outputs[0]))

print(output_text)
```

## 模型量化
如果你没有足够的GPU资源进行推理，可以利用[llama.cpp](https://github.com/ggerganov/llama.cpp)进行4-bit模型量化后进行CPU推理。请按以下步骤操作：

1. 请按[llama.cpp](https://github.com/ggerganov/llama.cpp)中的教程，完成`Get the code`和`Build`两步。

2. 使用我们提供的脚本`convert_hf-7b_to_llama-pth.py`将HuggingFace格式的Laywer LLaMA转换成LLaMA原始格式（该脚本目前仅支持7B模型的转换）。`--hf_model_path`和`--hf_tokenizer_path`分别为HuggingFace格式的Lawyer LLaMA模型和分词器的路径，`--output`为输出的LLaMA原始格式的路径。

```bash
python convert_hf-7b_to_llama-pth.py \
--hf_model_path /path/to/laywer-llama/model \
--hf_tokenizer_path /path/to/laywer-llama/tokenizer  \
--output_dir /output/path
```

3. 将转换得到的LLaMA原始格式的模型放在`./models/laywer-llama`中。继续按照[llama.cpp](https://github.com/ggerganov/llama.cpp)中的教程，将Lawyer LLaMA转换成ggml FB16格式，并进行4-bit量化。
```
python3 convert.py models/lawyer-llama/
./quantize ./models/lawyer-llama/ggml-model-f16.bin ./models/lawyer-llama/ggml-model-q4_0.bin 2
make -j && ./main -m ./models/fakao_zixun/ggml-model-q4_0.bin -p "什么是夫妻共同财产？\n\n" -n 512 -t 16
```

