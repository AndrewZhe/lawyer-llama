# 模型推理
我们目前公开的最新版本Lawyer LLaMA是：
* **Lawyer LLaMA 2 (`lawyer-llama-13b-v2`)**: 以[quzhe/llama_chinese_13B](https://huggingface.co/quzhe/llama_chinese_13B)为底座，使用通用instruction和法律instruction进行SFT，配有婚姻相关法律检索模块。

## 运行环境
建议使用Python 3.8及以上版本。

主要依赖库如下：
* `transformers` >= 4.28.0 **注意：检索模块需要使用transformers <= 4.30**
* `sentencepiece` >= 0.1.97
* `gradio`

## 获取模型
1. 从[HuggingFace](https://huggingface.co/pkupie/lawyer-llama-13b-v2)下载 **Lawyer LLaMA 2 (`lawyer-llama-13b-v2`)**模型参数。


4. 从[HuggingFace](https://huggingface.co/pkupie/marriage_law_retrieval)下载法条检索模块，并运行其中的`python server.py`启动法条检索服务，默认挂在9098端口。

## 模型运行
### 使用命令行运行
```bash
python demo_cmd.py \
--checkpoint /path/to/model \
--classifier_url "http://127.0.0.1:9098/check_hunyin" \
--use_chat_mode
```

### 使用交互界面运行
运行以下命令启动交互网页，访问`http://127.0.0.1:7863`。
```bash
python demo_web.py \
--port 7863 \
--checkpoint /path/to/model \
--classifier_url "http://127.0.0.1:9098/check_hunyin"
```

如需使用nginx反向代理访问此服务，可参考https://github.com/LeetJoe/lawyer-llama/blob/main/demo/nginx_proxy.md （Credit to [@LeetJoe](https://github.com/LeetJoe)）
