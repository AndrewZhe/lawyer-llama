# 模型推理
我们目前公开了以下版本的Lawyer LLaMA：
* lawyer-llama-13b-beta1.0: 以[Chinese-LLaMA-13B](https://github.com/ymcui/Chinese-LLaMA-Alpaca)为底座，**未经过法律语料continual training**，使用通用instruction和法律instruction进行SFT，配有婚姻相关法律检索模块。

我们计划公开：
* 以ChatGLM为底座训练的模型
* 经过法律语料continual training的版本

## 运行环境
建议使用Python 3.8及以上版本。

主要依赖库如下：
* `transformers` >= 4.28.0
* `sentencepiece` >= 0.1.97
* `gradio`

## 获取模型
为了符合LLaMA的使用规范，我们发布的Lawyer LLaMA权重需要使用原始LLaMA权重文件进行解码（相关代码来自[point-alpaca](https://github.com/pointnetwork/point-alpaca/)）。

1. 通过[官方途径](https://github.com/facebookresearch/llama)获取LLaMA原始模型。

2. 通过[Hugging Face](https://huggingface.co/pkupie/lawyer-llama-13b-beta1.0)或者[百度网盘](https://pan.baidu.com/s/1cE9_c8er3NASpDkFou-B9g?pwd=lwhx)（提取码：lwhx）获取Lawyer LLaMA权重。

3. 利用原始LLaMA文件中的`7B/consolidated.00.pth`文件，运行以下bash命令，使用`decrypt.py`对Lawyer LLaMA模型文件进行解码。
```bash
for f in "/path/to/model/pytorch_model"*".enc"; \
    do if [ -f "$f" ]; then \
       python3 decrypt.py "$f" "/path/to_original_llama/7B/consolidated.00.pth" "/path/to/model"; \
    fi; \
done
```
将以上命令中的`/path/to/model/`替换成下载后的Lawyer LLaMA所在路径。

4. 从[百度网盘](https://pan.baidu.com/s/1V9wsQR4ndKNqWRl8lGhOaw?pwd=r0vx)（提取码：r0vx）下载法条检索模块，并运行其中的`python server.py`启动法条检索服务，默认挂在9098端口。

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
