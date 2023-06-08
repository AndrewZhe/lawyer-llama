import gradio as gr
import requests
import json
from transformers import LlamaForCausalLM, LlamaTokenizer, TextIteratorStreamer
import torch
import threading
import argparse

class StoppableThread(threading.Thread):
    """Thread class with a stop() method. The thread itself has to check
    regularly for the stopped() condition."""

    def __init__(self,  *args, **kwargs):
        super(StoppableThread, self).__init__(*args, **kwargs)
        self._stop_event = threading.Event()

    def stop(self):
        self._stop_event.set()

    def stopped(self):
        return self._stop_event.is_set()

def json_send(url, data=None, method="POST"):
    headers = {"Content-type": "application/json",
               "Accept": "text/plain", "charset": "UTF-8"}
    if method == "POST":
        if data != None:
            response = requests.post(url=url, headers=headers, data=json.dumps(data))
        else:
            response = requests.post(url=url, headers=headers)
    elif method == "GET":
        response = requests.get(url=url, headers=headers)
    return json.loads(response.text)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--checkpoint", type=str, default="")
    parser.add_argument("--classifier_url", type=str, default="")
    parser.add_argument("--load_in_8bit", action="store_true")
    args = parser.parse_args()
    checkpoint = args.checkpoint
    classifier_url = args.classifier_url

    print("Loading model...")
    tokenizer = LlamaTokenizer.from_pretrained(checkpoint)
    if args.load_in_8bit:
        model = LlamaForCausalLM.from_pretrained(checkpoint, device_map="auto", load_in_8bit=True)
    else:
        model = LlamaForCausalLM.from_pretrained(checkpoint, device_map="auto", torch_dtype=torch.float16)
    print("Model loaded.")

    with gr.Blocks() as demo:
        chatbot = gr.Chatbot()
        input_msg = gr.Textbox(label="Input")
        with gr.Row():
            generate_button = gr.Button('Generate', elem_id='generate', variant='primary')
            clear_button = gr.Button('Clear', elem_id='clear', variant='secondary')

        def user(user_message, chat_history):
            
            user_message = user_message.strip()
            return "", chat_history + [[user_message, None]]
        
        
        def bot(chat_history):

            # extract user inputs from chat history and retrieve law articles
            current_user_input = chat_history[-1][0]
            
            if len(current_user_input) == 0:
                yield chat_history[:-1]
                return

            # 检索法条
            history_user_input = [x[0] for x in chat_history]
            input_to_classifier = " ".join(history_user_input)
            data = {"input": input_to_classifier}
            result = json_send(classifier_url, data, method="POST")
            retrieve_output = result['output']
            
            # 构造输入
            if len(retrieve_output) == 0:
                input_text = "你是人工智能法律助手“Lawyer LLaMA”，能够回答与中国法律相关的问题。\n"
                for history_pair in chat_history[:-1]:
                    input_text += f"### Human: {history_pair[0]}\n### Assistant: {history_pair[1]}\n"
                input_text += f"### Human: {current_user_input}\n### Assistant: "
            else:
                input_text = f"你是人工智能法律助手“Lawyer LLaMA”，能够回答与中国法律相关的问题。请参考给出的\"参考法条\"，回复用户的咨询问题。\"参考法条\"中可能存在与咨询无关的法条，请回复时不要引用这些无关的法条。\n"
                for history_pair in chat_history[:-1]:
                    input_text += f"### Human: {history_pair[0]}\n### Assistant: {history_pair[1]}\n"
                input_text += f"### Human: {current_user_input}\n### 参考法条: {retrieve_output[0]['text']}\n{retrieve_output[1]['text']}\n{retrieve_output[2]['text']}\n### Assistant: "

            print("=== Input ===")
            print("input_text: ", input_text)

            inputs = tokenizer(input_text, return_tensors="pt").to("cuda")
            streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

            # Run the generation in a separate thread, so that we can fetch the generated text in a non-blocking way.
            generation_kwargs = dict(inputs, streamer=streamer, max_new_tokens=400, do_sample=False, repetition_penalty=1.1)
            thread = StoppableThread(target=model.generate, kwargs=generation_kwargs)
            thread.start()


            # 开始流式生成
            chat_history[-1][1] = ""
            for new_text in streamer:
                chat_history[-1][1] += new_text
                yield chat_history
            
            streamer.end()
            thread.stop()
            print("Output: ", chat_history[-1][1])

        def stop(invitation_code):
            global stop_everything
            stop_everything[invitation_code] = True


        input_msg.submit(user, [input_msg, chatbot], [input_msg, chatbot], queue=False).then(
            bot, [chatbot], chatbot
        )
        generate_button.click(user, [input_msg, chatbot], [input_msg, chatbot], queue=False).then(
            bot, [chatbot], chatbot
        )

    demo.queue()
    demo.launch(share=False, server_port=args.port, server_name='0.0.0.0')

