import argparse
import os
from nanovllm import LLM, SamplingParams
from transformers import AutoTokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    path = os.path.expanduser("~/huggingface/Qwen3-0.6B/")
    tokenizer = AutoTokenizer.from_pretrained(path)#初始化分词器及模型
    llm = LLM(path, enforce_eager=False, tensor_parallel_size=1, device=args.device)
# LLM是对LLMEngine的包装，主要是为了对齐vllm的行为，代码参考llm.py和LLMEngine
    sampling_params = SamplingParams(temperature=0.6, max_tokens=256)#处理控制文本的几个开关
    prompts = [
        "introduce yourself",
        "list all prime numbers within 100",
        "The future of AI is",
    ]
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],#"content"：说了什么内容
            tokenize=False,
            add_generation_prompt=True,
        )#使用tokenizer模板转换prompt
        #['<|im_start|>user\nintroduce yourself<|im_end|>\n<|im_start|>assistant\n', 
        #'<|im_start|>user\nlist all prime numbers within 100<|im_end|>\n<|im_start|>assistant\n']assistant指模型该自动补全回复了
        for prompt in prompts
    ]
    outputs = llm.generate(prompts, sampling_params)#从提示中生成文本。输出是一个包含提示、生成的文本和其他信息的RequestOutput对象列表

    for prompt, output in zip(prompts, outputs):
        print("\n")
        print(f"Prompt: {prompt!r}")
        print(f"Completion: {output['text']!r}")


if __name__ == "__main__":
    main()
