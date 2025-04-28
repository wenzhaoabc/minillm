import json
from jinja2 import Template


def test_chat_template():
    # 加载 tokenizer_config.json
    with open("tokenizer/tokenizer_config.json", "r", encoding="utf-8") as f:
        config = json.load(f)

    # 获取 chat_template
    chat_template = config["chat_template"]

    # 定义测试消息
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the capital of France?"},
        {"role": "assistant", "content": "The capital of France is Paris."},
        {"role": "user", "content": "Thank you!"},
    ]

    # 渲染模板
    template = Template(chat_template)
    rendered_output = template.render(messages=messages)

    # 打印结果
    print("Rendered Chat Template:")
    print(rendered_output)


if __name__ == "__main__":
    test_chat_template()
