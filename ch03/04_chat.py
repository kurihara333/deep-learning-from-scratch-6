import os, sys
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.append('.')

from codebot.model import GPT
from codebot.tokenizer import BPETokenizer
from codebot.utils import generate, get_device

# 設定
device = get_device()
model_path = 'codebot/model_sft.pt'
# model_path = 'codebot/model_grpo.pt'
tokenizer_path = 'codebot/merge_rules.pkl'
max_new_tokens = 200
temperature = 1.0

def format_prompt(user_message):
    return f"### Instruction:\n{user_message}\n\n### Response:\n"

# モデルとトークナイザの読み込み
tokenizer = BPETokenizer.load_from(tokenizer_path)
model = GPT.load_from(model_path, device=device)

print("使用デバイス:", device)
print("使用モデル:", model_path)
print(f"モデルパラメータ数: {sum(p.numel() for p in model.parameters()):,}")

while True:
    try:
        user_input = input("\nYou: ").strip()
    except EOFError:
        print("\n(入力終了)")
        break

    if not user_input:
        continue

    # プロンプトのフォーマットと生成
    prompt = format_prompt(user_input)
    print(f"\n--- モデルに渡す実際のプロンプト ---\n{prompt!r}")

    response = generate(model, tokenizer, prompt, max_new_tokens, temperature)
    print(f"\n--- 生成された生テキスト(抽出前) ---\n{response!r}")

    # アシスタントの応答部分のみ抽出
    if "### Response:" in response:
        response = response.split("### Response:")[-1].strip()

    # 改行が含まれているかで出力形式を切り替える
    if "\n" in response:
        print(f"Bot:\n{response}")
    else:
        print(f"Bot: {response}")