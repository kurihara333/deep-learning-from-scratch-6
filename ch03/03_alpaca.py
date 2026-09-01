import os, sys
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.append('.')

import json
from codebot.tokenizer import BPETokenizer

# トークナイザの読み込み
tokenizer = BPETokenizer.load_from('codebot/merge_rules.pkl')

# JSONデータの読み込み
with open('codebot/tiny_codes_sft.json') as f:
    data = json.load(f)

print(f"データ件数: {len(data)}")

# 1つ目のサンプルを取り出す
item = data[0]
print(item)
# {'instruction': 'Hello', 'response': 'Hello. What can I help you with?'}

# Alpaca形式に変換
text = f"### Instruction:\n{item['instruction']}\n\n### Response:\n{item['response']}<|endoftext|>"
print(text)
# ### Instruction:
# Hello
#
# ### Response:
# Hello. What can I help you with?<|endoftext|>

# トークン化
ids = tokenizer.encode(text)
print(ids)
# [35, 35, 35, 962, 519, 117, 389, 58, 10, 846, 10, 10, 35, 35, 35, 752, 568, 58, 10, 846, 46, 840, 104, 277, 280, 356, 473, 708, 108, 112, 930, 657, 63, 999]

print(f"\n文字数: {len(text)}, トークン数: {len(ids)} (圧縮率 {len(text)/len(ids):.2f}文字/トークン)")

# デコードして元のテキストに戻るか確認(エンコード⇔デコードの往復)
decoded = tokenizer.decode(ids)
print("\nデコード結果:")
print(decoded)
print("元のtextと完全一致:", decoded == text)

# 別サンプル(3件目)も見てみる
item2 = data[2]
print(f"\n--- 別サンプル(3件目) ---")
print("instruction:", item2['instruction'])
print("response:", item2['response'])