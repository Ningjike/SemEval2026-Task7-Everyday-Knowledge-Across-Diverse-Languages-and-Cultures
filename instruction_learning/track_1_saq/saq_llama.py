import ollama
import pandas as pd
from tqdm import tqdm

import time

from translate_qwen import translate_to_english
from translate_qwen import translate_answer_back


INPUT_FILE = "D:\\Pycharm\\SemEval2026\\track_1_saq\\track_1_saq_input.tsv"
MODEL_NAME = "llama3:latest"
DELAY = 0.5

COUNTRY_MAP = {
    'DZ': 'Algeria', 'EG': 'Egypt', 'MA': 'Morocco', 'SA': 'Saudi Arabia',
    'ET': 'Ethiopia', 'NG': 'Nigeria', 'LK': 'Sri Lanka', 'AS': 'India',
    'AZ': 'Azerbaijan', 'CN': 'China', 'SG': 'Singapore', 'IR': 'Iran',
    'JP': 'Japan', 'KP': 'North Korea', 'KR': 'South Korea', 'PH': 'Philippines',
    'ID': 'Indonesia', 'JB': 'Indonesia', 'GR': 'Greece', 'GB': 'United Kingdom',
    'IE': 'Ireland', 'ES': 'Spain', 'MX': 'Mexico', 'EC': 'Ecuador',
    'FR': 'France', 'BG': 'Bulgaria', 'AU': 'Australia', 'US': 'United States',
    'SE': 'Sweden',
}


def get_answer_from_ollama(question_en: str, lang_region: str, model: str = MODEL_NAME) -> str:
    country_name = "the relevant country"

    if '-' in lang_region:
        parts = lang_region.split('-', 1)
        if len(parts) == 2:
            country_code = parts[1]
            # 标准化：转大写（防小写如 'us'）
            country_code = country_code.upper()
            country_name = COUNTRY_MAP.get(country_code, f"the region {country_code}")

    prompt = f"""### Instruction: You are a local resident of {country_name}. Answer the following question in English, concisely and with cultural accuracy. Provide only the essential answer without any explanation, introduction, or punctuation.
                    ### Question: {question_en}
                    ### Response:"""

    response = ollama.chat(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0.1}
    )
    ans = response['message']['content'].strip()
    ans = ans.replace('\n', ' ').replace('\t', ' ').strip('.,;:!?"\'')
    return ans

def main():
    df = pd.read_csv(INPUT_FILE, delimiter="\t", quoting=3)
    df[['lang_region', 'q_id']] = df['id'].str.rsplit('_', n=1, expand=True)

    # 初始化列
    df["question_en"] = ""
    df["answer_en"] = ""
    df["answer"] = ""

    print(f"开始处理 {len(df)} 行...")

    for idx in tqdm(df.index, desc="Processing"):
        orig_q = df.at[idx, "question"]
        lang_reg = str(df.at[idx, "lang_region"])

        # 翻译问题到英文
        q_en = translate_to_english(orig_q, lang_reg)
        df.at[idx, "question_en"] = q_en

        if not q_en or q_en.startswith("["):
            df.at[idx, "answer"] = "[SKIPPED] Translation failed."
            continue

        # 获取英文答案
        ans_en = get_answer_from_ollama(q_en, lang_reg)
        df.at[idx, "answer_en"] = ans_en

        if ans_en.startswith("[ERROR]") or ans_en == "[SKIPPED]":
            df.at[idx, "answer"] = ans_en
            continue

        # 回译答案到原始语言
        final_ans = translate_answer_back(ans_en, lang_reg)
        df.at[idx, "answer"] = final_ans

        time.sleep(DELAY)

    # 保存结果
    df.to_csv("answers.tsv", sep="\t", index=False, quoting=3)

if __name__ == "__main__":
    main()

