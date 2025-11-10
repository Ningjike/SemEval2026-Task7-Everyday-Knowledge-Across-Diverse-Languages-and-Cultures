# translate_Qwen.py
import openai
from openai import APIError
import pandas as pd
from googletrans import Translator, LANGUAGES

import time

# 配置API客户端
client = openai.OpenAI(
    api_key="XXXX",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

def translate_to_english(text: str, lang_region: str, max_retries=2) -> str:
    if not text or pd.isna(text):
        return ""

    # 如果是英文，直接返回
    if lang_region.startswith("en"):
        return str(text).strip()

    # 提取语言代码
    lang_code = lang_region.split('-')[0].lower()

    for attempt in range(max_retries):
        try:
            source_lang = "auto" if attempt > 0 else lang_code

            completion = client.chat.completions.create(
                model="qwen-mt-turbo",
                messages=[{"role": "user", "content": str(text)}],
                extra_body={
                    "translation_options": {
                        "source_lang": source_lang,
                        "target_lang": "en"
                    }
                },
                timeout=10
            )
            time.sleep(0.8)  # 控制 QPS，避免 429
            return completion.choices[0].message.content.strip()

        except APIError as e:
            error_code = getattr(e, 'code', None)
            error_message = str(e)

            # 如果是 400 + “不支持语种”，且还没重试过，尝试用 auto
            if attempt == 0 and error_code == "invalid_parameter_error" and "不支持当前设置的语种" in error_message:
                print(f"[Fallback] 语言 {lang_code} 不支持，尝试自动识别... (text='{str(text)[:30]}...')")
                continue
            else:
                print(f"[ERROR] 翻译失败 ({lang_region}): {error_message}")
                break

        except Exception as e:
            print(f"[UNEXPECTED ERROR] {e}")
            break

    return str(text)



def translate_with_google(text: str, target_lang: str) -> str:
    try:
        translator = Translator()
        result = translator.translate(text, src='en', dest=target_lang)
        return result.text
    except Exception as e:
        print(f"[Google Translate Error] {e}")
        return text

def translate_answer_back(english_answer: str, lang_region: str, max_retries=2) -> str:
    """
    将英文答案翻译回原始语言（根据 lang_region 中的语言部分）
    """
    if not english_answer or pd.isna(english_answer) or english_answer.startswith("["):
        return english_answer

    # 提取语言代码
    lang_code = lang_region.split('-')[0].lower()

    # 如果本来就是英文，无需回译
    if lang_region.startswith("en"):
        return english_answer

    for attempt in range(max_retries):
        try:
            target_lang = "auto"
            completion = client.chat.completions.create(
                model="qwen-mt-turbo",
                messages=[{"role": "user", "content": english_answer}],
                extra_body={
                    "translation_options": {
                        "source_lang": "en",
                        "target_lang": lang_code
                    }
                },
                timeout=10
            )
            time.sleep(0.8)  # 控制请求频率
            translated = completion.choices[0].message.content.strip()
            return translated

        except APIError as e:
            error_code = getattr(e, 'code', None)
            error_msg = str(e)
            if attempt == 0 and error_code == "invalid_parameter_error" and "不支持当前设置的语种" in error_msg:
                print(f"[Back-trans Fallback] 语言 {lang_code} 不支持，跳过回译 (ans='{english_answer[:30]}...')")
                return english_answer
            else:
                print(f"[ERROR] 回译失败 ({lang_region}): {error_msg}")
                break
        except Exception as e:
            print(f"[UNEXPECTED ERROR in back-translation] {e}")
            break

    return english_answer

