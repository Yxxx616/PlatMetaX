
from openai import OpenAI, error
import time
def process_string(prompt):
    try:
        client = OpenAI(
            api_key = "sk-QNGrTzzP8BHcpsUfBtpQ8xmwrzfKqhyBZvHz9sVhM4q4tF8a",
            base_url = "https://api.moonshot.cn/v1",
        )
        
        completion = client.chat.completions.create(
            model = "moonshot-v1-8k",
            messages = [
                {"role": "system", "content": prompt}
            ],
            temperature = 0.3,
        )
        time.sleep(30)
        return completion.choices[0].message.content
    except error.RateLimitError as e:
        time.sleep(60)  # 等待 1 秒后重试
        return process_string(prompt)
    