from openai import OpenAI

base_url = "https://api.aimlapi.com/v1"
api_key = "c1b1d7df5eef4e5999962faa66ec81b7"
system_prompt = "You are a doctor managing diabetes and hypertension. Provide daily insights on food, sleep, and exercise to help regulate blood sugar and blood pressure."
user_prompt = "The patient has hypertension and high blood sugar. What should they eat, how much sleep do they need, and what type of exercise is best?"

api = OpenAI(api_key=api_key, base_url=base_url)


def main():
    completion = api.chat.completions.create(
        model="gpt-4o-2024-05-13",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.7,
        max_tokens=256,
    )

    response = completion.choices[0].message.content

    print("User:", user_prompt)
    print("AI:", response)


if __name__ == "__main__":
    main()
