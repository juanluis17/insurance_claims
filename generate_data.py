import os
import openai
import pandas as pd
import time

openai.api_key = os.getenv("OPENAI_API_KEY")


def main():
    domains = ["life", "car", "home", "health", "sports"]
    iterations = 25
    index = 0
    examples = []
    for domain in domains:
        for i in range(0, iterations):
            try:
                chat_completion = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=[{"role": "user",
                                                                                                 "content": f"Could you generate me a long realistic example of insurance claim in the {domain} domain."}])
                example = chat_completion.choices[0].message.content
                examples.append([index, example.strip(), domain])
                index += 1
                time.sleep(0.5)
            except:
                print("Error")
                time.sleep(15)
        time.sleep(3)

    dataset = pd.DataFrame(examples, columns=["id", "text", "label"])
    if os.path.exists("insurance_dataset.csv"):
        tmp_dataset = pd.read_csv("insurance_dataset.csv", header=0)
        dataset = pd.concat([tmp_dataset, dataset])

    dataset.to_csv("insurance_dataset.csv", index=False)
    print(index)


if __name__ == '__main__':
    main()
