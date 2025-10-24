import csv
import sys
import re
from openai import OpenAI
import datetime
from tqdm import tqdm

def gpt(prompt):
    # Ensure prompt is a string
    if not isinstance(prompt, str):
        prompt = str(prompt)

    response = client.chat.completions.create(
        model="gpt-4.1",
        messages=[
            {
               "role": "system",
               "content": "Assistant is a large language model trained by OpenAI."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        max_tokens=4096,
    )

    parsed = response.choices[0].message.content

    # print(parsed) #debugging

    return parsed

def parser():
    prompts = []

    with open(sys.argv[1]) as file:
       reader = csv.reader(file)

       for row in reader:
           prompts.append(row[0])
    file.close()

    return prompts

def clean(resp):
    pattern = r"(?i)(^certainly.*?[.:!])(?:\s*(.*?:))?"
    cleaned_text = re.sub(pattern, "", resp)

    return cleaned_text


def wordsLen(str):
    return len(str.split())



api_key = open("gpt_api_key.txt").read().strip()

#main method:
client = OpenAI(api_key=api_key)

prompts = parser()

outputfile = sys.argv[1].replace('.csv', '_responses_4.csv')
with open(outputfile, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    for prompt in tqdm(prompts):
        cleaned_response = ""
        while len(cleaned_response) < 100:
            response = gpt(prompt)
            cleaned_response = clean(response)

        writer.writerow([datetime.datetime.now(), str(prompt), response, wordsLen(response)])

csvfile.close()
