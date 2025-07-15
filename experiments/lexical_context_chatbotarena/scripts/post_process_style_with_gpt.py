import os
import json
import openai
import argparse
from tqdm import tqdm
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def analyze_comparison(text):
    """
    Analyze the input text to determine whether it implies:
    - A is better than B
    - B is better than A
    - They are equally good

    :param text: Input text describing the comparison.
    :return: Analysis result as a string.
    """

    # Prepare the prompt
    prompt = f"""
    Originally, a judge was asked to determine the relationship between A and B, on providing verdicts on three categories: "content", "style", and "overall". If the judge worked properly, the verdicts should be in a JSON format already like:
    {{
        "content": "X",
        "style": "Y",
        "overall": "Z"
    }}

    Analyze the following text and determine the relationship between A and B (it might already be in the correct JSON format, if not please make one for me). Respond with one of the following:
    - A: 'A is better than B'
    - B: 'B is better than A'
    - C: 'A and B are equally good'

    Note that A might also be reffered to as Assistant A or first option, and B might also be reffered to as Assistant B or second option.

    Text: "{text}"

    You must strictly output the JSON format like this:
    {{
        "content": "X",
        "style": "X",
        "overall": "X"
    }}
    where X is one of "A", "B", or "C" based on your analysis. Do not include any additional information. Your output will be parsed by `json.loads` in Python, so don't include any other text.
    """

    for i in range(10):
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ],
        )
        try:
            response = completion.choices[0].message.content.strip()
            response = response.strip("```").strip("json").strip()
            parsed = json.loads(response)
            assert "content" in parsed
            assert "style" in parsed
            assert "overall" in parsed
            for v in parsed.values():
                assert v in ["A", "B", "C"]
        except:
            import ipdb

            ipdb.set_trace()
            print("failed at attempt", i + 1)
    # Extract and return the result
    return parsed


def run(
    input_path,
    output_path,
):
    # Read input text from file
    data = []
    with open(input_path, "r") as f:
        for line in f:
            x = json.loads(line)
            data.append(x)
    print("len(data):", len(data))

    outputs = []
    if os.path.exists(output_path):
        with open(output_path, "r") as f:
            for line in f:
                x = json.loads(line)
                outputs.append(x)
        num_done = len(outputs)
    else:
        num_done = 0
    print("num_done = {}".format(num_done))

    for i in tqdm(range(num_done, len(data))):
        x = data[i]  # ['data_path', 'data', 'prompt_text', 'response']
        response = x["response"]

        # Get the analysis result
        processed = analyze_comparison(response)
        x["processed"] = processed

        # print(i, processed)
        with open(output_path, "a") as f:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    # Argument parser setup
    parser = argparse.ArgumentParser(
        description="Analyze comparison between A and B from input text."
    )
    parser.add_argument(
        "--input_path",
        type=str,
        help="Path to the input file containing text to analyze.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        help="Path to the output file to save the analysis result.",
    )

    args = parser.parse_args()
    run(args.input_path, args.output_path)
