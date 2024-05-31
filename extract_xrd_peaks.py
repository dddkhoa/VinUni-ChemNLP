import requests
import base64

PROMPT = """
You are an experienced material scientist. Analyze the given figure, which is the result of an X-ray diffraction (XRD) analysis of metal-organic frameworks (MOF). In XRD analysis, X-rays interact with the crystalline structure of a material, generating peaks in the resulting figure. The figure may contain one or more lines representing different compounds.

For each line of the figure:

1. Identify the top 5 peaks based on their intensity, ordered from highest to lowest intensity. Intensity refers to the relative height of a peak in the recorded pattern.
2. Report the width of each top peak in the x-axis range (2Theta values) based on the full width at the base of the peak. For example, if the base of the peak starts at 10 degrees and ends at 15 degrees, report it as [10, 15] degrees. If you cannot clearly determine the base of a peak due to merging with neighboring peaks, return "N/A".

Please strictly adhere to the templates provided.

***
Template for your answer:
Compound: [Compound name]
Peak 1: [start,end]
Peak 2: [start,end]
Peak 3: [start,end]
Peak 4: [start,end]
Peak 5: [start,end]
---
Compound: [Compound name]
Peak 1: [start,end]
Peak 2: [start,end]
Peak 3: [start,end]
Peak 4: [start,end]
Peak 5: [start,end]
---
...
***
"""

few_shot_images = ["ICL_examples/2.png", "ICL_examples/3.png"]


def encode_image(image_path):
    """Encodes the image for OpenAI API using base64 encoding.

    Args:
        image_path (str): Path to the image file.

    Returns:
        str: Base64 encoded string of the image data.
    """

    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def analyze_image(api_key, test_image):
    """Analyzes the image using OpenAI's API.

    Args:
        image_path (str): Path to the image file.
        api_key (str): OpenAI API key.

    Returns:
        str: API response with image analysis text, or an empty string if an error occurs.
    """
    prompt = PROMPT
    #    base64_image = encode_image(image_path)
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
    payload = {
        "model": "gpt-4-vision-preview",
        "messages": [
            {"role": "system", "content": prompt},
            {
                "role": "user",
                "content": [
                    #    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{encode_image(few_shot_images[0])}"
                        },
                    },
                ],
            },
            {
                "role": "assistant",
                "content": """
                    Compound: Cotton
                    Peak 1: [21,22.8]
                    Peak 2: [13,14]
                    Peak 3: [16,17]
                    Peak 4: N/A
                    Peak 5: N/A
                    ---
                    Compound: HKUST
                    Peak 1: [11,11.5]
                    Peak 2: [9,10]
                    Peak 3: [18,19]
                    Peak 4: N/A
                    Peak 5: N/A
                    ---
                    Compound: Sample
                    Peak 1: [21,22.7]
                    Peak 2: [13,14]
                    Peak 3: [16,17]
                    Peak 4: [18,19]
                    Peak 5: [19,20]
                """,
            },
            {
                "role": "user",
                "content": [
                    #    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{encode_image(few_shot_images[1])}"
                        },
                    },
                ],
            },
            {
                "role": "assistant",
                "content": """
                    Compound: N/A
                    Peak 1: [35.5,36.5]
                    Peak 2: [32.5,33.5]
                    Peak 3: [34.5,35]
                    Peak 4: [56.5,57.5]
                    Peak 5: [47,48]
                """,
            },
            {
                "role": "user",
                "content": [
                    #    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{encode_image(test_image)}"
                        },
                    },
                ],
            },
        ],
        "max_tokens": 300,
    }

    try:
        response = requests.post(
            "https://api.openai.com/v1/chat/completions", headers=headers, json=payload
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
    except Exception as e:
        print(e)
        return ""


def main():
    api_key = ""
    test_image = ""  # TODO: input test image path here

    test_peaks_result = analyze_image(api_key, test_image)
    print(test_peaks_result)
