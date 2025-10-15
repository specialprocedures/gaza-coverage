import argparse
import json
import os
import sys

DEFAULT_PROMPT = """
Extract quotes from the following article and return as structured JSON using the following fields.
    name: The full name of the person or organisation being quoted,
            - Do not use a person's first or last name only, unless they are only referred to that way in the article.
            - Even then, if they are famous (e.g., "Donald Trump", "Joe Biden"), always use their full name.
    organisation: The organisation represented by the speaker, 
                  - Use the full name of the organisation rather than just the acronym.
    role: The speaker's position within that organisation, if given in the article.
    nationality: The nationality of the speaker, if known, as a 3-letter country code e.g. USA, GBR, ISR, PLE.
                 Return an empty string if unknown.
    quote: The text of the quote, if the quote breaks across the sentence, merge it into on single quote,
    message: A summary of the quote, written so that it can be understood out of context. 
             Do not include the speaker in the message, reproduce as if it was being spoken directly.
             Ensure that the message can be understood independently of the article, and that all subjects of the message are clear. 
                For example:
                    - do just not say "I agree with the government's decision", say "I agree with the German government's decision to increase funding for healthcare".
                    - do not just say "The report is very important", say "The report, published by Greenpeace, which is about climate change, is very important".

There may be several or no quotes in the article, please ensure that you return all quotes. One speaker may have several quotes.

Include direct quotes only, do not include paraphrased quotes.

Here is the article:


"""

SCHEMA = {
    "description": "A list of quotes extracted from the article",
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "organisation": {"type": "string"},
            "role": {"type": "string"},
            "nationality": {"type": "string"},
            "quote": {"type": "string"},
            "message": {"type": "string"},
        },
        "required": [
            "name",
            "organisation",
            "role",
            "nationality",
            "quote",
            "message",
        ],
    },
}


def create_request(
    record: dict, prompt: str = DEFAULT_PROMPT, schema: dict = SCHEMA
) -> dict:
    """
    Create a request dictionary for the Gemini API based on the article record.

    Args:
        record (dict): newsapi.ai article data.
        prompt (str): The prompt to prepend to the article text.
    Returns:
        dict: A dictionary formatted for the Gemini API request.
    """

    # Build the request text
    request_text = f"{prompt}<article>{record["body"]}</article>"

    # Construct the request dictionary
    request_dict = {
        "key": record["uri"],
        "request": {
            "contents": [
                {"role": "user", "parts": [{"text": request_text}]},
            ],
        },
        "generationConfig": str(
            {
                "responseMimeType": "application/json",
                "responseSchema": schema,
                "temperature": 0,
            }
        ),
    }
    return request_dict


def main(input_file: str, output_file: str):
    """
    Main function to read articles from input_file, process them, and write requests to output_file.

    Args:
        input_file (str): Path to the input JSONL file containing articles.
        output_file (str): Path to the output JSONL file for Gemini API requests.
    """
    with open(input_file, "r", encoding="utf-8") as infile, open(
        output_file, "w", encoding="utf-8"
    ) as outfile:
        data = json.load(infile)
        for record in data:
            request_dict = create_request(record)
            outfile.write(json.dumps(request_dict) + "\n")
    print(f"Processed articles from {input_file} and wrote requests to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prepare news articles for LLM processing."
    )
    parser.add_argument(
        "input_file", type=str, help="Path to the input JSONL file with articles."
    )
    parser.add_argument(
        "output_file", type=str, help="Path to the output JSONL file for requests."
    )
    args = parser.parse_args()

    if not os.path.isfile(args.input_file):
        print(f"Error: Input file {args.input_file} does not exist.")
        sys.exit(1)

    main(args.input_file, args.output_file)
