# %%
import os
import json
import argparse
import time
from dotenv import load_dotenv

from google import genai
from google.genai import types
from pydantic import BaseModel
from pydantic import TypeAdapter


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract quotes from articles using Gemini API"
    )
    parser.add_argument(
        "input",
        type=str,
        help="Path to the input JSON file containing articles",
    )
    parser.add_argument(
        "output",
        type=str,
        help="Path to the output JSONL file for batch requests",
    )
    parser.add_argument(
        "--upload-file",
        type=str,
        default="data/upload.jsonl",
        help="Path to the file to upload to Gemini File API",
    )
    parser.add_argument(
        "--file-upload-name",
        type=str,
        default="quote-extraction-requests",
        help="Name for the uploaded file in Gemini",
    )
    parser.add_argument(
        "--batch-job-name",
        type=str,
        default="quote-extraction-job",
        help="Name for the batch job in Gemini",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="quote_prompt.txt",
        help="Custom prompt for the Gemini API",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gemini-2.5-flash",
        help="Gemini model to use for extraction",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="If set, will not make any API calls, just prepare the data",
    )
    return parser.parse_args()


def load_api_key():
    load_dotenv(".env")
    try:
        return os.environ["GEMINI_API_KEY"]
    except KeyError:
        raise ValueError("GEMINI_API_KEY not found in environment variables")


class Quote(BaseModel):
    name: str
    organisation: str
    role: str
    nationality: str
    quote: str
    message: str


list_of_quotes_adapter = TypeAdapter(list[Quote])
list_of_quotes_schema = list_of_quotes_adapter.json_schema()

schema = {
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
        "required": ["name", "organisation", "role", "nationality", "quote", "message"],
    },
}


# %%
def create_request(record: dict, prompt: str):

    article_text = record["body"]

    request_text = f"<article>{article_text}</article>"

    request_dict = {
        "key": record["uri"],
        "request": {
            "contents": [
                {"parts": [{"text": prompt + request_text}]},
            ],
        },
        "config": {
            "responseMimeType": "application/json",
            "responseSchema": schema,
        },
    }
    return request_dict


DEFAULT_PROMPT = """
Extract quotes from the following article and return as structured JSON using the following fields.
    name: The full name of the person or organisation being quoted,
    organisation: The organisation represented by the speaker, 
                  Use the full name of the organisation rather than just the acronym.
    role: The speaker's position within that organisation.
    nationality: The nationality of the speaker, if known, as a 3-letter country code e.g. USA, GBR, ISR, PLE.
                 Return an empty string if unknown.
    quote: The text of the quote, if the quote breaks across the sentence, merge it into on single quote,
    message: A summary of the quote, written so that it can be understood out of context. 
             Do not include the speaker in the message, reproduce as if it was being spoken directly.
             Ensure that the message can be understood independently of the article, and that the subject of the message is clear. 
             For example, do not say "The report is very important", say "The report, which is about climate change, is very important".
             And do not say "The proposals are bad", say "The proposals, which would see taxes rise for small businesses, are bad."

In the people or organisations fields, consider that the results will be bulk processed and deduplication should be minimised.
Try to present proper names as they would appear in wikipedia, where possible.

There may be several or no quotes in the article, please ensure that you return all quotes. One speaker may have several quotes.

Include direct quotes, e.g., "This is a quote", said the speaker, and indirect quotes, e.g. The speaker said this is a quote.

Here is the article:


"""


def main():
    """
    Main function to extract quotes from articles using Gemini API.
    """

    # Load args and api key from env
    args = parse_args()
    api_key = load_api_key()
    input_path = args.input
    output_path = args.output

    # Load input data
    with open(input_path, "r") as f:
        data = json.load(f)

    # Instantiate Gemini client
    client = genai.Client(api_key=api_key)

    # Load custom prompt if provided
    if args.prompt and os.path.exists(args.prompt):
        with open(args.prompt, "r") as f:
            prompt = f.read()
    else:
        prompt = DEFAULT_PROMPT

    # Prepare batch requests
    with open(args.upload_file, "w") as f:
        for item in data:
            f.write(json.dumps(create_request(item, prompt=prompt)) + "\n")

    print(f"Prepared batch requests in {args.upload_file}")

    if args.dry_run:
        print("Dry run mode, not making API calls.")
        return

    # Upload the file to the File API
    uploaded_file = client.files.upload(
        file=args.upload_file,
        config=types.UploadFileConfig(
            display_name=args.file_upload_name, mime_type="application/jsonl"
        ),
    )
    print(f"Uploaded file: {uploaded_file.name}")

    # Create batch job
    file_batch_job = client.batches.create(
        model=args.model,
        src=uploaded_file.name,
        config={
            "display_name": args.batch_job_name,
        },
    )
    print(f"Created batch job: {file_batch_job.name}")

    job_name = file_batch_job.name
    batch_job = client.batches.get(name=job_name)

    completed_states = set(
        [
            "JOB_STATE_SUCCEEDED",
            "JOB_STATE_FAILED",
            "JOB_STATE_CANCELLED",
            "JOB_STATE_EXPIRED",
        ]
    )

    print(f"Polling status for job: {job_name}")
    batch_job = client.batches.get(name=job_name)  # Initial get

    while batch_job.state.name not in completed_states:
        print(f"Current state: {batch_job.state.name}")
        time.sleep(30)  # Wait for 30 seconds before polling again
        batch_job = client.batches.get(name=job_name)

    print(f"Job finished with state: {batch_job.state.name}")
    if batch_job.state.name == "JOB_STATE_FAILED":
        print(f"Error: {batch_job.error}")

    if batch_job.state.name == "JOB_STATE_SUCCEEDED":
        # The output is in another file.
        result_file_name = batch_job.dest.file_name
        print(f"Results are in file: {result_file_name}")

        print("\nDownloading and parsing result file content...")
        file_content_bytes = client.files.download(file=result_file_name)
        file_content = file_content_bytes.decode("utf-8")

        output_list = []
        for line in file_content.splitlines():
            if line.strip():
                output_list.append(json.loads(line))

        with open(output_path, "w") as f:
            json.dump(output_list, f, indent=4, ensure_ascii=False)

        print(f"Results written to {output_path}")


if __name__ == "__main__":
    main()
