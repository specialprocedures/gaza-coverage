# %%
import os
import json
import argparse
import time
from dotenv import load_dotenv
import sys

from google import genai
from google.genai import types


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


def setup() -> tuple:
    """
    Setup Gemini client, load input data and prompt.

    Returns:
        tuple: Gemini client, input data, and prompt string.
    """

    # Load args and api key from env
    args = parse_args()
    api_key = load_api_key()

    # Load input data
    with open(args.input, "r") as f:
        data = json.load(f)

    # Instantiate Gemini client
    client = genai.Client(api_key=api_key)

    # Load custom prompt if provided
    if args.prompt and os.path.exists(args.prompt):
        with open(args.prompt, "r") as f:
            prompt = f.read()
    else:
        prompt = DEFAULT_PROMPT

    return args, client, data, prompt


def write_jsonl(
    data: list, path: str, prompt: str = DEFAULT_PROMPT, schema: dict = SCHEMA
):
    """
    Write a list of dictionaries to a JSONL file. Each dictionary is processed
    into the format required by the Gemini API.

    Args:
        data (list): List of dictionaries to write.
        path (str): Path to the output JSONL file.
        prompt (str): The prompt to prepend to each article text.
    """
    with open(path, "w") as f:
        for item in data:
            processed_item = create_request(item, prompt=prompt, schema=schema)
            f.write(json.dumps(processed_item) + "\n")


def get_processed_data(
    client: genai.Client, file_batch_job, args: argparse.Namespace
) -> str | None:
    """
    Poll the batch job status until completion and write the results to a file.

    Args:
        client: The Gemini API client.
        file_batch_job: The batch job object returned from job creation.
        args: The command line arguments.
    Returns:
        The content of the result file if the job succeeded, else None.
    """

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

        with open(args.output, "w") as f:
            json.dump(output_list, f, indent=4, ensure_ascii=False)

        print(f"Results written to {args.output}")

        return file_content

    elif batch_job.state.name == "JOB_STATE_FAILED":
        print(f"Error: {batch_job.error}")

    elif batch_job.state.name == "JOB_STATE_CANCELLED":
        print("Job was cancelled.")

    elif batch_job.state.name == "JOB_STATE_EXPIRED":
        print("Job has expired.")

    return


def main():
    """
    Main function to extract quotes from articles using Gemini API.
    """

    # Setup client, data and prompt
    args, client, data, prompt = setup()

    # Write jsonl for upload
    write_jsonl(data, path=args.upload_file, prompt=prompt, schema=SCHEMA)
    print(f"Prepared batch requests in {args.upload_file}")

    # Quit if dry run
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

    # Poll for job completion and get results
    job_file_content = get_processed_data(client, file_batch_job, args)

    if job_file_content:
        print("Batch processing completed successfully.")
        sys.exit(0)
    else:
        print("Batch processing did not complete successfully.")
        sys.exit(1)


if __name__ == "__main__":
    main()
