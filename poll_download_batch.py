from extract import load_api_key, get_processed_data, parse_args
from google import genai
import sys
import argparse


def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Poll Gemini batch job and download results."
    )
    parser.add_argument(
        "--job-id",
        type=str,
        required=True,
        help="The batch job ID to poll.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/quotes_extracted.jsonl",
        help="Path to the output JSONL file.",
    )
    parser.add_argument(
        "--api-key-env",
        type=str,
        default="GENAI_API_KEY",
        help="Environment variable name for the Gemini API key.",
    )
    return parser.parse_args()


def main():
    """
    Main function to poll the status of a batch job and download the results
    when complete.
    """
    # Load args and api key from env
    args = parse_args()
    api_key = load_api_key()
    # Instantiate Gemini client
    client = genai.Client(api_key=api_key)

    file_batch_job = client.batches.get(name=args.job_id)

    # Get the batch job by ID
    job_file_content = get_processed_data(client, file_batch_job, args)

    if job_file_content:
        print("Batch processing completed successfully.")
        sys.exit(0)
    else:
        print("Batch processing did not complete successfully.")
        sys.exit(1)


if __name__ == "__main__":
    main()
