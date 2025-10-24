from google import genai
import sys
import argparse

from gemini.extract import load_api_key


def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Cancel a Gemini batch job.")
    parser.add_argument(
        "--job-id",
        type=str,
        required=True,
        help="The batch job ID to cancel.",
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
    Main function to cancel a batch job.
    """
    # Load args and api key from env
    args = parse_args()
    api_key = load_api_key()

    # Instantiate Gemini client
    client = genai.Client()

    # Get the batch job by ID
    batch_job_to_cancel = client.batches.get(name=args.job_id)
    if not batch_job_to_cancel:
        print(f"Batch job with ID {args.job_id} not found.")
        sys.exit(1)

    print(f"Cancelling batch job with ID {args.job_id}...")
    # Cancel a batch job
    client.batches.cancel(name=batch_job_to_cancel.name)

    print(f"Batch job with ID {args.job_id} has been cancelled.")


if __name__ == "__main__":
    main()
