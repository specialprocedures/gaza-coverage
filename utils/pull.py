import os
import argparse
import json
from pathlib import Path
import requests
from dotenv import load_dotenv
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(
        prog="pull.py",
        description="Pulls data from newsapi.ai and saves it to a local json file.",
    )

    parser.add_argument(
        "query_json",
        default="query.json",
        help="The json query in newsapi format, created via the API sandbox.",
    )
    parser.add_argument(
        "output_json",
        default="output.json",
        help="The output path to which the results will be saved.",
    )
    parser.add_argument(
        "--api_key",
        help="Your newsapi.ai API key. Can be passed as an argument, set as an environment variable (NEWSAPI_API_KEY), or stored in a .env.",
    )

    args = parser.parse_args()

    return args


def get_api_key(args_api_key: str):
    """Gets API key for request, starting with cli args, proceeding through environment vars and .env file.

    Args:
        args_api_key (str): An API key passed via argparse, if any.

    Returns:
        api_key: The API key for use in the script.
    """

    # If the user passed an API key in the cli, use that
    if args_api_key:
        return args_api_key

    # Next check for an environment variable
    env_key = os.getenv("NEWSAPI_API_KEY")
    if env_key:
        return env_key

    # Finally try for a dotenv
    load_dotenv()
    dotenv_key = os.getenv("NEWSAPI_API_KEY")
    if dotenv_key:
        return dotenv_key

    raise TypeError(
        "Error: No API key found. Provide via --api_key, NEWSAPI_API_KEY env var, or .env file."
    )


def get_page(query_params: dict, page: int):
    """Returns a single page of articles based on the query parameters
    and page number provided.

    Args:
        query_params (dict): The parameters for the query, built using newsapi sandbox
        page (int): The results page to pull

    Returns:
        data (dict): The response data from the request
    """
    # Set endpoint
    ENDPOINT = "https://eventregistry.org/api/v1/article/getArticles"

    # Set page to pull
    query_params["articlesPage"] = page

    # Pull down the data, do basic checks and instantiate the data object from request
    r = requests.post(ENDPOINT, json=query_params)
    r.raise_for_status()
    data = r.json()["articles"]

    return data


def main(args):
    """Main function to pull data from newsapi.ai and save to a local json file.
    Args:
        args (argparse.Namespace): The parsed arguments from argparse.
    """

    # check output dir exists
    out_path = Path(args.output_json)

    if not out_path.parent.exists():
        raise FileNotFoundError(f"Output directory {out_path.parent} does not exist.")

    # Get API key from cli, env or dotenv
    API_KEY = get_api_key(args.api_key)

    # Load query parameters from JSON file
    with open(args.query_json, "r") as file:
        query_params = json.load(file)

    # Add the API key to the query parameters
    query_params["apiKey"] = API_KEY

    all_articles = []
    page = 1
    num_pages = 0

    # Instantiate progress bar
    with tqdm() as pbar:

        # Loop until we hit the total pages
        while True:

            # Check we're not at the end
            if page > 1 and page > num_pages:
                break

            # Pull data
            data = get_page(query_params, page)
            num_pages = data["pages"]

            # Set progress total
            if pbar.total != num_pages:
                pbar.total = num_pages
                pbar.refresh()

            # Update the output object
            all_articles.extend(data["results"])

            # Paginate
            page += 1
            pbar.update(1)

    with open(args.output_json, "w") as f:
        json.dump(all_articles, f, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    args = parse_args()
    main(args=args)
