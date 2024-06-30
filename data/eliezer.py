import os
import multiprocessing as mp
import numpy as np
import tiktoken
import requests
from tqdm import tqdm
from bs4 import BeautifulSoup
import re
import json

# Set up directories and constants
local_dir = "yudkowsky_lesswrong"
shard_size = int(1e6)  # 1M tokens per shard

# Create the local directory if it doesn't exist yet
DATA_CACHE_DIR = os.path.join(os.path.dirname(__file__), local_dir)
os.makedirs(DATA_CACHE_DIR, exist_ok=True)

local_data_file = os.path.join(
    DATA_CACHE_DIR, "documents.json"
)  # so we don't have to fetch the data again

# GraphQL endpoint and queries
graphql_endpoint = "https://www.lesswrong.com/graphql"
user_id = "nmk3nLpQE89dMRzzN"

posts_query = (
    """
{
  posts(input: {
    terms: {
      view: "userPosts"
      userId: "%s"
      limit: 9999
      meta: null
    }
  }) {
    results {
      _id
      htmlBody
    }
  }
}
"""
    % user_id
)

comments_query = (
    """
{
  comments(input: {
    terms: {
      view: "userComments"
      userId: "%s"
      limit: 9999
    }
  }) {
    results {
      _id
      htmlBody
    }
  }
}
"""
    % user_id
)


def strip_html(html):
    if not html:
        return ""
    try:
        soup = BeautifulSoup(html, "html.parser")
        return soup.get_text()
    except:
        print("FAILED TO STRIP; returning original html")
        return html


# Function to fetch data from the GraphQL endpoint
def fetch_data(query):
    graphql_endpoint = "https://www.lesswrong.com/graphql"
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36",
        "Origin": "https://www.lesswrong.com",
        "Referer": "https://www.lesswrong.com/",
    }
    response = requests.post(graphql_endpoint, json={"query": query}, headers=headers)
    response.raise_for_status()
    data = response.json()
    # print data headers
    key = "posts" if "posts" in data["data"] else "comments"
    return [strip_html(item["htmlBody"]) for item in data["data"][key]["results"]]


# Fetch posts and comments
if os.path.exists(local_data_file):
    print("Loading from local file")
    with open(local_data_file, "r") as file:
        documents = json.load(file)
else:
    print("Fetching posts and comments")
    # Fetch posts and comments
    posts = fetch_data(posts_query)
    comments = fetch_data(comments_query)
    documents = posts + comments

    np.random.shuffle(documents)

    # Save the documents to the local file
    with open(local_data_file, "w") as file:
        json.dump(documents, file)

print(str(len(documents)) + " documents")
print("Avg len for document:", sum(len(doc) for doc in documents) / len(documents))

# Initialize the tokenizer
enc = tiktoken.get_encoding("gpt2")
eot = enc._special_tokens["<|endoftext|>"]


def tokenize(doc):
    # Tokenize a single document and return a numpy array of uint16 tokens
    tokens = [eot]  # The special token delimits all documents
    tokens.extend(enc.encode_ordinary(doc))
    tokens_np = np.array(tokens)
    assert (0 <= tokens_np).all() and (
        tokens_np < 2**16
    ).all(), "Token dictionary too large for uint16"
    tokens_np_uint16 = tokens_np.astype(np.uint16)
    return tokens_np_uint16


def write_datafile(filename, tokens_np):
    np.save(filename, tokens_np)


if __name__ == "__main__":
    # Tokenize all documents and write output shards, each of shard_size tokens (last shard has remainder)
    nprocs = max(1, os.cpu_count() // 2)
    with mp.Pool(nprocs) as pool:
        shard_index = 0
        # Preallocate buffer to hold current shard
        all_tokens_np = np.empty((shard_size,), dtype=np.uint16)
        token_count = 0
        progress_bar = None
        for tokens in pool.imap(tokenize, documents, chunksize=16):

            # Is there enough space in the current shard for the new tokens?
            if token_count + len(tokens) < shard_size:
                # Simply append tokens to current shard
                all_tokens_np[token_count : token_count + len(tokens)] = tokens
                token_count += len(tokens)
                # Update progress bar
                if progress_bar is None:
                    progress_bar = tqdm(
                        total=shard_size, unit="tokens", desc=f"Shard {shard_index}"
                    )
                progress_bar.update(len(tokens))
            else:
                # Write the current shard and start a new one
                filename = os.path.join(
                    DATA_CACHE_DIR, f"lesswrong_train_{shard_index:06d}"
                )
                # Split the document into whatever fits in this shard; the remainder goes to next one
                remainder = shard_size - token_count
                progress_bar.update(remainder)
                all_tokens_np[token_count : token_count + remainder] = tokens[
                    :remainder
                ]
                write_datafile(filename, all_tokens_np)
                shard_index += 1
                progress_bar = None
                # Populate the next shard with the leftovers of the current doc
                all_tokens_np[0 : len(tokens) - remainder] = tokens[remainder:]
                token_count = len(tokens) - remainder

        # Write any remaining tokens as the last (val) shard
        if token_count != 0:
            filename = os.path.join(DATA_CACHE_DIR, f"lesswrong_val_{shard_index:06d}")
            write_datafile(filename, all_tokens_np[:token_count])
