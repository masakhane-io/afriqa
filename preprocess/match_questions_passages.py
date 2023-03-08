import re
import os
import pandas as pd
import json
import argparse

def regex_match(text, pattern):
    """Test if a regex pattern is contained within a text."""
    try:
        pattern = re.compile(
            pattern,
            flags=re.IGNORECASE + re.UNICODE + re.MULTILINE,
        )
    except BaseException:
        return False
    return pattern.search(text) is not None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query_file", type=str, required=True)
    parser.add_argument("--passage__annotation_file", type=str, required=True)

    args = parser.parse_args()

    query_df = pd.read_csv(args.query_file, sep="\t", header=0, dtype=object)
    
    with open(args.passage__annotation_file, "r") as f:
        passage_annotation = json.load(f)

    print("Number of queries: ", len(passage_annotation["data"]))




if __name__ == "__main__":
    main()