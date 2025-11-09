#!/usr/bin/env python3
"""
Standalone script to generate submission files
"""

import argparse
import yaml
from src.submission import SubmissionGenerator


def main():
    parser = argparse.ArgumentParser(description='Generate submission files')
    parser.add_argument('--model', choices=['rf', 'nn', 'ensemble'], default='ensemble',
                        help='Model to use for submission')
    parser.add_argument('--output', default='submission.csv',
                        help='Output file name')

    args = parser.parse_args()

    # Load configuration
    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)

    print("=== Generating Submission File ===")

    submission_gen = SubmissionGenerator(config)

    if args.model == 'ensemble':
        submission = submission_gen.generate_ensemble_submission(args.output)
    else:
        submission = submission_gen.generate_submission(args.model, args.output)

    if submission is not None:
        print(f"✅ Submission file '{args.output}' generated successfully!")
        print(f"📊 Total predictions: {len(submission)}")
        print(f"🔢 First 5 predictions: {list(submission['Label'].head())}")
    else:
        print("❌ Failed to generate submission file")


if __name__ == "__main__":
    main()