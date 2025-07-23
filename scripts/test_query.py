import argparse
import sys
import os

# Add project root to sys.path to allow imports from utils
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from utils.app_utils import query_llmpedia_new, log_debug
from utils.db import db_utils # Needed implicitly for db connection setup
from dotenv import load_dotenv

load_dotenv() # Load environment variables from .env file

# Simple progress callback function
def print_progress(message: str):
    print(f"[PROGRESS] {message}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test the query_llmpedia_new function.")
    parser.add_argument("question", type=str, help="The question to ask LLMpedia.")
    parser.add_argument("--deep-research", action="store_true", help="Enable deep research mode.")
    parser.add_argument("--iterations", type=int, default=3, help="Max iterations for deep research.")
    parser.add_argument("--max-sources", type=int, default=10, help="Maximum number of sources to consider.")
    parser.add_argument("--response-length", type=int, default=4000, help="Target response length in words.")
    parser.add_argument(
        "--debug", 
        type=str, 
        choices=['true', 'false'], 
        default='false', 
        help="Enable debug logging (true/false)."
    )
    parser.add_argument( 
        "--query-model", 
        type=str, 
        default="claude-3-7-sonnet-20250219",
        help="LLM model for query generation and decision making."
    )
    parser.add_argument(
        "--response-model", 
        type=str, 
        default="claude-3-7-sonnet-20250219",
        help="LLM model for response generation."
    ) 

    args = parser.parse_args()

    # Interpret the string args.debug as a boolean
    effective_debug = (args.debug == 'true')

    if effective_debug:
        print("--- Arguments ---")
        print(f"Question: {args.question}")
        print(f"Deep Research: {args.deep_research}")
        print(f"Iterations: {args.iterations}")
        print(f"Max Sources: {args.max_sources}")
        print(f"Response Length: {args.response_length}")
        print(f"Query/Decision Model: {args.query_model}")
        print(f"Response Model: {args.response_model}")
        print("-----------------")

    # Ensure DB connection is possible (using dotenv for credentials)
    try:
        # Attempt to access the dictionary to check if loaded
        if db_utils.db_params:
             print("Database connection parameters loaded.")
        else:
            # This case might not be reachable if db_params is defined but empty,
            # but included for completeness.
            raise ValueError("db_params dictionary is empty or not loaded.")
    except AttributeError:
        print("Error: db_utils.db_params dictionary not found.")
        print("Make sure database utilities are initialized correctly and .env is loaded.")
        sys.exit(1)
    except Exception as e:
        print(f"Error accessing DB parameters: {e}")
        print("Make sure your .env file is set up correctly and db_utils defines db_params.")
        sys.exit(1)

    print(f"\n--- Running query_llmpedia_new ({'Deep' if args.deep_research else 'Standard'} Search) ---")
    
    try:
        answer, referenced_codes, relevant_codes = query_llmpedia_new(
            user_question=args.question,
            response_length=args.response_length,
            query_llm_model=args.query_model, 
            rerank_llm_model=args.query_model, # Use same model for rerank for simplicity
            response_llm_model=args.response_model,
            max_sources=args.max_sources,
            debug=effective_debug,
            progress_callback=print_progress,
            deep_research=args.deep_research,
            deep_research_iterations=args.iterations
        )

        print("\n--- Results ---")
        if args.deep_research:
            print("[NOTE: Deep research response is synthesized SOLELY from the internal scratchpad.]")
            
        print("\n## Answer:")
        print(answer)
        print("\n## Referenced arXiv Codes:")
        print(referenced_codes if referenced_codes else "None")
        print("\n## Additional Relevant arXiv Codes:")
        print(relevant_codes if relevant_codes else "None")
        print("-------------")

    except Exception as e:
        print(f"\n--- ERROR --- ")
        print(f"An error occurred during the query: {e}")
        import traceback
        traceback.print_exc()
        print("-------------") 