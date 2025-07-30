#!/usr/bin/env python3
"""
Simple test script for the deep_research.py implementation.
"""

import time
import argparse
from deep_research import deep_research_query

def progress_reporter(message: str):
    """Progress callback function to display intermediate results."""
    timestamp = time.strftime("%H:%M:%S")
    print(f"[{timestamp}] {message}")

def test_basic_query(user_question: str):
    """Test the deep research implementation with a basic query."""
    print("🔬 Testing Deep Research Implementation")
    print("=" * 50)
    
    print(f"Research Question: {user_question}")
    print("\n🚀 Starting deep research process with progress reporting...")
    print("-" * 50)
    
    try:
        # Run deep research with progress reporting and verbose mode
        title, workflow_id, response, referenced_arxiv_codes, referenced_reddit_codes, additional_arxiv_codes, additional_reddit_codes = deep_research_query(
            user_question=user_question,
            max_agents=3,
            max_sources_per_agent=15,
            response_length=250,
            llm_model="openai/gpt-4.1-nano",
            progress_callback=progress_reporter,
            verbose=True
        )
        
        print("\n" + "=" * 50)
        print("✅ Research completed successfully!")
        print("=" * 50)
        print("\n📋 Results Summary:")
        print(f"Response length: {len(response.split())} words")
        print(f"Referenced arXiv papers: {len(referenced_arxiv_codes)}")
        print(f"Referenced Reddit posts: {len(referenced_reddit_codes)}")
        print(f"Additional arXiv papers: {len(additional_arxiv_codes)}")
        print(f"Additional Reddit posts: {len(additional_reddit_codes)}")
        
        print("\n📄 Full Response:")
        print("-" * 30)
        print(response)
        
        if referenced_arxiv_codes:
            print(f"\n📚 Referenced arXiv Papers:")
            for code in referenced_arxiv_codes[:5]:
                print(f"  - arxiv:{code}")
        
        if referenced_reddit_codes:
            print(f"\n💬 Referenced Reddit Posts:")
            for code in referenced_reddit_codes[:5]:
                print(f"  - r/{code}")
        
        if additional_arxiv_codes:
            print(f"\n📖 Additional arXiv Papers:")
            for code in additional_arxiv_codes[:5]:
                print(f"  - arxiv:{code}")
        
        if additional_reddit_codes:
            print(f"\n💭 Additional Reddit Posts:")
            for code in additional_reddit_codes[:5]:
                print(f"  - r/{code}")
    except Exception as e:
        print(f"\n❌ Test failed with error: {str(e)}")
        import traceback
        traceback.print_exc()

def main():
    """Main function to handle command line arguments."""
    parser = argparse.ArgumentParser(
        description="Test the deep research implementation with a custom query.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python test_deep_research.py "What evidence is there of LLMs being self-aware?"
  python test_deep_research.py "How do transformer attention mechanisms work?"
  python test_deep_research.py "What are the latest advances in RAG systems?"
        """
    )
    
    parser.add_argument(
        "query",
        type=str,
        help="The research question to investigate (enclose in quotes if it contains spaces)"
    )
    
    args = parser.parse_args()
    test_basic_query(args.query)

if __name__ == "__main__":
    main()
    