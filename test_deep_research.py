#!/usr/bin/env python3
"""
Simple test script for the deep_research.py implementation.
"""

import time
from deep_research import deep_research_query

def progress_reporter(message: str):
    """Progress callback function to display intermediate results."""
    timestamp = time.strftime("%H:%M:%S")
    print(f"[{timestamp}] {message}")

def test_basic_query():
    """Test the deep research implementation with a basic query."""
    print("🔬 Testing Deep Research Implementation")
    print("=" * 50)
    
    # Test query
    user_question = "What evidence is there of LLMs being self-aware?"
    
    print(f"Research Question: {user_question}")
    print("\n🚀 Starting deep research process with progress reporting...")
    print("-" * 50)
    
    try:
        # Run deep research with progress reporting and verbose mode
        response, referenced_codes, additional_codes = deep_research_query(
            user_question=user_question,
            max_agents=5,
            max_sources_per_agent=25,
            response_length=300,
            llm_model="gemini/gemini-2.5-flash",
            progress_callback=progress_reporter,
            verbose=True
        )
        
        print("\n" + "=" * 50)
        print("✅ Research completed successfully!")
        print("=" * 50)
        print("\n📋 Results Summary:")
        print(f"Response length: {len(response.split())} words")
        print(f"Referenced papers: {len(referenced_codes)}")
        print(f"Additional relevant papers: {len(additional_codes)}")
        
        print("\n📄 Full Response:")
        print("-" * 30)
        print(response)
        
        if referenced_codes:
            print(f"\n📚 Referenced Papers:")
            for code in referenced_codes[:5]:
                print(f"  - arxiv:{code}")
        
        if additional_codes:
            print(f"\n📖 Additional Relevant Papers:")
            for code in additional_codes[:5]:
                print(f"  - arxiv:{code}")
    except Exception as e:
        print(f"\n❌ Test failed with error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_basic_query()
    