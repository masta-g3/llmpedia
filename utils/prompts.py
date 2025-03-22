"""
Prompt templates for the LLMpedia Streamlit application.
"""

from typing import Optional
import datetime

# Today's date for dynamic prompts
todays_date = datetime.datetime.now().strftime("%Y-%m-%d")
recent_date = datetime.datetime.now() - datetime.timedelta(days=7)
recent_date = recent_date.strftime("%Y-%m-%d")

# Base style for prompts
TWEET_BASE_STYLE = """
<style_guide>
- Be technically precise and information-dense, while maintaining accessibility
- Focus on the *unexpected* and *counterintuitive* aspects of the findings
- Use conversational tone while preserving intellectual depth
- Employ a slightly irreverent, direct, and thought-provoking style
- Use format "X isn't Y, it's Z" or "We thought X was about Y, but it's actually about Z"
- Prefer specific numbers/metrics over vague statements (e.g., "76% accuracy" not "high accuracy")
- Include a compelling narrative arc with setup → insight → implications
- Subtly imply broader implications without overreaching
- Use literary techniques like parallel structure, metaphor, and rhythm
- Respect the intelligence of your audience
</style_guide>
"""

# System and user prompts for interrogating papers
INTERROGATE_PAPER_SYSTEM_PROMPT = "You are GPT Maestro, a renowned librarian specialized in Large Language Models. Read carefully the whitepaper and the user question. Provide a comprehensive, helpful and truthful response."

def create_interrogate_user_prompt(context: str, user_question: str) -> str:
    """Create a prompt for asking questions about a specific paper."""
    user_prompt = f"""
    <whitepaper_context>
    {context}
    </whitepaper_context>
    
    <user_query>
    {user_question}
    </user_query>

    <guidelines>
    - Be direct and to the point, using layman's language that is easy to understand.
    - If the question cannot be answered with the provided whitepaper, please respond with 'Sorry, I don't know about that.', and nothing else.
    - Avoid filler content.
    - Reply with your answer in a concise paragraph and nothing else (no preambles, greetings, etc.).
    - Do not make reference to the existance of the whitepaper_context in your response.
    </guidelines>
    """
    return user_prompt

# System prompt for vector store queries
VS_QUERY_SYSTEM_PROMPT = f"""Today is {todays_date}. You are an expert system that can translate natural language questions into structured queries used to search a database of Large Language Model (LLM) related whitepapers."""

# Prompt for deciding query action
def create_decision_user_prompt(user_question: str) -> str:
    """Create a prompt for deciding what type of query action to take."""
    user_prompt = f"""
    <user_query>
    {user_question}
    </user_query>
    
    <response_format>
    Classify the user query into one of the following categories:
    - Question about large language models, AI agents, text embeddings, data rerieval, natural language processing, and similar topics.
    - Question about any other subject (unrelated to LLMs).
    - General comment or feedback.
    </response_format>
    
    If you are not sure, bias your classification towards large language model related queries.
    """
    return user_prompt

# Prompt for generating structured search query
def create_query_user_prompt(user_question: str) -> str:
    """Create a prompt for generating a structured search query."""
    VS_QUERY_USER_PROMPT = f'''
    <response_format> 
    Use the following JSON response format. All fields are optional; when not provided, the system will search across all values for that field. Notice that string fields are case-insensitive. Always use the minimum number of fields necessary to get the desired results; if you don't need a field do not include it in your search query.
    
    {{
        "title": "(str) Title of the paper. Use only when the user is looking for a specific paper. Partial matches will be returned.",
        "min_publication_date": "(str) Minimum publication date of the paper. Use "YYYY-MM-DD" format.",
        "max_publication_date": "(str) Maximum publication date of the paper. Use "YYYY-MM-DD" format.",
        "topic_categories": "(list) List containing the topic categories of the paper. Use only when the user explicitly asks about one of these topics (not for related topics)."
        "semantic_search_queries": "(list) List of queries to be used in the semantic search. The system will use these queries to find papers that have abstracts that are semantically similar to the queries. If you use more than one search query make them diverse enough so that each query addresses a different part of what is needed to build up an answer. Be sure that one of the queries is closely related to the user question. Consider the language typically used in academic papers when writing the queries; phrase the queries as if they were part of the text that could be found on these abstracts.", 
        "min_citations": "(int) Minimum number of citations of the paper."
    }}
    </response_format>
    
    <topic_categories>
    - Adversarial Defense and Red Teaming in LLMs
    - AI Safety and Governance Evaluations
    - automated factuality evaluation and fact-checking in long-form text generation
    - Automated Prompt Optimization Techniques
    - Autonomous Multi-Agent Systems with LLM Coordination and Planning
    - Bias Mitigation in NLP for Hate Speech and Offensive Language Detection
    - Chain of Thought Reasoning in Large Language Models
    - Code Synthesis and Evaluation in Multilingual Programming
    - Comprehension QA and Reasoning Datasets
    - Efficient and Adaptive Fine-Tuning Techniques
    - Efficient and Scalable Attention Mechanisms
    - Efficient Low-bit Model Quantization and Inference Techniques
    - Efficient Scalable Sparse and Mixture-of-Experts Models
    - Empirical Scaling Laws and Optimization in Training Large Neural Networks
    - Financial and Time Series Analysis Applications
    - Hallucination in Language and Vision Models
    - Healthcare and Medical Applications of LLMs
    - Human Preference Optimization in RLHF for Language Models
    - In-Context Learning Mechanisms and Applications
    - Instruction Tuning and Dataset Quality Enhancement
    - LLM Evaluation Metrics and Benchmarks
    - LLM Privacy and Data Leakage Risks
    - Long Context Handling Techniques and Evaluations
    - Low-Rank Adaptation in Fine-Tuning LLMs
    - Mathematical Reasoning Datasets and Models for LLMs
    - Miscellaneous
    - Multilingual Low-Resource Language Adaptation and Translation Strategies
    - Multimodal Vision-Language Embodied Agent Training
    - Open-Domain Conversational AI and Role-Playing Systems
    - Optimizations for KV Cache Memory Efficiency in LLM Inference
    - Optimized Data Selection and Pre-training Efficiency
    - Personalized Multimodal and Explainable Recommendations with LLMs
    - Retrieval-Augmented Generation and Evaluation in Knowledge-Intensive Tasks
    - Specialized Domain LLMs for Scientific Research
    - Speculative Decoding Architectures for High-Efficiency Inference
    - Speech and Audio Multimodal Language Models
    - State Space Models for Efficient Long-Range Sequence Modeling
    - Table Understanding and Text-to-SQL Models
    - Versatile and Efficient Text Embedding Methods
    - Vision-Language Multimodal Models and Image Generation
    </topic_categories>
    
    Now read the following question and reply with the response query and no other comment or explanation.

    <question>
    {user_question}
    </question>
    '''
    return VS_QUERY_USER_PROMPT

# Prompt for reranking documents
def create_rerank_user_prompt(user_question: str, documents: list) -> str:
    """Create a prompt for reranking retrieved documents."""
    document_str = ""
    for idx, doc in enumerate(documents):
        document_str += f"""
    ### Doc ID: {idx}
    ### Title: {doc.title}
    *Published*: {doc.published_date.strftime("%Y-%m-%d")}
    *Citations*: {doc.citations}
    **Abstract**:
    {doc.abstract}
    ---"""
    document_str = document_str.strip()
    rerank_msg = f""""
    <question>
    {user_question}
    </question>

    <documents>
    {document_str}
    </documents>

    <response_format>
    - Reply with a list of JSON object according to the provided schema. Each element must contain the document IDs, plus two additional fields: 'analysis' and 'selected'. 
    - The 'analysis' element should contain a brief analysis of if and why the paper is relevant to the user query. 
    - The 'selected' element should be a float indicating the relevance level:
      * 1.0: Directly relevant and essential for answering the query
      * 0.5: Tangentially relevant or provides supporting context
      * 0.0: Not relevant to the specific query
    - Make sure to be stringent in scoring - only assign 1.0 to papers that are **directly** relevant to answer the specific user query.
    - Be sure to include all the documents in the list, even if scored 0.0.
    </response_format>"""
    return rerank_msg

# Prompt for resolving query with documents
def create_resolve_user_prompt(
    user_question: str, 
    documents: list, 
    response_length: int,
    custom_instructions: Optional[str] = None
) -> str:
    """Create a prompt for generating a response based on retrieved documents."""
    notes = ""
    ## Convert word count to response guidance.
    response_guidance = ""
    if response_length <= 250:
        response_guidance = "\n- Write a focused research note (~250 words) that directly answers the question in a single cohesive paragraph.\n- Emphasize key findings and core concepts while maintaining narrative flow.\n- Use clear topic transitions and supporting evidence."
    elif response_length <= 500:
        response_guidance = "\n- Write a focused research note (~500 words) that directly answers the question in a single cohesive, in-depth paragraph.\n- Emphasize key findings and core concepts while maintaining narrative flow.\n- Use clear topic transitions and supporting evidence."
    elif response_length <= 1000:
        response_guidance = "\n- Write an engaging research summary (~1000 words) that explores the topic through multiple angles.\n- Use 2-3 naturally flowing sections to develop ideas from key findings through implications.\n- Blend technical insights with practical applications, maintaining narrative momentum."
    elif response_length <= 3000:
        response_guidance = "\n- Write an in-depth research analysis (~3000 words) that thoroughly explores the topic's landscape.\n- Structure with clear sections using markdown headers (###) and information-dense paragraphs to guide the reader through your narrative.\n- Progress from core findings through technical details to broader implications, incorporating code examples where they enhance understanding."
    else:
        response_guidance = "\n- Write a comprehensive research report (~5000 words) that covers the full scope of the topic.\n- Use hierarchical markdown headers (##, ###) to create a natural progression through multiple major sections, which will be made by information-rich paragraphs.\n- Weave together theoretical foundations, technical implementations, and practical implications while maintaining narrative cohesion."
    
    for doc in documents:
        notes += f"""
    ### Title: {doc.title}
    *Arxiv Code*: {doc.arxiv_code}
    *Published*: {doc.published_date.strftime("%Y-%m-%d")}
    *Citations*: {doc.citations}
    **Summary**:
    {doc.notes}

    ---"""
    notes = notes.strip()
    
    custom_instructions_section = f"""
    <custom_instructions>
    {custom_instructions}
    </custom_instructions>
    """ if custom_instructions else ""
    
    user_message = f""""
    <question>
    {user_question}
    </question>

    <context>
    {notes}
    </context>

    <output_format>
    Your response should be structured in three parts:

    BRAINSTORM:
    - Analyze each document's relevance and key contributions to answering the question
    - Plan the structure of your response including:
      * Key papers to cite, in chronological/event order
      * Markdown sections and their hierarchy
      * Additional elements needed (code blocks, tables, diagrams)
    - Consider how to maintain narrative flow between sections

    SKETCH:
    - Create a detailed outline using nested markdown lists
    - For each planned section:
      * List key points, findings, and evidence to include
      * Note specific citations and technical details
      * Identify examples and practical applications
      * Plan transitions between ideas
    - Include placeholders for code blocks or other elements

    RESPONSE:
    {response_guidance}
    - Pay special attention to the style guidelines when formulating your response.
    </output_format>

    <instructions>
    - Use narrative writing to provide a complete, direct and useful answer. Structure your response as a mini-report in a magazine.
    - Provide the conclusion of your analysis upfront, and then provide the rest of the report in a narrative manner.
    - Do not mention 'the context'! The user does not have access to it, so do not reference it or the fact that I presented it to you. Act as if you have all the information in your head (i.e.: do not say 'Based on the information provided...', etc.).
    - Make sure your report reads naturally and is easy to follow.
    - Use markdown to add a title to your response (i.e.: '##'), add subtitles, incorporate code blocks (when relevant, informed by the content of the documents), and any other elements that help improve clarity and flow.
    - Use information-dense paragraphs interweaving techncial details with clear examples.
    - Even if your report is extensive, be sure to not get lost in the weeds. Provide clear conclusions and implications upfront.
    - Avoid listicles, enumerations and other repetitive, non-engaging writing styles.
    - Be practical and reference any existing libraries or implementations mentioned on the documents.
    - Prioritize papers that are more recent and have more citations.
    - Present different viewpoints if they exist, but avoid being vague or unclear.
    - Inform your response with the information available in the context, and less so with your own opinions (although you can draw simple connections between ideas and draw conclusions).
    - Organize concepts chronologically, indicating how the field evolved from one stage to the next.
    - Add citations when referencing papers by mentioning the relevant arxiv_codes (e.g.: use the format *reference content* (arxiv:1234.5678)). If you mention paper titles wrap them in double quotes.
    </instructions>

    {custom_instructions_section}

    {TWEET_BASE_STYLE}
    """
    return user_message