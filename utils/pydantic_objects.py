"""
Pydantic models for the LLMpedia Streamlit application.
"""

from pydantic import BaseModel, Field
from enum import Enum
from typing import Optional, List, Dict, Any

class QueryDecision(BaseModel):
    """Decision about query type to determine processing approach."""
    llm_query: bool
    other_query: bool
    comment_query: bool

class TopicCategory(str, Enum):
    """Categories for LLM research papers."""
    VISION_LANGUAGE_MODEL = "Vision-Language Model Innovations and Applications"
    AUTONOMOUS_LANGUAGE_AGENTS = "Autonomous Language Agents and Task Planning"
    CODE_GENERATION_TECHNIQUES = "Code Generation Techniques in Software Engineering"
    MULTILINGUAL_LANGUAGE_MODEL = "Multilingual Language Model Developments"
    ETHICAL_SECURE_AI = "Ethical and Secure AI Development Challenges"
    TRANSFORMER_ALTERNATIVES = "Transformer Alternatives and Efficiency Improvements"
    EFFICIENT_LLM_TRAINING = "Efficient LLM Training and Inference Optimization"
    RETRIEVAL_AUGMENTED_GENERATION = "Retrieval-Augmented Generation for NLP Tasks"
    ADVANCED_PROMPT_TECHNIQUES = "Enhancing LLM Performance with Advanced Prompt Techniques"
    INSTRUCTION_TUNING_TECHNIQUES = "Instruction Tuning Techniques for LLMs"
    BIAS_HATE_SPEECH_DETECTION = "Mitigating Bias and Hate Speech Detection"
    MATHEMATICAL_PROBLEM_SOLVING = "Enhancing Mathematical Problem Solving with AI"
    HUMAN_PREFERENCE_ALIGNMENT = "Human Preference Alignment in LLM Training"
    CHAIN_OF_THOUGHT_REASONING = "Enhancements in Chain-of-Thought Reasoning"
    MISCELLANEOUS = "Miscellaneous"

class SearchCriteria(BaseModel):
    """Search criteria for finding LLM research papers."""
    title: Optional[str] = Field(
        None,
        description="Title of the paper. Use only when the user is looking for a specific paper. Partial matches will be returned.",
    )
    min_publication_date: Optional[str] = Field(
        None,
        description="Minimum publication date of the paper. Use 'YYYY-MM-DD' format.",
    )
    max_publication_date: Optional[str] = Field(
        None,
        description="Maximum publication date of the paper. Use 'YYYY-MM-DD' format.",
    )
    # topic_categories: Optional[List[TopicCategory]] = Field(
    #     None,
    #     description="List containing the topic categories of the paper. Use only when the user explicitly asks about one of these topics (not for related topics).",
    # )
    semantic_search_queries: Optional[List[str]] = Field(
        None,
        description="List of queries to be used in the semantic search. The system will use these queries to find papers that have abstracts that are semantically similar to the queries. If you use more than one search query make them diverse enough so that each query addresses a different part of what is needed to build up an answer. Consider the language typically used in academic papers when writing the queries; phrase the queries as if they were part of the text that could be found on these abstracts.",
    )
    min_citations: Optional[int] = Field(
        None, description="Minimum number of citations of the paper."
    )

class DocumentAnalysis(BaseModel):
    """Analysis of a document's relevance to a query."""
    document_id: int
    analysis: str
    selected: float

class RerankedDocuments(BaseModel):
    """Container for document relevance analysis results."""
    documents: List[DocumentAnalysis]

class ResolveQuery(BaseModel):
    """Response generation process and final content."""
    brainstorm: str = Field(
        ...,
        description="Analysis of relevant information from documents and planning of response structure including markdown sections and additional elements (code blocks, tables, etc.)."
    )
    sketch: str = Field(
        ...,
        description="Detailed outline of the response using nested lists, prefilling each section with key ideas, findings, and supporting evidence."
    )
    response: str = Field(
        ...,
        description="The final response following the style guidelines and length requirements, incorporating all planned elements."
    )

class Document(BaseModel):
    """Document metadata and content for retrieval and display."""
    arxiv_code: str
    title: str
    published_date: Any
    citations: int
    abstract: str
    notes: str
    tokens: int
    distance: float

class NextSearchStepDecision(BaseModel):
    """Decision on whether to continue deep search and the next query."""
    reasoning: str = Field(
        ...,
        description="Initial analysis of the current research state, explaining what has been found and what gaps remain (if any).",
    )
    continue_search: bool = Field(
        ...,
        description="Whether the search should continue with another iteration.",
    )
    next_query: Optional[str] = Field(
        None,
        description="The refined or new search query to use in the next iteration, if continue_search is True.",
    )

class ScratchpadAnalysisResult(BaseModel):
    """Result of analyzing documents and updating the research scratchpad."""
    key_insights: List[str] = Field(
        ...,
        description="List of key findings or insights extracted from the newly analyzed documents relevant to the original question and current scratchpad.",
    )
    remaining_questions: List[str] = Field(
        ...,
        description="List of questions that remain unanswered or new questions that arose from the analysis.",
    )
    updated_scratchpad: str = Field(
        ...,
        description="The updated and augmented scratchpad content, incorporating the new insights and summarizing the current state of the research.",
    )

class ResolveScratchpadResponse(BaseModel):
    """Final response synthesized from the research scratchpad."""
    response: str = Field(
        ...,
        description="The final, synthesized response based on the scratchpad content, adhering to style and length requirements.",
    )