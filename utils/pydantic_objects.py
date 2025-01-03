from pydantic import BaseModel, Field
from enum import Enum
from typing import Optional, List, Dict

################
## SUMMARIZER ##
################


class Contribution(BaseModel):
    headline: str = Field(..., description="Headline of the main contribution.")
    description: str = Field(..., description="Description of the main contribution.")


class Takeaways(BaseModel):
    headline: str = Field(..., description="Headline of the main takeaway.")
    description: str = Field(..., description="Description of the main takeaway.")
    applied_example: str = Field(
        ..., description="Applied example related to the main takeaway."
    )


class PaperReview(BaseModel):
    main_contribution: Contribution = Field(
        ..., description="The main contribution of the paper."
    )
    takeaways: Takeaways = Field(..., description="The main takeaways from the paper.")
    category: str = Field(..., description="The primary focus category of the paper.")
    novelty_analysis: str = Field(..., description="Analysis of the paper's novelty.")
    novelty_score: int = Field(
        ..., description="Score representing the novelty of the paper."
    )
    technical_analysis: str = Field(
        ..., description="Analysis of the paper's technical depth."
    )
    technical_score: int = Field(
        ..., description="Score representing the technical depth of the paper."
    )
    enjoyable_analysis: str = Field(
        ..., description="Analysis of the paper's readability and engagement level."
    )
    enjoyable_score: int = Field(
        ..., description="Score representing the enjoyability of reading the paper."
    )


##################
## VECTOR STORE ##
##################

class LLMVerifier(BaseModel):
    analysis: str = Field(
        ...,
        description="The paper's analysis on its relevance to LLMs or text embeddings.",
    )
    is_related: bool = Field(
        ...,
        description="Indicates if the paper is directly related to LLMs or text embeddings.",
    )


class InterestingPaperSelection(BaseModel):
    analysis: str = Field(
        ...,
        description="Analysis explaining why this abstract was selected as the most interesting.",
    )
    selected_arxiv_code: str = Field(
        ...,
        description="Arxiv code of the selected paper.",
    )


class QueryDecision(BaseModel):
    llm_query: bool
    other_query: bool
    comment_query: bool


class TopicCategory(str, Enum):
    VISION_LANGUAGE_MODEL = "Vision-Language Model Innovations and Applications"
    AUTONOMOUS_LANGUAGE_AGENTS = "Autonomous Language Agents and Task Planning"
    CODE_GENERATION_TECHNIQUES = "Code Generation Techniques in Software Engineering"
    MULTILINGUAL_LANGUAGE_MODEL = "Multilingual Language Model Developments"
    ETHICAL_SECURE_AI = "Ethical and Secure AI Development Challenges"
    TRANSFORMER_ALTERNATIVES = "Transformer Alternatives and Efficiency Improvements"
    EFFICIENT_LLM_TRAINING = "Efficient LLM Training and Inference Optimization"
    RETRIEVAL_AUGMENTED_GENERATION = "Retrieval-Augmented Generation for NLP Tasks"
    ADVANCED_PROMPT_TECHNIQUES = (
        "Enhancing LLM Performance with Advanced Prompt Techniques"
    )
    INSTRUCTION_TUNING_TECHNIQUES = "Instruction Tuning Techniques for LLMs"
    BIAS_HATE_SPEECH_DETECTION = "Mitigating Bias and Hate Speech Detection"
    MATHEMATICAL_PROBLEM_SOLVING = "Enhancing Mathematical Problem Solving with AI"
    HUMAN_PREFERENCE_ALIGNMENT = "Human Preference Alignment in LLM Training"
    CHAIN_OF_THOUGHT_REASONING = "Enhancements in Chain-of-Thought Reasoning"
    MISCELLANEOUS = "Miscellaneous"


class SearchCriteria(BaseModel):
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
    topic_categories: Optional[List[TopicCategory]] = Field(
        None,
        description="List containing the topic categories of the paper. Use only when the user explicitly asks about one of these topics (not for related topics).",
    )
    semantic_search_queries: Optional[List[str]] = Field(
        None,
        description="List of queries to be used in the semantic search. The system will use these queries to find papers that have abstracts that are semantically similar to the queries. If you use more than one search query make them diverse enough so that each query addresses a different part of what is needed to build up an answer. Consider the language typically used in academic papers when writing the queries; phrase the queries as if they were part of the text that could be found on these abstracts.",
    )
    min_citations: Optional[int] = Field(
        None, description="Minimum number of citations of the paper."
    )

    # @model_validator(mode="before")
    # def validate_fields(cls, values):
    #     if not any(values.values()):
    #         raise ValueError("At least one field must be provided")
    #     if (
    #         values.get("semantic_search_queries")
    #         and len(values["semantic_search_queries"]) > 3
    #     ):
    #         raise ValueError("semantic_search_queries must contain at most 3 items")
    #     if values.get("topic_categories"):
    #         for category in values["topic_categories"]:
    #             if category not in (item.value for item in TopicCategory):
    #                 raise ValueError(f"Invalid topic category: {category}")
    #     return values


class DocumentAnalysis(BaseModel):
    document_id: int
    analysis: str
    selected: bool


class RerankedDocuments(BaseModel):
    documents: List[DocumentAnalysis]



###################
## WEEKLY REVIEW ##
###################

class WeeklyReview(BaseModel):
    # scratchpad_papers: str = Field(
    #     ...,
    #     description="List of ~30 interesting papers, their main themes and contributions.",
    # )
    # scratchpad_themes: str = Field(
    #     ..., description="At least 3 common themes identified among the papers."
    # )
    # themes_mapping: Dict[str, List[str]] = Field(
    #     ..., description="Mapping of themes to papers."
    # )
    new_developments_findings: str = Field(
        ..., description="New developments and findings."
    )


class ExternalResource(BaseModel):
    arxiv_code: str = Field(..., description="Arxiv code of the paper.")
    url: str = Field(
        ...,
        description="URL of the github repository or project website. Make sure to copy verbatim from context.",
    )
    title: str = Field(..., description="Title of the repository or project.")
    description: str = Field(
        ...,
        description="Brief description of the content of the repository or project. Explain what is the purpose of the underlying resource or model.",
    )


class ExternalResources(BaseModel):
    resources: List[ExternalResource] = Field(
        ..., description="List of external resources mentioned in the context."
    )


###############
## Q&A MODEL ##
###############

class QnaPair(BaseModel):
    question: str = Field(
        ...,
        description="Very specific question that does not make reference to the text.",
    )
    answer: str = Field(
        ..., description="Detailed answer to the question with citation."
    )


class QnaSet(BaseModel):
    qna_pairs: list[QnaPair] = Field(..., description="List of Q&A pairs.")


############
## TWEETS ##
############

class TweetScratchpad(BaseModel):
    content: str = Field(..., description="Content of the tweet.")
    structure: str = Field(..., description="Structure of the tweet.")
    broader_discussion: str = Field(..., description="How to connect to broader discussions?")


class TweetEditScratchpad(BaseModel):
    prohibited_phrases: str = Field(..., description="Analysis of prohibited phrases and planned rephrasing")
    structural_analysis: str = Field(..., description="Analysis of structural patterns and planned reorganization")
    revision_plan: str = Field(..., description="Specific changes to implement and how to preserve core message")


class Tweet(BaseModel):
    scratchpad: TweetScratchpad = Field(..., description="Freeform text for brainstorming initial tweet")
    tweet: str = Field(..., description="Initial version of the tweet")
    edit_scratchpad: TweetEditScratchpad = Field(..., description="Analysis and planning for tweet revision")
    edited_tweet: str = Field(..., description="Revised version avoiding prohibited phrases and ensuring structural uniqueness")


class TweetEdit(BaseModel):
    repetition_analysis: List[str] = Field(
        ...,
        description="List of identified structural patterns and phrases that match recent tweets"
    )
    suggested_tweet: str = Field(
        ...,
        description="The revised tweet with minimal but impactful changes to reduce repetitiveness"
    )
    edit_rationale: str = Field(
        ...,
        description="Explanation of how the changes reduce repetitiveness while preserving style requirements"
    )
