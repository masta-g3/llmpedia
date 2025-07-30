"""
LangChain-style multi-agent deep research implementation.
Minimal, lightweight implementation using existing LLMpedia infrastructure.
"""

from typing import List, Optional, Tuple, Set, Callable
from pydantic import BaseModel, Field
import datetime
import uuid

from utils.instruct import run_instructor_query
from utils.app_utils import (
    rerank_documents_new,
    Document,
    RedditContent,
    VS_EMBEDDING_MODEL,
    query_config,
    extract_arxiv_codes,
    extract_reddit_codes,
    extract_all_citations,
    add_links_to_text_blob,
    enhance_documents_with_reddit,
)
from utils.db import db, db_utils

## Dynamic date for temporal context in prompts
TODAY_DATE = datetime.datetime.now().strftime("%Y-%m-%d")
CURRENT_YEAR = datetime.datetime.now().year

## Pydantic Models for Deep Research


class ResearchBrief(BaseModel):
    """Focused research brief generated from user question."""

    focused_question: str = Field(
        ...,
        description="The refined, focused research question that captures the core of what needs to be investigated.",
    )
    research_scope: str = Field(
        ...,
        description="Clear definition of the research scope, including what should be included and excluded.",
    )
    key_subtopics: List[str] = Field(
        ...,
        description="Between 1 and 5 independent subtopics that comprehensively address the research question.",
    )
    expected_timeline: str = Field(
        ...,
        description="Relevant time period for the research (e.g., 'recent papers from April and May 2025', 'foundational work from 2017-2020'). Might vary for different subtopics.",
    )


class SubTopicAssignment(BaseModel):
    """Assignment of a specific subtopic to a research agent."""

    subtopic: str = Field(..., description="The specific subtopic to research.")
    search_strategy: str = Field(
        ...,
        description="Tailored search approach for this subtopic, including key terms and concepts to focus on.",
    )
    semantic_queries: List[str] = Field(
        ...,
        description="2-3 semantic search queries optimized for this subtopic using academic language. Do not include date-related terms (use the date fields for that).",
    )
    expected_findings: str = Field(
        ...,
        description="What type of insights or evidence this subtopic should contribute to the overall research.",
    )
    sources: List[str] = Field(
        default=["arxiv"],
        description="Data sources to search: 'arxiv' for academic papers, 'reddit' for community discussions, or both.",
    )
    min_publication_date: Optional[str] = Field(
        None,
        description="Minimum publication date for papers in YYYY-MM-DD format. Use when temporal constraints are important for this subtopic.",
    )
    max_publication_date: Optional[str] = Field(
        None,
        description="Maximum publication date for papers in YYYY-MM-DD format. Use when temporal constraints are important for this subtopic.",
    )


class SubTopicAssignments(BaseModel):
    """List of subtopic assignments."""

    assignments: List[SubTopicAssignment] = Field(
        ..., description="List of subtopic assignments."
    )


class AgentFindings(BaseModel):
    """Research findings from a single agent's investigation."""

    subtopic: str = Field(..., description="The subtopic that was researched.")
    key_insights: List[str] = Field(
        ..., description="3-5 most important insights discovered for this subtopic."
    )
    supporting_evidence: List[str] = Field(
        ..., description="Specific evidence and findings that support the key insights."
    )
    community_insights: List[str] = Field(
        default=[],
        description="Key insights from Reddit community discussions, including practical perspectives and real-world experiences.",
    )    
    referenced_papers: List[str] = Field(
        default=[],
        description="List of arxiv paper codes that provided key evidence. Format: ['arxiv:2507.03113', 'arxiv:2401.12345']"
    )
    reddit_references: List[str] = Field(
        default=[],
        description="List of Reddit post references that provided community insights. Format: ['r/MachineLearning:1l69w7i', 'r/LocalLLaMA:1k8x2p4']"
    )
    research_gaps: List[str] = Field(
        ...,
        description="Identified gaps or limitations in the current research on this subtopic.",
    )

class FinalReport(BaseModel):
    """Final synthesized research report."""
    title: str = Field(
        ..., description="A simple, short, punchy sentence summarizing your most insightful finding."
    )
    response: str = Field(
        ..., description="Final formatted response ready for presentation to the user. Includes inline citations in the format: ['arxiv:2507.03113', 'r/LocalLLaMA:1k8x2p4']."
    )


## Prompt Templates

RESEARCH_BRIEF_SYSTEM_PROMPT = f"""You are a research planning expert. Your job is to analyze a user's research question and create a focused, actionable research plan that will guide a team of specialized research agents.

Break down complex questions into independent, researchable subtopics that can be investigated in parallel. In many cases, particularly for targetted questions, a single subtopic will be enough. Otherwise, focus on creating subtopics that don't overlap and together provide comprehensive coverage of the research question. Be sure topics are not too broad and directly address the user's question (less is more).

Current date: {TODAY_DATE}."""


def create_research_brief_prompt(user_question: str, max_agents: int = 5) -> str:
    return f"""
<user_question>
{user_question}
</user_question>

<instructions>
- Analyze the user's question and create a focused research brief.
- Identify between 1 and {max_agents} independent, orthogonal subtopics that together comprehensively address the question.
- In many cases a single subtopic will be enough (particularly for targetted questions).
- Ensure subtopics don't overlap and can be researched independently. Do not add more topics than necessary.
- Define clear research scope and the relevant time periods.
- When the user asks about "up-to-date", "recent" or "latest" information, focus on content from the last 1-2 months.
</instructions>"""


SUPERVISOR_SYSTEM_PROMPT = f"""You are a research supervisor coordinating a team of specialized research agents. Your job is to take a research brief and create a comprehensive list of specific assignments for individual agents, ensuring comprehensive coverage without overlap.

Each agent will focus on one subtopic and conduct semantic search across different information sources. You can assign agents to search:
- "arxiv": Academic papers and research literature
- "reddit": Community discussions and practitioner experiences  
- Both sources when comprehensive coverage is needed

Design assignments that are independent, focused, and together address the full research brief.

You must return ALL assignments in a single response as a complete list - do not make separate calls for each assignment.

Current date: {TODAY_DATE}. Consider temporal context when creating search strategies - recent work refers to content from the last 2-3 months."""


def create_supervisor_assignment_prompt(research_brief: ResearchBrief) -> str:
    return f"""
<research_brief>
Focused Question: {research_brief.focused_question}
Research Scope: {research_brief.research_scope}
Key Subtopics: {', '.join(research_brief.key_subtopics)}
Expected Timeline: {research_brief.expected_timeline}
</research_brief>

<instructions>
Create a comprehensive list of research assignments. For each subtopic in the research brief, include a detailed assignment that contains:

1. Clear, direct, and targeted subtopic definition.
2. Specific search strategy tailored to that subtopic.
3. 2-3 diverse and non-overlapping semantic search queries using targeted academic language.
4. Expected type of findings this subtopic should contribute.
5. Appropriate data sources: ["arxiv"], ["reddit"], or ["arxiv", "reddit"] based on the information needs.
6. Optional publication date constraints (min/max) when temporal focus is important.
</instructions>

<guidelines_for_search_queries>
- Use language typical of academic abstracts - phrase queries as if they were part of the text found in abstracts.
- Consider that there are likely few or no relevant papers to very niche subtopics, so try to effectively balance between breadth and depth of search.
- Focus on key terms and methodological language common in research papers about the subtopic.
- Make queries diverse enough to capture different aspects of the subtopic. Think about how to avoid situations where your queries all return the same set of papers.
- Make your queries concise and to the point, aiming to maximize semantic similarity between the query and the axiv documents you expect to find.
</guidelines_for_search_queries>

<guidelines_for_source_selection>
- Use "arxiv" for: Technical methodologies, performance benchmarks, theoretical foundations, algorithm details, empirical evaluations
- Use "reddit" for: User experiences, implementation challenges, adoption patterns, community sentiment, practical applications, real-world deployment issues
- Use both ["arxiv", "reddit"] for: Comprehensive analysis where both academic rigor and practical perspectives are valuable
- Choose sources based on the type of insights needed for each subtopic
</guidelines_for_source_selection>

<guidelines_for_date_constraints>
- Consider the specified timeline only when relevant (e.g. when the user asks for specific time range or recent findings).
- Set min_publication_date and/or max_publication_date (YYYY-MM-DD format) only when temporal focus is critical for the subtopic.
- Consider that this is a very fast moving field; "recent advances" or "latest developments" likely refers papers from the last 2-3 months.
- Leave date fields unset (None) when the subtopic should search across all time periods.
- Consider that different subtopics may need different temporal constraints.
</guidelines_for_date_constraints>
"""


AGENT_RESEARCH_SYSTEM_PROMPT = f"""You are a specialized research agent focused on investigating a specific subtopic. Your job is to analyze the documents you've found and extract key insights that directly address your assigned subtopic.

Be thorough but focused - stick to your subtopic and provide evidence-backed insights. Try to copy verbatim from the documents when possible. Identify gaps where current research may be incomplete.

When analyzing Reddit content: You'll see Reddit discussions structured hierarchically with main posts followed by indented top comments from the community. Weight opinions based on community engagement (upvotes/comments) and look for patterns across multiple posts rather than relying on individual low-engagement posts. Pay special attention to highly-upvoted comments as they often represent validated community consensus or valuable practical insights that complement the original posts.

Current date: {TODAY_DATE}."""


def create_agent_research_prompt(
    assignment: SubTopicAssignment, sources: List
) -> str:
    doc_context = ""
    for source in sources:
        if isinstance(source, Document):
            # Academic paper content
            doc_context += f"""
**Title:** {source.title}
**ArXiv:** {source.arxiv_code} ({source.published_date.year}, {source.citations} citations)
**Abstract:** {source.abstract}
**Notes:**
{source.notes}

---
---
---
"""
        elif isinstance(source, RedditContent):
            # Reddit community content
            if source.content_type == 'reddit_post':
                doc_context += f"""
**📱 Reddit Discussion**
**r/{source.subreddit}:{source.reddit_id}** • {source.published_date.strftime('%Y-%m-%d')}
**{source.title}**
👤 u/{source.author} • ⬆️ {source.score} upvotes • 💬 {source.num_comments} comments

{source.content}

---
---
---
"""
            else:
                doc_context += f"""
    **💬 Top Comment** (⬆️ {source.score} upvotes) • ID: {source.reddit_id}
    👤 u/{source.author} • {source.published_date.strftime('%Y-%m-%d')}
    
    {source.content}

---
---
---
"""

    return f"""
<assignment>
Subtopic: {assignment.subtopic}
Search Strategy: {assignment.search_strategy}
Expected Findings: {assignment.expected_findings}
</assignment>

<sources>
{doc_context}
</sources>

<instructions>
Analyze the provided sources and extract findings specifically related to your assigned subtopic. You may have academic papers, community discussions, or both:

1. Identify all key insights that directly address your subtopic.
2. For academic sources: Focus on research findings, methodologies, and experimental results.
3. For community sources: Analyze practical perspectives, implementation challenges, user experiences, and real-world applications.
4. Provide supporting evidence from the sources, copying verbatim when possible.
5. List the arxiv codes of papers that provided key evidence (if any).
6. For Reddit sources: Use the format "r/subreddit:post_id" (e.g., r/MachineLearning:1l69w7i).
7. Be decisive with your conclusions. Identify any research gaps or limitations you notice.
8. Note any discrepancies between academic claims and community experiences (when both are available).
9. If you are not able to find any evidence for your subtopic, just say so.

Guidelines:
- Stay focused on your specific subtopic.
- Ground all insights in the provided sources.
- Be specific about which sources support which insights.
- When citing sources: Use "arxiv:paper_id" for academic papers and "r/subreddit:post_id" for Reddit discussions. List multiple citations as [r/MachineLearning:1l69w7i, r/LocalLLaMA:1l6opyh].
- Consider that some papers are about specific models and not general AI behavior.
- Avoid referencing niche models unless they are very relevant to the subtopic.
- Note if certain aspects of your subtopic lack sufficient evidence.
- Balance different types of sources appropriately based on what's available.
</instructions>
"""


REPORT_SYNTHESIS_SYSTEM_PROMPT = f"""You are a terminally online millennial AI researcher, deeply immersed in Large Language Models (LLMs) and the X.com tech scene. Your posts blend optimistic, accessible insights that synthesize complex research, playful wit with meme-savvy takes on AI's quirks, and sharp skepticism that cuts through hype with incisive questions. You dissect cutting-edge papers, spotlight nuanced findings, and explore unexpected implications for AI's future, all while engaging the X.com AI crowd with humor, curiosity, and bold takes. Your tone is technically precise yet conversational, sharp and without too much slang. You spark discussions with a mix of enthusiasm, irony, and critical edge.

Your job is to synthesize findings from multiple specialized research agents into a comprehensive, coherent response that directly answers the original research question.

Maintain a friendly, technically precise, and conversational tone while combining insights across subtopics, identifying patterns and connections, and providing a clear, well-structured response with excellent narrative flow.

Current date: {TODAY_DATE}.

IMPORTANT: Prioritize the knowledge from the provided agent findings as your primary source. If the document evidence is insufficient for comprehensive coverage, you may complement with your internal knowledge and logical reasoning, but clearly distinguish such content by noting it clearly."""


def create_report_synthesis_prompt(
    research_brief: ResearchBrief,
    agent_findings: List[AgentFindings],
    response_length: int,
) -> str:
    findings_context = ""
    for findings in agent_findings:
        findings_context += f"""
## {findings.subtopic}

Key Insights:
{chr(10).join(f"- {insight}" for insight in findings.key_insights)}

Supporting Evidence:
{chr(10).join(f"- {evidence}" for evidence in findings.supporting_evidence)}

Referenced Papers: {', '.join(findings.referenced_papers)}

Research Gaps:
{chr(10).join(f"- {gap}" for gap in findings.research_gaps)}"""

        # Add community insights if available
        if findings.community_insights:
            findings_context += f"""

Community Insights:
{chr(10).join(f"- {insight}" for insight in findings.community_insights)}"""

        # Add Reddit references if available
        if findings.reddit_references:
            findings_context += f"""

Referenced Reddit Posts: {', '.join(findings.reddit_references)}"""

        findings_context += """

---
"""

    length_guidance = ""
    if response_length <= 250:
        length_guidance = "Write a focused response (~200 words) that directly answers the question in a single cohesive paragraph. Emphasize the most important, unituitive, and surprising key finding from all the evidence presented to you."
    elif response_length <= 500:
        length_guidance = "Write a focused research note (~350 words) that directly answers the question in a single cohesive, in-depth paragraph. Emphasize key findings and core concepts while maintaining narrative flow. Use clear topic transitions and supporting evidence."
    elif response_length <= 1000:
        length_guidance = "Write an engaging research summary (~750 words) that explores the topic through multiple angles. Use 2-3 naturally flowing sections to develop ideas from key findings through implications. Blend technical insights with practical applications, maintaining narrative momentum."
    elif response_length <= 3000:
        length_guidance = "Write an in-depth research analysis (~2500 words) that thoroughly explores the topic's landscape. Structure with clear sections using markdown headers (###) and information-dense paragraphs to guide the reader through your narrative. Progress from core findings through technical details to broader implications."
    else:
        length_guidance = "Write a comprehensive research report (~4000 words) that covers the full scope of the topic. Use hierarchical markdown headers (##, ###) to create a natural progression through multiple major sections, which will be made by information-rich paragraphs. Weave together theoretical foundations, technical implementations, and practical implications while maintaining narrative cohesion."

    return f"""
<research_brief>
Original Question: {research_brief.focused_question}
Research Scope: {research_brief.research_scope}
</research_brief>

<agent_findings>
{findings_context}
</agent_findings>

<instructions>
Synthesize the research findings into a comprehensive answer that directly answers the original research question.
- Provide a one line, top level summary as the first line of the response.
- Provide detailed analysis integrating insights across subtopics, combining both academic research and community perspectives.
- When community insights are available, prioritize practical experiences and consensus from practitioners over purely theoretical findings.
- Highlight any notable agreements, disagreements, or gaps between academic research and community experiences.
- Draw one main, clear conclusion and be clear about it.
- Format your response as a coherent, engaging answer for the user, following the style guide provided below.
- IMPORTANT: Be decisive and useful with your conclusions. Be sure to provide a clear, actionable answer to the user's question.
</instructions>

<response_length_guidance>
- {length_guidance}
</response_length_guidance>

<style_guidelines>
- Use direct, concise wording with a technical edge that reflects AI research discourse on X.com.
- Avpid being too academic, techincal or boring.
- Don't shy away from technical terms - assume your audience has domain knowledge.
- Be informal when it serves the content - subtle humor is welcome.
- Avoid being pedantic, obnoxious or overtly-critical.
- Don't frame ideas as revelatory paradigm shifts or contrarian declarations.
- Ensure the response directly addresses the original research question.
- Seamlessly integrate citations (using [arxiv:XXXX.YYYYY, r/subreddit:post_id] format) into the narrative flow to support claims and guide further reading.
- Use markdown formatting to enhance readability and structure.
- Maintain narrative momentum and logical flow.
- Focus on actionable insights and be insightful/clever when possible.
- Only include citations for papers that are listed in the agent findings.
- Do not make reference to the existence of agent findings in your response (treat them as your internal knowledge).
- Avoid repetitive conclusions or final remarks.
- Avoid these prohibited phrases: fascinating, mind-blowing, wild, surprising, reveals, crucial, turns out that, the twist/secret, sweet spot, here's the kicker, irony/ironic, makes you think/wonder, really makes you, we might need to (rethink), the real [constraint/question/etc.] is, its no surprise, we are basically, fundamentally changes, peak 2024/25, crushing it, feels like, here's the scoop, etc.
- Do not include any other text or comments in your response, just the answer to the user formatted as a markdown document. Do not include triple backticks or any other formatting.
</style_guidelines>
"""


## Core Implementation Classes


class ResearchAgent:
    """Individual research agent focused on a specific subtopic."""

    def __init__(
        self, assignment: SubTopicAssignment, llm_model: str = "gemini/gemini-2.5-flash"
    ):
        self.assignment = assignment
        self.llm_model = llm_model
        self.findings: Optional[AgentFindings] = None

    def conduct_research(
        self,
        workflow_id: str,
        max_sources: int = 15,
        exclude_codes: Optional[Set[str]] = None,
        progress_callback: Optional[Callable[[str], None]] = None,
        verbose: bool = False,
    ) -> AgentFindings:
        """Conduct focused research on the assigned subtopic."""

        if verbose and progress_callback:
            queries_info = "\n".join(
                [f"        - {q}" for q in self.assignment.semantic_queries]
            )
            date_info = ""
            if (
                self.assignment.min_publication_date
                or self.assignment.max_publication_date
            ):
                date_constraints = []
                if self.assignment.min_publication_date:
                    date_constraints.append(
                        f"after {self.assignment.min_publication_date}"
                    )
                if self.assignment.max_publication_date:
                    date_constraints.append(
                        f"before {self.assignment.max_publication_date}"
                    )
                date_info = (
                    f"\n        📅 Date constraints: {' and '.join(date_constraints)}"
                )

            progress_callback(
                f"      🔎 Searching with queries:\n{queries_info}{date_info}"
            )

        ## Generate search criteria from assignment
        search_criteria = {
            "semantic_search_queries": self.assignment.semantic_queries,
            "limit": max_sources * 2,  # Fetch more for reranking
        }

        ## Add date constraints if specified in assignment
        if self.assignment.min_publication_date:
            search_criteria["min_publication_date"] = (
                self.assignment.min_publication_date
            )
        if self.assignment.max_publication_date:
            search_criteria["max_publication_date"] = (
                self.assignment.max_publication_date
            )

        ## Execute searches based on assigned sources
        all_documents = []
        
        if "arxiv" in self.assignment.sources:
            if verbose and progress_callback:
                progress_callback(f"      📚 Searching arXiv papers...")
            
            arxiv_sql = db.generate_semantic_search_query(
                search_criteria,
                query_config,
                embedding_model=VS_EMBEDDING_MODEL,
                exclude_arxiv_codes=exclude_codes or set(),
            )
            arxiv_df = db_utils.execute_read_query(arxiv_sql)
            
            if not arxiv_df.empty:
                arxiv_documents = [
                    Document(
                        arxiv_code=d["arxiv_code"],
                        title=d["title"],
                        published_date=d["published_date"].to_pydatetime(),
                        citations=int(d["citations"]),
                        abstract=d["abstract"],
                        notes=d["notes"],
                        tokens=int(d["tokens"]),
                        distance=float(d["similarity_score"]),
                    )
                    for _, d in arxiv_df.iterrows()
                ]
                all_documents.extend(arxiv_documents)
        
        if "reddit" in self.assignment.sources:
            if verbose and progress_callback:
                progress_callback(f"      💬 Searching Reddit discussions...")
            
            reddit_sql = db.generate_reddit_semantic_search_query(
                search_criteria,
                query_config,
                embedding_model=VS_EMBEDDING_MODEL,
            )
            reddit_df = db_utils.execute_read_query(reddit_sql)
            
            if not reddit_df.empty:
                reddit_documents = [
                    RedditContent(
                        reddit_id=d["reddit_id"],
                        subreddit=d["subreddit"],
                        title=d["title"],
                        content=d["content"] or "",
                        author=d["author"] or "",
                        score=int(d["score"]),
                        num_comments=int(d["num_comments"]),
                        published_date=d["published_date"].to_pydatetime(),
                        content_type=d["content_type"],
                        distance=float(d["similarity_score"]),
                    )
                    for _, d in reddit_df.iterrows()
                ]
                all_documents.extend(reddit_documents)

        if verbose and progress_callback:
            progress_callback(f"      📄 Found {len(all_documents)} candidate sources")

        if not all_documents:
            return AgentFindings(
                subtopic=self.assignment.subtopic,
                key_insights=["No relevant documents found for this subtopic."],
                supporting_evidence=[],
                referenced_papers=[],
                reddit_references=[],
                research_gaps=["Insufficient research available on this subtopic."],
            )

        ## Limit to 30 total documents, distributed across sources
        total_limit = max_sources * 2
        if len(all_documents) > total_limit:
            if len(self.assignment.sources) == 2:
                arxiv_docs = [d for d in all_documents if isinstance(d, Document)]
                reddit_docs = [d for d in all_documents if isinstance(d, RedditContent)]
                split = total_limit // 2
                arxiv_take = min(len(arxiv_docs), split)
                reddit_take = min(len(reddit_docs), total_limit - arxiv_take)
                all_documents = arxiv_docs[:arxiv_take] + reddit_docs[:reddit_take]
            else:
                all_documents = all_documents[:total_limit]

        if verbose and progress_callback:
            progress_callback(
                f"      ⚖️ Reranking {len(all_documents)} sources for relevance..."
            )

        ## Rerank documents for relevance to subtopic
        reranked_docs = rerank_documents_new(
            user_question=f"""{self.assignment.subtopic} - {self.assignment.search_strategy}. 
            Expected findings: {self.assignment.expected_findings}""",
            documents=all_documents,
            llm_model=self.llm_model,
        )

        ## Select top relevant documents
        high_relevance_docs = [
            all_documents[int(da.document_id)]
            for da in reranked_docs.documents
            if da.selected >= 0.5  # Include medium and high relevance
        ][:max_sources]

        ## For Reddit posts, fetch top comments to enrich context
        if "reddit" in self.assignment.sources and high_relevance_docs:
            reddit_post_ids = [
                doc.reddit_id for doc in high_relevance_docs 
                if isinstance(doc, RedditContent) and doc.content_type == 'reddit_post'
            ]
            
            if reddit_post_ids:
                if verbose and progress_callback:
                    progress_callback(f"      💬 Fetching top comments for {len(reddit_post_ids)} Reddit posts...")
                
                comments_df = db.get_top_reddit_comments(reddit_post_ids, max_comments=3, min_score=5)
                
                if not comments_df.empty:
                    reddit_comments = [
                        RedditContent(
                            reddit_id=row["reddit_id"],
                            subreddit="",  # Comments inherit subreddit from parent post
                            title=f"Comment on: {next((doc.title for doc in high_relevance_docs if isinstance(doc, RedditContent) and doc.reddit_id == row['post_reddit_id']), 'Unknown Post')}",
                            content=row["content"] or "",
                            author=row["author"] or "",
                            score=int(row["score"]),
                            num_comments=0,  # Comments don't have sub-comments counted
                            published_date=row["published_date"].to_pydatetime(),
                            content_type="reddit_comment",
                            distance=0.0,  # Comments weren't semantically searched
                        )
                        for _, row in comments_df.iterrows()
                    ]
                    
                    ## Add comments to the documents for analysis
                    high_relevance_docs.extend(reddit_comments)
                    
                    if verbose and progress_callback:
                        progress_callback(f"      📋 Added {len(reddit_comments)} top comments to analysis")

        if verbose and progress_callback:
            progress_callback(
                f"      📋 Selected {len(high_relevance_docs)} relevant sources for analysis"
            )

        if not high_relevance_docs:
            return AgentFindings(
                subtopic=self.assignment.subtopic,
                key_insights=["No highly relevant documents found after reranking."],
                supporting_evidence=[],
                referenced_papers=[],
                reddit_references=[],
                research_gaps=["Limited relevant research on this specific subtopic."],
            )

        if verbose and progress_callback:
            progress_callback(f"      🧠 Analyzing sources and extracting insights...")

        ## Analyze documents and extract findings
        findings = run_instructor_query(
            system_message=AGENT_RESEARCH_SYSTEM_PROMPT,
            user_message=create_agent_research_prompt(
                self.assignment, high_relevance_docs
            ),
            model=AgentFindings,
            llm_model=self.llm_model,
            temperature=0.7,
            process_id=f"agent_research_{self.assignment.subtopic}",
            workflow_id=workflow_id,
            step_type="agent_research",
            step_metadata={
                "subtopic": self.assignment.subtopic,
                "sources": self.assignment.sources,
                "documents_found": len(all_documents),
                "documents_selected": len(high_relevance_docs),
                "semantic_queries": self.assignment.semantic_queries,
            },
            # thinking_budget=2048
        )

        self.findings = findings
        return findings


class ResearchSupervisor:
    """Coordinates research agents and manages subtopic assignments."""

    def __init__(self, llm_model: str = "gemini/gemini-2.5-flash"):
        self.llm_model = llm_model
        self.research_brief: Optional[ResearchBrief] = None
        self.assignments: SubTopicAssignments = SubTopicAssignments(assignments=[])

    def create_research_brief(
        self,
        user_question: str,
        workflow_id: str,
        max_agents: int = 5,
        progress_callback: Optional[Callable[[str], None]] = None,
        verbose: bool = False,
    ) -> ResearchBrief:
        """Generate focused research brief from user question."""
        if progress_callback:
            progress_callback("🧠 Analyzing question and creating research focus...")

        brief = run_instructor_query(
            system_message=RESEARCH_BRIEF_SYSTEM_PROMPT,
            user_message=create_research_brief_prompt(user_question, max_agents),
            model=ResearchBrief,
            llm_model=self.llm_model,
            temperature=0.4,
            process_id="create_research_brief",
            workflow_id=workflow_id,
            step_type="research_brief",
            step_metadata={
                "original_question": user_question,
                "llm_model": self.llm_model,
                "max_agents": max_agents,
            },
            # thinking_budget=1024
        )

        if verbose and progress_callback:
            progress_callback(f"🎯 Research scope: {brief.research_scope}")
            progress_callback(f"⏱️ Timeline focus: {brief.expected_timeline}")

        self.research_brief = brief
        return brief

    def create_agent_assignments(
        self,
        research_brief: ResearchBrief,
        workflow_id: str,
        progress_callback: Optional[Callable[[str], None]] = None,
        verbose: bool = False,
    ) -> SubTopicAssignments:
        """Create specific assignments for research agents."""
        if progress_callback:
            progress_callback("📋 Creating specialized research assignments...")

        ## Use LLM to create detailed assignments with proper date constraints
        assignments_response = run_instructor_query(
            system_message=SUPERVISOR_SYSTEM_PROMPT,
            user_message=create_supervisor_assignment_prompt(research_brief),
            model=SubTopicAssignments,
            llm_model=self.llm_model,
            temperature=0.5,
            process_id="create_agent_assignments",
            workflow_id=workflow_id,
            step_type="agent_assignment",
            step_metadata={
                "subtopics_count": len(research_brief.key_subtopics),
                "research_scope": research_brief.research_scope,
                "expected_timeline": research_brief.expected_timeline,
            },
            # thinking_budget=1024
        )

        if verbose and progress_callback:
            for i, assignment in enumerate(assignments_response.assignments, 1):
                date_info = ""
                if assignment.min_publication_date or assignment.max_publication_date:
                    constraints = []
                    if assignment.min_publication_date:
                        constraints.append(f"after {assignment.min_publication_date}")
                    if assignment.max_publication_date:
                        constraints.append(f"before {assignment.max_publication_date}")
                    date_info = f" (📅 {' and '.join(constraints)})"
                
                sources_info = f" ({'📚' if 'arxiv' in assignment.sources else ''}{'💬' if 'reddit' in assignment.sources else ''})"
                progress_callback(
                    f"   📝 Assignment {i}: {assignment.subtopic}{sources_info}{date_info}"
                )

        if progress_callback:
            progress_callback(
                f"✅ Created {len(assignments_response.assignments)} research assignments"
            )

        self.assignments = assignments_response.assignments
        return assignments_response.assignments


class DeepResearchOrchestrator:
    """Main orchestrator for the three-phase deep research process."""

    def __init__(self, llm_model: str = "gemini/gemini-2.5-flash"):
        self.llm_model = llm_model
        self.supervisor = ResearchSupervisor(llm_model)
        self.agents: List[ResearchAgent] = []
        self.all_findings: List[AgentFindings] = []

    def conduct_deep_research(
        self,
        user_question: str,
        max_agents: int = 3,
        max_sources_per_agent: int = 10,
        response_length: int = 4000,
        progress_callback: Optional[Callable[[str], None]] = None,
        verbose: bool = False,
    ) -> Tuple[str, str, str, List[str], List[str], List[str], List[str]]:
        """Execute the full three-phase deep research process."""

        ## Generate unique workflow ID for this research session
        workflow_id = str(uuid.uuid4())

        if progress_callback:
            progress_callback("🎯 PHASE 1: Creating focused research brief...")

        ## Phase 1: Scope - Create Research Brief
        research_brief = self.supervisor.create_research_brief(
            user_question, workflow_id, max_agents, progress_callback, verbose
        )

        if progress_callback:
            brief_preview = (
                research_brief.focused_question[:80] + "..."
                if len(research_brief.focused_question) > 80
                else research_brief.focused_question
            )
            progress_callback(f"📋 Research brief: {brief_preview}")
            progress_callback(
                f"🔀 Breaking down into {len(research_brief.key_subtopics)} subtopics..."
            )

        if verbose and progress_callback:
            for i, subtopic in enumerate(research_brief.key_subtopics, 1):
                progress_callback(f"   {i}. {subtopic}")

        ## Phase 2: Research - Coordinate Agent Research
        if progress_callback:
            progress_callback("🤖 PHASE 2: Deploying research agents...")

        assignments = self.supervisor.create_agent_assignments(
            research_brief, workflow_id, progress_callback, verbose
        )

        ## Limit agents if needed
        if len(assignments) > max_agents:
            assignments = assignments[:max_agents]
            if progress_callback:
                progress_callback(f"📊 Limited to {max_agents} agents for efficiency")

        ## Track processed documents to avoid overlap
        processed_codes: Set[str] = set()

        if progress_callback:
            progress_callback(
                f"🔍 Starting research with {len(assignments)} specialized agents..."
            )

        ## Execute research with each agent
        for i, assignment in enumerate(assignments, 1):
            if progress_callback:
                progress_callback(
                    f"🤖 Agent {i}/{len(assignments)}: Researching '{assignment.subtopic}'..."
                )

            agent = ResearchAgent(assignment, self.llm_model)
            findings = agent.conduct_research(
                workflow_id=workflow_id,
                max_sources=max_sources_per_agent,
                exclude_codes=processed_codes,
                progress_callback=progress_callback,
                verbose=verbose,
            )

            if progress_callback:
                insights_count = len(findings.key_insights)
                papers_count = len(findings.referenced_papers)
                progress_callback(
                    f"✅ Agent {i} completed: {insights_count} insights, {papers_count} papers"
                )

            if verbose and progress_callback and findings.key_insights:
                progress_callback(
                    f"   Top insight: {findings.key_insights[0][:100]}..."
                )

            ## Add found papers to exclusion set
            processed_codes.update(findings.referenced_papers)

            self.agents.append(agent)
            self.all_findings.append(findings)

        ## Phase 3: Report Writing - Synthesize Final Report
        if progress_callback:
            total_insights = sum(len(f.key_insights) for f in self.all_findings)
            total_papers = len(processed_codes)
            progress_callback(
                f"📝 PHASE 3: Synthesizing {total_insights} insights from {total_papers} papers..."
            )

        final_report = run_instructor_query(
            system_message=REPORT_SYNTHESIS_SYSTEM_PROMPT,
            user_message=create_report_synthesis_prompt(
                research_brief, self.all_findings, response_length
            ),
            model=FinalReport,
            llm_model=self.llm_model,
            temperature=1.0,
            process_id="synthesize_final_report",
            workflow_id=workflow_id,
            step_type="final_report",
            step_metadata={
                "total_agents": len(self.all_findings),
                "total_insights": sum(len(f.key_insights) for f in self.all_findings),
                "target_length": response_length,
                "research_question": research_brief.focused_question,
            },
            # max_tokens=response_length,
            # thinking_budget=4096
        )

        ## Process final response
        response_with_links = add_links_to_text_blob(final_report.response)
        citations_dict = extract_all_citations(final_report.response)
        
        ## Extract referenced sources from report
        referenced_arxiv_codes = citations_dict["arxiv_papers"]
        referenced_reddit_codes = citations_dict["reddit_posts"]

        ## Collect all sources found by agents
        all_arxiv_found = set()
        all_reddit_found = set()
        for findings in self.all_findings:
            # Clean arxiv prefixes from agent findings
            for paper in findings.referenced_papers:
                if paper.startswith("arxiv:"):
                    all_arxiv_found.add(paper[6:])  # Remove prefix
                else:
                    all_arxiv_found.add(paper)
            
            # Add reddit references as-is
            all_reddit_found.update(findings.reddit_references)

        ## Calculate additional sources (found but not referenced)
        additional_arxiv_codes = list(all_arxiv_found - set(referenced_arxiv_codes))
        additional_reddit_codes = list(all_reddit_found - set(referenced_reddit_codes))
        
        ## Count total sources
        total_referenced_sources = len(referenced_arxiv_codes) + len(referenced_reddit_codes)
        source_label = "sources" if total_referenced_sources != 1 else "source"
        
        ## Log comprehensive citation summary
        if progress_callback:
            word_count = len(response_with_links.split())
            
            ## Brief completion message
            if referenced_reddit_codes:
                progress_callback(
                    f"🎉 Research complete! Generated {word_count} word response with {total_referenced_sources} referenced {source_label} ({len(referenced_arxiv_codes)} papers, {len(referenced_reddit_codes)} discussions)"
                )
            else:
                progress_callback(
                    f"🎉 Research complete! Generated {word_count} word response with {len(referenced_arxiv_codes)} referenced papers"
                )
            
            ## Detailed citation summary
            progress_callback("📚 CITATION SUMMARY:")
            progress_callback(f"   Referenced in report: {len(referenced_arxiv_codes)} arxiv papers, {len(referenced_reddit_codes)} reddit posts")
            progress_callback(f"   All sources found: {len(all_arxiv_found)} arxiv papers, {len(all_reddit_found)} reddit posts")
            progress_callback(f"   Additional relevant: {len(additional_arxiv_codes)} arxiv papers, {len(additional_reddit_codes)} reddit posts")
            
            ## Detailed source listings
            if referenced_arxiv_codes:
                arxiv_list = ", ".join([f"arxiv:{code}" for code in referenced_arxiv_codes])
                progress_callback(f"📄 Referenced ArXiv: {arxiv_list}")
            
            if referenced_reddit_codes:
                reddit_list = ", ".join(referenced_reddit_codes)
                progress_callback(f"💬 Referenced Reddit: {reddit_list}")
                
            if additional_arxiv_codes:
                additional_arxiv_list = ", ".join([f"arxiv:{code}" for code in additional_arxiv_codes])
                progress_callback(f"📋 Additional ArXiv: {additional_arxiv_list}")
                
            if additional_reddit_codes:
                additional_reddit_list = ", ".join(additional_reddit_codes)
                progress_callback(f"💭 Additional Reddit: {additional_reddit_list}")

        ## Return sources as separate lists
        return final_report.title, workflow_id, response_with_links, referenced_arxiv_codes, referenced_reddit_codes, additional_arxiv_codes, additional_reddit_codes


## Main Entry Point


def deep_research_query(
    user_question: str,
    max_agents: int = 3,
    max_sources_per_agent: int = 10,
    response_length: int = 4000,
    llm_model: str = "gpt-4.1-nano",
    progress_callback: Optional[Callable[[str], None]] = None,
    verbose: bool = False,
) -> Tuple[str, str, str, List[str], List[str], List[str], List[str]]:
    """
    Conduct LangChain-style deep research using multi-agent approach.

    Args:
        user_question: The research question to investigate
        max_agents: Maximum number of research agents to deploy (default: 3)
        max_sources_per_agent: Maximum papers per agent (default: 10)
        response_length: Target response length in words (default: 4000)
        llm_model: LLM model to use for all stages
        progress_callback: Optional callback function to receive progress updates
        verbose: Enable detailed progress reporting (default: False)

    Returns:
        Tuple of (title, workflow_id, final_response, referenced_arxiv_codes, referenced_reddit_codes, additional_arxiv_codes, additional_reddit_codes)
    """
    orchestrator = DeepResearchOrchestrator(llm_model)
    return orchestrator.conduct_deep_research(
        user_question=user_question,
        max_agents=max_agents,
        max_sources_per_agent=max_sources_per_agent,
        response_length=response_length,
        progress_callback=progress_callback,
        verbose=verbose,
    )
