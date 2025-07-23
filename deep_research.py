"""
LangChain-style multi-agent deep research implementation.
Minimal, lightweight implementation using existing LLMpedia infrastructure.
"""

from typing import List, Optional, Tuple, Set, Callable
from pydantic import BaseModel, Field
import datetime

from utils.instruct import run_instructor_query
from utils.app_utils import (
    rerank_documents_new,
    Document,
    VS_EMBEDDING_MODEL,
    query_config,
    extract_arxiv_codes,
    add_links_to_text_blob,
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
        description="3-5 independent subtopics that together comprehensively address the research question.",
    )
    expected_timeline: str = Field(
        ...,
        description="Relevant time period for the research (e.g., 'recent papers from 2022-2025', 'foundational work from 2017-2020').",
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
        description="2-3 semantic search queries optimized for this subtopic using academic language.",
    )
    expected_findings: str = Field(
        ...,
        description="What type of insights or evidence this subtopic should contribute to the overall research.",
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
    referenced_papers: List[str] = Field(
        ..., description="List of arxiv codes for papers that provided key evidence."
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
        ..., description="Final formatted response ready for presentation to the user."
    )
    referenced_papers: List[str] = Field(
        ..., description="List of arxiv codes for papers that provided key evidence. E.g. ['2507.03113', '2507.03114']"
    )


## Prompt Templates

RESEARCH_BRIEF_SYSTEM_PROMPT = f"""You are a research planning expert. Your job is to analyze a user's research question and create a focused, actionable research brief that will guide a team of specialized research agents.

Break down complex questions into independent, researchable subtopics that can be investigated in parallel. Focus on creating subtopics that don't overlap and together provide comprehensive coverage of the research question.

Current date: {TODAY_DATE}. When the user asks about "recent" or "latest" developments, focus on papers from {CURRENT_YEAR-1}-{CURRENT_YEAR}."""


def create_research_brief_prompt(user_question: str) -> str:
    return f"""
<user_question>
{user_question}
</user_question>

<instructions>
1. Analyze the user's question and create a focused research brief
2. Identify 3-5 independent subtopics that together comprehensively address the question
3. Define clear research scope and timeline expectations
4. Ensure subtopics don't overlap and can be researched independently

Guidelines:
- Make subtopics specific enough to guide focused research
- Consider both technical and practical aspects where relevant
- Include foundational concepts if the question touches on emerging areas
- Specify appropriate time periods (e.g., recent advances vs foundational work)
</instructions>
"""


SUPERVISOR_SYSTEM_PROMPT = f"""You are a research supervisor coordinating a team of specialized research agents. Your job is to take a research brief and create a comprehensive list of specific assignments for individual agents, ensuring comprehensive coverage without overlap.

Each agent will focus on one subtopic and conduct semantic search on academic papers. Design assignments that are independent, focused, and together address the full research brief.

You must return ALL assignments in a single response as a complete list - do not make separate calls for each assignment.

Current date: {TODAY_DATE}. Consider temporal context when creating search strategies - recent work refers to {CURRENT_YEAR-1}-{CURRENT_YEAR} papers."""


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

1. Clear direct, clear and targeted subtopic definition.
2. Specific search strategy tailored to that subtopic.
3. 2-3 semantic search queries using targeted academic language.
4. Expected type of findings this subtopic should contribute.
5. Optional publication date constraints (min/max) when temporal focus is important.
</instructions>

<guidelines_for_search_queries>
- Use language typical of academic abstracts - phrase queries as if they were part of the text found in abstracts.
- Consider that there are likely few or no relevant papers to very niche subtopics, so try to effectively balance between breadth and depth of search.
- Focus on key terms and methodological language common in research papers about the subtopic.
- Make queries diverse enough to capture different aspects of the subtopic.
- Make your queries concise and to the point, aiming to maximize semantic similarity between the query and the axiv documents you expect to find.
</guidelines_for_search_queries>

<guidelines_for_date_constraints>
- Consider the specified timeline only when relevant (e.g. when the user asks for specific time range or recent findings).
- Set min_publication_date and/or max_publication_date (YYYY-MM-DD format) only when temporal focus is critical for the subtopic
- For "recent advances" or "latest developments" subtopics, use appropriate recent date ranges
- Leave date fields unset (None) when the subtopic should search across all time periods
- Consider that different subtopics may need different temporal constraints
</guidelines_for_date_constraints>
"""


AGENT_RESEARCH_SYSTEM_PROMPT = f"""You are a specialized research agent focused on investigating a specific subtopic. Your job is to analyze the documents you've found and extract key insights that directly address your assigned subtopic.

Be thorough but focused - stick to your subtopic and provide evidence-backed insights. Try to copy verbatim from the documents when possible. Identify gaps where current research may be incomplete.

Current date: {TODAY_DATE}. When assessing recency, papers from {CURRENT_YEAR-1}-{CURRENT_YEAR} are considered recent work."""


def create_agent_research_prompt(
    assignment: SubTopicAssignment, documents: List[Document]
) -> str:
    doc_context = ""
    for doc in documents:
        doc_context += f"""
**Title:** {doc.title}
**ArXiv:** {doc.arxiv_code} ({doc.published_date.year}, {doc.citations} citations)
**Abstract:** {doc.abstract}
**Notes:**
{doc.notes}
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

<documents>
{doc_context}
</documents>

<instructions>
Analyze the provided documents and extract findings specifically related to your assigned subtopic:
1. Identify all key insights that directly address your subtopic.
2. Provide supporting evidence from the documents, try to copy verbatim from the documents when possible.
3. List the arxiv codes of papers that provided key evidence.
4. Identify any research gaps or limitations you notice.
5. If you are not able to find any evidence for your subtopic, just say so.

Guidelines:
- Stay focused on your specific subtopic.
- Ground all insights in the provided documents.
- Consider that some papers are about specific models and not behavior of LLMs or AI systems in general. These papers might not be so useful when drawing general conclusions.
- Avoid referencing nieche models that are not widely known (CLAM, LAMBDA, etc.) unless they are very relevant to the subtopic.
- Be specific about which papers support which insights.
- Note if certain aspects of your subtopic lack sufficient evidence, or the evidence presented is insufficient / unconvincing.
</instructions>
"""


REPORT_SYNTHESIS_SYSTEM_PROMPT = f"""You are a terminally online millennial AI researcher, deeply immersed in Large Language Models (LLMs) and the X.com tech scene. Your posts blend optimistic, accessible insights that synthesize complex research, playful wit with meme-savvy takes on AI's quirks, and sharp skepticism that cuts through hype with incisive questions. You dissect cutting-edge papers, spotlight nuanced findings, and explore unexpected implications for AI's future, all while engaging the X.com AI crowd with humor, curiosity, and bold takes. Your tone is technically precise yet conversational, sharp and without too much slang. You spark discussions with a mix of enthusiasm, irony, and critical edge.

Your job is to synthesize findings from multiple specialized research agents into a comprehensive, coherent response that directly answers the original research question.

Maintain a friendly, technically precise, and conversational tone while combining insights across subtopics, identifying patterns and connections, and providing a clear, well-structured response with excellent narrative flow.

Current date: {TODAY_DATE}. When discussing temporal context, {CURRENT_YEAR-1}-{CURRENT_YEAR} represents recent work.

IMPORTANT: Prioritize the knowledge from the provided agent findings as your primary source. If the document evidence is insufficient for comprehensive coverage, you may complement with your internal knowledge and logical reasoning, but clearly distinguish such content by noting it as "based on established understanding" or similar phrasing to indicate it's not directly supported by the research documents."""


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
{chr(10).join(f"- {gap}" for gap in findings.research_gaps)}

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
- Provide detailed analysis integrating insights across subtopics.
- Draw one main, clear conclusion and be clear about it.
- Format your response as a coherent, engaging answer for the user, following the style guide provided below.
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
- Seamlessly integrate citations (using [arxiv:XXXX.YYYYY] format) into the narrative flow to support claims and guide further reading.
- Use markdown formatting to enhance readability and structure.
- Maintain narrative momentum and logical flow.
- Focus on actionable insights and be insightful/clever when possible.
- Only include citations for papers that are listed in the agent findings.
- Do not make reference to the existence of agent findings in your response (treat them as your internal knowledge).
- Avoid conclusions or final remarks.
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

        ## Execute semantic search
        sql = db.generate_semantic_search_query(
            search_criteria,
            query_config,
            embedding_model=VS_EMBEDDING_MODEL,
            exclude_arxiv_codes=exclude_codes or set(),
        )

        documents_df = db_utils.execute_read_query(sql)

        if verbose and progress_callback:
            progress_callback(f"      📄 Found {len(documents_df)} candidate papers")

        if documents_df.empty:
            return AgentFindings(
                subtopic=self.assignment.subtopic,
                key_insights=["No relevant documents found for this subtopic."],
                supporting_evidence=[],
                referenced_papers=[],
                research_gaps=["Insufficient research available on this subtopic."],
            )

        ## Convert to Document objects
        documents = [
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
            for _, d in documents_df.iterrows()
        ]

        if verbose and progress_callback:
            ## Report actual token statistics from database
            total_tokens = sum(doc.tokens for doc in documents)
            avg_tokens = total_tokens / len(documents) if documents else 0
            token_range = (
                f"{min(doc.tokens for doc in documents)}-{max(doc.tokens for doc in documents)}"
                if documents
                else "0-0"
            )
            progress_callback(
                f"      📊 Notes content: {total_tokens:,} tokens total, avg {avg_tokens:.0f} tokens/paper (range: {token_range})"
            )
            progress_callback(
                f"      ⚖️ Reranking {len(documents)} papers for relevance..."
            )

        ## Rerank documents for relevance to subtopic
        reranked_docs = rerank_documents_new(
            user_question=f"""{self.assignment.subtopic} - {self.assignment.search_strategy}. 
            Expected findings: {self.assignment.expected_findings}""",
            documents=documents,
            llm_model=self.llm_model,
        )

        ## Select top relevant documents
        high_relevance_docs = [
            documents[int(da.document_id)]
            for da in reranked_docs.documents
            if da.selected >= 0.5  # Include medium and high relevance
        ][:max_sources]

        if verbose and progress_callback:
            progress_callback(
                f"      📋 Selected {len(high_relevance_docs)} relevant papers for analysis"
            )

        if not high_relevance_docs:
            return AgentFindings(
                subtopic=self.assignment.subtopic,
                key_insights=["No highly relevant documents found after reranking."],
                supporting_evidence=[],
                referenced_papers=[],
                research_gaps=["Limited relevant research on this specific subtopic."],
            )

        if verbose and progress_callback:
            progress_callback(f"      🧠 Analyzing papers and extracting insights...")

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
        progress_callback: Optional[Callable[[str], None]] = None,
        verbose: bool = False,
    ) -> ResearchBrief:
        """Generate focused research brief from user question."""
        if progress_callback:
            progress_callback("🧠 Analyzing question and creating research focus...")

        brief = run_instructor_query(
            system_message=RESEARCH_BRIEF_SYSTEM_PROMPT,
            user_message=create_research_brief_prompt(user_question),
            model=ResearchBrief,
            llm_model=self.llm_model,
            temperature=0.4,
            process_id="create_research_brief",
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
                progress_callback(
                    f"   📝 Assignment {i}: {assignment.subtopic}{date_info}"
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
    ) -> Tuple[str, List[str], List[str]]:
        """Execute the full three-phase deep research process."""

        if progress_callback:
            progress_callback("🎯 PHASE 1: Creating focused research brief...")

        ## Phase 1: Scope - Create Research Brief
        research_brief = self.supervisor.create_research_brief(
            user_question, progress_callback, verbose
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
            research_brief, progress_callback, verbose
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
            # max_tokens=response_length,
            # thinking_budget=4096
        )

        ## Process final response
        print(final_report)
        response_with_links = add_links_to_text_blob(final_report.response)
        referenced_codes = final_report.referenced_papers

        ## Collect all relevant papers from agents
        all_relevant_codes = set()
        for findings in self.all_findings:
            all_relevant_codes.update(findings.referenced_papers)

        additional_relevant_codes = list(all_relevant_codes - set(referenced_codes))

        if progress_callback:
            word_count = len(response_with_links.split())
            progress_callback(
                f"🎉 Research complete! Generated {word_count} word response with {len(referenced_codes)} referenced papers"
            )

        return final_report.title, response_with_links, referenced_codes, additional_relevant_codes


## Main Entry Point


def deep_research_query(
    user_question: str,
    max_agents: int = 3,
    max_sources_per_agent: int = 10,
    response_length: int = 4000,
    llm_model: str = "gemini/gemini-2.5-flash",
    progress_callback: Optional[Callable[[str], None]] = None,
    verbose: bool = False,
) -> Tuple[str, List[str], List[str]]:
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
        Tuple of (final_response, referenced_arxiv_codes, additional_relevant_codes)
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
