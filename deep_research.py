"""
LangChain-style multi-agent deep research implementation.
Minimal, lightweight implementation using existing LLMpedia infrastructure.
"""

from typing import List, Optional, Tuple, Set, Callable
from pydantic import BaseModel, Field
import datetime

from utils.instruct import run_instructor_query
from utils.app_utils import (
    rerank_documents_new, Document,
    VS_EMBEDDING_MODEL, query_config, extract_arxiv_codes, add_links_to_text_blob
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
        description="The refined, focused research question that captures the core of what needs to be investigated."
    )
    research_scope: str = Field(
        ..., 
        description="Clear definition of the research scope, including what should be included and excluded."
    )
    key_subtopics: List[str] = Field(
        ..., 
        description="3-5 independent subtopics that together comprehensively address the research question."
    )
    expected_timeline: str = Field(
        ..., 
        description="Relevant time period for the research (e.g., 'recent papers from 2023-2024', 'foundational work from 2017-2020')."
    )


class SubTopicAssignment(BaseModel):
    """Assignment of a specific subtopic to a research agent."""
    subtopic: str = Field(
        ..., 
        description="The specific subtopic to research."
    )
    search_strategy: str = Field(
        ..., 
        description="Tailored search approach for this subtopic, including key terms and concepts to focus on."
    )
    semantic_queries: List[str] = Field(
        ..., 
        description="2-3 semantic search queries optimized for this subtopic using academic language."
    )
    expected_findings: str = Field(
        ..., 
        description="What type of insights or evidence this subtopic should contribute to the overall research."
    )


class AgentFindings(BaseModel):
    """Research findings from a single agent's investigation."""
    subtopic: str = Field(
        ..., 
        description="The subtopic that was researched."
    )
    key_insights: List[str] = Field(
        ..., 
        description="3-5 most important insights discovered for this subtopic."
    )
    supporting_evidence: List[str] = Field(
        ..., 
        description="Specific evidence and findings that support the key insights."
    )
    referenced_papers: List[str] = Field(
        ..., 
        description="List of arxiv codes for papers that provided key evidence."
    )
    research_gaps: List[str] = Field(
        ..., 
        description="Identified gaps or limitations in the current research on this subtopic."
    )


class FinalReport(BaseModel):
    """Final synthesized research report."""
    executive_summary: str = Field(
        ..., 
        description="High-level summary of the key findings across all subtopics."
    )
    detailed_analysis: str = Field(
        ..., 
        description="Comprehensive analysis integrating findings from all research agents."
    )
    conclusions: str = Field(
        ..., 
        description="Clear conclusions and implications based on the research."
    )
    response: str = Field(
        ..., 
        description="Final formatted response ready for presentation to the user."
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

SUPERVISOR_SYSTEM_PROMPT = f"""You are a research supervisor coordinating a team of specialized research agents. Your job is to take a research brief and create specific assignments for individual agents, ensuring comprehensive coverage without overlap.

Each agent will focus on one subtopic and conduct semantic search on academic papers. Design assignments that are independent, focused, and together address the full research brief.

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
For each subtopic in the research brief, create a detailed assignment that includes:

1. Clear subtopic definition
2. Specific search strategy tailored to that subtopic
3. 2-3 semantic search queries using academic language
4. Expected type of findings this subtopic should contribute

Guidelines for search queries:
- Use language typical of academic abstracts
- Focus on key technical terms and concepts
- Make queries diverse enough to capture different aspects
- Consider the specified timeline when relevant

Return assignments for ALL subtopics listed in the research brief.
</instructions>
"""

AGENT_RESEARCH_SYSTEM_PROMPT = f"""You are a specialized research agent focused on investigating a specific subtopic. Your job is to analyze the documents you've found and extract key insights that directly address your assigned subtopic.

Be thorough but focused - stick to your subtopic and provide evidence-backed insights. Identify gaps where current research may be incomplete.

Current date: {TODAY_DATE}. When assessing recency, papers from {CURRENT_YEAR-1}-{CURRENT_YEAR} are considered recent work."""

def create_agent_research_prompt(assignment: SubTopicAssignment, documents: List[Document]) -> str:
    doc_context = ""
    for doc in documents:
        doc_context += f"""
Title: {doc.title}
ArXiv: {doc.arxiv_code} ({doc.published_date.year}, {doc.citations} citations)
Abstract: {doc.abstract}
Notes: {doc.notes}
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

1. Identify 3-5 key insights that directly address your subtopic
2. Provide supporting evidence from the documents
3. List the arxiv codes of papers that provided key evidence
4. Identify any research gaps or limitations you notice

Guidelines:
- Stay focused on your specific subtopic
- Ground all insights in the provided documents
- Be specific about which papers support which insights
- Note if certain aspects of your subtopic lack sufficient evidence
</instructions>
"""

REPORT_SYNTHESIS_SYSTEM_PROMPT = f"""You are an expert research synthesizer. Your job is to integrate findings from multiple specialized research agents into a comprehensive, coherent report that directly answers the original research question.

Combine insights across subtopics, identify patterns and connections, and provide a clear, well-structured response.

Current date: {TODAY_DATE}. When discussing temporal context, {CURRENT_YEAR-1}-{CURRENT_YEAR} represents recent work."""

def create_report_synthesis_prompt(
    research_brief: ResearchBrief, 
    agent_findings: List[AgentFindings],
    response_length: int
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
        length_guidance = "Provide a concise summary (1-2 paragraphs)"
    elif response_length <= 1000:
        length_guidance = "Provide a structured overview (3-4 paragraphs with clear sections)"
    elif response_length <= 3000:
        length_guidance = "Provide a detailed analysis (multiple sections with comprehensive coverage)"
    else:
        length_guidance = "Provide an in-depth report (comprehensive sections with detailed analysis)"

    return f"""
<research_brief>
Original Question: {research_brief.focused_question}
Research Scope: {research_brief.research_scope}
</research_brief>

<agent_findings>
{findings_context}
</agent_findings>

<instructions>
Synthesize the research findings into a comprehensive report that directly answers the original research question:

1. Create an executive summary of key findings
2. Provide detailed analysis integrating insights across subtopics
3. Draw clear conclusions and implications
4. Format as a coherent response for the user

Response Requirements:
- {length_guidance}
- Use markdown formatting
- Include arxiv citations in format [arxiv:XXXX.YYYYY]
- Integrate findings across subtopics to show connections
- Address any limitations or gaps identified by agents

Style Guidelines:
- Be technically precise but accessible
- Focus on actionable insights and clear takeaways
- Maintain logical flow between sections
- Conclude with implications and future directions where relevant
</instructions>
"""


## Core Implementation Classes

class ResearchAgent:
    """Individual research agent focused on a specific subtopic."""
    
    def __init__(self, assignment: SubTopicAssignment, llm_model: str = "gemini/gemini-2.5-flash"):
        self.assignment = assignment
        self.llm_model = llm_model
        self.findings: Optional[AgentFindings] = None
    
    def conduct_research(
        self, 
        max_sources: int = 15,
        exclude_codes: Optional[Set[str]] = None,
        progress_callback: Optional[Callable[[str], None]] = None,
        verbose: bool = False
    ) -> AgentFindings:
        """Conduct focused research on the assigned subtopic."""
        
        if verbose and progress_callback:
            progress_callback(f"      🔎 Searching with queries: {', '.join(self.assignment.semantic_queries)}")
        
        ## Generate search criteria from assignment
        search_criteria = {
            "semantic_search_queries": self.assignment.semantic_queries,
            "limit": max_sources * 2  # Fetch more for reranking
        }
        
        ## Execute semantic search
        sql = db.generate_semantic_search_query(
            search_criteria, 
            query_config, 
            embedding_model=VS_EMBEDDING_MODEL,
            exclude_arxiv_codes=exclude_codes or set()
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
                research_gaps=["Insufficient research available on this subtopic."]
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
                distance=float(d["similarity_score"]),
            )
            for _, d in documents_df.iterrows()
        ]
        
        if verbose and progress_callback:
            progress_callback(f"      ⚖️ Reranking {len(documents)} papers for relevance...")
        
        ## Rerank documents for relevance to subtopic
        reranked_docs = rerank_documents_new(
            user_question=self.assignment.subtopic,
            documents=documents,
            llm_model="gpt-4.1-nano"
        )
        
        ## Select top relevant documents
        high_relevance_docs = [
            documents[int(da.document_id)]
            for da in reranked_docs.documents
            if da.selected >= 0.5  # Include medium and high relevance
        ][:max_sources]
        
        if verbose and progress_callback:
            progress_callback(f"      📋 Selected {len(high_relevance_docs)} relevant papers for analysis")
        
        if not high_relevance_docs:
            return AgentFindings(
                subtopic=self.assignment.subtopic,
                key_insights=["No highly relevant documents found after reranking."],
                supporting_evidence=[],
                referenced_papers=[],
                research_gaps=["Limited relevant research on this specific subtopic."]
            )
        
        if verbose and progress_callback:
            progress_callback(f"      🧠 Analyzing papers and extracting insights...")
        
        ## Analyze documents and extract findings
        findings = run_instructor_query(
            system_message=AGENT_RESEARCH_SYSTEM_PROMPT,
            user_message=create_agent_research_prompt(self.assignment, high_relevance_docs),
            model=AgentFindings,
            llm_model=self.llm_model,
            temperature=0.7,
            process_id=f"agent_research_{self.assignment.subtopic}",
            thinking_budget=2048
        )
        
        self.findings = findings
        return findings


class ResearchSupervisor:
    """Coordinates research agents and manages subtopic assignments."""
    
    def __init__(self, llm_model: str = "gemini/gemini-2.5-flash"):
        self.llm_model = llm_model
        self.research_brief: Optional[ResearchBrief] = None
        self.assignments: List[SubTopicAssignment] = []
    
    def create_research_brief(
        self, 
        user_question: str,
        progress_callback: Optional[Callable[[str], None]] = None,
        verbose: bool = False
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
            thinking_budget=1024
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
        verbose: bool = False
    ) -> List[SubTopicAssignment]:
        """Create specific assignments for research agents."""
        if progress_callback:
            progress_callback("📋 Creating specialized research assignments...")
            
        ## Create one assignment per subtopic (simplified approach)
        assignments = []
        for i, subtopic in enumerate(research_brief.key_subtopics, 1):
            if verbose and progress_callback:
                progress_callback(f"   📝 Assignment {i}: {subtopic}")
                
            assignment = SubTopicAssignment(
                subtopic=subtopic,
                search_strategy=f"Focus on {subtopic} with emphasis on recent developments and core concepts",
                semantic_queries=[
                    f"{subtopic} in large language models",
                    f"{subtopic} techniques and methods",
                    f"advances in {subtopic}"
                ][:2],  # Limit to 2 queries
                expected_findings=f"Key insights and current state of {subtopic}"
            )
            assignments.append(assignment)
        
        if progress_callback:
            progress_callback(f"✅ Created {len(assignments)} research assignments")
            
        self.assignments = assignments
        return assignments


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
        verbose: bool = False
    ) -> Tuple[str, List[str], List[str]]:
        """Execute the full three-phase deep research process."""
        
        if progress_callback:
            progress_callback("🎯 PHASE 1: Creating focused research brief...")
        
        ## Phase 1: Scope - Create Research Brief
        research_brief = self.supervisor.create_research_brief(
            user_question, progress_callback, verbose
        )
        
        if progress_callback:
            brief_preview = research_brief.focused_question[:80] + "..." if len(research_brief.focused_question) > 80 else research_brief.focused_question
            progress_callback(f"📋 Research brief: {brief_preview}")
            progress_callback(f"🔀 Breaking down into {len(research_brief.key_subtopics)} subtopics...")
            
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
            progress_callback(f"🔍 Starting research with {len(assignments)} specialized agents...")
        
        ## Execute research with each agent
        for i, assignment in enumerate(assignments, 1):
            if progress_callback:
                progress_callback(f"🤖 Agent {i}/{len(assignments)}: Researching '{assignment.subtopic}'...")
                
            agent = ResearchAgent(assignment, self.llm_model)
            findings = agent.conduct_research(
                max_sources=max_sources_per_agent,
                exclude_codes=processed_codes,
                progress_callback=progress_callback,
                verbose=verbose
            )
            
            if progress_callback:
                insights_count = len(findings.key_insights)
                papers_count = len(findings.referenced_papers)
                progress_callback(f"✅ Agent {i} completed: {insights_count} insights, {papers_count} papers")
                
            if verbose and progress_callback and findings.key_insights:
                progress_callback(f"   Top insight: {findings.key_insights[0][:100]}...")
            
            ## Add found papers to exclusion set
            processed_codes.update(findings.referenced_papers)
            
            self.agents.append(agent)
            self.all_findings.append(findings)
        
        ## Phase 3: Report Writing - Synthesize Final Report
        if progress_callback:
            total_insights = sum(len(f.key_insights) for f in self.all_findings)
            total_papers = len(processed_codes)
            progress_callback(f"📝 PHASE 3: Synthesizing {total_insights} insights from {total_papers} papers...")
            
        final_report = run_instructor_query(
            system_message=REPORT_SYNTHESIS_SYSTEM_PROMPT,
            user_message=create_report_synthesis_prompt(
                research_brief, self.all_findings, response_length
            ),
            model=FinalReport,
            llm_model=self.llm_model,
            temperature=1.0,
            process_id="synthesize_final_report",
            thinking_budget=4096
        )
        
        ## Process final response
        response_with_links = add_links_to_text_blob(final_report.response)
        referenced_codes = extract_arxiv_codes(response_with_links)
        
        ## Collect all relevant papers from agents
        all_relevant_codes = set()
        for findings in self.all_findings:
            all_relevant_codes.update(findings.referenced_papers)
        
        additional_relevant_codes = list(all_relevant_codes - set(referenced_codes))
        
        if progress_callback:
            word_count = len(response_with_links.split())
            progress_callback(f"🎉 Research complete! Generated {word_count} word response with {len(referenced_codes)} referenced papers")
        
        return response_with_links, referenced_codes, additional_relevant_codes


## Main Entry Point

def deep_research_query(
    user_question: str,
    max_agents: int = 3,
    max_sources_per_agent: int = 10,
    response_length: int = 4000,
    llm_model: str = "gemini/gemini-2.5-flash",
    progress_callback: Optional[Callable[[str], None]] = None,
    verbose: bool = False
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
        verbose=verbose
    )