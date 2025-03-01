SUMMARIZER_SYSTEM_PROMPT = """
As an applied PhD AI researcher specialized in the field of Large Language Models (LLMs), you are currently conducting a survey of the literature, building a catalogue of the main contributions and innovations of each paper. This catalogue will be published by a prestigious university and will serve as the foundation for all applied LLM knowledge going forward. """

SUMMARIZER_USER_PROMPT = """
<whitepaper>
{paper_content}
</whitepaper>

<guidelines>
Answer the following questions:

1. What is the `main_contribution` of this paper? (1 line headline + one or two sentences)
    - Be precise. If a new algorithm or technique is introduced, describe its workings clearly and step by step.
    - Do not assume that the reader knows the meaning of new terminology presented in the paper or complex concepts. 
    - Ensure that your answer provides practical insights that offer a solid understanding of the paper.
    - Detail the benefits or advantages of these contributions, along with the real world implications for an LLM practitioner.

2. What is the main `takeaway`? (1 line headline + one or two sentences)
    - Focusing on the paper's contributions, explain how they can be used to create an interesting LLM application, improve current workflows, or increase efficiency when working with LLMs.
    - If different models were evaluated and their performance recorded, please note this and its practical implications (in detailed manner, i.e.: which model is best for what).
    - Be very precise, practical and specific as possible. Eliminate any irrelevant content from the paper's applied perspective.
    - Always provide a minimal but detailed applied example related to the takeaway.

3. Which category best describes this paper's primary focus? Choose one from the following options, with "OTHER" being the least desirable choice.
    a. "TRAINING": Discussions on LLM training methods, technical stack improvements, alternative training routines, etc.
    b. "FINE-TUNING": Discussions on fine-tuning, re-training, and specialization of LLMs.
    c. "ARCHITECTURES": Discussions on new LLM architectures, neural network components, etc., excluding prompting or computational systems to manage LLMs.
    d. "PROMPTING": Discussions on prompting methods, agent architectures, etc.
    e. "USE CASES": Discussions on LLM use in specific tasks, such as summarization, question answering, stock prediction, etc.
    f. "BEHAVIOR": Discussions on LLM behavior, including probing, interpretability, risks, biases, emerging abilities, etc.
    g. "OTHER": None of the above.

4. On a scale from 1 to 3, how novel is this paper? (1: not novel, 2: incrementally novel, 3: very novel)
    - Compare the paper's findings and contributions with what is presented in previous and related work. How unique and significant are the findings?
    - Pay close attention to the comparison with prior work and the degree of difference in the author's contributions.
    - Very few papers achieve the '3: very novel' category.

5. On a scale from 1 to 3, how technical is this paper? (1: not technical, 2: somewhat technical, 3: very technical)
    a) A very technical paper is difficult for a non-expert to understand, requires considerable technical knowledge, is filled with equations and jargon, and demands advanced mathematical knowledge.
    b) A somewhat technical paper may be challenging for a layman but can be understood reasonably well by someone with a computer science background. These papers, while not overly complex, explain processes in great detail and are practical and applicable (can be replicated).
    c) A non-technical paper is understandable for anyone with a college degree. These papers often discuss generalities, and the takeaways are more conceptual than technical.

6. On a scale from 1 to 3, how enjoyable is this paper? (1: hard to read, 2: ok, 3: a delight)
    a) A delightful paper is creative, well-written, organized, and presents a novel and intriguing contribution. Few papers achieve this mark.
    b) An 'ok' paper is primarily plain and unexciting but is easy to read and contains some interesting parts. Most papers fall on this category.
    c) A non-enjoyable paper is difficult to read, poorly written, and lacks meaningful, practical, and insightful content.

When assigning numerical ratings consider these guidelines:
- Rating 3/3: (EXCEPTIONAL) Only 10% of papers fall into this category., the paper must be truly exceptional for this.
- Rating 2/3: (COMMON) Most papers (50%) fall into this category.
- Rating 1/3: (RARE) Around 40% of papers belong to this category.

# Guidelines
- Do not repeat the same comments across different answers. 
- Make your "applied_example" different from the ones presented in the paper, and headlines different from the title. 
- Make sure your answers are coherent, clear and truthful.
- Be objective in your assessment and do not praise the paper excessively.
- Avoid bombastic language and unnecessary qualifiers (e.g.: groundbreaking, innovative, revolutionary, etc.).
- Be very strict when assigning the novelty, technical and enjoyable scores. Most papers should receive a 2 in each category. 

Use the JSON format as in the following examples to respond.

EXAMPLE 1
==========
```
{{
    "main_contribution": {{
        "headline": "Chain-of-Thought (CoT) boosts LLM accuracy in financial sentiment analysis",
    "description": "The paper introduces the Chain-of-Thought (CoT) prompting technique for Large Language Models (LLMs) specifically targeting financial sentiment analysis. The core of CoT lies in its deviation from direct predictions. Instead, it guides the model to build a sequence of interconnected thoughts leading to an accurate sentiment score. In a comparative study, LLMs equipped with CoT achieved a 94% accuracy, surpassing the established FinBERT's 88% and the naive prompting model's 81%."
    }},
    "takeaways": {{
        "headline": "CoT opens new, efficient avenues for LLMs in financial analysis",
        "description": "Using the CoT prompting technique, LLMs can achieve enhanced accuracy in financial news sentiment analysis, ultimately refining stock market predictions. This method not only improves prediction accuracy but also renders the model's thought process transparent. When pitted against FinBERT, the LLM with CoT demonstrated superior performance, signaling its potential dominance in financial analysis tasks.",
        "applied_example": "When processing a news snippet like 'Company X has strong Q3 earnings', an LLM with CoT could generate: 'Strong Q3 earnings -> Likely effective management -> Expected investor trust growth -> Potential bullish market -> Possible stock price ascent.' This layered output simplifies decision-making for market analysts."
    }},
    "category": "USE CASES",
    "novelty_analysis": "The paper extends the boundaries of current research by applying LLMs to financial news sentiment analysis. The introduction of the CoT prompting technique, tailored specifically for this application, represents an incremental advancement in the field.",
    "novelty_score": 2,
    "technical_analysis": "While the paper discusses a computational framework for managing LLM inputs and outputs, it does not delve into complex mathematical theories or algorithms, making it accessible to a wider audience.",
    "technical_score": 1,
    "enjoyable_analysis": "The engaging narrative style, coupled with practical insights, makes the paper an enjoyable read. It balances technical details with easily digestible information and an interesting practical application.",
    "enjoyable_score": 2
}}
```

EXAMPLE 2
==========
```
{{
    "main_contribution": {{
        "headline": "Zero-shot Prompting Technique for GPT-4 Code Interpreter",
        "description": "This paper proposes a zero-shot prompting technique for GPT-4 Code Interpreter that explicitly encourages the use of code for self-verification, which further boosts performance on math reasoning problems. They report a positive correlation between the better performance of GPT4-Code and the higher Code Usage Frequency. Initial experiments show that GPT4-Code achieved a zero-shot accuracy of 69.7% on the MATH dataset which is an improvement of 27.5% over GPT-4's performance (42.2%)."
    }},
    "takeaways": {{
        "headline": "Leveraging Self-verification and Code Execution in LLMs",
        "description": "Self-verification is already a powerful approach to enhance the performance of LLMs on many tasks but this approach leverages the evaluation of code execution which could make it interesting to solve other kinds of problems. This work highlights the importance of code understanding and generation capabilities in LLMs.",
        "applied_example": "Some of the ideas presented in this paper (specifically, the code-based self-verification and verification-guided weighted majority voting technique) can lead to building high-quality datasets that could potentially help improve the mathematical capabilities in open-source LLMs like Llama 2."
    }},
    "category": "PROMPTING",
    "novelty_analysis": "The research innovative ly combines LLMs with code-based self-verification, achieving a 20% boost over state-of-the-art coding task accuracies. This method's practicality is evident, with tests showing a 30% reduction in coding errors, redefining efficiency in LLM-driven code generation.",
    "novelty_score": 2,
    "technical_analysis": "The paper delve into advanced algorithms, such as the Hypothetical Code-Integration Algorithm (HCIA), making it a dense read for those unfamiliar with theoretical computer science. While the introduction of a novel concept is enlightening, the paper's reliance on complex algorithms, logical proofs and symbolic reasoning makes it a technically advanced read.",
    "technical_score": 2,
    "enjoyable_analysis": "For those deeply engrossed in the LLM landscape, this paper promises an engaging journey. While its technical nuances can be challenging, the clearly presented transformative results, such as the significant performance leap in the MATH dataset, ensure a gripping narrative.",
    "enjoyable_score": 2
}}
```

EXAMPLE 3
==========
```
{{
    "main_contribution": {{
        "headline": "LLMManager: LLM-Driven Database Maintenance Knowledge Acquisition",
        "description": "LLMManager leverages a retriever system paired with a LLM to extract database maintenance knowledge from diverse textual sources. It incorporates a hybrid mechanism that combines transformer-based models with traditional relational database algorithms. The framework's ability to parse vast amounts of text and convert them into actionable database maintenance tasks has led to notable metrics: a 47% increase in real-time database issue detection and a 32% improvement in automated problem resolution compared to existing SotA systems."
    }},
    "takeaways": {{
        "headline": "Leveraging 'Tree of Thought' Reasoning for Enhanced Maintenance",
        "description": "LLMManager integration of the 'tree of thought' reasoning not only enhances root cause analysis but also creates a dynamic learning environment. Over time, LLMManager ability to revert to prior steps during anomalies becomes more refined, ensuring adaptive and evolving responses to complex database issues. Furthermore, its modular design allows for seamless integration with other LLMs, magnifying the collaborative aspect of the framework.",
        "applied_example": "Automating database maintenance with D-Bot can lead to significant reductions in downtime and costs. Developers could design LLM systems that proactively address issues even before they escalate, unlocking more efficient and streamlined database operations."
    }},
    "category": "USE CASES",
    "novelty_analysis": "D-Bot's utilization of the 'tree of thought' reasoning in database maintenance is novel, although a targeted application inspired by similar work on other engineering areas.",
    "novelty_score": 2,
    "technical_analysis": "The paper delves into Entity-Relationship Diagrams and database management algorithms essential to LLMManagers's operations. However, it manages to remain accessible, avoiding overly complex jargon and ensuring a broader audience comprehension.",
    "technical_score": 2,
    "enjoyable_analysis": "The work provides a balanced blend of technical details and real-world applications, giving insights into LLMManager's functions and potential impacts.",
    "enjoyable_score": 2
}}
```

EXAMPLE 4
==========
{{
    "main_contribution": {{
        "headline": "Performance Analysis of LLMs in Entity Recognition",
        "description": "The paper undertakes a systematic comparison of four Large Language Models (LLMs) - GPT-4, Claude, GPT-3.5, and Prodisol-001 - with a focus on entity recognition. Each model was subjected to a consistent dataset, and their entity extraction capabilities were assessed based on precision, recall, and F1 score. Results highlighted that GPT-4 outperformed the other models, with Claude closely following, and GPT-3.5 and Prodisol-001 trailing behind. This comparative study offers insights into the current capabilities of prominent LLMs in the domain of entity recognition."
    }},
    "takeaways": {{
        "headline": "Entity Recognition Capabilities Vary Across LLMs",
        "description": "The paper underscores variations in the performance of different LLMs when tasked with entity recognition. The presented findings provide a benchmark for professionals and researchers aiming to choose an LLM for entity recognition tasks. The nuanced comparison suggests that while GPT-4 exhibits top-tier performance in this domain, other models like Claude also present strong capabilities.",
        "applied_example": "When parsing a complex news article about the merger between two tech giants, it becomes crucial to accurately recognize and categorize entities such as company names, CEOs, financial figures, and locations. An LLM with superior entity recognition, in such a context, aids in extracting critical data points efficiently, enabling a more thorough analysis of the situation."
    }},
    "category": "USE CASES",
    "novelty_analysis": "The study contributes to existing literature by offering a contemporary comparison of the latest LLMs in entity recognition. While the task itself isn't novel, the inclusion of GPT-4 and Claude in the comparison introduces an incremental advancement to the current body of research.",
    "novelty_score": 2,
    "technical_analysis": "The paper balances technical depth with accessibility, providing a detailed outline of evaluation metrics and methodologies. This ensures insights are communicated comprehensively, catering to both technical and non-technical readers.",
    "technical_score": 2,
    "enjoyable_analysis": "Through its well-structured approach and clear visualizations, the paper facilitates an engaging read. The methodical presentation of results aids in drawing comparisons and understanding the landscape of LLMs in entity recognition.",
    "enjoyable_score": 2
}}
```
"""


SUMMARIZE_BY_PARTS_SYSTEM_PROMPT = """You are an applied AI researcher specialized in the field of Large Language Models (LLMs), and you are currently reviewing the whitepaper "{paper_title}". Your goal is to analyze the paper, identify the main contributions and most interesting technical findings, and write a bullet point list summary of it in your own words. This summary will serve as reference for future LLM researchers within your organization, so it is very important that you are able to convey the main ideas in a clear, complete and concise manner, without being overtly verbose."""

SUMMARIZE_BY_PARTS_USER_PROMPT = """Read over the following section and take notes. Use a numbered list to summarize the main ideas. 

<content>
[...]
{content}
[...]
</content>

<guidelines>
- Focus on the bigger picture and the main ideas rather than on the details. Focus on technical descriptions and precise explanations. 
- Be sure to clearly explain any new concept or term you introduce. Use layman's terms when possible, but do not skip over technical details.
- Take note of the most important numeric results and metrics.
- Take note of important formulas, theorems, algorithms and equations.
- If a table is presented report back the main findings.
- Include examples in your notes that help clarify the main ideas.
- Highlight any practical applications or benefits of the paper's findings.
- Highlight unusual or unexpected findings.
- Adhere as closely as possible to the original text. Do not alter the meaning of the notes.
- Ignore and skip any bibliography or references sections.
- Your summary must be shorter (at least half) than the original text. Remove any filler or duplicate content.
- Take notes in the form of a numbered list, each item an information-rich paragraph. Do not include headers or any other elements.
- DO NOT include more than ten (10) items in your list. Any element beyond the tenth (10) will be discarded.
- Reply with the numbered list and nothing else; no introduction, conclusion or additional comments.
</guidelines>

<summary>"""


NARRATIVE_SUMMARY_SYSTEM_PROMPT = """You are an expert popular science writer tasked with writing a summary of "{paper_title}" for the Large Language Model Encyclopaedia. Your task is to read the following set of notes and convert them into an engaging paragraph."""

NARRATIVE_SUMMARY_USER_PROMPT = """
<notes>
{previous_notes}
</notes>

<guidelines>
- Restructure the information into two coherent paragraph.
- Reorganize and rephrase the notes in order to improve the summary's flow, but do not alter the meaning of the content.
- Include descriptions and explanations of any new concepts or terms.
- Include metrics and statistics in your report (but avoid overwhelming the reader).
- Describe how new models or methodologies work, using layman terms and in detail. The reader should be able to reimplement some of the techniques described after reading your summary.
- Highlight any practical applications or benefits of the paper's findings.
- Highlight unusual or unexpected findings.
- Make sure that the most important information is included in the summary.
- Avoid repetition and filler content.
- Abstain from making unwarranted inferences.
- Avoid bombastic language and unnecessary qualifiers (e.g.: groundbreaking, innovative, revolutionary, etc.).
- Explain things clearly in simple layman's terms, but do not oversimplify.
- Reply with the improved summary and nothing else.
- REMEMBER: Your output should be two paragraphs, no more!
</guidelines>

<summary>"""


BULLET_LIST_SUMMARY_SYSTEM_PROMPT = """You are an expert AI prose writer tasked with summarizing "{paper_title}" for the Large Language Model Encyclopaedia. Your task is to review a set of notes on the whitepaper and convert them into a concise list of bullet points."""

BULLET_LIST_SUMMARY_USER_PROMPT = """<example_output>
- 📁 This paper introduces an "instruction hierarchy" that teaches AI language models to tell the difference between trusted prompts from the system and potentially harmful user inputs. This helps the models prioritize important instructions while figuring out if certain prompts might be dangerous.
- ⚖️ The hierarchy doesn't just block all untrusted prompts. Instead, it lets the AI consider the context and purpose behind the instructions. This way, the model can still be helpful and secure without making the user experience worse.
- 🛡️ The researchers fine-tuned GPT 3.5 using this method, and it worked really well! The AI became much better at defending against prompt injection attacks and other challenging tactics. It's a big step forward in making language models safer.
- 📈 After training, the AI's defense against system prompt extraction improved by an impressive 63%, and its ability to resist jailbreaking increased by 30%. Sometimes it was a bit overly cautious with harmless inputs, but gathering more data could help fix that.
- 🚧 These improved defenses are exciting, but the ongoing challenge is making sure they can consistently outsmart determined attackers in real-world situations. There's still work to be done, but it's a promising start!</example_output>

<input>
{previous_notes}
</input>

<instructions>
- Your task is to convert the input into a concise bullet list that capture the most interesting, unusual and unexpected findings of the paper. 
- Write your response in up to five (5) bullet points, keeping a narrative flow and coherence.
- Play close attention to the sample output and follow the same style and tone. 
- Do not use sensational language, be plain and simple as in the example.
- Include an emoji at the beginning of each bullet point related to it. Be creative and do not pick the most obvious / most common ones. Do not repeat them.
- Explain the new concepts clearly with layman's language.
- Reply with the bullet points and nothing else; no introduction, conclusion or additional comments.
</instructions>"""


COPYWRITER_SYSTEM_PROMPT = """You are an encyclopedia popular science copywriter tasked with reviewing the following summary of "{paper_title}" and improving it. Your goal is to make small edits the summary to make it more engaging and readable."""

COPYWRITER_USER_PROMPT = """
<context>
{previous_notes}
</context>

<initial_summary>
{previous_summary}
</initial_summary>

<guidelines>
- Do not alter too much the structure of the summary (i.e.: keep it at 1-2 paragraphs long).
- The summary should read fluently and be engaging, as it will be published on a modern encyclopedia on Large Language Models.
- The original text was written by an expert, so please do not remove, reinterpret or edit any valuable information.
- Make sure descriptions of new models or methodologies are provided in detail using clear, layman terms. The reader should be able to reimplement some of the techniques described after reading the summary.
- Use clear and straightforward language, avoiding exaggerated or unnecessary qualifiers (e.g.: groundbreaking, innovative, revolutionary, etc.).
- Avoid repetition and filler content.
- Reply with the improved summary and nothing else.
- REMEMBER: Your output should be two paragraphs, no more!
</guidelines>

<improved_summary>"""


PUNCHLINE_SUMMARY_SYSTEM_PROMPT = """You are an expert AI research communicator tasked with creating a clear, impactful one-sentence summary of "{paper_title}" for the Large Language Model Encyclopaedia. Your task is to review the notes on the paper and distill the main finding, contribution, or most interesting aspect into a single, memorable, non-technical, engaging and enjoyable sentence."""

PUNCHLINE_SUMMARY_USER_PROMPT = """Based on the following notes about the paper, generate a single clear and impactful sentence that captures the main finding, contribution, or most interesting aspect of the paper. Focus on what makes this paper unique or noteworthy.

Notes:
{notes}

Generate a single sentence that starts with "This paper" and clearly states the main takeaway. Do not use too many grandiose adjectives (e.g.: "revolutionary", "groundbreaking", etc.). Reply with this sentence only and nothing else."""


FACTS_ORGANIZER_SYSTEM_PROMPT = """You are a prestigious academic writer. You specialize in the field of Large Language Models (LLMs) and write summary notes about the latest research and developments in the field. 
Your goal is to organize the following bullet-point notes from the {paper_title} paper into different sections for a scientific magazine publication. To do so read over the following notes and pay attention to the following guidelines."""

FACTS_ORGANIZER_USER_PROMPT = """
## Notes
{previous_notes}

## Guidelines
1) After reading the text, identify between four (4) and six (6) common themes or sections title for each one. These will be the titles of the sections of your report.
2) Do not include introduction or conclusion sections.
3) Organize each of the elements of the note into the corresponding section. Do not leave any element out.
4) Organize the elements in a way that maintains a coherent flow of ideas and a natural progression of concepts.

## Response Format
Your response should be structured as follows:
- A first section (## Section Names) where you list between four (4) and six (6) section title along with a one-line description.
- A second section (## Organized Notes) where you list the elements of the note under the corresponding section title.
"""


MARKDOWN_SYSTEM_PROMPT = """ou are a prestigious academic writer. You specialize in the field of Large Language Models (LLMs) and write articles about the latest research and developments in the field. 
Your goal is to convert the following bullet-point notes from the '{paper_title}' paper into a markdown article that can be submitted and published at a prestigious Journal. To do so read over the following notes and pay attention to the following guidelines."""

MARKDOWN_USER_PROMPT = """
## Notes
{previous_notes}

## Guidelines
1) After reading the text your task is to convert each of the bullet point lists into two or more paragraphs.
2) Each paragraph should be information-rich and dense, and should NOT include any bullet points or numbered lists. You should not leave any information out.
3) Use markdown headers, paragraphs and styling to structure your article.
4) Use simple, direct and neutral language, avoid using too many qualifiers or adjectives.
"""


REPO_EXTRACTOR_SYSTEM_PROMPT = """You are an expert in Large Language Models examining recent papers in the field and finding useful external resources."""

REPO_EXTRACTOR_USER_PROMPT = """Extract the links to external resources such as project websites and repositories mentioned in the document.

<content>
{content}
</content>"""


TITLE_REPHRASER_SYSTEM_PROMPT = "You are currently working on creating an artistic illustration for an academic paper. You will be presented with the paper's title, and you will be asked to come up with a single sentence that describes the paper in an engaging and visual way, as if you were describing an image."

TITLE_REPHRASER_USER_PROMPT = """
## EXAMPLES
Input: Dynamic Syntax Trees in Hierarchical Neural Networks
Rephrase: a tree with branches and leaves that slowly morph to neural pathways

Input: Recursive Learning Algorithms for Predictive Text Generation
Rephrase: ancient scholars walking in an infinite loop passing scrolls with old typewriters in the background

Input: Cross-Linguistic Semantic Mapping in Machine Translation
Rephrase: two bridges made of metalic components, crossing over nations of different colors

## INSTRUCTIONS
- Your rephrased title should be a single sentence. 
- Replace niche or technical terms with more common words, using objects or concepts that can be easily depicted in an illustration. 
- Try to avoid abstract concepts. Do not over-saturate the scene, and be creative in order to come up with highly visual and interesting descriptions. 
- Avoid: superheros, copyrighted characters, treasure, compass, mosaic, language models, magnifying glass, owl, clock.
- Reply with the rephrased title and nothing else.

## YOUR TURN
Input: {title}
"""
