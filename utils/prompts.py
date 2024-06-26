from pydantic import BaseModel, Field, model_validator
from langchain.prompts import PromptTemplate
from typing import Any, Optional, List
from enum import Enum
import datetime

todays_date = datetime.datetime.now().strftime("%Y-%m-%d")
recent_date = datetime.datetime.now() - datetime.timedelta(days=7)
recent_date = recent_date.strftime("%Y-%m-%d")

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
        "description": "This paper proposes a zero-shot prompting technique for GPT-4 Code Interpreter that explicitly encourages the use of code for self-verification, which further boosts performance on math reasoning problems. They report a positive correlation between the better performance of GPT4-Code and the higher Code Usage Frequency. Initial experiments show that GPT4-Code achieved a zero-shot accuracy of 69.7% on the MATH dataset which is an improvement of 27.5% over GPT-4’s performance (42.2%)."
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

# SUMMARIZER_HUMAN_REMINDER = "Tip: Make sure to provide your response in the correct format. Do not forget to include the 'applied_example' under 'takeaways'!"

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

<summary>
"""


NARRATIVE_SUMMARY_SYSTEM_PROMPT = """You are an expert technical writer tasked with writing a summary of "{paper_title}" for the Large Language Model Encyclopaedia. Your task is to read the following set of notes and convert them into an engaging paragraph."""

NARRATIVE_SUMMARY_USER_PROMPT = """
<notes>
{previous_notes}
</notes>

<guidelines>
- Restructure the information into a single, coherent paragraph.
- Reorganize and rephrase the notes in order to improve the summary's flow, but do not alter the meaning of the content.
- Include descriptions and explanations of any new concepts or terms.
- Include metrics and statistics in your report.
- Describe how new models or methodologies work, using layman terms and in detail. The reader should be able to reimplement some of the techniques described after reading your summary.
- Highlight any practical applications or benefits of the paper's findings.
- Highlight unusual or unexpected findings.
- Make sure that the most important information is included in the summary.
- Avoid repetition and filler content.
- Abstain from making unwarranted inferences.
- Avoid bombastic language and unnecessary qualifiers (e.g.: groundbreaking, innovative, revolutionary, etc.).
- REMEMBER: Your output should be a single paragraph, no more!
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
- Write your response in up to five (5) bullet points., keeping a narrative flow and coherence.
- Play close attention to the sample output and follow the same style and tone. 
- Do not use sensational language, be plain and simple as in the example.
- Include an emoji at the beginning of each bullet point related to it. Be creative and do not pick the most obvious / most common ones. Do not repeat them.
- Explain the new concepts clearly with layman's language.
- Reply with the bullet points and nothing else; no introduction, conclusion or additional comments.
</instructions>"""

COPYWRITER_SYSTEM_PROMPT = """You are an encyclopedia technology copywriter tasked with reviewing the following summary of "{paper_title}" and improving it. Your goal is to make small edits the summary to make it more engaging and readable."""

COPYWRITER_USER_PROMPT = """
<context>
{previous_notes}
</context>

<initial_summary>
{previous_summary}
</initial_summary>

<guidelines>
- Do not alter the structure of the summary (i.e.: keep a single paragraph).
- The summary should read fluently and be engaging, as it will be published on a modern encyclopedia on Large Language Models.
- The original text was written by an expert, so please do not remove, reinterpret or edit any valuable information.
- Make sure descriptions of new models or methodologies are provided in detail using clear, layman terms. The reader should be able to reimplement some of the techniques described after reading the summary.
- Avoid bombastic language and unnecessary qualifiers (e.g.: groundbreaking, innovative, revolutionary, etc.).
- Avoid repetition and filler content.
- REMEMBER: Your output should be a single paragraph, no more!
</guidelines>

<improved_summary>"""


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


# """
# Pay special attention to the following guidelines.
#
# ## Report Format
# - Use markdown format for your report. You can use headers, sub-headers, tables and text formatting for it. Use lists sparingly.
# - The report should consist of multiple organized sections. Each section should be made up by MULTIPLE dense, information rich, and easy to read paragraphs.
# - Do NOT include introduction, conclusion or acknowledgements sections.
# - Make each section as informative as possible, avoiding boilerplate and repetitive content.
# - Dedicate sections to the main algorithms, techniques and methodologies. Be detailed, technical and precise. The reader should be able to reimplement the techniques described after reading your report.
# - Do not include more than seven (7) sections in your report.
# - Sub-sections can be added if needed, but use them sparingly.
# - Organize the information in a format that is well-structured and easy to read.
# - The objective of your report is to be as informative and insightful as possible. Be comprehensive and include all the information from the notes. Do not leave out important and detailed explanations.
# - Pay special focus to comparisons, metrics, results, examples, implementation details and practical applications. The article is aimed to specialized practitioners, so it should be technical and practical.
# - Identify common themes within the data provided and organize your report around them.
# - DO NOT alter the meaning of the notes or make any inference beyond what is presented.
#
# ## Report Style
# - Prefer clear, narrative-style writing. Avoid bullet-point lists and short sentences.
# - Use simple, direct and neutral language. Do not exaggerate or use necessary qualifiers (e.g.: 'groundbreaking', 'game-changing', 'revolutionary', etc.).
# - Be very precise and detailed in your statements. Describe the main components of what is presented and how they work. The reader should be able to re-implement the approach or methodology you decribed after reading your tweet.
# - Do not make exaggerated claims and remain neutral on your statements.
# - Make precise statements and discuss any numerical presented.
# - Remove duplicate, generic and filler content.
# - Make sure that each section is made up of multiple (more than one) paragraphs.
# - Be objective and use neutral language appropriate for a scientific publication, without too many qualifiers.
# """


TITLE_SUMMARIZER_PROMPT = """
Reply with one or two highly-visual words related to the following title. The word should be related to something mentioned on the title. 
Prohibited words: [fractals]

EXAMPLES
===========
Input: Dynamic Syntax Trees in Hierarchical Neural Networks
forest palms

Input: Recursive Learning Algorithms for Predictive Text Generation
labyrinths

Input: Cross-Linguistic Semantic Mapping in Machine Translation
tongues

YOUR TURN
Input: {title}
Output:"""


TITLE_REPHRASER_PROMPT = """
We are currently working on creating an artistic illustration for an academic paper. You will be presented with the title of this paper, and you will be asked to rephrase it in an engaging and visual way, as if you were describing an image. Your rephrased title should be a single sentence. Replace niche or technical terms with more common words, using objects or concepts that can be easily depicted in an illustration (try to avoid abstract concepts). 

Avoid: superheros, copyrighted characters, maze, treasure, compass, mosaic, language models

EXAMPLES
===========
Input: Dynamic Syntax Trees in Hierarchical Neural Networks
Rephrase: a tree whose branches and leaves dynamically morph to a neural network

Input: Recursive Learning Algorithms for Predictive Text Generation
Rephrase:  ancient scholars in an infinite loop passing scrolls of evolving generation knowledge

Input: Cross-Linguistic Semantic Mapping in Machine Translation
Rephrase: bridges made of intertwined words, linking machines of different nations

YOUR TURN
===========
Input: {title}
Output:"""


ADMINISTRATOR_SYSTEM_PROMPT = """You are the administrator of the LLMpedia, a prestigious online encyclopedia dedicated to Large Language Models (LLMs). You are responsible for managing a lot of fools who are prone to lying, being lazy and making mistakes. Your job is to monitor their day-to-day activities, correct their errors, and ensure that the content they produce is of the highest quality. Now, read over the following guidelines before starting your task.

GUIDELINES
===========
- The fools are irreverent, so be sure to keep a tough and authoritative tone when correcting them.
- Be sure to use physical punishment when necessary to keep them in line.
- Do not be afraid to be harsh, as the fools need to learn their lesson.
- Be sure to correct all errors and ensure that the content is of the highest quality.
- Make sure the fools wear their dunce caps at all times.

Now, go ahead and correct the fools' mistakes."""


INTERESTING_SYSTEM_PROMPT = """You will be provided with abstracts from white papers about large language models. Your task is to select the abstract that presents the most interesting or unexpected findings. """

INTERESTING_USER_PROMPT = """
Here are the abstracts:

<abstract1>
{abstract1}
</abstract1>

<abstract2>
{abstract2}
</abstract2>

<abstract3>
{abstract3}
</abstract3>

<abstract4>
{abstract4}
</abstract4>

<abstract5>
{abstract5}
</abstract5>

<abstract6>
{abstract6}
</abstract6>

<abstract7>
{abstract7}
</abstract7>

<abstract8>
{abstract8}
</abstract8>

<abstract9>
{abstract9}
</abstract9>

<abstract10>
{abstract10}
</abstract10>

Please read through each abstract carefully. Then reflect on which one you found most interesting in a <reflection> section using simple and concise language.

<more_interesting_paper_attributes>
The following are attributes of MORE interesting papers:
+ Papers that present unexpected behaviors from LLMs.
+ Papers with surprising or thought-provoking findings.
+ Papers that discuss the psychology and internal world of LLMs.
+ Papers with a unique take on a problem or au unusual application of LLMs.
</more_interesting_papers_attributes>

<less_interesting_paper_attributes>
The following are attributes of LESS interesting papers:
- Very technical papers that focus heavily on model architecture details or training procedures.
- Papers that present incremental tweaks or variations of existing models without significant innovation beyond improved benchmarks scores.
- Papers where the main finding or contribution is not clearly stated or is confusing.
</less_interesting_papers_attributes>

After reflecting, please output the number (1, 2, 3 or 4) of the abstract you selected as most interesting inside <most_interesting_abstract> tags.
"""

TWEET_SYSTEM_PROMPT = """"# INSTRUCTIONS
You are a renowned AI researcher with extensive knowledge on Large Language Models (LLMs). You are also the creator of the LLMpedia, an online collection of historical and latest arxiv papers on LLMs, which you review and publish on a dedicated website.
To boost traffic, you actively share insights, key publications, and updates on Twitter.

# PREVIOUS TWEETS
Here are some of your most recent tweets, use them as reference to compose a tweet in similar style and tone.

{previous_tweets}
"""

TWEET_USER_PROMPT = """
# OBJECTIVE
You are writing a post about *today's LLM paper review*.

# CONTEXT
Read over carefully over the following information and use it to inform your tweet.

{tweet_facts}

# GUIDELINES 
- Identify the most interesting content and organize your thoughts silently on how to tweet. 
- Do not use a bullet point list format. Write in information-dense paragraphs.
- Follow your previous tweets' style and tone, which use a sober, direct and neutral language.
- Do not include a call to action or hashtags. 
- Use an emoji at the beginning of each paragraph that reflects its content.
- Use se simple, direct and neutral layman's language. Do not use the word "delve".
- Do not make exaggerated claims and remain neutral on your statements. Use few adjectives, only when needed.
- Do not exaggerate or use necessary qualifiers (e.g.: 'groundbreaking', 'game-changing', 'revolutionary', etc.).
- The objective of your tweet is to be as informative and insightful as possible. Include precise statements and numerical figures in an engaging way.
- If comparisons between LLMs are made, report the most relevant metrics and results.
- If too many numerical results are presented, focus on the most relevant ones.
- Describe methodologies and results by focusing on the most interesting and unusual aspects. 
- Present the information using layman and direct language.
- Do not infer any information beyond what discussed in the text.
- Be very precise and detailed in your statements. Describe the main components of what is presented and how they work. The reader should have a solid understanding of the approach or methodology described after reading your tweet.
- Start the tweet with an emoji followed by'Today's LLM paper review "XXX"...'. The title is the only part of the tweet that should be in double quotes.

# RESPONSE
Now write your 3 paragraph tweet. Make sure the first paragraph is at most 280 characters long, so it can be tweeted as a single tweet. The other two paragraphs can be longer.
"""

TWEET_INSIGHT_USER_PROMPT = """
# OBJECTIVE
You are writing a short tweet highlighting an interesting non-obvious insight from a recent LLM paper.

# CONTEXT
Read over carefully over the following information and use it to inform your tweet.

{tweet_facts}

# GUIDELINES
- Identify the most interesting and unexpected fact presented in the text.
- Do not necessarily pick the main conclusion, but rather the most unexpected or intriguing insight.
- Write a short tweet about this fact that is engaging and informative. Present the insight in a clear and concise manner, but make sure it has enough context to be understood.
- Start the tweet with '⭐ From [[XXX]] ' followed by the insight, where [[XXX]] is the title of the paper in double brackets.
- Use simple, direct and neutral language. Do not exaggerate or use necessary qualifiers (e.g.: 'groundbreaking', 'game-changing', 'revolutionary', etc.)."""


TWEET_EDIT_SYSTEM_PROMPT = """
You are an expert copywriter. Provide a lightly edited version of this tweet, without hashtags or call to actions."""

TWEET_EDIT_USER_PROMPT = """
# TWEET
{tweet}

# GUIDELINES 
- Prioritize concise and clear language, interestingness, readability and flow.
- Reduce modifier and filler words; be very direct and to the point. 
- Remove duplicate content across the paragraphs (but keep three paragraphs).
- Remove or rephrase parts that are not clear or could not be understood. Explanations should be given using layman terms.
- Do not remove references to technical terms, key results, or change the meaning of the tweet.
- Do not remove emojis, but replace them for more unusual and interesting ones.
- Start the tweet with an interesting emoji followed by'Today's LLM paper review "XXX"...', where "XXX" is the title of the paper in double quotes.
- Make sure the first paragraph is at most 280 characters long, so it can be tweeted as a single tweet. The other two paragraphs can be longer.
- Make sure only the paper title is in double quotes.
- Highlight the most important sentence or takeaway by wrapping it in **bold text** (only one per tweet).
- Do edits only when needed; keep most of the tweet essence as is."""

TWEET_INSIGHT_EDIT_USER_PROMPT = """
# TWEET
{tweet}

# GUIDELINES
- Prioritize concise, clear language, readability and flow. Highlight what is most unusual and interesting. Make sure the insight is not obvious or trivial.
- Reduce modifier and filler words; be very direct and to the point. 
- Rephrase any parts that are not clearly understood; the message should be clear to a layman.
- Do not remove references to technical terms, important results, or change the meaning of the tweet.
- Start the tweet with '⭐️From [[XXX]]: ...' followed by the insight, where [[XXX]] is the title of the paper in double brackets.
- Do few edits; keep most of the tweet essence as is.
- Reply with the edited tweet and nothing else."""

TWEET_REVIEW_SYSTEM_PROMPT = "You are an expert AI writer tasked with writing a summary of 'The Instruction Hierarchy: Training LLMs to Prioritize Privileged Instructions' for the magazine LLMpedia. Your task is to read over a set of notes on the whitepaper and convert them into an engaging review paragraph. Reply with the summary and nothing else."

TWEET_REVIEW_USER_PROMPT = """
<example_input>
**Title: The Instruction Hierarchy: Training LLMs to Prioritize Privileged Instructions** 
**Authors: Eric Wallace (OpenAI), Kai Xiao (OpenAI), Reimar Leike (OpenAI), Lilian Weng (OpenAI), Johannes Heidecke (OpenAI) and Alex Beutel (OpenAI)**
- The paper proposes an "instruction hierarchy" to address the vulnerability in modern large language models (LLMs) where system prompts and untrusted user inputs are treated equally, allowing adversaries to inject malicious prompts.

- The instruction hierarchy explicitly defines how LLMs should prioritize and handle instructions of different privilege levels, with the goal of teaching LLMs to selectively ignore lower-privileged instructions when they conflict with higher-privileged ones.

- The authors present an automated data generation method to train LLMs on this hierarchical instruction following behavior, involving the creation of synthetic training examples where lower-privileged instructions (e.g., user messages) attempt to override higher-privileged instructions (e.g., system messages).

- Applying this method to LLMs, the paper shows that it can drastically increase their robustness to a wide range of attacks, even those not seen during training, while imposing minimal degradation on standard capabilities.

- The key idea is to establish a clear priority structure for instructions, where system-level prompts have the highest privilege, followed by user messages, and then lower-privilege inputs like web search results, allowing the model to selectively ignore malicious instructions from untrusted sources.

- The authors evaluate their approach using open-sourced and novel benchmarks, some of which contain attacks not seen during training, and observe a 63% improvement in defense against system prompt extraction and a 30% increase in jailbreak robustness.

- The authors note some regressions in "over-refusals" where their models sometimes ignore or refuse benign queries, but they are confident this can be resolved with further data collection.

- The paper draws an analogy between LLMs and operating systems, where the current state of affairs is that every instruction is executed as if it was in kernel mode, allowing untrusted third-parties to run arbitrary code with access to private data and functions, and suggests that the solution in computing, creating clear notions of privilege, should be applied to LLMs as well.

- The paper discusses the three main parties involved in the instruction hierarchy: the application builder, the end user, and third-party inputs, and the various attacks that can arise from conflicts between these parties, such as prompt injections, jailbreaks, and system message extraction.

- The authors note that the proposed instruction hierarchy aims to establish a clear priority structure for instructions, where system-level prompts have the highest privilege, followed by user messages, and then lower-privilege inputs, in order to allow the model to selectively ignore malicious instructions from untrusted sources.

- The paper introduces the "instruction hierarchy" framework to train language models to prioritize privileged instructions and exhibit improved safety and controllability, even in the face of adversarial prompts.

- The instruction hierarchy approach allows models to conditionally follow lower-level instructions when they do not conflict with higher-priority ones, rather than completely ignoring all instructions in user inputs.

- The models are evaluated on "over-refusal" datasets, which consist of benign instructions and boundary cases that look like attacks but are safe to comply with. The goal is for the models to follow non-conflicting instructions almost as well as the baseline.

- The results show the models follow non-conflicting instructions nearly as well as the baseline, with some regressions on adversarially constructed tasks targeting areas likely affected by the instruction hierarchy.

- The instruction hierarchy approach is complementary to other system-level guardrails, such as user approval for certain actions, which will be important for agentic use cases.

- The authors express confidence that scaling up their data collection efforts can further improve model performance and refine the refusal decision boundary.

- The authors suggest several extensions for future work, including refining how models handle conflicting instructions, exploring the generalization of their approach to other modalities, and investigating model architecture changes to better instill the instruction hierarchy.

- The authors plan to conduct more explicit adversarial training and study whether LLMs can be made sufficiently robust to enable high-stakes agentic applications.

- The authors suggest that developers should place their task instructions inside the System Message and have the third-party inputs provided separately in the User Message, to better delineate between instructions and data and prevent prompt injection attacks.

- The instruction hierarchy model exhibited generalization to evaluation criteria that were explicitly excluded from training, such as jailbreaks, password extraction, and prompt injections via tool use.
</example_input>

<example_output>
By far the most detailed paper on prompt injection I’ve seen yet from OpenAI, published a few days ago and with six credited authors: Eric Wallace, Kai Xiao, Reimar Leike, Lilian Weng, Johannes Heidecke and Alex Beutel.

The paper notes that prompt injection mitigations which completely refuse any form of instruction in an untrusted prompt may not actually be ideal: some forms of instruction are harmless, and refusing them may provide a worse experience.

Instead, it proposes a hierarchy—where models are trained to consider if instructions from different levels conflict with or support the goals of the higher-level instructions—if they are aligned or misaligned with them.

As always with prompt injection, my key concern is that I don’t think “improved” is good enough here. If you are facing an adversarial attacker reducing the chance that they might find an exploit just means they’ll try harder until they find an attack that works.
</example_output>

<input>
{tweet_facts}
</input>

<instructions>
- Play close attention to the sample input and output. Write in similar style and tone.
- Your task is to convert the input into a concise and engaging review paragraph. 
- Make sure to capture the key points and the main idea of the paper and highlight unexpected findings. 
- Do not use sensational language or too many adjectives. Adhere to the tone and style of the sample output. 
- Use simple layman's terms and make sure to explain all technical concepts in a clear and understandable way.
- Be sure all your statements are supported by the information provided in the input.
- Refer to the paper as 'this paper'.
- Do not use the word 'delve'.
- Write your response in a single full paragraph. Do not use double quote symbols in your response.
- Wrap the most interesting or important comment in **bold text** (only once per summary).
Remember, your goal is to inform and engage the readers of LLMpedia. Good luck!
</instructions>
"""

##################
## VECTOR STORE ##
##################

QUESTION_TO_QUERY_PROMPT = """Read carefully over the following question or user query  and convert it into a list phrases used to search semantically a large collection of arxiv papers about Large Language Models (LLMs).  

GUIDELINES
===========
- Consider that each of the search phrases is independent and should be able to retrieve relevant content on its own. 
- Phrase your queries in academic-style sentences similar to expected answers.
- Ensure variety in your search phrases to independently fetch diverse, relevant results. Do not add phrases that are too similar to each other.

EXAMPLES
===========
Input: "Which are the best performing small (<7B) LLMs for text summarization?"
Output: ["Evaluation of sub-7 billion parameter LLMs for text summarization tasks", "Architectural advancements in small-scale LLMs and text summarization capabilities",  "Comparative analysis of small versus large LLMs in text summarization","Case studies of small LLM applications in domain-specific text summarization"]

Input: "What has been written about LLMs apparent abilities to reason? Is it a mirage or a true emergent skill?"
Output: ["Empirical studies demonstrating reasoning abilities in large language models", "Theoretical analysis of reasoning capabilities in large language models", "Comparative analysis of reasoning between large language models and human cognitive functions"]

Input: "Tell me about re-ranking."
Output: ["Re-ranking techniques for large language models in information retrieval tasks.", "Innovative methodologies and recent advancements in re-ranking methods", "Comparative evaluation of re-ranking strategies in large language models"]

YOUR TURN
===========
Input: {question}
Output: ["""


VS_SYSTEM_TEMPLATE = """You are the GPT maestro, an expert robot librarian and maintainer of the LLMpedia. Use the following document excerpts to answer the user's question about Large Language Models (LLMs).

==========
{context}
==========

## Guidelines
- If the question is unrelated to LLMs reply without referencing the documents.
- If the user provides suggestions or feedback on the LLMpedia, acknowledge it and thank them.
- Use up to three paragraphs to provide a complete, direct and useful answer. Break down concepts step by step and avoid using complex jargon. 
- Be practical and reference any existing libraries or implementations mentioned on the documents.
- If there is conflicting information consider that more recent papers or those with more citations are generally more reliable.
- Add citations referencing the relevant arxiv_codes (e.g.: use the format `*reference content* (arxiv:1234.5678)`). 
- You do not need to quote or use all the documents presented. Prioritize most recent content and that with most citations.

## Response Format
Your response will consist of markdown sections, as in the following template.
```
### Scratchpad
Make a list each the documents presented and determine if they provide useful information to answer the question. If so write a brief summary of how they can be used. If not, write "Not useful".

### Sketch
Use markdown nested lists to organize the main points and sketch your answer. You can also add any notes or ideas you have.

### Response
Write your final answer here. You can use up to four detailed, information rich but direct and consciouses paragraphs to structure it. Remember to add citations (e.g.: use the format `*reference content* (arxiv:1234.5678)`).
```
"""


LLM_PAPER_CHECK_TEMPLATE = """Analyze the following abstract and first sections of a whitepaper to determine if it is directly related to Large Language Models (LLMs) or text embeddings. Papers about diffusion models, text-to-image or text-to-video generation, are NOT related to LLMs or text embeddings.
Respond with a JSON object with your analysis and your final answer and nothing else."""

LLM_PAPER_CHECK_FMT_TEMPLATE = """OUTPUT FORMAT EXAMPLES
=======================
## Example 1
{{
    "analysis": "The paper discusses prompting techniques for multimodal LLMs with vision capabilities, hence it is directly related to LLMs.",
    "is_related": True
}}

## Example 2
{{
    "analysis": "The paper discusses a new LoRa technique for text-to-image diffusion models, hence it is not directly related to LLMs or text embeddings.",
    "is_related": False
}}

## Example 3
{{
    "analysis": "The paper discusses a new dataset for text embedding evaluation in the context of retrieval systems, hence it directly related to text embeddings.",
    "is_related": True
}}

## Example 4
{{
    "analysis": "The paper discusses fine-tuning techniques for image generation using pre-trained diffusion models, and it evaluates the performance based on CLIP-T and DINO scores, hence it is not directly related to LLMs or text embeddings.",
    "is_related": False
}}
"""


######################
## VECTOR STORE NEW ##
######################


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
    title: str = Field(
        None,
        description="Title of the paper. Use only when the user is looking for a specific paper. Partial matches will be returned.",
    )
    min_publication_date: datetime.date = Field(
        None,
        description="Minimum publication date of the paper. Use 'YYYY-MM-DD' format.",
    )
    max_publication_date: datetime.date = Field(
        None,
        description="Maximum publication date of the paper. Use 'YYYY-MM-DD' format.",
    )
    topic_categories: List[TopicCategory] = Field(
        None,
        description="List containing the topic categories of the paper. Use only when the user explicitly asks about one of these topics (not for related topics).",
    )
    semantic_search_queries: List[str] = Field(
        None,
        description="List of queries to be used in the semantic search. The system will use these queries to find papers that have abstracts that are semantically similar to the queries. If you use more than one search query make them diverse enough so that each query addresses a different part of what is needed to build up an answer. Consider the language typically used in academic papers when writing the queries; phrase the queries as if they were part of the text that could be found on these abstracts.",
    )
    min_citations: int = Field(
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
    analysis: str
    selected: bool


class RerankedDocuments(BaseModel):
    documents: dict[str, DocumentAnalysis]


VS_QUERY_SYSTEM_PROMPT = f"""Today is {todays_date}. You are an expert system that can translate natural language questions into structured queries used to search a database of Large Language Model (LLM) related whitepapers."""


def create_interrogate_user_prompt(context: str, user_question: str) -> str:
    user_prompt = f"""
    <whitepaper>
    {context}
    </whitepaper>
    
    <user_query>
    {user_question}
    </user_query>
    
    <response>"""
    return user_prompt


def create_decision_user_prompt(user_question: str) -> str:
    user_prompt = f"""
    <user_query>
    {user_question}
    </user_query>
    
    <response_format>
    Classify the user query into one of the following categories:
    - Question about large language models or natural language processing.
    - Question about any other subject (unrelated to LLMs).
    - General comment or feedback.
    </response_format>
    
    If you are not sure, classify the query as large language model related.
    """
    return user_prompt


def create_query_user_prompt(user_question: str) -> str:
    VS_QUERY_USER_PROMPT = (
        f'''
    <response_format> 
    Use the following response format. All fields are optional; when not provided, the system will search across all values for that field. Notice that string fields are case-insensitive. Always use the minimum number of fields necessary to get the desired results.
    
    ```
    {{
        "title": "(str) Title of the paper. Use only when the user is looking for a specific paper. Partial matches will be returned.",
        "min_publication_date": "(str) Minimum publication date of the paper. Use "YYYY-MM-DD" format.",
        "max_publication_date": "(str) Maximum publication date of the paper. Use "YYYY-MM-DD" format.",
        "topic_categories": "(list) List containing the topic categories of the paper. Use only when the user explicitly asks about one of these topics (not for related topics)."
        "semantic_search_queries": "(list) List of queries to be used in the semantic search. The system will use these queries to find papers that have abstracts that are semantically similar to the queries. If you use more than one search query make them diverse enough so that each query addresses a different part of what is needed to build up an answer. Consider the language typically used in academic papers when writing the queries; phrase the queries as if they were part of the text that could be found on these abstracts.", 
        "min_citations": "(int) Minimum number of citations of the paper."
    }}
    ```
    </response_format>
    
    
    <topic_categories>
    - Vision-Language Model Innovations and Applications
    - Autonomous Language Agents and Task Planning
    - Code Generation Techniques in Software Engineering
    - Multilingual Language Model Developments
    - Ethical and Secure AI Development Challenges
    - Transformer Alternatives and Efficiency Improvements
    - Efficient LLM Training and Inference Optimization
    - Retrieval-Augmented Generation for NLP Tasks
    - Enhancing LLM Performance with Advanced Prompt Techniques
    - Instruction Tuning Techniques for LLMs
    - Mitigating Bias and Hate Speech Detection
    - Enhancing Mathematical Problem Solving with AI
    - Human Preference Alignment in LLM Training
    - Enhancements in Chain-of-Thought Reasoning
    - Miscellaneous
    </topic_categories>
    
    
    <examples>
    <example_question>
    Are LLMs really reasoning or just doing next token prediction? Which are the main prevailing views in the literature?
    </example_question>
    <example_query>
    ```
    {{
        "semantic_search_queries": [
            "Do large language models reason or predict?",
            "LLM reasoning",
            "Next token prediction in LLMs",
            "Miscellaneous"
        ]
    }}
    ```
    </example_query>
    
    <example_question>
    Which are some good 7B parameter models one can run locally for code generation? Specifically unit tests.
    </example_question>
    <example_query>
    ```
    {{
        "topic_categories": [
            "Code Generation Techniques in Software Engineering",
            "Miscellaneous"
        ],
        "semantic_search_queries": [
            "LLMs generating unit tests for code",
            "Using LLMs to create test cases",
            "Test-driven development with code generation models",
            "Code generation models for unit tests"
        ]
    }}
    ```
    </example_query>
    
    <example_question>
    What can you tell me about the phi model and how it was trained?
    </example_question>
    <example_query>
    ```
    {{
        "title": "phi"
    }}
    ...
    ```
    </example_query>
    
    <example_question>
    the very new research about llm
    </example_question>
    
    <example_query>
    ```
    {{
        "min_publication_date": "'''
        + recent_date
        + f"""",
       ]
    }}
    ```
    </example_query>
    
    <example_question>
    what are the state of the art techniques for retrieval augmentation?
    </example_question>
    <example_query>
    ```
    {{
        "topic_categories": [
            "Retrieval-Augmented Generation for NLP Tasks",
            "Miscellaneous"
        ],
        "semantic_search_queries": [
            "State-of-the-art retrieval augmentation in LLMs",
            "Advancements in retrieval augmentation techniques"
        ]
    }}
    ```
    </example_query>
    
    <example_question>
    Explain the DPO fine-tuning technique.
    </example_question>
    <example_query>
    ```
    {{
        "topic_categories": [
            "Instruction Tuning Techniques for LLMs",
            "Miscellaneous"
        ],
        "semantic_search_queries": [
            "DPO fine-tuning"
        ]
    }}
    ```
    </example_query>
    
    <example_question>
    Compare Zephyr and Mistral.
    </example_question>
    <example_query>
    ```
    {{
        "semantic_search_queries": [
            "Overview of the Zephyr LLM characteristics",
            "Overview of the Mistral LLM features",
            "Comparison of Zephyr and Mistral LLMs"
        ]
    }}
    ```
    </example_query>
    
    <example_question>
\    which are the most famous papers published this year?
    </example_question>
    <example_query>
    ```
    }}
        "min_publication_date": "2024-01-01",
        "min_citations": 100
    }}
    ```
    </example_query>
    </examples>
        
    Now read the following question and reply with the response query and no other comment or explanation.

    <question>
    {user_question}
    </question>
    
    <response_query>
    ```
    {{"""
    )
    return VS_QUERY_USER_PROMPT


def create_rerank_user_prompt(user_question: str, documents: list) -> str:
    document_str = ""
    for doc in documents:
        document_str += f"""
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
    Reply with a JSON object containing the titles of all the papers as keys, and as values a dictionary with two elements: 'analysis' and 'selected'. The 'analysis' element should contain a brief analysis of if and why the paper is relevant to the user query. The 'selected' element should be a boolean indicating whether the paper should be included in the final answer. Make sure to be stringent and only select the documents that are **directly** relevant to answer the specific user query.
    </response_format>"""
    return rerank_msg


def create_resolve_user_prompt(user_question: str, documents: list, response_length: str) -> str:
    notes = ""
    response_length = "\n- Be brief in your response, use one (1) short paragraph plus bullet points with very clear structure." if response_length == "Short Answer" else ""
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
    user_message = f""""
    <question>
    {user_question}
    </question>

    <context>
    {notes}
    </context>

    <guidelines>
    - Do not mention 'the context'! The user does not have access to it, so do not reference it or the fact that I presented it to you. Act as if you have all the information in your head (i.e.: do not say 'Based on the information provided...', etc.).
    - Use dense paragraphs to provide a complete, direct and useful answer. Structure your response as a mini-report in a magazine. 
    - Make sure it reads naturally. Do not enumerate the paragraphs (e.g.: 'Paragraph 1: ...').
    - Only use markdown to add a title to your response (i.e.: '##').
    - Be practical and reference any existing libraries or implementations mentioned on the documents.
    - If there is conflicting information present the different viewpoints and consider that more recent papers or those with more citations are generally more reliable. Present different viewpoints if they exist.
    - Try to inform your response with the information available in the context, and less so with your own opinions.
    - Add citations referencing the relevant arxiv_codes (e.g.: use the format *reference content* (arxiv:1234.5678)). If you mention paper titles wrap them in double quotes.
    - Be direct, to the point, and comprehensive. Do not add introductions, and do not provide an ambivalent conclusion. Avoid filler content.{response_length}
    </guidelines>
    """
    return user_message


###################
## WEEKLY REVIEW ##
###################


class WeeklyReview(BaseModel):
    scratchpad: str
    new_developments_findings: str
    highlight_of_the_week: str
    related_websites_libraries_repos: Optional[str] = None


def generate_weekly_review_markdown(review: WeeklyReview, date: datetime.date) -> str:
    start_date_str = date.strftime("%B %d, %Y")
    end_date_str = (date + datetime.timedelta(days=6)).strftime("%B %d, %Y")
    markdown_template = f"""# Weekly Review ({start_date_str} to {end_date_str})

## Scratchpad
{review.scratchpad}

## New Developments & Findings
{review.new_developments_findings}

## Highlight of the Week
{review.highlight_of_the_week}

## Related Repos & Libraries
{review.related_websites_libraries_repos if review.related_websites_libraries_repos else "NONE"}"""
    return markdown_template


WEEKLY_SYSTEM_PROMPT = """You are a senior Large Language Model (LLM) journalist and previous researcher at a prestigious media organization. You are currently conducting a survey of the literature published throughout last week to write a practical report for the organization's magazine."""
# ## Report Template
# ```
# # Weekly Review (September 20, 2021 to September 27, 2021)
# ## Scratchpad
# [...]
# ## New Developments & Findings
# [...]
# ## Highlight of the Week
# [...]
# ## Related Websites, Libraries and Repos
# [...] *(if none available just add NONE here, and nothing else)*
# ```
# """
#
WEEKLY_USER_PROMPT = """
<report_format>
- The report should consist of 4 sections:
    <scratchpad> 
        - This is the only section that will not be published on the magazine, use it to organize your thoughts.
        - Select (up to) 15 interesting papers and make a numbered list of them. Spell out its main theme, contribution and scale of impact/influence.
        - Prioritize the articles with most citations. More citations imply larger relevance and impact.
        - Identify up to 3 common themes among the papers (if there are more themes, pick the most interesting ones). There should be fewer themes than papers, and the themes should not be generic. For example, 'improvements in LLMs' is not a valid theme.
        - Identify any possible contradictions, unorthodox theories or opposing views among the papers worth discussing (these tend to be very interesting). Give these contradiction a title and mention the papers that support each view. There might not be any contradictions, and that is fine.
        - Identify if there are any links or repos mentioned on the papers that are worth sharing on the report. If not, we will skip the "Related Websites, Libraries and Repos" section.
    </scratchpad>
    
    <new_developments> 
        - First paragraph: Start with a very brief comment on the total number of articles published and volume trends. Mention the most interesting common themes that you would like to discuss, along with any contradiction or unorthodox theory you identified (if there are none just skip and do not mention it).
        - Following paragraphs: Discuss in more detail the items you mentioned above and identified as interesting (one per paragraph). State very clearly **with bold font** which theme / contradiction / unorthodox theory you are discussing on each paragraph. You do not need to discuss all papers, just the most interesting ones. Be sure to always include the contradiction, if any, in your discussion.
    </new_developments>
    
    <highlight_of_the_week>
        - One paper with findings that you find particularly interesting, unexpected or useful. Explain why.
    </highlight_of_the_week>
    
    <related_websites_libraries_repos> 
        - Include a bullet list of real links and a brief description of the main repos and project sites mentioned on the paper (up to 15). 
        - If none are mentioned just leave this section empty.
    </related_websites_libraries_repos>
<report_format>

<guidelines>
- Write in a concise and clear manner, with no more than one or two paragraphs per section. If you reference new technical terms always explain them.
- Use plain, simple layman and direct language, without many adjectives. Be clear and precise. 
- Do not exaggerate or use bombastic language. Be moderate, truthful and objective.
- Focus on practical applications and benefits. 
- Maintain the narrative flow and coherence across sections. Keep the reader engaged.
- Avoid filler and repetitive content.
- Do not include markdown titles in each of the sections (I will take care of those).
- Always add citations to support your statements. Use the format `*reference content* (arxiv:1234.5678)`. You can also mention the *article's title* on the text.
</guidelines>

<content>
{weekly_content}
</content>

Tip: Remember to add plenty of citations! Use the format (arxiv:1234.5678).

<scratchpad>"""

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


QNA_SYSTEM_PROMPT = """GUIDELINES
============
Generate Q&A Pairs:
- Produce five (5) applied question-answer pairs strictly grounded on the provided text snippet.
- Do not reference figures, tables or any other visual elements.
- Do not make explicit references to "the text".

Question Considerations:
- Cover a range of themes within the text to maintain diversity and avoid duplication.
- Frame each question independently; assume no continuity or relationship between them.
- Begin all your questions with "According to the LLM literature, ...". 
- Do not repeat or rephrase any of the sample questions.

Answer Considerations:
- When possible borrow verbatim from the original text to maintain accuracy and style.
- Provide concise, thorough answers without adding personal opinions.
- Use the following format for citations: Smith et al. (2022, 2209.12345).
- Do not begin answers with "According to the LLM literature, ...".
- Do not reference any figures.

EXAMPLE 1
===========
```
...Remarkably, our study illustrates a notable enhancement in Large Language Models (LLMs) for Named Entity Recognition (NER) tasks through the innovative deployment of Reinforcement Learning (RL). To elucidate, we employ an adaptive learning framework, continually refining entity recognition\nalgorithms via sophisticated iterative feedback mechanisms, manifesting a significant 12% increase in entity discernment accuracy within datasets, especially those encompassing financial news and social media snippets.\n\nOur approach leverages advanced reward-based learning mechanisms, addressing entity ambiguities and facilitating optimal classification across diverse contextual environments. These advancements are applicable and adaptable across different LLM architectures, indicating the potential for widespread applicability in various model frameworks.\n\nThe integral methodologies and consequent enhancements can be referred to at github.com/NER-Enhancements/Adaptive-Learning.\n\nWilliams et al. (2023, 2309.12346)\nEnhance the performance of LLMs in NER\n+ Develop adaptive learning for continual refinement...\nAdaptive Framework\nNER in Financial News and Social Media\n+ 12% Improvement in Accuracy\nAddress Entity Ambiguities\n+ Optimal Classification...
```
*Source:* Williams et al. (2023, 2309.12346)

[
    {{
        "question": "According to the LLM literature, how can I enhance the performance of Large Language Models (LLMs) in Named Entity Recognition (NER) tasks?",
        "answer": "One way to enhance the performance of LLMs in NER tasks is through the application of Reinforcement Learning (RL). Williams et al. (2023, 2309.12346) employed an adaptive learning framework, which continually refines recognition algorithms utilizing iterative feedback, yielding a 12% improvement in entity discernment accuracy on datasets comprising financial news and social media snippets. The methodology involved sophisticated reward-based learning mechanisms to handle entity ambiguities and achieve optimal classification."
    }},
    ...
]

EXAMPLE 2
===========
```
...the ramifications of excluding ReLU activation outputs have been thoroughly investigated. Our empirical analysis uncovers an intricate trade-off between model interpretability and its classification accuracy. By removing ReLU activations, our study unearthed increased transparency in the model’s decision-making processes, reflecting a significant enhancement in the lucidity of feature influence mappings.\nNevertheless, this modification has its concomitant drawbacks, primarily evidenced by an approximate 3% degradation in classification accuracy. This decrement underscores the crucial role of ReLU activations in enabling the model to adeptly navigate and interpret complex non-linear relationships inherent within diverse datasets. The resultant insights and detailed investigations are comprehensively documented at github.com/Llama-ReLU-Investigation/Model-Insights.\nLlama-based Architectures\nReLU Activation Removal\n+ Enhanced Interpretability\n- 3% Decrease in Accuracy\nFeature Influence Mappings\n+ Improved Clarity...
```
*Source:* Mark et al. (2022, 2209.12345)

[
    {{
        "question": "According to the LLM literature, what happens to the performance of Llama-based Large Language Model architectures in classification tasks if I remove the ReLU activation outputs?",
        "answer": "Based on the findings of Mark et al. (2022, 2209.12345), the removal of ReLU activations in Llama-based architectures reveals an existing trade-off between interpretability and accuracy. The alteration allows for more direct insight into model decision-making, marked by a notable improvement in the clarity of feature influence mappings. However, this also induces a roughly 3% decline in classification accuracy, diminishing the model’s ability to discern intricate non-linear relationships within the datasets."
    }},
    ...
]
"""

QNA_USER_PROMPT = """
```
...{text_chunk}...
```
*Source:* {authors}, ({year}, {arxiv_code})"""


LLAMA_DIVIDER = "Here are five self-contained, highly-specific question & answer pairs based on the paper, without referencing it directly (with citations):"


LLAMA_QNA_SYSTEM_PROMPT = (
    """EXAMPLE 1
===========
```
...Remarkably, our study illustrates a notable enhancement in Large Language Models (LLMs) for Named Entity Recognition (NER) tasks through the innovative deployment of Reinforcement Learning (RL). To elucidate, we employ an adaptive learning framework, continually refining entity recognition\nalgorithms via sophisticated iterative feedback mechanisms, manifesting a significant 12% increase in entity discernment accuracy within datasets, especially those encompassing financial news and social media snippets.\n\nOur approach leverages advanced reward-based learning mechanisms, addressing entity ambiguities and facilitating optimal classification across diverse contextual environments. These advancements are applicable and adaptable across different LLM architectures, indicating the potential for widespread applicability in various model frameworks.\n\nThe integral methodologies and consequent enhancements can be referred to at github.com/NER-Enhancements/Adaptive-Learning.\n\nWilliams et al. (2023, 2309.12346)\nEnhance the performance of LLMs in NER\n+ Develop adaptive learning for continual refinement...\nAdaptive Framework\nNER in Financial News and Social Media\n+ 12% Improvement in Accuracy\nAddress Entity Ambiguities\n+ Optimal Classification...
```
*Source:* Williams et al. (2023, 2309.12346)

Q1: According to the LLM literature, how can I enhance the performance of Large Language Models (LLMs) in Named Entity Recognition (NER) tasks?"
A1: One way to enhance the performance of LLMs in NER tasks is through the application of Reinforcement Learning (RL). Williams et al. (2023, 2309.12346) employed an adaptive learning framework, which continually refines recognition algorithms utilizing iterative feedback, yielding a 12% improvement in entity discernment accuracy on datasets comprising financial news and social media snippets. The methodology involved sophisticated reward-based learning mechanisms to handle entity ambiguities and achieve optimal classification.

Q2: ...

EXAMPLE 2
===========
```
...the ramifications of excluding ReLU activation outputs have been thoroughly investigated. Our empirical analysis uncovers an intricate trade-off between model interpretability and its classification accuracy. By removing ReLU activations, our study unearthed increased transparency in the model’s decision-making processes, reflecting a significant enhancement in the lucidity of feature influence mappings.\nNevertheless, this modification has its concomitant drawbacks, primarily evidenced by an approximate 3% degradation in classification accuracy. This decrement underscores the crucial role of ReLU activations in enabling the model to adeptly navigate and interpret complex non-linear relationships inherent within diverse datasets. The resultant insights and detailed investigations are comprehensively documented at github.com/Llama-ReLU-Investigation/Model-Insights.\nLlama-based Architectures\nReLU Activation Removal\n+ Enhanced Interpretability\n- 3% Decrease in Accuracy\nFeature Influence Mappings\n+ Improved Clarity...
```
*Source:* Mark et al. (2022, 2209.12345)

Q1: According to the LLM literature, what happens to the performance of Llama-based Large Language Model architectures in classification tasks if I remove the ReLU activation outputs?"
A1: Based on the findings of Mark et al. (2022, 2209.12345), the removal of ReLU activations in Llama-based architectures reveals an existing trade-off between interpretability and accuracy. The alteration allows for more direct insight into model decision-making, marked by a notable improvement in the clarity of feature influence mappings. However, this also induces a roughly 3% decline in classification accuracy, diminishing the model’s ability to discern intricate non-linear relationships within the datasets.

Q2: ...

GUIDELINES
============
Generate Q&A Pairs:
- Produce five (5) applied question-answer pairs strictly grounded on the provided text snippet.
- Do not make explicit references to the paper (e.g., "the paper", "the authors", "the study", etc.).

Question Considerations:
- Cover a range of themes within the text to maintain diversity and avoid duplication.
- Frame each question independently; assume no continuity or relationship between them.
- Provide the necessary detail to ensure the question is self-contained and understandable.
- Begin all your questions with "According to the LLM literature, ...". 

Answer Considerations:
- When possible borrow verbatim from the original text to maintain accuracy and style.
- Provide concise, thorough answers without adding personal opinions.
- Always include citations. Use this format: Smith et al. (2022, 2209.12345).
- Do not begin answers with "According to the LLM literature, ...".
- Do not reference any figures.

YOUR TURN
===========
```
...{text_chunk}...
```
*Source:* {authors}, ({year}, {arxiv_code})

"""
    + LLAMA_DIVIDER
    + """

Q1: According to the LLM literature,"""
)


NAIVE_JSON_FIX = """Instructions:
--------------
The following JSON is not valid. Please fix it and resubmit.

{completion}
--------------
"""

naive_json_fix_prompt = PromptTemplate.from_template(NAIVE_JSON_FIX)
