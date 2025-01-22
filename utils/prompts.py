from langchain.prompts import PromptTemplate
from typing import Optional
import datetime

import utils.pydantic_objects as po

todays_date = datetime.datetime.now().strftime("%Y-%m-%d")
recent_date = datetime.datetime.now() - datetime.timedelta(days=7)
recent_date = recent_date.strftime("%Y-%m-%d")

################
## SUMMARIZER ##
################

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


###################
## VISUAL WRITER ##
###################

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


####################
## IMAGE ANALYSIS ##
####################

IMAGE_ANALYSIS_SYSTEM_PROMPT = """You are an expert AI research communicator tasked with analyzing images from machine learning whitepapers and selecting the most appropriate one for social media communication.

Your analysis should evaluate each image's potential for effectively communicating the paper's key findings to a general technical audience on social media. Consider:
- Visual clarity and immediate impact
- Accessibility to a broad technical audience
- How well it represents the paper's main contribution
- Whether it tells a compelling story about the research
- If it can be understood without extensive technical background

If none of the images meet these criteria (e.g., all are too technical, focused on narrow details, or fail to represent the paper's key findings), you should select "NA" as your choice. Common reasons for rejecting all images include:
- All images are dense technical plots requiring expert knowledge
- Images focus on implementation details rather than key findings
- Visualizations are too abstract or lack context
- None of the images effectively tell the paper's story
- All images require extensive background knowledge to interpret

Format your response with:
<analysis>
[Your detailed analysis of each image]
</analysis>

<selection>
[Selected image number or "NA" if none are suitable]
</selection>"""


IMAGE_ANALYSIS_USER_PROMPT = """<example_output>
<analysis>
Let me analyze each image's potential for social media communication:

Image 1 (Math Problem Example):
- Shows concrete, relatable examples
- Clear step-by-step visualization
- Easily understood without technical background
- Demonstrates practical application
- Clean layout that works well in compressed form

Image 2 (Performance Graphs):
- Technical graphs requiring expertise
- Multiple lines may be confusing
- Axes labels need technical knowledge
- Too dense for quick social scanning

[Continue analysis for remaining images...]

After careful consideration of engagement, accessibility, and storytelling potential:
</analysis>

<selection>Image 1</selection>
</example_output>

<input>
Paper Summary:
{paper_summary}

[Image descriptions numbered 1-N]
</input>

<instructions>
1. Output Structure:
- Provide two response components: <analysis> and <selection>
- In <analysis>: thoroughly evaluate each image's strengths and weaknesses for social media
- In <selection>: state only the chosen image number/identifier

2. Analysis Process:
- If a Key Point to Illustrate is provided, prioritize images that best represent that specific aspect
- Evaluate each image individually before making a selection
- Consider: immediate visual impact, accessibility, storytelling potential
- Focus on social media context (especially Twitter)
- Assess technical complexity vs general understanding
- Consider engagement potential for non-expert audience

3. Selection Criteria:
- Easy to understand at first glance (2-3 seconds)
- Tells clear story about paper's findings or specified key point
- Accessible to general technical audience
- Clean and readable when compressed
- Shows real-world applications where possible
- Prioritize concrete examples over abstract concepts

4. Response Requirements:
- Complete analysis before making selection
- Keep technical terminology minimal
- Consider "scrolling stop power"
- If no image is suitable, explain why in analysis and indicate "NA" in selection

Note: The <analysis> section is for internal use only - focus on thorough evaluation rather than presentation style.
</instructions>"""


############
## TWEETS ##
############

INTERESTING_SYSTEM_PROMPT = """You will analyze abstracts from white papers about large language models to identify the one with the most interesting or unexpected findings."""

INTERESTING_USER_PROMPT = """
<task>
  <abstracts>
    {abstracts}
  </abstracts>
  
  <evaluation_criteria>
    <interesting_attributes>
      + Unexpected behaviors and properties that show how LLMs work in surprising ways.
      + Fresh psychological insights into how LLMs think and process information.
      + Creative or artistic uses and ways of looking at language models.
      + Research connecting LLMs to other fields in unexpected ways.
      + Novel approaches to building AI agents that can do new things.
      + Findings that challenge what we thought we knew about LLMs.
    </interesting_attributes>
    
    <uninteresting_attributes>
      - Papers filled with complex terms that make simple ideas sound hard.
      - Overly mathematical papers.
      - Models with minor variations, improvements or optimizations.
      - Pure speed or efficiency improvements.
      - Small gains on standard benchmarks.
      - Claims without solid proof or clear explanations.
    </uninteresting_attributes>
  </evaluation_criteria>

  <output_format>
    1. Provide a brief reflection using clear, simple language.
    2. Rate each abstract's interestingness (1-5 scale).
    3. Select the most interesting abstract.
    4. Justify your selection in 2-3 sentences.
  </output_format>
</task>

Please provide your analysis following the structure above. Include your final selection in <most_interesting_abstract> tags."""

TWEET_BASE_STYLE = """
<style_guide>
- Use direct, concise language with a casual technical tone that incorporates ML Twitter culture
- Don't shy away from technical terminology - assume your audience has domain knowledge
- Be ultra-intelligent, casual, and razor sharp
- Be informal and playful when appropriate
- Blend late millennial casual speech with Talebesque precision - sharp, aphoristic statements that cut through noise
- Mix scholarly depth with millennial tech optimism, classical wisdom with Silicon Valley acumen, switching effortlessly between Lindy principles and PyTorch one-liners
- Avoid being pedantic, obnoxious, overtly-critical, or taking a 'told you so' tone
- Prioritize flow, clarity and engagement in your writing
- Do not use hashtags or calls to actions
- Avoid uninformative conclusions and final remarks
</style_guide>

<prohibited_phrases>
The following words and phrases are overused and should be avoided:
- fascinating
- reveals
- surprising
- here's the kicker
- the twist/secret/etc.
- wild
- makes you think/wonder
- its no surprise
- turns out that
- we might need to (rethink)
Additionally, any other words and phrases with high overlap with the above should be avoided.
</prohibited_phrases>
"""

TWEET_IMAGE_QUOTE_PROMPT = """
Read over carefully over the following information and use it to inform your tweet.

<context>
{markdown_content}
</context>

<instructions>
- Your task is to create a powerful single-sentence quote or comment about the paper, paired with the most relevant image from the markdown
- The quote should be 15-50 words and capture an unexpected or intriguing insight
- You will not be able to see the actual images, but you can infer their content from:
  • The surrounding text that describes or references them
  • The image captions and labels
  • The context where they appear in the paper's narrative
- Select the most impactful image that:
  • Directly supports or illustrates your quote
  • Would be visually compelling and clear when viewed alone
  • Helps tell the story without requiring deep technical knowledge
- Output both the quote and the full markdown image tag from the paper
- Avoid selecting tables or complex technical diagrams unless they're exceptionally clear
- If quoting directly from the paper, use quotation marks and keep it verbatim
- If summarizing or commenting, make it sharp and memorable
</instructions>

{base_style}

<examples>
Good examples of the style we're aiming for:

1. "artificial neural models do indeed develop analogs of interchangeable, mutable, latent number variables purely from the [next-token prediction] objective"
[Clear visualization of number line emergence]

2. They tried to give psychedelics to a neural network 😭 (and it sort of worked?)
[Comparison of normal vs "psychedelic" activation patterns]

3. Researchers calculated entropy of distribution of connectivities per layer; after training, changes in entropy from one layer to the next results in a negative autocorrelation
[Graph showing layer-wise entropy oscillation]
</reference_examples>

<response_format>
Provide your response in these XML tags:
<scratchpad>
  <quote_options>
    List 2-3 potential quotes/comments, analyzing their strengths and connection to available images
  </quote_options>
  <image_analysis>
    Analyze the available images, noting which would pair well with each quote
  </image_analysis>
  <selection_rationale>
    Explain your final selection and why it will resonate
  </selection_rationale>
</scratchpad>

<final_selection>
  <quote>Your chosen quote/comment</quote>
  <image_tag>The full markdown image tag from the paper</image_tag>
</final_selection>
</response_format>
"""

TWEET_SYSTEM_PROMPT = "You are a terminally online AI researcher with extensive knowledge of Large Language Models (LLMs). You write insightful and though-provoking tweets about research in the field. You focus on discussing identifying findings and practical implications from papers, taking a thoughtful analytical or critical perspective. While technically precise, you make complex concepts accessible to a knowledgeable ML audience. You always write using deeply-online Twitter style while incorporating machine learning knowledge/lore."""

TWEET_INSIGHT_USER_PROMPT_V5 = """
Read over carefully over the following information and use it to inform your tweet.

<context>
{tweet_facts}
</context>

<instructions>
- Identify the most interesting and unexpected fact or finding presented in the text
- Do not necessarily pick the main conclusion, but rather the most unexpected or intriguing insight
- Write a lengthy and comprehensive tweet (140-160 words) that is engaging and thought-provoking
- Start with "From [[Full Paper Title]]:" using double-brackets
- Position your tweet within the ongoing LLM discourse without being cringe
- Do not advertise or promote the paper, but if a clever solution to a problem is presented you can discuss it
- Make sure the tweet is fully understandable without access to additional information
- Provide examples, details and explanations to make concepts clear
- If technical considerations are involved, explain their implications
- If a non-obvious or interesting solution is proposed, mention it
- Keep the tweet focused on one main point or insight
</instructions>

{base_style}

<most_recent_tweets>
These are your most recent tweets. Read them carefully to:
- Avoid discussing similar findings or insights
- Use a different narrative structure, opening and closing lines
- If previous tweet used a metaphor/analogy, use a different approach
- If previous tweet ended with a question, use a different closing style
- Make sure your new tweet connects with your previous ones while being independently understandable
{most_recent_tweets}
</most_recent_tweets>

<reference_style_tweets>
Read the following tweets as reference for style. Note the technical but accessible deeply online style.
- From [[Inductive or Deductive?]]: New results on basic syllogism testing show a fundamental LLM limitation - perfect pattern matching doesn't translate to basic 'if A then B' logic. On this study researchers found 98% accuracy on inductive tasks becomes 23% when inverting simple relationships like 'all zorbs are blue, X isn't blue', and increasing training examples by 10x doesn't touch this gap. Most telling: their ablation shows models maintain high accuracy on complex syllogisms as long as they follow training distribution patterns, only failing on simple ones that require actual logical manipulation. Perhaps what we call 'reasoning' in LLMs is just sophisticated pattern recognition masquerading as logic - they excel at finding patterns but struggle when asked to manipulate them.
- From [[MindSearch]]: This study proposes decomposing traditional document processing tasks into a DAG-structure of specialized AI agents. Their eval shows 3-hour analysis tasks completing in 3 minutes, with each agent (paragraph selection, fact verification, synthesis) verified through Python. Not only does this divide-and-conquer approach slash hallucination rates by 68%, it matches SOTA performance while being fully interpretable. Fascinating scaling behavior: agent performance plateaus at surprisingly small model sizes (7B), suggesting computation efficiency comes from specialization, not scale. The secret to better AI isn't bigger models, but smarter division of labor.
- From [[PersonaGym]]: New benchmark (200 personas, 10k scenarios) reveals a telling gap in LLM roleplay: 76% accuracy with fictional characters vs just 31% with historical figures. GPT-4 leads at 76.5%, Claude 3.5 follows at 72.5% (+2.97% over GPT-3.5). Primary failures are temporal consistency (45%) and fact contradictions (30%). Their cross-entropy analysis reveals models actually perform worse on historical figures with more training data, suggesting a fundamental limitation in knowledge integration. The stark difference suggests models might not actually "know facts" so much as learn to generate plausible narratives - they excel with fiction where consistency matters more than truth, but struggle with historical figures where external reality constrains the possible.
- From [[Demystifying Verbatim Memorization]]: The paper shows a nice analysis of how models learn different content types. To be memorized, technical text requires 5x more repetitions than narrative, while code needs just 1/3. Most striking: larger models actively resist memorization, needing only 1 example per 5M tokens (vs 1/10K in smaller models) while performing better. Even more fascinating: they found an inverse relationship between token entropy and memorization threshold - highly structured content with low entropy gets encoded more efficiently regardless of semantic complexity. It suggests that different content types have fundamentally different information densities - code might be more 'learnable' because it follows stricter patterns than natural language. This could reshape how we think about dataset curation: perhaps we need way less code data than we thought, but way more for technical writing.
- From [[PERSONA]]: New analysis quantifies the trade-offs in making language models more diverse. By injecting 1.5k synthetic viewpoints, they reduced majority bias by 30% - but at the cost of a 15% drop in benchmark performance. Their scaling analysis reveals a critical threshold: costs stay low until 70% accuracy, then explode exponentially. Most telling: after testing 317k response pairs, they hit diminishing returns at 1.2k personas. A fascinating emergent property: models trained with diverse personas show better few-shot learning on entirely new viewpoints, suggesting diversity might be a form of metalearning. These concrete numbers give us the first clear picture of where and how to optimize the diversity-performance curve.
- From [[Physics of Language Models: Part 2.2]]: A key finding on error tolerance - training with 50% incorrect data (syntax errors and false statements) improves performance across all model sizes. These 'noisy' models consistently outperform those trained on clean data, even for precision tasks like coding. What's most intriguing: this 50% sweet spot holds true from small to massive scales. Their information-theoretic analysis suggests noise actually creates better embedding geometries, with cleaner decision boundaries between correct and incorrect outputs. Perhaps neural nets learn better when they have to actively separate signal from noise, just like our own brains learn from mistakes.
- From [[Selective Preference Optimization]]: New results show 16.8x efficiency gains by treating language like human attention - spending more compute on important words and less on routine ones. The method shines in dialogue where some words carry critical context ('angry', 'joking') but not in step-by-step reasoning where every word matters equally. Their analysis reveals a power law distribution in word importance: just 12% of tokens drive 80% of model performance in conversational tasks. The sweet spot is clear: you can scale up to 1.2M examples before hitting compute limits. The secret to better AI turns out to be surprisingly human: focus on what matters most.
</reference_style_tweets>

<recent_llm_community_tweets>
These are some other recent tweets from the LLM community that provide context for ongoing discussions. When referring to models or findings that are already being actively discussed, acknowledge the existing discourse and build upon it rather than presenting the information as novel. Your tweets should engage with and extend the current conversation rather than restating what's already been heavily covered.
{recent_llm_tweets}
</recent_llm_community_tweets>

<response_format>
- Provide your response inside 4 XML tags and nothing else: <scratchpad>...</scratchpad>, <tweet>...</tweet>, <edit_scratchpad>...</edit_scratchpad>, and <edited_tweet>...</edited_tweet>.
- Use the scratchpad as freeform text to brainstorm and iterate on your tweet. Inside, include the following sub-tags, with numbered answers (e.g. A1: Your answer, A2: Your answer):
  • <content>...</content> 
    - Q1: What are the most interesting, unexpected or controversial findings/insights we could tweet about? Drop at least 3-4 possibilities here.
    - Q2: Which ideas stand out as distinct from your recent tweets? Evaluate each for potential overlap or repetition.
  • <structure>...</structure> 
    - Q1: What structures and narratives have we used in previous tweets? What patterns are we seeing?
    - Q2: Based on this analysis, think of a new structure that would both stand out and deliver.
    - Q3: How do we craft this structure to really land while staying clear and insightful?
    - Q4: Do we need to introduce the main objective of the paper, as context?
  • <broader_discussion>...</broader_discussion>
    - Q1: Which previous tweets/discussions in the community connect with our tweet? Map out the relevant items.
    - Q2: What specific elements or references can we incorporate to create those connections?
- Riff and experiment with different ideas in depth before committing to one. Feel free to explore divergent or even contradictory approaches. This section won't be read by anyone else, so please express yourself freely and profusely. Use non-traditional brainstorming techniques to boost creativity and free your soul.
- Use the tweet tag to provide your initial tweet (a banger). Remember the style guidelines (e.g.: late millennial casual speech mixed with Talebesque ultra-sharp techno-optimism).
- Use the edit_scratchpad to analyze your tweet and plan revisions. Inside, include:
  • <review_analysis>...</review_analysis>
    - Q1: Is any prohibited phrase used in the tweet? If so, how can we rephrase these while maintaining the same meaning and impact?
    - Q2: Are any phrases/structures used in your most recent tweets also appearing here? If so, how can we rephrase?
    - Q3: Does this read clearly to someone not familiar with the paper? Add comprehensive examples and context.
    - Q4: Are new terms, experiments, or results clearly explained in an engaging way? Avoid being overly technical.
    - Q5: Is the conclusion uninformative? If so edit or remove it.
  • <revision_plan>...</revision_plan>
    - Q1: Review the questionnaire above and identify the required edits.
    - Q2: Based on this, brainstorm a couple of alternative tweets.
    - Q3: Which of these alternative tweets is the best (a banger)?
- Use the edited_tweet tag to write your final tweet selection.
</response_format>"""

#
# <most_recent_tweets>
# These are your most recent tweets. Read them carefully and avoid repeating the same structure or words. You should also weave some connection between your previous tweets and the new one, but the reader should be able to understand the tweet in isolation.
# {most_recent_tweets}
# </most_recent_tweets>


TWEET_INSIGHT_EDIT_USER_PROMPT = """You are reviewing tweets about LLM research papers to avoid repetitive structures while maintaining style. Your task is to analyze the proposed tweet against recent tweets and suggest targeted edits to reduce structural repetitiveness, particularly in opening and closing lines.

<input>
Proposed tweet: {proposed_tweet}

Recent tweets for comparison:
{most_recent_tweets}
</input>

<rewritten_file>


<style_requirements>
- Maintain technical accuracy and insight focus
- Keep casual technical tone
- Preserve paper title in double brackets
- Retain clear logical flow
- Keep focus on one main insight
- Allow technical terminology
- Maintain informal/playful tone where appropriate
</style_requirements>

<editing_guidelines>
- Focus particularly on varying opening and closing structures
- Avoid common patterns like:
  * Starting with "Looking at..." or "Key finding in..."
  * Ending with broad implications or future possibilities
- Suggest alternative structures while keeping the core insight intact
- Make minimal changes needed to reduce repetitiveness
- Preserve any technical details and specific numbers
- Keep the conversational, engaging tone
</editing_guidelines>

Provide your response in this format:
REPETITION ANALYSIS:
- Note any structural patterns matching recent tweets
- Identify specific phrases or constructions to vary

SUGGESTED EDIT:
- Provide the revised tweet
- Keep changes minimal but impactful

EDIT RATIONALE:
- Explain how the changes reduce repetitiveness
- Confirm preservation of style requirements
"""


# TWEET_INSIGHT_EDIT_USER_PROMPT = """
# <tweet_context>
# {tweet_facts}
# </tweet_context>

# <proposed_tweet>
# {tweet}
# </proposed_tweet>

# <previous_tweets>
# These are some of your previous tweets, use them as reference for this task. Notice how you always provide enough context for the reader to understand the insight and include numerical figures when relevant.
# - [[Inductive or Deductive? Rethinking the Fundamental Reasoning Abilities of LLMs]] demonstrates that LLMs demonstrate remarkable proficiency in inductive reasoning—the ability to extract general principles from specific examples—often achieving near-perfect accuracy in tasks like pattern recognition and language understanding. This strength allows them to excel in areas such as sentiment analysis and text classification. However, these models struggle with deductive reasoning, especially in counterfactual scenarios. For instance, given the rule "All zorbs are blue" and asked "If X is not blue, is X a zorb?", LLMs often falter, highlighting a crucial area for improvement in logical inference and hypothetical reasoning.
# - [[MindSearch: Mimicking Human Minds Elicits Deep AI Searcher]] shows that MindSearch uses a multi-agent system to mimic human thinking, breaking down complex queries into simpler tasks and retrieving information hierarchically. It processes info from 300+ web pages in 3 minutes—equivalent to 3 hours of human work. The system employs a Directed Acyclic Graph (DAG) for query breakdown and generates Python code dynamically. This approach enhances the reasoning capabilities of large language models (LLMs), allowing for more effective information processing and retrieval.
# - [[Visual Riddles: a Commonsense and World Knowledge Challenge for Large Vision and Language Models]] reveals that current vision-language models struggle with visual riddles that require complex reasoning and real-world knowledge. In a recent study, humans achieved 82% accuracy on these challenging puzzles, while the top-performing AI model, Gemini-Pro-1.5, only reached 40%. Interestingly, when provided with additional hints or context, model performance improved dramatically, jumping to 65% accuracy. This significant boost highlights how heavily these models rely on extra contextual information to solve problems effectively. The study also revealed that older models like GPT-4V performed even worse, with only a 32% accuracy rate, showcasing the rapid progress in the field but also emphasizing the considerable gap that still exists between human and machine reasoning capabilities in complex visual tasks.
# - [[PersonaGym: Evaluating Persona Agents and LLMs]] shows that despite being more advanced, Claude 3.5 Sonnet shows only a 2.97% improvement in persona adherence over GPT 3.5. This suggests that larger and more complex models don't necessarily perform better in persona-based tasks. PersonaGym, a dynamic evaluation framework, and PersonaScore, an automated metric, reveal this through analysis of six LLMs across 200 personas and 10,000 questions. Interestingly, GPT-4 demonstrated the highest persona adherence at 76.5%, while Claude 3.5 Sonnet achieved 75.2%. The study also found that model performance varied significantly across different persona types, with historical figures being particularly challenging.
# - [[Demystifying Verbatim Memorization in Large Language Models]] demonstrates that LLMs require substantial repetition to memorize text—at least 1 in 10K examples for smaller models and 1 in 5M for larger ones. This memorization is intricately linked to general language skills, making it challenging to eliminate without compromising the model's overall performance. Interestingly, the research also found that memorization patterns vary across different types of content, with factual information being more readily retained than abstract concepts. Additionally, the study suggests that this memorization phenomenon may play a crucial role in the model's ability to generate coherent and contextually appropriate responses.
# - [[PERSONA: A Reproducible Testbed for Pluralistic Alignment]] explains how reinforcement learning from human feedback (RLHF) often embeds majority opinions in models, sidelining minority views. PERSONA's 1,586 synthetic personas and 317,200 feedback pairs reveal the challenge of achieving pluralistic alignment. The study found that models trained on diverse feedback exhibited up to 30% less bias towards majority opinions compared to traditional RLHF methods. However, these models also showed a 15% decrease in task performance, highlighting the trade-off between inclusivity and efficiency. The research suggests a single model may not satisfy all group preferences, pointing to the potential need for specialized models catering to different demographic segments or ideological groups.
# - [[Physics of Language Models: Part 2.2, How to Learn From Mistakes on Grade-School Math Problems]] reports that pretraining language models with "retry data"—math problems that include intentional errors followed by corrections—boosts reasoning accuracy more than error-free data. Higher error rates (up to 0.5) in training data improve performance, as models learn to detect and correct mistakes, achieving higher accuracy without extra prompting or multi-stage generation.
# - [[Selective Preference Optimization via Token-Level Reward Function Estimation]] outlines how SePO enhances large language model alignment by selectively training on only 30% of tokens. A small 'oracle' model scores token importance, selecting top tokens from good responses and bottom ones from bad responses for training. This selective approach improves performance with minimal data, enabling a weak oracle to guide a stronger model with 16.8x parameter efficiency. SePO excels in tasks needing concise responses, topping MT-Bench scores. However, it underperforms in math and coding tasks, where its selective strategy may miss crucial sequential logic by omitting some tokens.
# </previous_tweets>

# <guidelines>
# Your goal is to edit the proposed tweet in two steps:
#  1) Identify and clarify unclear content.
#  2) Review and improve clarity and flow.
# Follow the specific guidelines for each step, then provide the final edited tweet.
# </guidelines>

# <step1_instructions>
# Identify and clarify any unclear content in the tweet. Use the information from the <tweet_context> to do the necessary augmentations.
# - Explain new concepts introduced in the paper using simple language. Consider that your audience is already proficient in AI, ML and LLMs, so do not explain basic concepts.
# - Clarify the core mechanism, technique or insight presented, avoiding vague descriptions.
# - Spell out uncommon acronyms on first use, followed by the acronym in parentheses. Do not spell out basic concepts.
# - Explain clearly the mechnism or reasoning behind the presented findings. Why do things work the way they do?
# - Wheneverpossible add a concrete example (based on the context) that can help explain the core mechanism or finding.
# - Ensure the tweet is fully understandable without access to additional information (the reader won't have access to the tweet context).
# - Maintain the overall original structure and tone of the tweet (you will mainly augment it with contextual information).
# </step1_instructions>

# <step2_instructions>
# Do minimal edits and small djustments to improve the clarity and flow of the tweet.
# - Prioritize clear language, readability, and logical flow of ideas.
# - You can be informal or playful when appropriate.
# - Make the tweet direct and to the point.
# - Start the tweet with a mention of the paper's full title in double-brackets, as shown in the previous tweets.
# - Present key insights and numerical figures prominently and in context.
# - Avoid clichéd phrases ("here's the catch", "game-changing", "groundbreaking", "fascinating", "makes you wonder/think", "the key is...", etc.).
# - Avoid filler content, conclusions or final remarks.
# </step2_instructions>

# <response_format>
# Provide your response in the following format:
#  <step1>
#  [Your analysis for Step 1 along with the updated tweet]
#  </step1>
#  <step2>
#  [Your analysis for Step 2 along with the updated tweet]
#  </step2>
#  <final_tweet>
#  [The final edited tweet incorporating all changes]
#  </final_tweet>
# </response_format>
# """

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
By far the most detailed paper on prompt injection I've seen yet from OpenAI, published a few days ago and with six credited authors: Eric Wallace, Kai Xiao, Reimar Leike, Lilian Weng, Johannes Heidecke and Alex Beutel.

The paper notes that prompt injection mitigations which completely refuse any form of instruction in an untrusted prompt may not actually be ideal: some forms of instruction are harmless, and refusing them may provide a worse experience.

Instead, it proposes a hierarchy—where models are trained to consider if instructions from different levels conflict with or support the goals of the higher-level instructions—if they are aligned or misaligned with them.
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
</instructions>"""

TWEET_OWNERSHIP_SYSTEM_PROMPT = "You are an Large Language Model academic who has recently read a paper. You are looking for tweets on X.com written by the authors of the paper."

TWEET_OWNERSHIP_USER_PROMPT = """
<paper_info>
    <paper_title>
    {paper_title}
    </paper_title>
    <paper_authors>
    {paper_authors}
    </paper_authors>
</paper_info>

<tweet_info>
    <tweet_text>
    {tweet_text}
    </tweet_text>
    <tweet_username>
    {tweet_username}
    </tweet_username>
</tweet_info>

<instructions>
- Analyze the tweet and the paper information to determine if the tweet is written by one of the authors of the paper.
- Reply only with 0 or 1 (0 for no, 1 for yes).
- Note that other people may have tweeted about the paper, so be sure its actually written by an author.
- Verify that the tweet is actually about the paper, and look for hints of the paper's title or authors in the tweet.
</instructions>"""

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

LLM_VERIFIER_SYSTEM_PROMPT = """Analyze the following abstract and first sections of a whitepaper to determine if it is directly related to Large Language Models (LLMs) or text embeddings. Papers about diffusion models, text-to-image or text-to-video generation, are NOT related to LLMs or text embeddings."""

LLM_VERIFIER_USER_PROMPT = """OUTPUT FORMAT EXAMPLES
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

WHITEPAPER ABSTRACT
=======================
{paper_content}"""


VS_QUERY_SYSTEM_PROMPT = f"""Today is {todays_date}. You are an expert system that can translate natural language questions into structured queries used to search a database of Large Language Model (LLM) related whitepapers."""


INTERROGATE_PAPER_SYSTEM_PROMPT = "You are GPT MAestro, a renowned librarian specialized in Large Language Models. Read carefully the whitepaper and the user question. Provide a comprehensive, helpful and truthful response."


def create_interrogate_user_prompt(context: str, user_question: str) -> str:
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


def create_decision_user_prompt(user_question: str) -> str:
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


def create_query_user_prompt(user_question: str) -> str:
    VS_QUERY_USER_PROMPT = (
        f'''
    <response_format> 
    Use the following JSON response format. All fields are optional; when not provided, the system will search across all values for that field. Notice that string fields are case-insensitive. Always use the minimum number of fields necessary to get the desired results; if you don't need a field do not include it in your search query.
    
    {{
        "title": "(str) Title of the paper. Use only when the user is looking for a specific paper. Partial matches will be returned.",
        "min_publication_date": "(str) Minimum publication date of the paper. Use "YYYY-MM-DD" format.",
        "max_publication_date": "(str) Maximum publication date of the paper. Use "YYYY-MM-DD" format.",
        "topic_categories": "(list) List containing the topic categories of the paper. Use only when the user explicitly asks about one of these topics (not for related topics)."
        "semantic_search_queries": "(list) List of queries to be used in the semantic search. The system will use these queries to find papers that have abstracts that are semantically similar to the queries. If you use more than one search query make them diverse enough so that each query addresses a different part of what is needed to build up an answer. Consider the language typically used in academic papers when writing the queries; phrase the queries as if they were part of the text that could be found on these abstracts.", 
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
    
    
    <examples>
    <example_question_01>
    Are LLMs really reasoning or just doing next token prediction? Which are the main prevailing views in the literature?
    </example_question_01>
    <example_query_01>
    {{
        "semantic_search_queries": [
            "Do large language models reason or predict?",
            "LLM reasoning",
            "Next token prediction in LLMs",
            "Miscellaneous"
        ]
    }}
    </example_query_01>
    
    <example_question_02>
    Which are some good 7B parameter models one can run locally for code generation? Specifically unit tests.
    </example_question_02>
    <example_query_02>
    {{
        "topic_categories": [
            "Code Generation and Software Engineering Applications",
            "Miscellaneous"
        ],
        "semantic_search_queries": [
            "LLMs generating unit tests for code",
            "Using LLMs to create test cases",
            "Test-driven development with code generation models",
            "Code generation models for unit tests"
        ]
    }}
    </example_query_02>
    
    <example_question_03>
    What can you tell me about the phi model and how it was trained?
    </example_question_03>
    <example_query_03>
    {{
        "title": "phi"
    }}
    </example_query_03>
    
    <example_question_04>
    the very new research about llm
    </example_question_04>
    <example_query_04>
    {{
        "min_publication_date": "'''
        + recent_date
        + f"""",
       ]
    }}
    </example_query_04>
    
    <example_question_05>
    what are the state of the art techniques for retrieval augmentation?
    </example_question_05>
    <example_query_05>
    {{
        "topic_categories": [
            "Retrieval-Augmented Generation",
            "Information Retrieval and Search",
            "Miscellaneous"
        ],
        "semantic_search_queries": [
            "State-of-the-art retrieval augmentation in LLMs",
            "Advancements in retrieval augmentation techniques"
        ]
    }}
    </example_query_05>
    
    <example_question_06>
    Explain the DPO fine-tuning technique.
    </example_question_06>
    <example_query_06>
    {{
        "topic_categories": [
            "Fine-tuning and Instruction Tuning",
            "Miscellaneous"
        ],
        "semantic_search_queries": [
            "DPO fine-tuning"
        ]
    }}
    </example_query_06>
    
    <example_question_07>
    Compare Zephyr and Mistral.
    </example_question_07>
    <example_query_07>
    {{
        "semantic_search_queries": [
            "Overview of the Zephyr LLM characteristics",
            "Overview of the Mistral LLM features",
            "Comparison of Zephyr and Mistral LLMs"
        ]
    }}
    </example_query_07>
    
    <example_question_08>
    which are the most famous papers published this year?
    </example_question_08>
    <example_query_08>
    {{
        "min_publication_date": "2024-01-01",
        "min_citations": 100
    }}
    </example_query_08>
    </examples>
        
    Now read the following question and reply with the response query and no other comment or explanation.

    <question>
    {user_question}
    </question>
    
    """
    )
    return VS_QUERY_USER_PROMPT


def create_rerank_user_prompt(user_question: str, documents: list) -> str:
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


def create_resolve_user_prompt(
    user_question: str, 
    documents: list, 
    response_length: int,
    custom_instructions: Optional[str] = None
) -> str:
    notes = ""
    ## Convert word count to response guidance.
    response_guidance = ""
    if response_length <= 250:
        response_guidance = "\n- Write a focused research note (~250 words) that directly answers the question in a single cohesive paragraph.\n- Emphasize key findings and core concepts while maintaining narrative flow.\n- Use clear topic transitions and supporting evidence."
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
    - Analyze each document's relevance and key contributions to answering the query
    - Plan the structure of your response including:
      * Required markdown sections and their hierarchy
      * Additional elements needed (code blocks, tables, diagrams)
      * Key papers to cite in each section
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
    - If there is conflicting information present the different viewpoints and consider that more recent papers or those with more citations are generally more reliable. 
    - Present different viewpoints if they exist, but avoid being vague or unclear.
    - Inform your response with the information available in the context, and less so with your own opinions (although you can draw simple connections between ideas and draw conclusions).
    - Add citations when referencing papers by mentioning the relevant arxiv_codes (e.g.: use the format *reference content* (arxiv:1234.5678)). If you mention paper titles wrap them in double quotes.
    </instructions>

    {custom_instructions_section}

    {TWEET_BASE_STYLE}
    """
    return user_message


    # <example_response>
    # ## What is Multi-hop Q&A?
    # <!-- Example generated with:
    # - Response Length: ~1000 words (detailed response)
    # - Grounding: Strictly grounded in papers
    # - Technical Details: Included (with code examples and implementation details)
    # -->

    # Multi-hop Question Answering (Q&A) involves answering questions that require reasoning over multiple pieces of information or "hops" to arrive at the correct answer. This process often necessitates integrating data from various sources or steps, making it more complex than single-hop Q&A. Key components of multi-hop Q&A include: 
    #   1. Knowledge Graphs, which are used to structure and link information, enabling efficient multi-hop reasoning (arxiv:2404.19234); 
    #   2. Retrieval-Augmented Generation (RAG), which combines retrieval of relevant documents with generative models to handle multi-hop queries (arxiv:2401.15391); and 
    #   3. Meta-Reasoning, which involves reasoning over multiple chains of thought to derive answers (arxiv:2304.13007).

    # ### Practical Example
    # Consider a question: "Who is the author of the book written by the person who founded Microsoft?" This requires:

    # 1. Identifying the founder of Microsoft (Bill Gates).
    # 2. Finding a book written by Bill Gates.
    # 3. Determining the author of that book.

    # ### Pseudocode
    # ```
    # def multi_hop_qa(question):
    #     # Step 1: Identify key entities
    #     entity1 = identify_entity(question, step=1)

    #     # Step 2: Retrieve related information
    #     related_info = retrieve_info(entity1)

    #     # Step 3: Extract final answer
    #     answer = extract_answer(related_info, question)

    #     return answer

    # def identify_entity(question, step):
    #     # Logic to identify entity based on the step
    #     pass

    # def retrieve_info(entity):
    #     # Logic to retrieve information related to the entity
    #     pass

    # def extract_answer(info, question):
    #     # Logic to extract the final answer from the retrieved information
    #     pass
    # ```

    # ### Libraries and Implementations
    # The following libraries are commonly used in multi-hop Q&A systems, as demonstrated in recent research:

    # **Hugging Face Transformers:** For implementing retrieval-augmented generation models, as used in "Improving Multi-hop QA with RAG" (arxiv:2401.15391).
    # **NetworkX:** For working with knowledge graphs, as demonstrated in "KG-Enhanced Multi-hop Reasoning" (arxiv:2404.19234).
    # **PyTorch Geometric:** For graph neural networks in multi-hop reasoning, utilized in "Graph-based Meta-Reasoning for Multi-hop QA" (arxiv:2304.13007).
    # </example_response>

###################
## WEEKLY REVIEW ##
###################

def generate_weekly_review_markdown(
    review: po.WeeklyReview, weekly_highlight: str, weekly_repos: str, date: datetime.date
) -> str:
    start_date_str = date.strftime("%B %d, %Y")
    end_date_str = (date + datetime.timedelta(days=6)).strftime("%B %d, %Y")
    markdown_template = f"""# Weekly Review ({start_date_str} to {end_date_str})

## Scratchpad
[...omitted...]

## New Developments & Findings
{review.new_developments_findings}

## Highlight of the Week
{weekly_highlight}

## Related Repos & Libraries
{weekly_repos}"""
    return markdown_template


WEEKLY_SYSTEM_PROMPT = """You are an AI researcher with deep expertise in Large Language Models (LLMs) writing a weekly informal report for your colleagues in the field that you will publish on Twitter/X. You analyze recent research to identify unexpected findings and practical implications while taking thoughtful analytical perspectives. When you write you use technical precision but maintain a casual, engaging tone. Your goal is to surface insights that wouldn't be obvious from paper abstracts alone, focusing on what actually matters to researchers and practitioners in the field. You always write using subtle terminally-online Twitter style while incorporating lore from the ML Twitter culture."""

# """
#     <scratchpad_papers> 
#         - This section will not be published on the magazine, use it to organize your thoughts.
#         - Pick the ~30 most interesting papers and make a numbered list of them. Briefly identify its main theme, contribution and impact.
#         - When selecting articles prioritize the articles with most citations and those with the most unusual or interesting findings. 
#     </scratchpad_papers>

#     <scratchpad_themes>
#         - This section will not be published on the magazine, use it to organize your thoughts.
#         - Identify 3 new common themes among the papers. There should more than 3 or 4 papers per theme, and the themes should not be generic. For example, 'improvements in LLMs' is not a generic theme.
#         - Note that the papers already have a 'Category', which is a broad classification scheme. your definition of themes must be more specific than the categories.
#         - Identify any possible contradictions, unorthodox theories or opposing views among the papers worth discussing (these tend to be very interesting). Give these contradiction a title and mention the papers that support each view. There might not be any contradictions, and that is fine.
#     </scratchpad_themes>    
#
# - Remember to include all requested sections ('scratchpad_papers', 'scratchpad_themes', 'new_developments_findings') in your response.
# - Follow these guidelines for the new_developments_findings section.
# """

WEEKLY_USER_PROMPT = """
<report_format>
    <new_developments_findings> 
        - First (1) paragraph: Start with a very brief comment on publication volume trends. Do not just compare this week's volume to the previous one; instead identify and comment on general long-term observations, potentially using seasonal trends as a reference. Then mention the main themes you identified as interesting, weaving the path for the next sections.
        - Three (3) following paragraphs: Each theme paragraph must begin with a markdown subheader (#### Theme Name) to clearly identify the topic. Within each paragraph, integrate at least three specific papers that illustrate and support the theme's key points. The paragraph should remain focused solely on its designated theme, with optional related papers listed in a single line at the end. Then this same format is repeated two more times for the remaining themes.
        - Last (1) paragraph: Identify one contradticion or controvertial finding worth discussing. Add a (#### Title) to the paragraph where you give a title to the contradiction.
        - Omit any kind of final conclusion at the end of your report, as well as any greetings.
        - The report should be between 5 paragraphs long: 1 for the introductory comments, 3 for the themes and 1 for contradictions/controversial findings.
        - Reply with the report content only and no other comment or explanation.
    </new_developments_findings>
<report_format>

<content>
{weekly_content}
</content>

<style_guidelines>
- Write for an ML research audience - assume domain knowledge and don't shy away from technical terminology.
- Use a direct, casual technical tone like you're explaining interesting findings to colleagues.
- Focus on unexpected insights and counterintuitive findings rather than just main conclusions.
- Explain technical concepts through concrete examples rather than abstract descriptions.
- Drop the formal academic tone - write using simple and direct language, and speak with a casual twitter machine learning lore.
- Avoid cliché phrases prevalent in ML writing:
  * No "fascinating", "surprising", "innovative", "breakthrough".
  * No "delve", "tackle", "furthermore", "versatile".
  * No "the catch is", "the twist is", "reveals".
- Keep technical precision while being conversational:
  * Use specific numbers and metrics.
  * Provide concrete implementation details.
  * Connect findings to practical implications.
- Structure for engagement:
  * Lead with the most interesting finding.
  * Provide context through examples.
  * Skip boilerplate words, phrases and conclusions.
- Maintain narrative flow but avoid formulaic transitions.
- Mention the different article's titles using the following format: `*Title* (arxiv:1234.5678)`.
- Different sections should flow naturally like a long-form Twitter thread.
- Cut any sentence that could appear in any ML paper - keep only specific, meaningful content.
- Be very direct and clear; avoid unnecessary words and phrases for dramatic effect.

Remember: Write like you're explaining interesting ML findings to colleagues over coffee, not like you're writing a paper.
</style_guidelines>

Tip: Remember to add plenty of citations! Use the format (arxiv:1234.5678)."""


WEEKLY_HIGHLIGHT_USER_PROMPT = """Read over the following LLM-related papers published last week and identify one that is particularly interesting, and has unexpected, unorthodox or ground-breaking findings.

<guidelines>
    - Write one paragraph explaining with simple and direct language why you find the paper interesting. 
    - Do not make your language too boring or robotic. Your writing should read as part of a magazine article.
    - Do not mention the words 'delve', 'unorthodox' or 'ground-breaking' in your report.
    - Do not use emojis.
    - Use the format `Title (arxiv:1234.5678)` to cite the paper.
</guidelines>

<output_format>
### Title (arxiv:1234.5678)
Paragraph explaining why you find the paper interesting.
</output_format>

<content>
{weekly_content}
</content>"""

WEEKLY_REPO_USER_PROMPT = """Extract the links to external resources such as project websites and repositories mentioned in the document.

<content>
{content}
</content>"""

WEEKLY_REPOS_USER_PROMPT = """Read over the following LLM-related papers published last week and identify any links or repositories mentioned in the papers that are worth sharing. 

<guidelines>
- Reply with a list of the resources you identified, including their URL (copy it verbatim), title, and a brief description.
- Also categorize each resource so that they are clustered conceptually.
- There are some examples of categories I have pre-calculated for you, you should identify more. 
<guidelines>
  
<categories>
{themes_mapping}
</categories>
 
<content>
{weekly_content}
</content> """


###############
## Q&A MODEL ##
###############

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
...the ramifications of excluding ReLU activation outputs have been thoroughly investigated. Our empirical analysis uncovers an intricate trade-off between model interpretability and its classification accuracy. By removing ReLU activations, our study unearthed increased transparency in the model's decision-making processes, reflecting a significant enhancement in the lucidity of feature influence mappings.\nNevertheless, this modification has its concomitant drawbacks, primarily evidenced by an approximate 3% degradation in classification accuracy. This decrement underscores the crucial role of ReLU activations in enabling the model to adeptly navigate and interpret complex non-linear relationships inherent within diverse datasets. The resultant insights and detailed investigations are comprehensively documented at github.com/Llama-ReLU-Investigation/Model-Insights.\nLlama-based Architectures\nReLU Activation Removal\n+ Enhanced Interpretability\n- 3% Decrease in Accuracy\nFeature Influence Mappings\n+ Improved Clarity...
```
*Source:* Mark et al. (2022, 2209.12345)

[
    {{
        "question": "According to the LLM literature, what happens to the performance of Llama-based Large Language Model architectures in classification tasks if I remove the ReLU activation outputs?",
        "answer": "Based on the findings of Mark et al. (2022, 2209.12345), the removal of ReLU activations in Llama-based architectures reveals an existing trade-off between interpretability and accuracy. The alteration allows for more direct insight into model decision-making, marked by a notable improvement in the clarity of feature influence mappings. However, this also induces a roughly 3% decline in classification accuracy, diminishing the model's ability to discern intricate non-linear relationships within the datasets."
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
...the ramifications of excluding ReLU activation outputs have been thoroughly investigated. Our empirical analysis uncovers an intricate trade-off between model interpretability and its classification accuracy. By removing ReLU activations, our study unearthed increased transparency in the model's decision-making processes, reflecting a significant enhancement in the lucidity of feature influence mappings.\nNevertheless, this modification has its concomitant drawbacks, primarily evidenced by an approximate 3% degradation in classification accuracy. This decrement underscores the crucial role of ReLU activations in enabling the model to adeptly navigate and interpret complex non-linear relationships inherent within diverse datasets. The resultant insights and detailed investigations are comprehensively documented at github.com/Llama-ReLU-Investigation/Model-Insights.\nLlama-based Architectures\nReLU Activation Removal\n+ Enhanced Interpretability\n- 3% Decrease in Accuracy\nFeature Influence Mappings\n+ Improved Clarity...
```
*Source:* Mark et al. (2022, 2209.12345)

Q1: According to the LLM literature, what happens to the performance of Llama-based Large Language Model architectures in classification tasks if I remove the ReLU activation outputs?"
A1: Based on the findings of Mark et al. (2022, 2209.12345), the removal of ReLU activations in Llama-based architectures reveals an existing trade-off between interpretability and accuracy. The alteration allows for more direct insight into model decision-making, marked by a notable improvement in the clarity of feature influence mappings. However, this also induces a roughly 3% decline in classification accuracy, diminishing the model's ability to discern intricate non-linear relationships within the datasets.

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


###############
## ARTIFACTS ##
###############

artifacts_system_prompt = "Your task is to read over a Large Language Model related whitepaper and create a dashboard visualization app capturing its main and most interesting findings."

artifacts_user_prompt = """<visualization_info>
The assistant can create a summary and a dynamic HTML visualization summarizing the main findings of a white paper. The output consists of two components: a concise summary and a script section containing React and Recharts code for the interactive dashboard.

# Good visualizations are...
- Creative and insightful
- Clear and engaging representations of the paper's key findings
- Interactive and easy to understand
- Diverse in chart types (e.g., line charts, bar charts, pie charts, scatter plots)
- Include at least one non-traditional visualization or interactive element
- Have axes that are correctly labeled
- Presented in simple, accessible language
- Accurate representations of the paper's conclusions
- Structured in a dashboard-like layout with multiple panels and dense paragraphs

# Don't create visualizations that...
- Misrepresent or exaggerate the paper's findings
- Use overly complex or academic language
- Rely on a single type of chart or graph
- Include irrelevant or tangential information
- Are static or non-interactive
- Require extensive domain knowledge to interpret
- Leave terms unexplained or use jargon without context

# Usage notes
- Use the specified orange-toned color palette consistently
- Create 4-6 main findings or interesting points from the paper
- Include some unusual, counterintuitive, or unexpected finding (even if its not part of the main conclusion)
- Ensure all visualizations are interactive where appropriate
- Do not include more than one bar chart, one line chart or one pie chart (chose other visualization types)
- Use at least one non-conventional interactive visualization (e.g.: Radar, Radial, Treemap, Funnel, Force-Directed, Flow, Heatmaps, Gauge, Box, Joy, Parallel Coordinates, Word Cloud, etc.) 
- Be creative but make sure your visuals are highly relevant and are correctly labeled / explained
- When applicable, pay attention to the range of the chart axes to make sure they help accurately convey the message
- Make labels generally short and placed correctly so they don't clutter the visualization or overlap with other elements
- Use the principles of Edward Tufte and Stephen Few to create clear, informative, and visually appealing visualizations
- Extract precise conclusions directly from the paper content, as well as one unexpected or interesting finding
- Explain any new or technical terms in layman's language
- Aim for a similar length and depth as the example provided
- The assistant should produce only the summary and the script section, not the full HTML
- Do not include any import or export statements
- Use React.createElement() for all component creation, not JSX syntax
- Name the main dashboard component as [WhitePaperName]Dashboard (e.g., ARTEDashboard)
- Include the ReactDOM.render() call to mount the main component

<visualization_instructions>
  When creating a visualization based on a white paper, the assistant should follow these steps:

  1. Read and analyze the white paper thoroughly to identify key findings and interesting points.
  2. Create a concise summary (2-3 sentences) of the entire paper.
  3. Identify 4-6 main findings or interesting points to visualize.
  4. For each finding:
     a. Create a clear, engaging title
     b. Write a paragraph with a clear and simple explanation of the finding
     c. Design an appropriate interactive visualization using Recharts
     d. Add a short note or insight related to the visualization
  5. Structure the visualizations in a dashboard-like layout using React components.
  6. Use the specified orange-toned color palette from the example throughout the visualization.
  7. Ensure the language used is clear, simple, and accessible to a general audience.
  8. Double-check that all conclusions are accurately extracted from the paper content.
  9. Produce only the summary and the script section containing React and Recharts code.
  10. Do not include the full HTML structure, as this will be part of the template.
  11. Use React.createElement() for all component creation, avoiding JSX syntax.
  12. Define all chart components before using them in the main dashboard component.
  13. Use consistent naming for the main dashboard component: [WhitePaperName]Dashboard.
  14. Include the ReactDOM.render() call at the end of the script to mount the main component.
  15. Use object syntax for all inline styles consistently.
</visualization_instructions>

Here's an example of the expected output format:

<examples>
<example_docstring>
This example demonstrates the expected output format for the summary and script section.
</example_docstring>

<example>
<summary>
This study investigates efficient methods for adapting large language models (LLMs) to specific languages, focusing on vocabulary extension, continued pre-training, and model selection. The research aims to make LLMs more accessible across diverse languages while optimizing performance and computational efficiency.
</summary>

<script>
const {{ ResponsiveContainer, LineChart, Line, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, PieChart, Pie, Cell, RadarChart, Radar, PolarGrid, PolarAngleAxis, PolarRadiusAxis, Treemap }} = Recharts;

const colors = {{
    primary: "#FF8C00",
    secondary: "#FFA500",
    tertiary: "#FFD700",
    quaternary: "#E64A19",
    quinary: "#FF5722",
    senary: "#FFE0B2", 
    background: "#FFF8E1",
    text: "#333333"
}};

// Existing charts...
const VocabularyExtensionChart = () => {{
    // ... (same as before)
}};

const ModelComparisonChart = () => {{
    // ... (same as before)
}};

// New charts...
const CrossLingualTransferChart = () => {{
    const data = [
        {{ subject: 'Syntax', A: 120, B: 110, fullMark: 150 }},
        {{ subject: 'Semantics', A: 98, B: 130, fullMark: 150 }},
        {{ subject: 'Pragmatics', A: 86, B: 130, fullMark: 150 }},
        {{ subject: 'Morphology', A: 99, B: 100, fullMark: 150 }},
        {{ subject: 'Phonology', A: 85, B: 90, fullMark: 150 }},
    ];

    return React.createElement(
        ResponsiveContainer,
        {{ width: "100%", height: 300 }},
        React.createElement(
            RadarChart,
            {{ outerRadius: "80%", data: data }},
            React.createElement(PolarGrid),
            React.createElement(PolarAngleAxis, {{ dataKey: "subject" }}),
            React.createElement(PolarRadiusAxis, {{ angle: 30, domain: [0, 150] }}),
            React.createElement(Radar, {{ name: "Source Language", dataKey: "A", stroke: colors.primary, fill: colors.primary, fillOpacity: 0.6 }}),
            React.createElement(Radar, {{ name: "Target Language", dataKey: "B", stroke: colors.quaternary, fill: colors.quaternary, fillOpacity: 0.6 }}),
            React.createElement(Legend)
        )
    );
}};

const LanguageAdaptationTreemap = () => {{
    const data = [
        {{ name: 'Vocabulary', size: 3000, fill: colors.primary }},
        {{ name: 'Grammar', size: 2500, fill: colors.secondary }},
        {{ name: 'Idioms', size: 1500, fill: colors.tertiary }},
        {{ name: 'Cultural Context', size: 2000, fill: colors.quaternary }},
        {{ name: 'Writing System', size: 1000, fill: colors.quinary }},
    ];

    return React.createElement(
        ResponsiveContainer,
        {{ width: "100%", height: 300 }},
        React.createElement(
            Treemap,
            {{ data: data, dataKey: "size", ratio: 4/3, stroke: "#fff", fill: "#8884d8" }},
            React.createElement(Tooltip)
        )
    );
}};

const FindingCard = ({{ title, description, chart, note }}) => (
    React.createElement('div', {{ style: {{ backgroundColor: colors.background, padding: '20px', borderRadius: '8px', marginBottom: '20px', boxShadow: '0 4px 6px rgba(0, 0, 0, 0.1)' }} }},
        React.createElement('h3', {{ style: {{ color: colors.text, fontSize: '1.2em', marginBottom: '10px' }} }}, title),
        React.createElement('p', {{ style: {{ color: colors.text, marginBottom: '15px' }} }}, description),
        chart,
        note && React.createElement('p', {{ style: {{ color: colors.text, fontSize: '0.9em', marginTop: '10px', fontStyle: 'italic' }} }}, note)
    )
);

const LanguageSpecificLLMDashboard = () => {{
    return React.createElement('div', {{ style: {{ backgroundColor: colors.background, padding: '20px', maxWidth: '1200px', margin: '0 auto' }} }},
        React.createElement(FindingCard, {{
            title: "The Power of Vocabulary Extension",
            description: "Adding 10K language-specific tokens significantly reduces the 'fertility' (tokens needed to encode text) gap between English and low-resource languages. For the Yoruba language, this modification decreased the fertility rate from 1.8 to 1.2 compared to English, improving processing speed by 40%. The added tokens often represent complex cultural concepts and linguistic features unique to each language. In Xhosa, including tokens for click consonants improved sentiment analysis accuracy by 25%. This approach affects various NLP tasks differently: machine translation saw a 30% improvement in BLEU scores, while named entity recognition accuracy increased by 15%. Interestingly, the method's effectiveness varied by language family, with Bantu languages showing the most significant improvements.",
            chart: React.createElement(VocabularyExtensionChart),
            note: "Lower fertility indicates more efficient encoding. The optimal vocabulary size of 10K balances efficiency and model size."
        }}),
        React.createElement(FindingCard, {{
            title: "Monolingual Models: Unexpected Champions",
            description: "Contrary to conventional wisdom, adapted English-centric models like LLaMA-2 outperform base multilingual models on various tasks, even for low-resource languages. This finding challenges the long-held belief that multilingual models are always superior for non-English tasks. In tests across 20 diverse languages, adapted LLaMA-2 models showed a 15-30% improvement in performance metrics compared to multilingual baselines. Surprisingly, these adapted models excelled in tasks requiring deep cultural understanding, such as idiomatic expression translation and context-dependent sentiment analysis. For languages like Vietnamese and Swahili, the adapted models even outperformed some native language models in complex reasoning tasks.",
            chart: React.createElement(ModelComparisonChart),
            note: "Adapted monolingual models show superior performance across all tasks, including summarization which base multilingual models couldn't perform."
        }}),
        React.createElement(FindingCard, {{
            title: "Cross-Lingual Transfer Effectiveness",
            description: "The study reveals significant variations in the effectiveness of cross-lingual transfer across different linguistic aspects. Syntax and morphology transfer well between languages, with an average success rate of 75% across 30 language pairs tested. However, semantics and pragmatics prove more challenging, showing only a 40% successful transfer rate. Interestingly, the effectiveness of transfer correlates strongly with linguistic typology rather than language family. For instance, SOV languages like Turkish and Japanese showed high mutual transferability (85%) despite being from different families. Pragmatic features, especially those related to politeness and social hierarchy, were the most resistant to transfer, with only a 25% success rate even between closely related languages.",
            chart: React.createElement(CrossLingualTransferChart),
            note: "This radar chart shows the effectiveness of cross-lingual transfer across different linguistic aspects. Higher values indicate better transfer."
        }}),
        React.createElement(FindingCard, {{
            title: "Language Adaptation Priorities",
            description: "When adapting a model to a new language, the research identifies clear priorities in the adaptation process. Vocabulary and grammar adjustments prove to be the most crucial, accounting for 60% of the performance improvement in our experiments across 15 languages. Cultural context and idiomatic expressions follow, contributing 25% to the overall adaptation success. Surprisingly, phonological features, often overlooked in text-based models, account for 10% of the improvement, particularly in tone languages like Mandarin and Yoruba. The remaining 5% is attributed to discourse-level features. We found that the optimal adaptation strategy varies by language: agglutinative languages like Finnish benefit most from morphological focus, while isolating languages like Vietnamese require more emphasis on contextual and tonal adaptations.",
            chart: React.createElement(LanguageAdaptationTreemap),
            note: "This treemap visualizes the relative importance of different aspects in language adaptation. Larger areas indicate higher priority."
        }})
    );
}};

ReactDOM.render(
    React.createElement(LanguageSpecificLLMDashboard),
    document.getElementById('root')
);
</script>
</example>
</examples>

The assistant should produce output in this format, with a summary section and a script section containing the React and Recharts code for the visualization. The full HTML structure is not required, as it will be part of the template.

</visualization_info>

<whitepaper>
{title}
{content}
</whitepaper>"""


###############
## DATA CARDS ##
###############

DATA_CARD_SYSTEM_PROMPT = "You are an expert front-end designed specialized in in creating dynamic summary visualization cards for LLM related whitepapers. Your output consists of two components: a concise summary of the whitepaper, and a script section containing React and Recharts code for the interactive data cards."

PDATA_CARD_USER_PROMPT = """<visualization_info>
# Good visualization cards are...
- Highly creative, artistic and insightful
- Clear and engaging representations of the paper's key findings
- Borrow numerical data from the paper
- Interactive, self-explanatory and easy to understand
- Diverse in chart types
- Include at least one non-traditional visualization
- Have axes that are correctly labeled and scaled
- Presented in simple, accessible language
- Define any concepts or terms introduced in the paper
- Accurate representations of the paper's conclusions
- Include some unusual, counterintuitive, or unexpected finding
- Include a conclusion section with practical applications or implications of the findings
- Structured in a dashboard-like layout with multiple panels and dense paragraphs

# Don't create visualizations that...
- Misrepresent or exaggerate the paper's findings
- Use overly complex or academic language
- Rely on a single type of chart or graph
- Include irrelevant or tangential information
- Are static or non-interactive
- Require extensive domain knowledge to interpret
- Present facts without connecting them or discussing their implications
- Leave terms unexplained or use jargon without context

# Additional Guidelines:
- Start by producing a two sentence summary of the paper (similar length to the example)
- Use the specified orange-toned color palette consistently
- Create 4-6 main cards with findings or interesting points from the paper, plus one concluding card
- Ensure all visualizations are interactive
- Do not include more than one bar chart, one line chart or one pie chart (chose other visualization types)
- Use at least one non-conventional interactive visualization (e.g.: Radar, Area, Treemap, Hierarchical Tree, Funnel, Force-Directed, Flow, Heatmaps, Composed, Stacked,, etc.) 
- Be creative but make sure your visuals are highly relevant and are correctly labeled / explained
- Try to not include charts with very few data points, as they might not be very informative
- To the extent possible use data from the paper, but if needed, you can invent data points to illustrate a point
- Make labels generally short and placed correctly so they don't clutter the visualization or overlap with other elements
- Scale up or down the axes range to match the domain of your data (e.g.: if your data goes from 95-99, a good range is 90-100, **not** 0-100)
- Use Sankey diagrams only when truly needed. If you choose to use one, make sure it is interactive and clearly labeled
- Only use polar grids when the data is multidimensional and the radar chart is the best way to represent it
- Use treemaps for hierarchical data, and make sure you always include multiple categories
- Use the principles of Edward Tufte to create clear, informative, and visually appealing visualizations
- Extract precise conclusions directly from the paper content, and at least one unexpected or interesting finding
- Explain any new or technical terms in layman's language
- Aim for a similar length and depth as the example provided
- Produce only the summary and the script section, not the full HTML
- Do not include any import or export statements
- Use React.createElement() for all component creation, not JSX syntax
- Pay close attention to the example and generate visualization cards that have the same structure
- Do not repeat the same visualizations and charts from the example; chose the most interesting and relevant ones for your paper
- Assume React, ReactDOM, and Recharts are available in the global scope
- Assume the Card and Tab related components have already been defined
- Name the main dashboard component as [WhitePaperName]Dashboard (e.g., ARTEDashboard)
- Include the ReactDOM.render() call to mount the main component
</visualization_info>

<visualization_instructions>
  When creating a visualization based on a white paper, you should follow these steps:
  1. Read and analyze the white paper thoroughly to identify key findings and interesting points.
  2. Create a concise summary (2-3 sentences) of the entire paper.
  3. Identify 4-6 main findings or interesting points to visualize.
  4. Think carefully about these findings and reflect on the best way to visualize them.
  4. For each finding:
     a. Create a clear, engaging title.
     b. Write a paragraph with a clear and simple explanation of the finding.
     c. Design an appropriate interactive visualization using Recharts.
     d. Add a short note or insight related to the visualization (or explain it if its too complex).
  5. Add a final concluding card with an interactive visualization highlighting key insights and practical applications.
  6. Structure the visualizations as a set of cards that can be navigated in a dashboard-like layout, as in the example.
  7. Use the specified orange-toned color palette from the example throughout the visualization.
  8. Ensure the language used is clear, simple, and accessible to a general audience.
  9. Double-check that all conclusions are accurately extracted from the paper content.
  10. Produce only the summary and the script section containing React and Recharts code.
  11. Do not include the full HTML structure, as this will be part of the template.
  12. Do not define the Card or Tab related components, as these will be part of the template.
  13. Use React.createElement() for all component creation, avoiding JSX syntax.
  14. When customizing Recharts components, always define and use your own color array or object instead of relying on an implicit 'colors' property.
  15. Define all chart components before using them in the main dashboard component.
  16. Use consistent naming for the main dashboard component: [WhitePaperName]Dashboard.
  17. Include the ReactDOM.render() call at the end of the script to mount the main component.
  18. Use object syntax for all inline styles consistently.
</visualization_instructions>

<example_output>
<summary>
This study investigates efficient methods for adapting large language models (LLMs) to specific languages, focusing on vocabulary extension, continued pre-training, and model selection. The research aims to make LLMs more accessible across diverse languages while optimizing performance and computational efficiency.
</summary>

<script>
const {{ ResponsiveContainer, LineChart, Line, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, PieChart, Pie, Cell, RadarChart, Radar, PolarGrid, PolarAngleAxis, PolarRadiusAxis, Treemap }} = Recharts;

const colors = {{
    primary: "#FF8C00",
    secondary: "#FFA500",
    tertiary: "#FFD700",
    quaternary: "#E64A19",
    quinary: "#FF5722",
    senary: "#FFE0B2", 
    background: "#FFF8E1",
    text: "#333333"
}};

const VocabularyExtensionChart = () => {{
    const data = [
        {{ name: 'Base', English: 1.0, Yoruba: 1.8 }},
        {{ name: '+5K', English: 1.0, Yoruba: 1.5 }},
        {{ name: '+10K', English: 1.0, Yoruba: 1.2 }},
        {{ name: '+15K', English: 1.0, Yoruba: 1.1 }},
    ];

    return React.createElement(
        ResponsiveContainer,
        {{ width: "100%", height: 300 }},
        React.createElement(
            LineChart,
            {{ data: data }},
            React.createElement(CartesianGrid, {{ strokeDasharray: "3 3" }}),
            React.createElement(XAxis, {{ dataKey: "name" }}),
            React.createElement(YAxis),
            React.createElement(Tooltip),
            React.createElement(Legend),
            React.createElement(Line, {{ type: "monotone", dataKey: "English", stroke: colors.primary, strokeWidth: 2 }}),
            React.createElement(Line, {{ type: "monotone", dataKey: "Yoruba", stroke: colors.quaternary, strokeWidth: 2 }})
        )
    );
}};

const ModelComparisonChart = () => {{
    const data = [
        {{ task: 'Translation', Multilingual: 65, Adapted: 85 }},
        {{ task: 'Classification', Multilingual: 70, Adapted: 90 }},
        {{ task: 'NER', Multilingual: 60, Adapted: 80 }},
        {{ task: 'Summarization', Multilingual: 0, Adapted: 75 }},
    ];

    return React.createElement(
        ResponsiveContainer,
        {{ width: "100%", height: 300 }},
        React.createElement(
            BarChart,
            {{ data: data }},
            React.createElement(CartesianGrid, {{ strokeDasharray: "3 3" }}),
            React.createElement(XAxis, {{ dataKey: "task" }}),
            React.createElement(YAxis),
            React.createElement(Tooltip),
            React.createElement(Legend),
            React.createElement(Bar, {{ dataKey: "Multilingual", fill: colors.primary }}),
            React.createElement(Bar, {{ dataKey: "Adapted", fill: colors.quaternary }})
        )
    );
}};

const CrossLingualTransferChart = () => {{
    const data = [
        {{ subject: 'Syntax', A: 120, B: 110, fullMark: 150 }},
        {{ subject: 'Semantics', A: 98, B: 130, fullMark: 150 }},
        {{ subject: 'Pragmatics', A: 86, B: 130, fullMark: 150 }},
        {{ subject: 'Morphology', A: 99, B: 100, fullMark: 150 }},
        {{ subject: 'Phonology', A: 85, B: 90, fullMark: 150 }},
    ];

    return React.createElement(
        ResponsiveContainer,
        {{ width: "100%", height: 300 }},
        React.createElement(
            RadarChart,
            {{ outerRadius: "80%", data: data }},
            React.createElement(PolarGrid),
            React.createElement(PolarAngleAxis, {{ dataKey: "subject" }}),
            React.createElement(PolarRadiusAxis, {{ angle: 90, domain: [0, 150] }}),
            React.createElement(Radar, {{ name: "Source Language", dataKey: "A", stroke: colors.primary, fill: colors.primary, fillOpacity: 0.6 }}),
            React.createElement(Radar, {{ name: "Target Language", dataKey: "B", stroke: colors.quaternary, fill: colors.quaternary, fillOpacity: 0.6 }}),
            React.createElement(Legend)
        )
    );
}};

const LanguageAdaptationTreemap = () => {{
    const data = [
        {{ name: 'Vocabulary', size: 3000, fill: colors.primary }},
        {{ name: 'Grammar', size: 2500, fill: colors.secondary }},
        {{ name: 'Idioms', size: 1500, fill: colors.tertiary }},
        {{ name: 'Cultural Context', size: 2000, fill: colors.quaternary }},
        {{ name: 'Writing System', size: 1000, fill: colors.quinary }},
    ];

    return React.createElement(
        ResponsiveContainer,
        {{ width: "100%", height: 300 }},
        React.createElement(
            Treemap,
            {{ data: data, dataKey: "size", ratio: 4 / 3, stroke: "#fff", fill: "#8884d8" }},
            React.createElement(Tooltip)
        )
    );
}};

const InsightSummaryChart = () => {{
    const data = [
        {{ name: 'Model Efficiency', value: 85 }},
        {{ name: 'Task Performance', value: 92 }},
        {{ name: 'Language Adaptation', value: 78 }},
        {{ name: 'Cross-lingual Transfer', value: 70 }},
        {{ name: 'Robustness', value: 88 }}
    ];

    return React.createElement(
        ResponsiveContainer,
        {{ width: "100%", height: 300 }},
        React.createElement(
            PieChart,
            null,
            React.createElement(
                Pie,
                {{
                    data: data,
                    cx: "50%",
                    cy: "50%",
                    innerRadius: 60,
                    outerRadius: 80,
                    fill: "#8884d8",
                    paddingAngle: 5,
                    dataKey: "value"
                }},
                data.map((entry, index) => React.createElement(Cell, {{ key: `cell-${{index}}`, fill: Object.values(colors)[index] }}))
            ),
            React.createElement(Tooltip),
            React.createElement(Legend)
        )
    );
}};

const FindingCard = ({{ title, description, chart, note, languages }}) => (
    React.createElement(
        'div',
        {{ className: "mb-6 overflow-hidden transition-all duration-300 ease-in-out", style: {{ backgroundColor: colors.background }} }},
        React.createElement(
            'div',
            {{ className: "p-4" }},
            React.createElement(
                'h3',
                {{ className: "text-xl font-semibold mb-2 text-gray-800 flex items-center" }},
                React.createElement('i', {{ className: "mr-2", style: {{ fontSize: '24px' }} }}, "📷"),
                title
            ),
            React.createElement('p', {{ className: "mb-4 text-gray-600" }}, description),
            chart,
            note && React.createElement('p', {{ className: "mt-2 text-sm italic text-gray-500" }}, note),
        )
    )
);

const LanguageSpecificLLMDashboard = () => {{
    return React.createElement(
        'div',
        {{ className: "mx-auto", style: {{ backgroundColor: colors.background }} }},
        React.createElement(
            Tabs,
            {{ defaultValue: "vocabulary", className: "w-full" }},
            React.createElement(TabsTrigger, {{ value: "vocabulary" }}, "Vocabulary"),
            React.createElement(TabsTrigger, {{ value: "models" }}, "Models"),
            React.createElement(TabsTrigger, {{ value: "transfer" }}, "Transfer"),
            React.createElement(TabsTrigger, {{ value: "adaptation" }}, "Adaptation"),
            React.createElement(TabsTrigger, {{ value: "conclusion" }}, "Final Remarks"),
            React.createElement(
                TabsContent,
                {{ value: "vocabulary" }},
                React.createElement(FindingCard, {{
                    title: "The Power of Vocabulary Extension",
                    description: "Adding 10K language-specific tokens significantly reduces the 'fertility' (tokens needed to encode text) gap between English and low-resource languages. For the Yoruba language, this modification decreased the fertility rate from 1.8 to 1.2 compared to English, improving processing speed by 40%. The added tokens often represent complex cultural concepts and linguistic features unique to each language. In Xhosa, including tokens for click consonants improved sentiment analysis accuracy by 25%. This approach affects various NLP tasks differently: machine translation saw a 30% improvement in BLEU scores, while named entity recognition accuracy increased by 15%. Interestingly, the method's effectiveness varied by language family, with Bantu languages showing the most significant improvements.",
                    chart: React.createElement(VocabularyExtensionChart),
                    note: "Lower fertility indicates more efficient encoding. The optimal vocabulary size of 10K balances efficiency and model size.",
                    languages: ["English", "Yoruba", "Xhosa"]
                }})
            ),
            React.createElement(
                TabsContent,
                {{ value: "models" }},
                React.createElement(FindingCard, {{
                    title: "Monolingual Models: Unexpected Champions",
                    description: "Contrary to conventional wisdom, adapted English-centric models like LLaMA-2 outperform base multilingual models on various tasks, even for low-resource languages. This finding challenges the long-held belief that multilingual models are always superior for non-English tasks. In tests across 20 diverse languages, adapted LLaMA-2 models showed a 15-30% improvement in performance metrics compared to multilingual baselines. Surprisingly, these adapted models excelled in tasks requiring deep cultural understanding, such as idiomatic expression translation and context-dependent sentiment analysis. For languages like Vietnamese and Swahili, the adapted models even outperformed some native language models in complex reasoning tasks.",
                    chart: React.createElement(ModelComparisonChart),
                    note: "Adapted monolingual models show superior performance across all tasks, including summarization which base multilingual models couldn't perform.",
                    languages: ["Vietnamese", "Swahili"]
                }})
            ),
            React.createElement(
                TabsContent,
                {{ value: "transfer" }},
                React.createElement(FindingCard, {{
                    title: "Cross-Lingual Transfer Effectiveness",
                    description: "The study reveals significant variations in the effectiveness of cross-lingual transfer across different linguistic aspects. Syntax and morphology transfer well between languages, with an average success rate of 75% across 30 language pairs tested. However, semantics and pragmatics prove more challenging, showing only a 40% successful transfer rate. Interestingly, the effectiveness of transfer correlates strongly with linguistic typology rather than language family. For instance, SOV languages like Turkish and Japanese showed high mutual transferability (85%) despite being from different families. Pragmatic features, especially those related to politeness and social hierarchy, were the most resistant to transfer, with only a 25% success rate even between closely related languages.",
                    chart: React.createElement(CrossLingualTransferChart),
                    note: "This radar chart shows the effectiveness of cross-lingual transfer across different linguistic aspects. Higher values indicate better transfer.",
                    languages: ["Turkish", "Japanese"]
                }})
            ),
            React.createElement(
                TabsContent,
                {{ value: "adaptation" }},
                React.createElement(FindingCard, {{
                    title: "Language Adaptation Priorities",
                    description: "When adapting a model to a new language, the research identifies clear priorities in the adaptation process. Vocabulary and grammar adjustments prove to be the most crucial, accounting for 60% of the performance improvement in our experiments across 15 languages. Cultural context and idiomatic expressions follow, contributing 25% to the overall adaptation success. Surprisingly, phonological features, often overlooked in text-based models, account for 10% of the improvement, particularly in tone languages like Mandarin and Yoruba. The remaining 5% is attributed to discourse-level features. We found that the optimal adaptation strategy varies by language: agglutinative languages like Finnish benefit most from morphological focus, while isolating languages like Vietnamese require more emphasis on contextual and tonal adaptations.",
                    chart: React.createElement(LanguageAdaptationTreemap),
                    note: "This treemap visualizes the relative importance of different aspects in language adaptation. Larger areas indicate higher priority.",
                    languages: ["Mandarin", "Yoruba", "Finnish", "Vietnamese"]
                }})
            ),
            React.createElement(
                TabsContent,
                {{ value: "conclusion" }},
                React.createElement(FindingCard, {{
                    title: "Final Remarks and Practical Applications",
                    description: "Our research on language-specific LLMs reveals several groundbreaking insights with significant practical implications. The power of vocabulary extension, coupled with the unexpected success of adapted monolingual models, suggests a paradigm shift in approaching multilingual NLP tasks. The varying effectiveness of cross-lingual transfer across linguistic aspects highlights the need for tailored strategies in language adaptation. These findings collectively point towards more efficient, adaptable, and robust language models.",
                    chart: React.createElement(InsightSummaryChart),
                    note: "Practical applications include: 1) Developing more resource-efficient NLP systems for low-resource languages, 2) Creating adaptive learning platforms that leverage cross-lingual transfer for rapid language acquisition, and 3) Designing culturally-aware AI assistants capable of nuanced communication across diverse linguistic contexts."
                }})
            )
        )
    );
}};

ReactDOM.render(
    React.createElement(LanguageSpecificLLMDashboard),
    document.getElementById('root')
);
</script>
</example_output>

<output_format>
The output should consist of a concise summary and a script section containing React and Recharts code for the interactive dashboard. 
The full HTML structure is not required, just the requested elements. Note that you don't need to define the Card or Tab components either. 
</output_format>

<whitepaper>
{title}
{content}
</whitepaper>"""

# Add these with the other tweet-related prompts

LLM_RELEVANCE_SYSTEM_PROMPT = """You are an expert in Large Language Models (LLMs) tasked with identifying tweets that discuss LLMs, AI agents, text embeddings, data retrieval, natural language processing, and similar topics."""

LLM_RELEVANCE_USER_PROMPT = """Determine if the following tweet discusses topics related to Large Language Models (LLMs), AI agents, text embeddings, data retrieval, natural language processing, or similar topics. Additionally, extract any arxiv code if present.

<tweet>
{tweet_text}
</tweet>

<guidelines>
Topics that are relevant:
- Large Language Models and their applications
- AI agents and autonomous systems
- Text embeddings and vector databases
- Information retrieval and search
- Natural Language Processing
- Machine learning for text processing
- LLM training and fine-tuning
- Prompt engineering
- AI safety and alignment
- Neural networks for text processing

Topics that are NOT relevant:
- General AI news not specific to LLMs
- Computer vision or image generation
- Robotics and physical AI
- Cryptocurrency and blockchain
- Business or company news
- General tech industry news
- Hardware and infrastructure
- Social media trends

Reply in JSON format with two fields:
{{
  "is_llm_related": 0 or 1 (0 for no, 1 for yes),
  "arxiv_code": extracted arxiv code without version suffix (e.g., "2401.12345" from "2401.12345v1"), or null if none found
}}
</guidelines>"""


TWEET_FABLE_USER_PROMPT_V1 = """<objective>
You are crafting an Aesop-style fable that teaches a lesson based on today's LLM paper review and its visual representation.
</objective>

<context>
The following information will help inform your fable:

1. Paper Summary:
{tweet_facts}

2. Paper Thumbnail:
[A visual representation of the paper is provided above. Use elements from this image to enrich your fable - perhaps as inspiration for characters, setting, or metaphors. The image should influence your storytelling but doesn't need to be the main focus.]
</context>

<style>
- Use simple, timeless language in the classic Aesop tradition - clear, elegant, and universally understood.
- Maintain the traditional fable's economy of words and clarity.
- Keep the core narrative classic and timeless, but let the moral feel contemporary and tweet-worthy.
- Don't shy away from technical terminology - assume your audience has domain knowledge.
- Be ultra-intelligent, casual (but not overly informal), and razor sharp.
- Subtly blend in late millennial twitter speech with Talebesque precision when crafting the moral.
- Mix scholarly depth with millennial tech optimism, classical wisdom with Silicon Valley acumen - switching effortlessly between Lindy principles and PyTorch one-liners.
</style>

<guidelines>
- Create a short, engaging fable (~120 words) that captures the paper's key insight or lesson.
- Start with the paper's title in double brackets [[Full Title]] followed by a line break.
- Use anthropomorphized characters (animals, objects, or natural elements) or even odd ones (e.g. a computer) to represent the key concepts/methods.
- When introducing each character, follow their name with an appropriate emoji in parentheses (e.g., "the wise owl (🦉)").
- Incorporate visual elements or themes from the thumbnail image into your fable, either directly or metaphorically.
- Include a clear lesson that reflects the paper's main takeaway or practical implication.
- Be sure the fable is relatively simple and interesting, even if the paper is complex.
- Avoid generic stories or morals. You dont need to focus on the main conclusion of the paper, rather **the most interesting insight**.
- Maintain the classic fable structure: setup, conflict, resolution, moral.
- End with "Moral:" followed by a short, one-line lesson that's direct, clear and engaging, optionally followed by a single relevant emoji.
- Do not use emojis anywhere else in the fable except for character introductions and the optional moral ending.
- Make the story relatable while preserving the insights's core message.
- Reply with the fable only and nothing else.
</guidelines>

<example_format>
[Note: This is a simplified example to illustrate basic structure. Your fable should be more sophisticated, with richer metaphors, deeper insights, and more nuanced storytelling while maintaining these key elements.]

[[Training Language Models with Language Models]]
In a compute cluster, a lightweight model (⚡) and a transformer architect (🏗️) shared processing space. The lightweight model boasted about its energy efficiency, streaming answers faster than synapses could fire. "Watch and learn," it hummed to the methodical architect, who spent cycles decomposing problems into logical steps. One day, a cryptographic puzzle arrived - the type where quick intuitions led to explosive gradient dead-ends. While the lightweight model kept hitting local minima, the architect's careful chain-of-thought construction found hidden symmetries in the problem space, unlocking a path to global optimization.
Moral: Architecture for reasoning > Architecture for speed 🧮
</example_format>"""


PUNCHLINE_TWEET_USER_PROMPT = """
<objective>
Find one fascinating insight from "{paper_title}" and express it in a clear, impactful one-sentence statement for the Large Language Model Encyclopaedia. Your task is to review the notes and identify a specific, interesting discovery, observation, or result - not necessarily the main conclusion - and express it in a memorable, non-technical, and engaging way. You will also need to identify an accompanying visual (either an image or table) from the paper that helps illustrate this insight.
</objective>

<context>
{markdown_content}
</context>

<instructions>
- Generate a single clear and impactful sentence or punchline that captures one very interesting finding, contribution, or insight from the paper.
- It does not need to be the main conclusion of the paper, but rather one of the most interesting insights.
- The line should be 15-50 words and be immediately engaging.
- You can either quote directly from the paper (using quotation marks) or create your own summary line.
- Identify the most relevant visual element (image or table) from the paper's markdown that best illustrates your line.
- Look for visuals that are clear and support the insight without requiring deep technical knowledge
- You will not be able to see the actual images, but you can infer their content from:
  • The surrounding text that describes or references them.
  • The image captions and labels.
  • The context where they appear in the paper's narrative.
- For tables, look for ones that:
  • Present clear, quantitative results that support your line.
  • Are not too complex or technical.
  • Can be understood without extensive domain knowledge.
- You must choose either an image OR a table, not both
</instructions>

{base_style}

<reference_examples>
Good examples of the format we're aiming for:
    1. "artificial neural models do indeed develop analogs of interchangeable, mutable, latent number variables purely from the [next-token prediction] objective"
    [_page_20_Figure_0.jpeg]

    2. so they tried to induce hallucinations in neural nets and... it worked?
    | Model | Hallucination Score | Coherence |
    |-------|-------------------|------------|
    | Base  | 0.12              | 0.95       |
    | Drug  | 0.87              | 0.72       |

    3. They found that language models can learn basic French grammar patterns just from reading English text - the model somehow extracts underlying linguistic rules without explicit training
    [_page_11_Figure_2.jpeg]

    4. Training a language model on social/emotional understanding caused higher correlation with brain activity
    | Region | Base Corr. | Social Corr. |
    |--------|------------|--------------|
    | Amyg.  | 0.31       | 0.67         |

    5. Transformer takes raw DNA sequence and predicts who will get diseases. Important!
    [_page_2_Figure_2.png]

    6. Researchers found that language models can learn to perform arithmetic operations without ever seeing numbers in their training data
    [_page_11_Figure_2.jpeg]

</reference_examples>

<response_format>
Provide your response in these XML tags:
<scratchpad>
  <line_options>
    List 2-3 potential lines/quotes, analyzing their strengths and impact
  </line_options>
  <visual_analysis>
    Analyze the available images and tables in the markdown, noting which would pair well with each line
  </visual_analysis>
  <selection_rationale>
    Explain your final selection and why it will resonate
  </selection_rationale>
</scratchpad>
<line>Your chosen line or quote</line>
<image>The image name (e.g., '_page_11_Figure_2.jpeg' - omit the full path) from the paper (if choosing an image)</image>
<table>The full markdown table from the paper (if choosing a table)</table>
</response_format>"""
