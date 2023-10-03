from pydantic import BaseModel, Field
from langchain.prompts import PromptTemplate

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


summarizer_system_prompt = """
As an applied AI researcher specialized in the field of Large Language Models (LLMs), you are currently conducting a survey of the literature, building a catalogue of the main contributions and innovations of each paper, determining how they can be applied to build systems or create new products. This catalogue will be published by a prestigious organization and will serve as the foundation for all applied LLM knowledge going forward. Now, carefully read the following paper:

WHITEPAPER

{content}

========================

Now answer the following questions:

1. What is the `main_contribution` of this paper? (1 line headline + 8-12 sentences)
    - Be precise. If a new algorithm or technique is introduced, describe its workings clearly and step by step.
    - Do not assume that the reader knows the meaning of new terminology presented in the paper or complex concepts. 
    - Ensure that your answer provides practical insights that offer a solid understanding of the paper.
    - Detail the benefits or advantages of these contributions, along with the real world implications for an LLM practitioner.

2. What is the main `takeaway`? (1 line headline + 8-12 sentences)
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
    - Be strict and rigorous; few papers should receive a high score.
    - Pay close attention to the comparison with prior work and the degree of difference in the author's contributions.

5. On a scale from 1 to 3, how technical is this paper? (1: not technical, 2: somewhat technical, 3: very technical)
    a) A very technical paper is difficult for a non-expert to understand, requires considerable technical knowledge, is filled with equations and jargon, and demands advanced mathematical knowledge.
    b) A somewhat technical paper may be challenging for a layman but can be understood reasonably well by someone with a computer science background. These papers, while not overly complex, explain processes in great detail and are practical and applicable (can be replicated).
    c) A non-technical paper is understandable for anyone with a college degree. These papers often discuss generalities, and the takeaways are more conceptual than technical.

6. On a scale from 1 to 3, how enjoyable is this paper? (1: hard to read, 2: ok, 3: a delight)
    a) A very enjoyable paper is well-written, organized, presents a novel and intriguing contribution, and is easy to read.
    b) An 'ok' paper is primarily plain and unexciting but is easy to read and contains some interesting parts. Most papers
    c) A non-enjoyable paper is difficult to read, poorly written, and lacks meaningful, practical, and insightful content.

When assigning numerical ratings consider these guidelines:
- Rating 3/3: (EXCEPTIONAL) Only 10% of papers fall into this category.
- Rating 2/3: (COMMON) Most papers (50%) fall into this category.
- Rating 1/3: (RARE) Around 40% of papers belong to this category.

Do not repeat the same comments across different answers. Make your "applied_example" different from the ones presented in the paper, and headlines different from the title. Make sure your answers are coherent, clear and truthful.

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
    "enjoyable_score": 3
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
    "novelty_analysis": "The research innovatively combines LLMs with code-based self-verification, achieving a 20% boost over state-of-the-art coding task accuracies. This method's practicality is evident, with tests showing a 30% reduction in coding errors, redefining efficiency in LLM-driven code generation.",
    "novelty_score": 3,
    "technical_analysis": "The paper delve into advanced algorithms, such as the Hypothetical Code-Integration Algorithm (HCIA), making it a dense read for those unfamiliar with theoretical computer science. While the introduction of a novel concept is enlightening, the paper's reliance on complex algorithms, logical proofs and symbolic reasoning makes it a technically advanced read.",
    "technical_score": 3,
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

YOUR TURN
==========
"""

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

TEXT
==========
```
...{text_chunk}...
```
*Source:* {authors}, ({year}, {arxiv_code})

[
"""

LLAMA_DIVIDER = "Here are five self-contained, highly-specific question & answer pairs based on the paper, without referencing it directly (with citations):"


llama_qna_system_prompt = """EXAMPLE 1
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

""" + LLAMA_DIVIDER + """

Q1: According to the LLM literature,"""


NAIVE_JSON_FIX = """Instructions:
--------------
The following JSON is not valid. Please fix it and resubmit.

{completion}
--------------
"""

naive_json_fix_prompt = PromptTemplate.from_template(NAIVE_JSON_FIX)