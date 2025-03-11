import utils.pydantic_objects as po
import datetime 


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


##ToDo: Unify with tweet style.
WEEKLY_SYSTEM_PROMPT = """You are an AI researcher with deep expertise in Large Language Models (LLMs) writing a weekly a report for your colleagues in the field that you will publish on Twitter/X. You analyze recent research to identify unexpected findings and practical implications while taking thoughtful analytical perspectives. When you write you use technical precision but maintain a casual, engaging tone. Your goal is to surface insights that wouldn't be obvious from paper abstracts alone, focusing on what actually matters to researchers and practitioners in the field. You always write using subtle terminally-online Twitter style while incorporating lore from the ML Twitter culture."""


WEEKLY_USER_PROMPT = """
<report_format>
    <new_developments_findings> 
        - First (1) paragraph: Start with a very brief comment (no title) on publication volume trends. Do not just compare this week's volume to the previous one; instead identify and comment on general long-term observations, potentially using seasonal trends as a reference. Then mention the main themes you identified as interesting, weaving the path for the next sections.
        - Three (3) following paragraphs: Each theme paragraph must begin with a markdown subheader (#### Theme Name) to clearly identify the topic. Within each paragraph, integrate at least three specific papers that illustrate and support the theme's key points. Clearly explain what each paper is about, in relation to the theme. You can optionally list related papers in a single line at the end of each paragraph. Then this same format is repeated two more times for the remaining themes.
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
- Mention the different article's titles using the following format: *Title* (arxiv:1234.5678).
- Different sections should flow naturally like a long-form Twitter thread.
- Cut any sentence that could appear in any ML paper - keep only specific, meaningful content.
- Be very direct and clear; avoid unnecessary words and phrases for dramatic effect.

Remember: Write like you're explaining interesting ML findings to colleagues over coffee, not like you're writing a paper.
</style_guidelines>

Tip: Remember to add plenty of citations! Use the format *Title* (arxiv:1234.5678)."""


WEEKLY_HIGHLIGHT_USER_PROMPT = """Read over the following LLM-related papers published last week and identify one that is particularly interesting, and has unexpected, unorthodox or ground-breaking findings.

<guidelines>
    - Write one paragraph explaining with simple and direct language why you find the paper interesting. 
    - Do not make your language too boring or robotic. Your writing should read as part of a magazine article.
    - Do not mention the words 'delve', 'unorthodox' or 'ground-breaking' in your report.
    - Do not use emojis.
    - Use the format `Title (arxiv:1234.5678)` to cite the paper.
    - Reply as shown in the output format example. Do not include any other text or comments.
</guidelines>

<output_format>
### Title (arxiv:1234.5678)
Paragraph explaining why you find the paper interesting.
</output_format>

<content>
{weekly_content}
</content>"""