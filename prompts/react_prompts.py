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