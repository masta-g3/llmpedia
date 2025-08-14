import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
from typing import Dict, Tuple, List
import time
import re
import markdown2
from html import escape as html_escape

import utils.app_utils as au
import utils.data_cards as dc
import utils.db.db as db
import utils.db.logging_db as logging_db


def create_sidebar(full_papers_df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    ## Filter sidebar.
    st.sidebar.markdown("# 📁 Filters")
    ## Filter by year or select all of them.
    year_cols = st.sidebar.columns((1, 2))
    _ = year_cols[0].markdown("#### Year")
    all_years = year_cols[0].checkbox("All ", value=st.session_state.all_years)
    st.session_state.all_years = all_years

    _ = year_cols[1].markdown("####")
    year = year_cols[1].slider(
        "Year",
        min_value=2016,
        max_value=2025,
        value=2025,
        step=1,
        label_visibility="collapsed",
        disabled=st.session_state.all_years,
    )

    search_term = st.sidebar.text_input("Search", "")
    search_opt_cols = st.sidebar.columns((1, 1))
    title_only = search_opt_cols[0].checkbox("`Title Only`", value=True)
    code_only = search_opt_cols[1].checkbox("`Arxiv Code`", value=False)
    categories = st.sidebar.multiselect(
        "Categories",
        list(full_papers_df["category"].unique()),
    )
    topics = st.sidebar.multiselect(
        "Topic Group",
        list(full_papers_df["topic"].unique()),
    )

    min_citations = st.sidebar.select_slider(
        "Min Citations",
        options=[0, 1, 5, 10, 100],
        value=0,
    )

    ## Sort by.
    sort_by = st.sidebar.selectbox(
        "Sort By",
        ["Published Date", "Last Updated", "Citations", "Random"],
    )

    ## Global image display preference.
    st.sidebar.markdown("---")
    st.sidebar.markdown("#### 🖼️ Image Display")
    image_preference = st.sidebar.radio(
        "Paper Images",
        options=["🎨 Art", "📄 Page"],
        index=0 if st.session_state.global_image_type == "artwork" else 1,
        key="global_image_toggle",
        horizontal=True,
    )
    st.session_state.global_image_type = (
        "artwork" if image_preference == "🎨 Art" else "first_page"
    )

    ## Year filter.
    if not st.session_state.all_years:
        papers_df = full_papers_df[full_papers_df["published"].dt.year == int(year)]
    else:
        papers_df = full_papers_df.copy()

    ## Search terms.
    if len(search_term) > 0 and title_only:
        search_term = search_term.lower()
        papers_df = papers_df[papers_df["title"].str.lower().str.contains(search_term)]
    elif len(search_term) > 0 and code_only:
        search_term = search_term.lower()
        papers_df = papers_df[
            papers_df["arxiv_code"].str.lower().str.contains(search_term)
        ]
        st.session_state.arxiv_code = search_term
    elif len(search_term) > 0:
        search_term = search_term.lower()
        papers_df = papers_df[
            papers_df["title"].str.lower().str.contains(search_term)
            | papers_df["arxiv_code"].str.lower().str.contains(search_term)
            | papers_df["authors"].str.lower().str.contains(search_term)
            | papers_df["summary"].str.lower().str.contains(search_term)
            | papers_df["contribution_title"].str.lower().str.contains(search_term)
            | papers_df["contribution_content"].str.lower().str.contains(search_term)
            | papers_df["takeaway_title"].str.lower().str.contains(search_term)
            | papers_df["takeaway_content"].str.lower().str.contains(search_term)
        ]

    ## Categories.
    if len(categories) > 0:
        papers_df = papers_df[papers_df["category"].isin(categories)]

    # Cluster.
    if len(topics) > 0:
        papers_df = papers_df[papers_df["topic"].isin(topics)]

    ## Citations.
    papers_df = papers_df[papers_df["citation_count"] >= min_citations]

    ## Order.
    if sort_by == "Last Updated":
        papers_df = papers_df.sort_values("updated", ascending=False)
    elif sort_by == "Published Date":
        papers_df = papers_df.sort_values("published", ascending=False)
    elif sort_by == "Citations":
        papers_df = papers_df.sort_values("citation_count", ascending=False)
    elif sort_by == "Random":
        papers_df = papers_df.sample(frac=1)

    return papers_df, year


@st.fragment
def parse_query_params():
    url_query = st.query_params
    if "arxiv_code" in url_query and len(st.session_state.arxiv_code) == 0:
        paper_code = url_query["arxiv_code"]
        # Visit logging now handled in main() - just handle paper navigation
        st.session_state.arxiv_code = paper_code
        click_tab(3)


def populate_paper_question(paper_code: str, name: str, question: str):
    """Populate paper question and auto-trigger send (modular, reusable)."""
    # Store the question in a separate state key to avoid widget state conflicts
    st.session_state[f"pending_question_{paper_code}{name}"] = question
    # Set flag to auto-trigger send on next render
    st.session_state[f"auto_send_{paper_code}{name}"] = True
    st.rerun(scope="fragment")


@st.fragment
def create_paper_card(paper: Dict, mode="closed", name=""):
    """Creates card UI for paper details."""
    # Main container with padding and border
    with st.container():
        # Top section with image and metadata
        img_cols = st.columns((1, 3))
        expanded = False
        if mode == "open":
            expanded = True
        paper_code = paper["arxiv_code"]

        # Image column
        try:
            if st.session_state.global_image_type == "first_page":
                image_url = f"https://arxiv-first-page.s3.us-east-1.amazonaws.com/{paper_code}.png"
            else:
                image_url = (
                    f"https://arxiv-art.s3.us-west-2.amazonaws.com/{paper_code}.png"
                )
            img_cols[0].image(image_url, use_container_width=True)
        except:
            pass

        # Metadata column
        meta_col = img_cols[1]

        # Title with link
        paper_title = paper["title"]
        paper_url = paper["url"]
        meta_col.markdown(
            f'<h2 style="margin-top: 0; margin-bottom: 0.5em;"><a href="{paper_url}" style="color: #FF4B4B; text-decoration: none;">{paper_title}</a></h2>',
            unsafe_allow_html=True,
        )

        # arXiv code badge
        meta_col.markdown(
            f"""<div style="margin-bottom: 0.75em;">
                <span style="
                    font-family: 'SF Mono', Monaco, 'Cascadia Code', 'Roboto Mono', Consolas, 'Courier New', monospace;
                    font-size: 0.75em;
                    background: linear-gradient(135deg, rgba(255, 75, 75, 0.1) 0%, rgba(255, 75, 75, 0.05) 100%);
                    color: #FF4B4B;
                    padding: 2px 6px;
                    border-radius: 4px;
                    border: 1px solid rgba(255, 75, 75, 0.2);
                    display: inline-block;
                    cursor: pointer;
                    user-select: all;
                    transition: all 0.2s ease;
                " 
                title="Click to copy arXiv code"
                onclick="navigator.clipboard.writeText('{paper_code}'); this.style.background='rgba(255, 75, 75, 0.2)'; setTimeout(() => this.style.background='linear-gradient(135deg, rgba(255, 75, 75, 0.1) 0%, rgba(255, 75, 75, 0.05) 100%)', 200);">
                    arXiv:{paper_code}
                </span>
            </div>""",
            unsafe_allow_html=True,
        )

        # Publication date
        pub_date = pd.to_datetime(paper["published"]).strftime("%d %b %Y")
        meta_col.markdown(
            f"<p style='margin-bottom: 0.5em; color: #666;'><span style='display: inline-flex; align-items: center;'>📅 <span style='margin-left: 4px;'>{pub_date}</span></span></p>",
            unsafe_allow_html=True,
        )

        # Topic with enhanced styling
        if "topic" in paper and not pd.isna(paper["topic"]):
            topic = paper["topic"]
            meta_col.markdown(
                f"""<p style='margin-bottom: 0.5em;'>
                <span style='
                    background-color: var(--secondary-background-color, rgba(128, 128, 128, 0.1)); 
                    padding: 4px 12px; 
                    border-radius: 12px; 
                    font-size: 0.95em;
                    color: var(--text-color, currentColor);
                '>{topic}</span></p>""",
                unsafe_allow_html=True,
            )

        punchline = paper.get("punchline", "")
        if isinstance(punchline, str) and punchline.strip():
            meta_col.markdown(
                f"<div class=\"trending-punchline\">{html_escape(punchline.strip())}</div>",
                unsafe_allow_html=True,
            )

        # Authors and citations in smaller text
        influential_citations = int(paper["influential_citation_count"])
        citation_count = int(paper["citation_count"])
        citation_text = f"{citation_count} citation{'s' if citation_count != 1 else ''}"
        if influential_citations > 0:
            citation_text += f" (⭐️ {influential_citations} influential)"

        meta_col.markdown(
            f"""<div style='margin: 0.5em 0;'>
            <p style='color: var(--text-color, #666); font-size: 0.9em; margin-bottom: 0.8em;'>{paper['authors']}</p>
            <div style='
                display: inline-flex; 
                align-items: center; 
                background-color: var(--secondary-background-color, rgba(128, 128, 128, 0.1)); 
                padding: 6px 12px; 
                border-radius: 12px;
                color: var(--text-color, currentColor);
            '>
            <span style='display: flex; align-items: center;'>
            <span style='margin-right: 4px;'>📊</span>
            <span style='font-size: 0.9em;'>{citation_text}</span>
            </span></div></div>""",
            unsafe_allow_html=True,
        )

        # Action buttons in a row with more spacing
        meta_col.markdown("<div style='margin: 1.5em 0;'>", unsafe_allow_html=True)
        action_btn_cols = meta_col.columns((1, 1, 1))

        # Report button
        report_log_space = meta_col.empty()
        report_btn = action_btn_cols[0].popover("⚠️ Report")
        if report_btn.checkbox(
            "Report bad image", key=f"report_v1_{paper_code}_{name}"
        ):
            logging_db.report_issue(paper_code, "bad_image")
            report_log_space.success("Reported bad image. Thanks!")
            time.sleep(3)
            report_log_space.empty()
        if report_btn.checkbox(
            "Report bad summary", key=f"report_v2_{paper_code}_{name}"
        ):
            logging_db.report_issue(paper_code, "bad_summary")
            report_log_space.success("Reported bad summary. Thanks!")
            time.sleep(3)
            report_log_space.empty()
        if report_btn.checkbox(
            "Report non-LLM paper", key=f"report_v3_{paper_code}_{name}"
        ):
            logging_db.report_issue(paper_code, "non_llm")
            report_log_space.success("Reported non-LLM paper. Thanks!")
            time.sleep(3)
            report_log_space.empty()
        if report_btn.checkbox(
            "Report bad data card", key=f"report_v4_{paper_code}_{name}"
        ):
            logging_db.report_issue(paper_code, "bad_datacard")
            report_log_space.success("Reported bad data-card. Thanks!")
            time.sleep(3)
            report_log_space.empty()

        # # Data card button
        # datacard_btn = action_btn_cols[1].button("📊 Data Card", key=f"dashboard_{paper_code}", type="primary")
        # if datacard_btn:
        #     with st.spinner("*Loading data card...*"):
        #         db.log_visit(f"data_card_{paper_code}")
        #         html_card = dc.generate_data_card_html(paper_code)
        #         if html_card:
        #             @st.dialog(paper_title, width="large")
        #             def render():
        #                 components.html(html_card, height=700, scrolling=True)
        #             render()
        #         else:
        #             error_container = st.empty()
        #             error_container.error("Data card not available yet. Check back soon!")
        #             time.sleep(2)
        #             error_container.empty()

    # Content sections using tabs
    tab_names = [
        "💬 Ask GPT Maestro",  # Enhanced title following current style
        "❗️ Takeaways",  # Concise overview
        "📝 Research Notes",  # More detailed analysis
        "📖 Full Paper",  # Complete in-depth content
    ]

    # More robust repo check
    has_repos = False
    try:
        if paper_code in st.session_state["repos"].index:
            paper_repos = st.session_state["repos"].loc[paper_code]
            if isinstance(paper_repos, pd.Series):
                has_repos = True
            elif isinstance(paper_repos, pd.DataFrame) and len(paper_repos) > 0:
                has_repos = True
    except Exception as e:
        st.error(f"Error checking repos: {e}")

    if has_repos:
        tab_names.append("💻 Code")

    tab_names.append("🔍 Similar Papers")

    # Only add Insight tab if we have an insight
    if "tweet_insight" in paper and not pd.isna(paper["tweet_insight"]):
        tab_names.append("🤖 Maestro's Insight")
    tabs = st.tabs(tab_names)

    with tabs[0]:  # Ask GPT Maestro
        # Enhanced chat interface with subtle improvements
        st.markdown(
            "<div style='font-size: 0.9em; opacity: 0.7; margin-bottom: 1em;'>💡 <em>Ask specific questions about this paper's methodology, findings, or implications</em></div>",
            unsafe_allow_html=True,
        )

        # Check for pending question from pills selection
        pending_question = st.session_state.get(
            f"pending_question_{paper_code}{name}", ""
        )
        if pending_question:
            # Clear the pending question after using it
            del st.session_state[f"pending_question_{paper_code}{name}"]

        paper_question = st.text_area(
            "Your question:",
            value=pending_question,
            height=100,
            key=f"chat_{paper_code}{name}",
            placeholder="e.g., 'What's the key innovation?' or 'How does this compare to previous work?'",
        )

        # Quick action suggestions (elegant pills interface)
        if not paper_question.strip():  # Only show when text area is empty

            # Question options mapping
            question_options = {
                "🔍 Key insights": "What are the key insights and main contributions of this paper?",
                "⚡ TL;DR": "Can you provide a concise summary of this paper's main findings?",
                "🧠 Methodology": "Can you explain the methodology and approach used in this research?",
                "🔗 Impact": "What are the practical implications and potential impact of this work?",
            }

            # Pills widget for quick questions
            selected_pill = st.pills(
                "Quick questions:",
                options=list(question_options.keys()),
                selection_mode="single",
                key=f"quick_pills_{paper_code}{name}",
                label_visibility="collapsed",
            )

            # Handle pill selection
            if selected_pill and selected_pill in question_options:
                populate_paper_question(
                    paper_code, name, question_options[selected_pill]
                )
                # Reset pills selection to avoid repeated triggers
                st.session_state[f"quick_pills_{paper_code}{name}"] = None
            # Subtle caption below pills
            st.markdown("</div>", unsafe_allow_html=True)

        # Check for auto-send trigger (from quick action buttons)
        auto_send = st.session_state.pop(f"auto_send_{paper_code}{name}", False)
        send_clicked = st.button(
            "Send", key=f"send_{paper_code}{name}", disabled=not paper_question.strip()
        )

        # Execute when Send is clicked OR auto-send is triggered
        if (send_clicked or auto_send) and paper_question.strip():
            with st.spinner("🤖 GPT Maestro is analyzing..."):
                response = au.interrogate_paper(
                    paper_question, paper_code, model="gpt-5-mini"
                )
                logging_db.log_qna_db(f"[{paper_code}] ::: {paper_question}", response)
                st.chat_message("assistant").write(response)

    with tabs[1]:  # Takeaways
        st.markdown("### ❗️ Takeaways")
        bullet_summary = (
            paper["bullet_list_summary"]
            if not pd.isna(paper["bullet_list_summary"])
            else "Not available yet, check back soon!"
        )
        bullet_summary_lines = bullet_summary.split("\n")
        numbered_summary = []
        number = 1

        # Regex pattern for matching emojis
        emoji_pattern = re.compile(
            "["
            "\U0001f1e0-\U0001f1ff"  # flags (iOS)
            "\U0001f300-\U0001f5ff"  # symbols & pictographs
            "\U0001f600-\U0001f64f"  # emoticons
            "\U0001f680-\U0001f6ff"  # transport & map symbols
            "\U0001f700-\U0001f77f"  # alchemical symbols
            "\U0001f780-\U0001f7ff"  # Geometric Shapes Extended
            "\U0001f800-\U0001f8ff"  # Supplemental Arrows-C
            "\U0001f900-\U0001f9ff"  # Supplemental Symbols and Pictographs
            "\U0001fa00-\U0001fa6f"  # Chess Symbols
            "\U0001fa70-\U0001faff"  # Symbols and Pictographs Extended-A
            "\U00002702-\U000027b0"  # Dingbats
            "\U000024c2-\U0001f251"
            "]+",
            flags=re.UNICODE,
        )

        for line in bullet_summary_lines:
            if line.strip().startswith("- "):
                # Remove the bullet point and clean the line
                clean_line = line.strip()[2:].strip()
                # Remove all emojis and extra spaces
                clean_line = emoji_pattern.sub("", clean_line).strip()
                numbered_summary.append(f"{number}. {clean_line}")
                number += 1
            else:
                # For non-bullet point lines, still remove emojis
                clean_line = emoji_pattern.sub("", line).strip()
                if clean_line:  # Only add non-empty lines
                    numbered_summary.append(clean_line)

        st.markdown("\n".join(numbered_summary))

    with tabs[2]:  # Research Notes
        st.markdown("### 📝 Research Notes")
        level_select = st.selectbox(
            "Detail",
            [
                "📝 High-Level Overview",
                "🔎 Detailed Research Notes",
            ],
            label_visibility="collapsed",
            index=0,
            key=f"level_select_{paper_code}{name}",
        )

        summary = (
            paper["recursive_summary"]
            if not pd.isna(paper["recursive_summary"])
            else paper["contribution_content"]
        )
        markdown_summary = paper["markdown_notes"]

        # Map selection to level values (level 1 is most detailed)
        level_map = {
            # "Most Detailed": 20,  # Most detailed level
            "Detailed": 10,  # Detailed level
            "Concise": 5,  # Concise level
            "Very Concise": 3,  # More concise
            # "Brief": 2,           # Brief level
            "Minimal": 1,  # Most concise level
        }

        if level_select == "📝 High-Level Overview":
            st.markdown(summary)

        elif level_select == "🔎 Detailed Research Notes":
            # Add level selector with predefined values
            level_select = st.select_slider(
                "Summary Level",
                options=level_map.keys(),
                value="Detailed",
                help="Adjust the level of detail in the research notes",
            )

            # Get notes based on selected level
            try:
                selected_level = level_map[level_select]
                detailed_notes = db.get_extended_notes(
                    paper["arxiv_code"], level=selected_level
                )

                if detailed_notes is None:
                    # If we're trying to get more detailed notes (lower level numbers)
                    if level_map[level_select] <= 2:
                        st.warning("No more detailed notes available for this paper")
                    # If we're trying to get more concise notes (higher level numbers)
                    else:
                        st.warning("No more concise notes available for this paper")
                elif pd.isna(detailed_notes):
                    st.warning("Notes currently unavailable at this level")
                else:
                    detailed_notes = detailed_notes.replace("#", "###")
                    detailed_notes = detailed_notes.replace("<summary>", "")
                    detailed_notes = detailed_notes.replace("</summary>", "")
                    # Add word count indicator
                    word_count = len(detailed_notes.split())
                    st.caption(f"📝 {word_count:,} words")
                    st.markdown(detailed_notes)
            except Exception as e:
                st.error(f"Error retrieving notes: {str(e)}")

        # Add Application Ideas section
        st.markdown("---")
        st.markdown("### 💡 Application Ideas")
        if not pd.isna(paper.get("takeaway_title")):
            st.markdown(f"#### {paper['takeaway_title']}")
            st.markdown(paper.get("takeaway_example", ""))

    with tabs[3]:  # Full Paper Content
        # Fetch paper content
        markdown_content, success = au.get_paper_markdown(paper_code)

        if success:
            # Create columns to center the content with some margin
            _, col, _ = st.columns([1, 8, 1])

            with col:
                # Convert markdown to HTML
                try:
                    html_content = markdown2.markdown(
                        markdown_content,
                        extras=[
                            "fenced-code-blocks",
                            "tables",
                            "header-ids",
                            "break-on-newline",
                            "latex",  # Add support for LaTeX conversion
                            "math",  # Additional math support
                        ],
                    )
                except Exception as e:
                    st.warning(
                        f"⚠️ LaTeX rendering failed. Falling back to plain text. Error: {str(e)}"
                    )
                    # Fallback to basic conversion without LaTeX support
                    html_content = markdown2.markdown(
                        markdown_content,
                        extras=[
                            "fenced-code-blocks",
                            "tables",
                            "header-ids",
                            "break-on-newline",
                        ],
                    )

                # Create an HTML string with GitHub markdown CSS (styling handled globally)
                full_html = f"""
                    <html>
                        <head>
                            <link href="https://cdn.jsdelivr.net/npm/github-markdown-css/github-markdown.min.css" rel="stylesheet">
                        </head>
                        <body class="markdown-body">
                            {html_content}
                        </body>
                    </html>
                """

                # Use the components.html to create a scrollable iframe
                components.html(full_html, height=800, scrolling=True)
        else:
            st.warning(markdown_content)

    # Code & Resources tab (shown if repos exist)
    tab_index = 4
    if has_repos:
        with tabs[tab_index]:
            paper_repos = st.session_state["repos"].loc[paper_code]
            if isinstance(paper_repos, pd.Series):
                paper_repos = pd.DataFrame([paper_repos])

            # Convert to list of dictionaries for easier iteration
            repos_list = paper_repos.to_dict("records")
            for idx, repo in enumerate(repos_list):
                st.markdown(f"### {repo['repo_title']}")
                st.markdown(
                    f"🔗 **Repository:** [{repo['repo_url']}]({repo['repo_url']})"
                )
                st.markdown(f"📝 **Description:** {repo['repo_description']}")
                # Only add separator if it's not the last repo
                if idx < len(repos_list) - 1:
                    st.markdown("---")
        tab_index += 1

    with tabs[tab_index]:  # Similar Papers
        papers_df = st.session_state["papers"]
        if paper_code in papers_df.index:
            ## Use lazy loading for similar docs
            similar_codes, similar_titles, publish_dates = au.get_similar_docs(paper_code, papers_df, n=5)
            if not similar_codes:
                st.write("Not available yet. Check back soon!")
            else:
                similar_df = papers_df.loc[similar_codes]
                generate_grid_gallery(
                    similar_df,
                    extra_key=f"_sim_{paper_code}",
                    n_cols=5,
                    image_type=st.session_state.global_image_type,
                )

    # GPT Maestro Insight tab (only shown if insight exists)
    if "tweet_insight" in paper and not pd.isna(paper["tweet_insight"]):
        with tabs[tab_index + 1]:
            st.markdown("### 🤖 GPT Maestro's Key Insight")
            st.markdown(f"{paper['tweet_insight']}")

    st.markdown("---")


@st.fragment
def generate_grid_gallery(df, n_cols=5, extra_key="", image_type="artwork"):
    """Create streamlit grid gallery of paper cards with flip effect."""
    n_rows = int(np.ceil(len(df) / n_cols))
    for i in range(n_rows):
        cols = st.columns(n_cols)
        for j in range(n_cols):
            if i * n_cols + j < len(df):
                paper_data = df.iloc[i * n_cols + j]
                paper_code = paper_data["arxiv_code"]
                paper_title = paper_data["title"].replace("\n", "")
                punchline = paper_data.get("punchline", "Summary not available.")

                # Sanitize for HTML
                safe_title = html_escape(paper_title)
                safe_punchline = html_escape(
                    punchline if pd.notna(punchline) else "Summary not available."
                )

                if image_type == "first_page":
                    image_url = f"https://arxiv-first-page.s3.us-east-1.amazonaws.com/{paper_code}.png"
                else:  # Default to artwork
                    image_url = (
                        f"https://arxiv-art.s3.us-west-2.amazonaws.com/{paper_code}.png"
                    )

                with cols[j]:
                    card_html = f"""
                    <div class="flip-card">
                      <div class="flip-card-inner">
                        <div class="flip-card-front">
                          <img src="{image_url}" alt="{safe_title}" onerror="this.style.display='none'; this.parentElement.style.justifyContent='center'; this.parentElement.innerHTML += '<div class=\\'flip-card-image-error-text\\'>Image not available</div>';">
                          <div class="flip-title">{safe_title}</div>
                        </div>
                        <div class="flip-card-back">
                          <div class="flip-card-back-content">{safe_punchline}</div>
                          <!-- Button handled by Streamlit below -->
                        </div>
                      </div>
                    </div>
                    """
                    st.markdown(card_html, unsafe_allow_html=True)

                    # Star and publish date (remains below the card)
                    star_count = paper_data["influential_citation_count"] > 0
                    publish_date = pd.to_datetime(paper_data["published"]).strftime(
                        "%b %d, %Y"
                    )
                    star = "⭐️" if star_count else ""
                    centered_code = f"""
                    <div class="centered" style="text-align: center; font-size: var(--font-size-sm); margin-top: calc(-1 * var(--space-sm)); margin-bottom: var(--space-sm);">
                        <code>{star} {publish_date}</code>
                    </div>
                    """
                    st.markdown(centered_code, unsafe_allow_html=True)

                    # Read More button remains a Streamlit button
                    if st.button(
                        "Read More",
                        key=f"focus_flip_{paper_code}{extra_key}",
                        help=(
                            punchline
                            if isinstance(punchline, str) and pd.notna(punchline)
                            else None
                        ),
                        use_container_width=True,
                    ):
                        st.session_state.arxiv_code = paper_code
                        click_tab(3)


@st.fragment
def generate_citations_list(df: pd.DataFrame) -> None:
    """Generate a formatted list of paper citations with clean Streamlit styling."""
    for i, (_, paper) in enumerate(df.iterrows()):
        # Extract paper information
        title = paper["title"].replace("\n", "")
        authors = paper["authors"]
        paper_url = paper["url"]
        paper_code = paper["arxiv_code"]
        publish_date = pd.to_datetime(paper["published"]).strftime("%b %d, %Y")
        citation_count = int(paper.get("citation_count", 0))
        influential_count = int(paper.get("influential_citation_count", 0))
        punchline = paper.get("punchline", "")

        # Title with click handler
        if st.button(
            title + (" ⭐️" if influential_count > 0 else ""),
            key=f"citation_title_{paper_code}_{i}",
            help=punchline if punchline else None,
            use_container_width=True,
        ):
            st.session_state.arxiv_code = paper_code
            click_tab(3)
        
        # Authors
        st.caption(authors)
        
        # Metadata row
        col1, col2, col3 = st.columns([1, 1, 1])
        with col1:
            st.markdown(f"📅 {publish_date}")
        with col2:
            citation_text = f"citation{'s' if citation_count != 1 else ''}"
            st.markdown(f"📊 {citation_count} {citation_text}")
        with col3:
            st.link_button(f"📄 arXiv:{paper_code}", paper_url)
        
        # Punchline if available
        if punchline:
            st.markdown(f"*{punchline}*")
        
        # Add subtle separator
        st.markdown("---")


@st.fragment
def generate_paper_table(df, extra_key=""):
    """Create a stylized table view of papers with key information."""
    # Table styles are now applied globally via apply_complete_app_styles()

    # Updated column width ratios for better spacing
    col_spec = [3.5, 0.9, 0.9, 1.2, 0.8]

    # Create a header row with styled headers
    header_cols = st.columns(col_spec)

    st.markdown("<div class='paper-header'>", unsafe_allow_html=True)
    header_cols[0].markdown("**Title**")
    header_cols[1].markdown("**Citations**")
    header_cols[2].markdown("**Influential**")
    header_cols[3].markdown("**Published**")
    header_cols[4].markdown("")  # Action column placeholder
    st.markdown("</div>", unsafe_allow_html=True)

    # Format function for titles
    def format_title(row):
        title = row["title"].replace("\n", "")
        star = "⭐️ " if row.get("influential_citation_count", 0) > 0 else ""
        return f"{star}{title}"

    # Create a simple table with all papers
    for i, paper in df.iterrows():
        paper_code = paper["arxiv_code"]
        title = format_title(paper)
        citations = int(paper.get("citation_count", 0))
        influential = int(paper.get("influential_citation_count", 0))
        published = pd.to_datetime(paper["published"]).strftime("%b %d, %Y")

        # Create a container for the row
        st.markdown("<div class='paper-row'>", unsafe_allow_html=True)

        # Create a row for each paper
        cols = st.columns(col_spec)

        # Get punchline for tooltip if available
        punchline = paper.get("punchline", "")
        if isinstance(punchline, str) and punchline:
            # Escape HTML and quotes in punchline to avoid breaking the HTML
            punchline = (
                punchline.replace("'", "&#39;")
                .replace('"', "&quot;")
                .replace("<", "&lt;")
                .replace(">", "&gt;")
            )
            punchline_text = f' title="{punchline}"'
        else:
            punchline_text = ""

        # Add URL to title
        paper_url = paper.get("url", "")
        title_html = f"<a href='{paper_url}' target='_blank' class='title-link' style='color: var(--arxiv-red);'{punchline_text}>{title}</a>"

        # Add authors truncated
        authors = paper.get("authors", "")
        if len(authors) > 70:
            authors = authors[:70] + "..."
        authors_html = f"<div style='font-size: 0.85em; color: var(--text-color, #666);'>{authors}</div>"

        # Combine title and authors
        cols[0].markdown(f"{title_html}{authors_html}", unsafe_allow_html=True)

        # Format counts with nice styling and SVG icons
        cols[1].markdown(
            f"""<div class='paper-cell' style='text-align: center;'>
            <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align: middle; opacity: 0.7; margin-right: 3px;"><path d="M17 6.1H3M21 12.1H3M21 18.1H3"></path></svg>
            {citations}
        </div>""",
            unsafe_allow_html=True,
        )

        cols[2].markdown(
            f"""<div class='paper-cell' style='text-align: center;'>
            <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align: middle; opacity: 0.7; margin-right: 3px;"><polygon points="12 2 15.09 8.26 22 9.27 17 14.14 18.18 21.02 12 17.77 5.82 21.02 7 14.14 2 9.27 8.91 8.26 12 2"></polygon></svg>
            {influential}
        </div>""",
            unsafe_allow_html=True,
        )

        # Date with nice styling and calendar icon
        cols[3].markdown(
            f"""<div class='paper-cell'>
            <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align: middle; opacity: 0.7; margin-right: 3px;"><rect x="3" y="4" width="18" height="18" rx="2" ry="2"></rect><line x1="16" y1="2" x2="16" y2="6"></line><line x1="8" y1="2" x2="8" y2="6"></line><line x1="3" y1="10" x2="21" y2="10"></line></svg>
            {published}
        </div>""",
            unsafe_allow_html=True,
        )

        # Button styled as HTML but triggered by Streamlit button with width constraint (set in CSS)
        if cols[4].button(
            "Read More", key=f"btn_{paper_code}_{extra_key}", use_container_width=True
        ):
            st.session_state.arxiv_code = paper_code
            click_tab(3)

        st.markdown("</div>", unsafe_allow_html=True)


def create_pagination(items, items_per_page, label="summaries", year=None):
    num_items = len(items)
    num_pages = num_items // items_per_page
    if num_items % items_per_page != 0:
        num_pages += 1

    st.session_state["num_pages"] = num_pages

    if not st.session_state.all_years and year is not None:
        st.markdown(f"**{num_items} papers found for {year}.**")
    else:
        st.markdown(f"**{num_items} papers found.**")
    st.markdown(f"**Pg. {st.session_state.page_number + 1} of {num_pages}**")
    prev_button, mid, next_button = st.columns((1, 10, 1))
    prev_clicked = prev_button.button("Prev", key=f"prev_{label}")
    next_clicked = next_button.button("Next", key=f"next_{label}")

    if prev_clicked and "page_number" in st.session_state:
        st.session_state.page_number = max(0, st.session_state.page_number - 1)
    if next_clicked and "page_number" in st.session_state:
        st.session_state.page_number = min(
            num_pages - 1, st.session_state.page_number + 1
        )

    start_index = st.session_state.page_number * items_per_page
    end_index = min(start_index + items_per_page, num_items)

    return items[start_index:end_index]


def create_bottom_navigation(label: str):
    num_pages = st.session_state["num_pages"]
    st.write(f"**Pg. {st.session_state.page_number + 1} of {num_pages}**")
    prev_button_btm, _, next_button_btm = st.columns((1, 10, 1))
    prev_clicked_btm = prev_button_btm.button("Prev", key=f"prev_{label}_btm")
    next_clicked_btm = next_button_btm.button("Next", key=f"next_{label}_btm")
    if prev_clicked_btm and "page_number" in st.session_state:
        st.session_state.page_number = max(0, st.session_state.page_number - 1)
        st.rerun()
    if next_clicked_btm and "page_number" in st.session_state:
        st.session_state.page_number = min(
            num_pages - 1, st.session_state.page_number + 1
        )
        st.rerun()


def _add_paper_fallbacks(paper: dict):
    """Add fallback values for missing paper detail fields."""
    if 'markdown_notes' not in paper:
        paper['markdown_notes'] = ""
    if 'recursive_summary' not in paper:
        paper['recursive_summary'] = ""
    if 'bullet_list_summary' not in paper:
        paper['bullet_list_summary'] = ""
    if 'tweet_insight' not in paper:
        paper['tweet_insight'] = ""
    if 'similar_docs' not in paper:
        paper['similar_docs'] = []


def display_paper_details_fragment(paper_code: str):
    """Displays the paper details card with lazy-loaded details."""
    st.session_state.details_canvas = st.session_state.details_canvas.empty()
    with st.session_state.details_canvas:
        if len(paper_code) > 0:
            if "papers" in st.session_state:
                full_papers_df = st.session_state.papers
                if paper_code in full_papers_df.index:
                    paper = full_papers_df.loc[paper_code].to_dict()
                    
                    ## Load details with spinner (cache handled internally)
                    with st.spinner("Loading paper details..."):
                        paper_details = au.hydrate_all_paper_details(paper_code)
                        paper.update(paper_details)
                    
                    ## Add fallbacks and render
                    _add_paper_fallbacks(paper)
                    create_paper_card(paper, mode="open", name="_focus")
                else:
                    st.error("Paper not found.")
            else:
                st.warning("Paper data is still loading...")


def click_tab(tab_num):
    js = f"""
    <script>
        (() => {{
            var tabs = window.parent.document.querySelectorAll("[id^='tabs-bui'][id$='-tab-{tab_num}']");
            if (tabs.length > 0) {{
                tabs[0].click();
            }} else {{
                console.log("Tab with id '-tab-{tab_num}' not found");
            }}
        }})();
    </script>
    """
    st.components.v1.html(js)
    if tab_num == 3:
        display_paper_details_fragment(st.session_state.arxiv_code)
    try:
        st.rerun(scope="fragment")
        time.sleep(1)
    except st.errors.StreamlitAPIException:
        pass


@st.fragment
def generate_mini_paper_table(
    df,
    n: int = 5,
    extra_key: str = "",
    metric_name: str = "Citations",
    metric_col: str = "citation_count",
    show_tweets_toggle: bool = False,
):
    """Create an enhanced card-based display of top papers for dashboard display."""
    # Trending card styles are now applied globally via apply_complete_app_styles()

    # Only take the top n papers
    display_df = df.head(n) if len(df) > n else df

    # Format function for titles
    def format_title(row):
        title = row["title"].replace("\n", "")
        star = "⭐ " if row.get("influential_citation_count", 0) > 0 else "&nbsp;&nbsp;"
        return f"{star} {title}&nbsp;&nbsp;&nbsp;&nbsp;"

    # Create enhanced card layout
    for rank, (_, paper) in enumerate(display_df.iterrows(), 1):
        paper_code = paper["arxiv_code"]
        # title = format_title(paper)
        title = paper["title"].replace("\n", "").strip()
        published = pd.to_datetime(paper["updated"]).strftime("%b %d, %Y")
        metric_value = int(paper.get(metric_col, 0))
        punchline = paper.get("punchline", "")
        authors = paper.get("authors", "")
        paper_url = paper.get("url", "")

        if len(authors) > 130:
            authors = authors[:130] + "... "

        # Image URL with fallback
        if st.session_state.global_image_type == "first_page":
            image_url = (
                f"https://arxiv-first-page.s3.us-east-1.amazonaws.com/{paper_code}.png"
            )
        else:
            image_url = f"https://arxiv-art.s3.amazonaws.com/{paper_code}.png"

        with st.container():
            show_star = paper.get("influential_citation_count", 0) > 0
            title_key = f"title_btn_{paper_code}_{extra_key}"
            title_large = rf"$\textsf{{\Large {title}}}$"
            if st.button(
                title_large, 
                type="tertiary", 
                icon="⭐" if show_star else None,
                key=title_key,
                use_container_width=True,
            ):
                st.session_state.arxiv_code = paper_code
                click_tab(3)

            # Main card content with image and text
            img_col, content_col = st.columns([2, 6], gap="medium")
            
            with img_col:
                # Display image with error handling
                try:
                    st.image(image_url, caption="", use_container_width=True)
                except:
                    st.markdown(
                        '<div style="width:80px; height:80px; background-color:var(--secondary-background-color, #f0f0f0); '
                        'display:flex; align-items:center; justify-content:center; border-radius:8px; '
                        'font-size:0.7em; color:var(--text-color,#999);">No Image</div>', 
                        unsafe_allow_html=True
                    )

            with content_col:
                # Unified card content with clean header badges
                st.markdown(f"""
                <div class="card-content">
                    <div class="card-header">
                        <div class="badge-left">📅 {published}</div>
                        <div class="badge-right">📊 {metric_value:,} {metric_name.lower()}</div>
                    </div>
                    {f'<div class="trending-punchline">{html_escape(punchline)}</div>' if punchline and pd.notna(punchline) else '&nbsp;'}
                    <div class="trending-metadata">
                        <span class="authors">👥 {html_escape(authors)}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Modern discussion section with custom styling
                if (
                    show_tweets_toggle
                    and "tweets" in paper
                    and paper["tweets"]
                    and len(paper["tweets"]) > 0
                ):
                    tweet_count = paper.get("tweet_count", len(paper["tweets"]))
                    
                    # Custom styled expander with CSS
                    expander_styles = """
                    <style>
                    .discussion-expander {
                        background: linear-gradient(180deg, var(--surface-light, #ffffff) 0%, var(--surface-light-alt, #fafbfc) 100%);
                        border: 1px solid rgba(179, 27, 27, 0.08);
                        border-radius: var(--radius-lg, 12px);
                        margin: var(--space-base, 1rem) 0;
                        overflow: hidden;
                        transition: all var(--transition-base, 0.3s ease);
                        box-shadow: var(--shadow-sm, 0 1px 3px rgba(0, 0, 0, 0.04));
                    }
                    
                    .discussion-header {
                        display: flex;
                        align-items: center;
                        padding: var(--space-base, 1rem);
                        background: linear-gradient(135deg, rgba(179, 27, 27, 0.03) 0%, rgba(179, 27, 27, 0.01) 100%);
                        border-bottom: 1px solid rgba(179, 27, 27, 0.06);
                        cursor: pointer;
                        transition: all var(--transition-fast, 0.15s ease);
                    }
                    
                    .discussion-header:hover {
                        background: linear-gradient(135deg, rgba(179, 27, 27, 0.05) 0%, rgba(179, 27, 27, 0.02) 100%);
                    }
                    
                    .discussion-icon {
                        margin-right: var(--space-sm, 0.5rem);
                        font-size: var(--font-size-lg, 1.1rem);
                        opacity: 0.8;
                    }
                    
                    .discussion-title {
                        font-weight: 600;
                        font-size: var(--font-size-sm, 0.95rem);
                        color: var(--text-color, currentColor);
                        flex: 1;
                    }
                    
                    .discussion-count {
                        font-size: var(--font-size-xs, 0.875rem);
                        opacity: 0.7;
                        background: rgba(179, 27, 27, 0.08);
                        padding: var(--space-xs, 0.25rem) var(--space-sm, 0.5rem);
                        border-radius: var(--radius-full, 50%);
                        margin-left: var(--space-sm, 0.5rem);
                    }
                    
                    .discussion-content {
                        padding: var(--space-base, 1rem);
                        padding-top: var(--space-sm, 0.5rem);
                    }
                    
                    .tweet-card {
                        background: rgba(255, 255, 255, 0.4);
                        border: 1px solid rgba(179, 27, 27, 0.04);
                        border-radius: var(--radius-base, 8px);
                        padding: var(--space-base, 1rem);
                        margin-bottom: var(--space-sm, 0.5rem);
                        transition: all var(--transition-fast, 0.15s ease);
                        position: relative;
                    }
                    
                    .tweet-card:hover {
                        border-color: rgba(179, 27, 27, 0.08);
                        transform: translateY(-1px);
                        box-shadow: var(--shadow-sm, 0 1px 3px rgba(0, 0, 0, 0.04));
                    }
                    
                    .tweet-card:last-child {
                        margin-bottom: 0;
                    }
                    
                    .tweet-author {
                        display: flex;
                        align-items: center;
                        margin-bottom: var(--space-xs, 0.25rem);
                        font-size: var(--font-size-sm, 0.95rem);
                    }
                    
                    .tweet-author-name {
                        font-weight: 600;
                        color: var(--text-color, currentColor);
                        margin-right: var(--space-xs, 0.25rem);
                    }
                    
                    .tweet-username {
                        font-family: var(--font-family-mono, Monaco, monospace);
                        font-size: var(--font-size-xs, 0.875rem);
                        opacity: 0.6;
                        background: rgba(179, 27, 27, 0.06);
                        padding: 2px var(--space-xs, 0.25rem);
                        border-radius: var(--radius-sm, 4px);
                    }
                    
                    .tweet-text {
                        font-size: var(--font-size-sm, 0.95rem);
                        line-height: 1.5;
                        color: var(--text-color, currentColor);
                        margin: var(--space-sm, 0.5rem) 0;
                        font-style: italic;
                    }
                    
                    .tweet-engagement {
                        display: flex;
                        align-items: center;
                        gap: var(--space-base, 1rem);
                        margin-top: var(--space-sm, 0.5rem);
                        font-size: var(--font-size-xs, 0.875rem);
                        opacity: 0.7;
                    }
                    
                    .tweet-metric {
                        display: flex;
                        align-items: center;
                        gap: var(--space-xs, 0.25rem);
                    }
                    
                    .tweet-link {
                        color: var(--arxiv-red, #b31b1b);
                        text-decoration: none;
                        font-weight: 500;
                        transition: opacity var(--transition-fast, 0.15s ease);
                    }
                    
                    .tweet-link:hover {
                        opacity: 0.8;
                        text-decoration: underline;
                    }
                    
                    /* Dark mode support */
                    @media (prefers-color-scheme: dark) {
                        .discussion-expander {
                            background: linear-gradient(180deg, var(--surface-dark, #0E1117) 0%, var(--surface-dark-alt, #13151b) 100%);
                            border-color: rgba(179, 27, 27, 0.15);
                            box-shadow: var(--shadow-dark-sm, 0 1px 3px rgba(0, 0, 0, 0.3));
                        }
                        
                        .discussion-header {
                            background: linear-gradient(135deg, rgba(179, 27, 27, 0.08) 0%, rgba(179, 27, 27, 0.03) 100%);
                            border-bottom-color: rgba(179, 27, 27, 0.12);
                        }
                        
                        .discussion-header:hover {
                            background: linear-gradient(135deg, rgba(179, 27, 27, 0.12) 0%, rgba(179, 27, 27, 0.05) 100%);
                        }
                        
                        .discussion-count {
                            background: rgba(179, 27, 27, 0.15);
                        }
                        
                        .tweet-card {
                            background: rgba(0, 0, 0, 0.2);
                            border-color: rgba(179, 27, 27, 0.08);
                        }
                        
                        .tweet-card:hover {
                            border-color: rgba(179, 27, 27, 0.15);
                            box-shadow: var(--shadow-dark-sm, 0 1px 3px rgba(0, 0, 0, 0.3));
                        }
                        
                        .tweet-username {
                            background: rgba(179, 27, 27, 0.12);
                        }
                    }
                    </style>
                    """
                    
                    st.markdown(expander_styles, unsafe_allow_html=True)
                    
                    with st.expander(f"💬 Discussion ({tweet_count} posts)", expanded=True):
                        # Display tweets with modern styling
                        for tweet_idx, tweet in enumerate(paper["tweets"][:3]):  # Show max 3 tweets
                            if tweet and isinstance(tweet, dict) and tweet.get("text", "").strip():
                                author = tweet.get("author", "Unknown")
                                username = tweet.get("username", "")
                                text = tweet.get("text", "")
                                like_count = tweet.get("like_count", 0)
                                repost_count = tweet.get("repost_count", 0)
                                reply_count = tweet.get("reply_count", 0)
                                tweet_link = tweet.get("link", "")
                                
                                # Truncate long tweets for clean display
                                if len(text) > 280:
                                    text = text[:277].strip() + "..."
                                
                                # Escape HTML in text content
                                safe_text = html_escape(text)
                                safe_author = html_escape(author)
                                safe_username = html_escape(username) if username else ""
                                
                                # Build engagement metrics
                                engagement_metrics = []
                                if like_count > 0:
                                    engagement_metrics.append(f'<span class="tweet-metric">❤️ {like_count:,}</span>')
                                if repost_count > 0:
                                    engagement_metrics.append(f'<span class="tweet-metric">🔄 {repost_count:,}</span>')
                                if reply_count > 0:
                                    engagement_metrics.append(f'<span class="tweet-metric">💬 {reply_count:,}</span>')
                                
                                engagement_html = "".join(engagement_metrics)
                                if tweet_link:
                                    if engagement_html:
                                        engagement_html += f'<a href="{tweet_link}" target="_blank" class="tweet-link">View on X →</a>'
                                    else:
                                        engagement_html = f'<a href="{tweet_link}" target="_blank" class="tweet-link">View on X →</a>'
                                
                                # Create clean tweet card
                                tweet_html = f"""
                                <div class="tweet-card">
                                    <div class="tweet-author">
                                        <span class="tweet-author-name">{safe_author}</span>
                                        {f'<span class="tweet-username">@{safe_username}</span>' if safe_username else ''}
                                    </div>
                                    <div class="tweet-text">{safe_text}</div>
                                    {f'<div class="tweet-engagement">{engagement_html}</div>' if engagement_html else ''}
                                </div>
                                """
                                
                                st.markdown(tweet_html, unsafe_allow_html=True)
                
            st.divider()



    # Summary information
    if len(df) > n:
        st.markdown(
            f'<div class="trending-summary">Showing top {n} of {len(df)} papers</div>',
            unsafe_allow_html=True,
        )


@st.fragment
def create_featured_paper_card(paper: Dict) -> None:
    """Display the weekly highlighted paper using enhanced discovery design."""

    # Section header with consistent styling
    header_html = """
    <div class="trending-panel-header">
        <div class="trending-panel-title">
            ⭐ Featured Paper
        </div>
        <div class="trending-panel-subtitle">
            Weekly highlight selected by GPT Maestro
        </div>
    </div>
    """
    st.markdown(header_html, unsafe_allow_html=True)

    paper_code = paper.get("arxiv_code", "")
    title = paper.get("title", "Featured Paper")
    punchline = paper.get("punchline", "No summary available.")
    if st.session_state.global_image_type == "first_page":
        image_url = (
            f"https://arxiv-first-page.s3.us-east-1.amazonaws.com/{paper_code}.png"
        )
    else:
        image_url = f"https://arxiv-art.s3.us-west-2.amazonaws.com/{paper_code}.png"
    paper_url = paper.get("url", f"https://arxiv.org/abs/{paper_code}")

    # HTML-escape potentially unsafe content
    safe_title = html_escape(title)
    safe_punchline = html_escape(punchline)

    # Enhanced discovery card with interactive actions
    card_html = f"""
    <div class="featured-card discovery-card">
        <div class="featured-image">
            <img src=\"{image_url}\" alt=\"{safe_title}\"
                 onerror=\"this.style.display='none'; this.parentElement.style.backgroundColor='var(--secondary-background-color, #f0f0f0)'; this.parentElement.innerHTML='<div style=\\'display:flex;align-items:center;justify-content:center;height:100%;font-size:0.7em;color:var(--text-color,#999);\\'>No Image</div>';\">
        </div>
        <div class="featured-content">
            <div class="featured-title"><a href=\"{paper_url}\" target=\"_blank\">{safe_title}</a></div>
            <div class="featured-punchline">{safe_punchline}</div>
            <!--
            <div class="discovery-actions">
                <a href="#" class="action-btn" onclick="navigator.clipboard.writeText('{paper_code}'); this.innerHTML='✓ Copied'; setTimeout(() => this.innerHTML='📋 Copy arXiv', 2000)">📋 Copy arXiv</a>
                <a href="{paper_url}" target="_blank" class="action-btn">📖 Read Paper</a>
                <a href="#" class="action-btn" onclick="document.querySelector('[data-testid=&quot;stTextInput&quot;] input').value='Tell me about {paper_code}'; document.querySelector('[data-testid=&quot;stTextInput&quot;] input').dispatchEvent(new Event('input', {{bubbles: true}}));">🔬 Research This</a>
            </div>
            -->
        </div>
    </div>
    """

    st.markdown(card_html, unsafe_allow_html=True)

    btn_cols = st.columns([2, 2, 2])
    with btn_cols[1]:
        if st.button(
            "View Details", key=f"featured_details_{paper_code}", use_container_width=True
        ):
            st.session_state.arxiv_code = paper_code
            click_tab(3)


def display_interesting_facts(facts_list, n_cols=2, papers_df=None):
    """Displays a grid of interesting facts from papers."""
    if not facts_list:
        st.info("No interesting facts found.")
        return

    # Interesting facts styles are now applied globally via apply_complete_app_styles()

    # Create a multi-column layout
    cols = st.columns(n_cols)

    # Distribute facts among columns
    for i, fact in enumerate(facts_list):
        col_idx = i % n_cols

        # Get topic if papers_df is provided
        topic = None
        topic_full = None
        if papers_df is not None and "arxiv_code" in fact:
            arxiv_code = fact["arxiv_code"]
            if arxiv_code in papers_df.index and "topic" in papers_df.columns:
                topic_full = papers_df.loc[arxiv_code, "topic"]
                # Handle NaN values safely
                if pd.notna(topic_full) and isinstance(topic_full, str):
                    topic = (
                        topic_full[:30] + "..." if len(topic_full) > 30 else topic_full
                    )

        with cols[col_idx]:
            # Create a container with padding and subtle border
            with st.container():
                # Build topic HTML safely
                topic_html = ""
                if topic and topic_full:
                    topic_html = (
                        f"<span class='fact-topic' title='{topic_full}'>{topic}</span>"
                    )

                st.markdown(
                    f"""<div class="fact-card discovery-card">
                    <div class="fact-content">{fact['fact']}</div>
                    <div class="fact-metadata">{topic_html}
                    <div class="fact-paper-link">
                        <a href="https://arxiv.org/abs/{fact['arxiv_code']}" target="_blank">
                            {fact['paper_title'][:75] + ('...' if len(fact['paper_title']) > 75 else '')}
                        </a>
                        </div>
                    </div>
                    <div class="discovery-actions">
                        <a href="#" class="action-btn" onclick="navigator.clipboard.writeText('{fact['arxiv_code']}'); this.innerHTML='✓ Copied'; setTimeout(() => this.innerHTML='📋 Copy arXiv', 2000)">📋 Copy arXiv</a>
                        <a href="#" class="action-btn" onclick="document.querySelector('[data-testid=&quot;stTextInput&quot;] input').value='Tell me more about {fact['arxiv_code']}'; document.querySelector('[data-testid=&quot;stTextInput&quot;] input').dispatchEvent(new Event('input', {{bubbles: true}}));">🔬 Research</a>
                    </div>
                    </div>""",
                    unsafe_allow_html=True,
                )


def display_tweet_summaries(df, max_entries: int = 8):
    """Display recent X.com LLM discussion summaries in a timeline carousel."""
    if df is None or df.empty:
        st.info("No recent discussions found.")
        return

    # Limit to desired number of entries and sort by timestamp (newest first)
    df = df.head(max_entries)
    df["tstp"] = pd.to_datetime(df["tstp"])
    df = df.sort_values("tstp", ascending=False)

    # Build complete HTML structure as one piece to preserve flexbox layout
    html_parts = []

    # Content only (no header - handled by toggle panel)
    html_parts.append(
        """
<div class="tweet-timeline-container">
    <div class="tweet-carousel">"""
    )

    # Add each discussion as a card
    for i, (_, row) in enumerate(df.iterrows()):
        timestamp = row["tstp"].strftime("%b %d, %H:%M")
        full_timestamp = row["tstp"].strftime("%B %d, %Y at %H:%M")

        # Clean and escape the summary content
        summary = str(row["response"]).strip()
        # Use html_escape to properly handle all special characters
        summary = html_escape(summary)

        # Calculate relative time
        time_diff = pd.Timestamp.now() - row["tstp"]
        if time_diff.days > 0:
            relative_time = (
                f"{time_diff.days} day{'s' if time_diff.days > 1 else ''} ago"
            )
        elif time_diff.seconds > 3600:
            hours = time_diff.seconds // 3600
            relative_time = f"{hours} hour{'s' if hours > 1 else ''} ago"
        else:
            relative_time = "Recent"

        # Add card to HTML - use simple concatenation to avoid f-string issues
        card_html = (
            '<div class="tweet-card" title="' + html_escape(full_timestamp) + '">'
        )
        card_html += (
            '<div class="tweet-timestamp">🕑 '
            + timestamp
            + " • "
            + relative_time
            + "</div>"
        )
        card_html += '<div class="tweet-content">' + summary + "</div>"
        card_html += "</div>"

        html_parts.append(card_html)

    # Footer
    html_parts.append(
        """
    </div>
    <div class="tweet-timeline-footer">Scroll horizontally to explore the timeline →</div>
</div>"""
    )

    # Render complete HTML structure as one piece
    complete_html = "".join(html_parts)
    st.markdown(complete_html, unsafe_allow_html=True)


def display_reddit_summaries(df, max_entries: int = 8):
    """Display recent Reddit LLM discussion summaries in a timeline carousel."""
    if df is None or df.empty:
        st.info("No recent discussions found.")
        return

    # Limit to desired number of entries and sort by timestamp (newest first)
    df = df.head(max_entries)
    df["tstp"] = pd.to_datetime(df["tstp"])
    df = df.sort_values("tstp", ascending=False)

    # Build complete HTML structure as one piece to preserve flexbox layout
    html_parts = []

    # Content only (no header - handled by toggle panel)
    html_parts.append(
        """
<div class="tweet-timeline-container">
    <div class="tweet-carousel">"""
    )

    # Add each discussion as a card
    for i, (_, row) in enumerate(df.iterrows()):
        timestamp = row["tstp"].strftime("%b %d, %H:%M")
        full_timestamp = row["tstp"].strftime("%B %d, %Y at %H:%M")

        # Clean and escape the summary content
        summary = str(row["response"]).strip()
        # Use html_escape to properly handle all special characters
        summary = html_escape(summary)

        # Calculate relative time
        time_diff = pd.Timestamp.now() - row["tstp"]
        if time_diff.days > 0:
            relative_time = (
                f"{time_diff.days} day{'s' if time_diff.days > 1 else ''} ago"
            )
        elif time_diff.seconds > 3600:
            hours = time_diff.seconds // 3600
            relative_time = f"{hours} hour{'s' if hours > 1 else ''} ago"
        else:
            relative_time = "Recent"

        # Add card to HTML - use simple concatenation to avoid f-string issues
        card_html = (
            '<div class="tweet-card" title="' + html_escape(full_timestamp) + '">'
        )
        card_html += (
            '<div class="tweet-timestamp">🕑 '
            + timestamp
            + " • "
            + relative_time
            + "</div>"
        )
        card_html += '<div class="tweet-content">' + summary + "</div>"
        card_html += "</div>"

        html_parts.append(card_html)

    # Footer
    html_parts.append(
        """
    </div>
    <div class="tweet-timeline-footer">Scroll horizontally to explore the timeline →</div>
</div>"""
    )

    # Render complete HTML structure as one piece
    complete_html = "".join(html_parts)
    st.markdown(complete_html, unsafe_allow_html=True)


def display_individual_tweet(tweet_data: Dict, compact: bool = True):
    """Display individual tweet card with author, content, metrics, and link."""
    # Extract tweet information
    author = tweet_data.get("author", "Unknown")
    username = tweet_data.get("username", "@unknown")
    text = tweet_data.get("text", "")
    like_count = tweet_data.get("like_count", 0)
    repost_count = tweet_data.get("repost_count", 0)
    reply_count = tweet_data.get("reply_count", 0)
    tweet_link = tweet_data.get("link", "")
    tweet_timestamp = tweet_data.get("tweet_timestamp")
    max_length = 512

    # Format timestamp
    if tweet_timestamp:
        try:
            timestamp = pd.to_datetime(tweet_timestamp)
            formatted_time = timestamp.strftime("%b %d, %H:%M")
        except:
            formatted_time = "Recent"
    else:
        formatted_time = "Recent"

    # Truncate text for compact view
    if compact and len(text) > max_length:
        text = text[:max_length] + "..."

    # Escape content for safe HTML
    safe_author = html_escape(author)
    safe_username = html_escape(username)
    safe_text = html_escape(text)

    # Create compact tweet card
    tweet_html = f"""
    <div class="individual-tweet-card">
        <div class="tweet-header">
            <div class="tweet-author">
                <strong>{safe_author}</strong> <span class="tweet-username">{safe_username}</span>
            </div>
            <div class="tweet-time">{formatted_time}</div>
        </div>
        <div class="tweet-text">{safe_text}</div>
        <div class="tweet-metrics">
            <span class="tweet-metric">💖 {like_count:,}</span>
            <span class="tweet-metric">🔄 {repost_count:,}</span>
            <span class="tweet-metric">💬 {reply_count:,}</span>
            <a href="{tweet_link}" target="_blank" class="tweet-link">View Tweet →</a>
        </div>
    </div>
    """

    st.markdown(tweet_html, unsafe_allow_html=True)


def render_research_header():
    """Render the research assistant header HTML."""
    header_html = """
    <div class="trending-panel-header">
        <div class="trending-panel-title">
            🤖 Online Research Assistant
        </div>
        <div class="trending-panel-subtitle">
            AI-powered research with cited sources • Ask questions about LLMs and arXiv papers
        </div>
    </div>
    """
    st.markdown(header_html, unsafe_allow_html=True)


def parse_research_progress_message(message: str) -> dict:
    """Parse progress messages into structured state dict."""
    import re

    updates = {}

    # Parse phase information
    if "PHASE 1:" in message:
        updates["current_phase"] = "Phase 1: Research Planning"
        updates["phase_details"] = "Analyzing question and creating research brief"
    elif "PHASE 2:" in message:
        updates["current_phase"] = "Phase 2: Multi-Agent Research"
        updates["phase_details"] = "Deploying specialized research agents"
    elif "PHASE 3:" in message:
        updates["current_phase"] = "Phase 3: Synthesis"
        updates["phase_details"] = "Combining findings into final response"
    elif "Agent " in message and "/" in message:
        # Parse agent progress: "🤖 Agent 2/4: Researching 'topic'..."
        agent_match = re.search(r"Agent (\d+)/(\d+)", message)
        if agent_match:
            updates["current_agent"] = int(agent_match.group(1))
            updates["agents_total"] = int(agent_match.group(2))
            updates["phase_details"] = (
                f"Agent {updates['current_agent']}/{updates['agents_total']} researching"
            )
    elif "completed:" in message:
        # Parse completion: "✅ Agent 1 completed: 3 insights, 5 papers"
        updates["agents_completed"] = updates.get("agents_completed", 0) + 1
        insight_match = re.search(r"(\d+) insights", message)
        papers_match = re.search(r"(\d+) papers", message)
        if insight_match:
            updates["insights_found"] = updates.get("insights_found", 0) + int(
                insight_match.group(1)
            )
        if papers_match:
            updates["papers_found"] = updates.get("papers_found", 0) + int(
                papers_match.group(1)
            )
    elif "Research complete!" in message:
        updates["current_phase"] = "Complete"
        updates["phase_details"] = "Research successfully completed"

    return updates


def add_insight_to_progress(progress_state: dict, insight: str):
    """Add a new insight to the progress state."""
    if "insights_list" not in progress_state:
        progress_state["insights_list"] = []
    progress_state["insights_list"].append(insight)
    progress_state["insights_found"] = len(progress_state["insights_list"])


def add_paper_to_progress(progress_state: dict, paper_info: dict):
    """Add a new paper to the progress state."""
    if "papers_list" not in progress_state:
        progress_state["papers_list"] = []
    progress_state["papers_list"].append(paper_info)
    progress_state["papers_found"] = len(progress_state["papers_list"])


def render_research_progress(status_widget, progress_state: dict):
    """Render progress display with structured HTML sections."""
    # Create main status label with current phase
    current_phase = progress_state.get("current_phase", "Initializing")
    agents_total = progress_state.get("agents_total", 0)
    agents_completed = progress_state.get("agents_completed", 0)
    current_agent = progress_state.get("current_agent", 0)

    main_label = f"{current_phase}"
    if agents_total > 0:
        # Show current agent number (1-based) instead of completed count when research is active
        if current_agent > 0 and agents_completed < agents_total:
            main_label += f" ({current_agent}/{agents_total} agents)"
        else:
            main_label += f" ({agents_completed}/{agents_total} agents)"

    # Update the main status label
    status_widget.update(label=main_label, expanded=True)

    # Get or create content container from progress_state
    if "content_container" not in progress_state:
        with status_widget:
            progress_state["content_container"] = st.empty()

    # Clear and update the content container
    with progress_state["content_container"].container():
        # Clean timeline header with total activity counter
        activity_log = progress_state.get("activity_log", [])
        total_activities = progress_state.get("total_activities", 0)
        displayed_activities = len(activity_log)
        insights_found = progress_state.get("insights_found", 0)
        papers_found = progress_state.get("papers_found", 0)

        timeline_header = f"""
        <div style="display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.75rem;">
            <div style="font-weight: 600; font-size: 0.9rem; color: var(--text-color);">📡 Research Timeline</div>
            <div style="width: 6px; height: 6px; background: #4caf50; border-radius: 50%; animation: pulse 2s ease-in-out infinite;"></div>
            <div style="font-size: 0.75rem; color: var(--text-color); opacity: 0.6;">({total_activities} total)</div>
        </div>
        <style>
        @keyframes pulse {{
            0% {{ opacity: 1; transform: scale(1); }}
            50% {{ opacity: 0.4; transform: scale(1.3); }}
            100% {{ opacity: 1; transform: scale(1); }}
        }}
        </style>
        """
        st.markdown(timeline_header, unsafe_allow_html=True)

        # Better layout: Timeline + compact stats
        layout_cols = st.columns([3, 1])  # Timeline takes 3/4, stats take 1/4

        # Timeline column (left - wider)
        with layout_cols[0]:
            # Timeline with truncation indicator
            if activity_log:
                # Show "earlier activities" if we have more total than displayed
                if total_activities > displayed_activities:
                    earlier_count = total_activities - displayed_activities
                    truncation_entry = (
                        '<div style="display: flex; margin-bottom: 0.1rem; opacity: 0.5;">'
                        '<div style="display: flex; flex-direction: column; align-items: center; margin-right: 0.75rem; min-width: 20px;">'
                        '<div style="width: 20px; height: 20px; display: flex; align-items: center; justify-content: center; font-size: 0.9rem; color: #999;">⋯</div>'
                        '<div style="width: 2px; height: 16px; background: rgba(179, 27, 27, 0.2); margin-top: 4px;"></div>'
                        "</div>"
                        f'<div style="flex: 1; font-size: 0.75rem; color: var(--text-color); opacity: 0.6; line-height: 1.3; padding-top: 2px; font-style: italic;">{earlier_count} earlier activities...</div>'
                        "</div>"
                    )
                    st.markdown(truncation_entry, unsafe_allow_html=True)

                # Timeline entries with enhanced visual hierarchy
                for i, activity in enumerate(activity_log):
                    # Clean up the activity message for display
                    clean_activity = (
                        activity.replace("🎯 ", "")
                        .replace("🤖 ", "")
                        .replace("📝 ", "")
                    )
                    if len(clean_activity) > 200:
                        clean_activity = clean_activity[:197] + "..."

                    # Add minimal log-style prefix: step number from total count
                    step_number = total_activities - len(activity_log) + i + 1
                    clean_activity = f"#{step_number} {clean_activity}"

                    # Enhanced activity type detection and styling
                    activity_lower = activity.lower()
                    is_latest = i == len(activity_log) - 1

                    # Determine activity type, icon, color, and status
                    if "phase" in activity_lower:
                        icon = "🎯"
                        color = "#b31b1b"
                        activity_type = "phase"
                        bg_color = "rgba(179, 27, 27, 0.08)"
                        icon_style = "font-weight: bold;"
                    elif "agent" in activity_lower and "completed" in activity_lower:
                        icon = "✅"
                        color = "#4caf50"
                        activity_type = "completion"
                        bg_color = "rgba(76, 175, 80, 0.1)"
                        icon_style = ""
                    elif "agent" in activity_lower:
                        icon = "🤖"
                        color = "#2196f3"
                        activity_type = "agent"
                        bg_color = "rgba(33, 150, 243, 0.08)"
                        icon_style = ""
                    elif "searching" in activity_lower or "found" in activity_lower:
                        icon = "🔍"
                        color = "#ff9800"
                        activity_type = "search"
                        bg_color = "rgba(255, 152, 0, 0.08)"
                        icon_style = ""
                    elif "reranking" in activity_lower or "filtering" in activity_lower:
                        icon = "⚖️"
                        color = "#9c27b0"
                        activity_type = "process"
                        bg_color = "rgba(156, 39, 176, 0.08)"
                        icon_style = ""
                    elif (
                        "analyzing" in activity_lower or "extracting" in activity_lower
                    ):
                        icon = "🧠"
                        color = "#673ab7"
                        activity_type = "analysis"
                        bg_color = "rgba(103, 58, 183, 0.08)"
                        icon_style = ""
                    elif (
                        "synthesizing" in activity_lower or "complete" in activity_lower
                    ):
                        icon = "📝"
                        color = "#4caf50"
                        activity_type = "synthesis"
                        bg_color = "rgba(76, 175, 80, 0.1)"
                        icon_style = ""
                    else:
                        icon = "📋"
                        color = "#9c27b0"
                        activity_type = "general"
                        bg_color = "rgba(156, 39, 176, 0.05)"
                        icon_style = ""

                    # Enhanced connector line styling based on activity relationship
                    next_activity = (
                        activity_log[i + 1] if i + 1 < len(activity_log) else ""
                    )
                    is_same_group = (
                        ("agent" in activity_lower and "agent" in next_activity.lower())
                        or (
                            "searching" in activity_lower
                            and (
                                "found" in next_activity.lower()
                                or "reranking" in next_activity.lower()
                            )
                        )
                        or (
                            "found" in activity_lower
                            and "reranking" in next_activity.lower()
                        )
                        or (
                            "reranking" in activity_lower
                            and "analyzing" in next_activity.lower()
                        )
                    )

                    if i == len(activity_log) - 1:
                        connector_line = ""
                    elif is_same_group:
                        connector_line = f'<div style="width: 2px; height: 16px; background: {color}; opacity: 0.4; margin-top: 4px;"></div>'
                    else:
                        connector_line = '<div style="width: 2px; height: 16px; background: rgba(179, 27, 27, 0.15); margin-top: 4px;"></div>'

                    # Activity status styling
                    if is_latest and not (
                        "completed" in activity_lower or "complete" in activity_lower
                    ):
                        # Current active item
                        opacity = "1"
                        pulse_animation = (
                            "animation: subtle-pulse 2s ease-in-out infinite;"
                        )
                        text_weight = "font-weight: 500;"
                    elif "completed" in activity_lower or "complete" in activity_lower:
                        # Completed items
                        opacity = "0.9"
                        pulse_animation = ""
                        text_weight = "font-weight: 400;"
                    else:
                        # Older items
                        opacity = "0.75"
                        pulse_animation = ""
                        text_weight = "font-weight: 400;"

                    # Spacing based on activity type (phase transitions get more space)
                    margin_bottom = "0.4rem" if activity_type == "phase" else "0.1rem"

                    # Enhanced timeline entry with visual grouping
                    timeline_entry = (
                        f'<div style="display: flex; margin-bottom: {margin_bottom}; padding: 0.3rem 0.5rem; border-radius: 6px; background: {bg_color}; transition: all 0.2s ease;">'
                        '<div style="display: flex; flex-direction: column; align-items: center; margin-right: 0.75rem; min-width: 20px;">'
                        f'<div style="width: 20px; height: 20px; background: {color}; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 0.7rem; {icon_style} {pulse_animation} box-shadow: 0 1px 3px rgba(0,0,0,0.1);">{icon}</div>'
                        f"{connector_line}"
                        "</div>"
                        f'<div style="flex: 1; font-size: 0.8rem; color: var(--text-color); opacity: {opacity}; line-height: 1.3; padding-top: 2px; {text_weight}">{clean_activity}</div>'
                        "</div>"
                    )

                    # Add subtle pulse animation for active items
                    if is_latest and not (
                        "completed" in activity_lower or "complete" in activity_lower
                    ):
                        if i == 0:  # Only add the CSS once
                            timeline_entry = (
                                "<style>"
                                "@keyframes subtle-pulse {"
                                "0% { box-shadow: 0 1px 3px rgba(0,0,0,0.1); }"
                                "50% { box-shadow: 0 2px 8px rgba(0,0,0,0.15); }"
                                "100% { box-shadow: 0 1px 3px rgba(0,0,0,0.1); }"
                                "}"
                                "</style>" + timeline_entry
                            )

                    st.markdown(timeline_entry, unsafe_allow_html=True)

        # Compact stats column (right - narrower)
        with layout_cols[1]:
            # Compact vertical stats using app's design system
            stats_html = f"""
            <div class="stats-card" style="
                background: linear-gradient(180deg, var(--surface-light) 0%, var(--surface-light-alt) 100%);
                border: 1px solid rgba(179, 27, 27, 0.08);
                border-radius: var(--radius-lg);
                padding: var(--space-base);
                box-shadow: var(--shadow-sm);
                transition: all var(--transition-base);
                position: relative;
                overflow: hidden;
            ">
                <div style="margin-bottom: var(--space-lg);">
                    <div style="font-weight: 600; font-size: var(--font-size-sm); color: var(--text-color); margin-bottom: var(--space-xs);">💡 Insights</div>
                    <div style="font-size: var(--font-size-2xl); font-weight: 700; color: #673ab7;">{insights_found}</div>
                </div>
                <div>
                    <div style="font-weight: 600; font-size: var(--font-size-sm); color: var(--text-color); margin-bottom: var(--space-xs);">📄 Papers</div>
                    <div style="font-size: var(--font-size-2xl); font-weight: 700; color: #2196f3;">{papers_found}</div>
                </div>
            </div>
            
            <style>
            @media (prefers-color-scheme: dark) {{
                .stats-card {{
                    background: linear-gradient(180deg, var(--surface-dark) 0%, var(--surface-dark-alt) 100%) !important;
                    border-color: rgba(179, 27, 27, 0.15) !important;
                    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.2) !important;
                }}
            }}
            </style>
            """
            st.markdown(stats_html, unsafe_allow_html=True)

        # Add some breathing room at the bottom
        st.markdown(
            "<div style='margin-bottom: 0.5rem;'></div>", unsafe_allow_html=True
        )


def get_initial_query_value() -> str:
    """Handle initial query from session state."""
    initial_query_value = ""
    if "query_to_pass_to_chat" in st.session_state:
        initial_query_value = st.session_state.query_to_pass_to_chat
        ## Store in text area's session state to preserve through reruns
        st.session_state["chat_user_question_area"] = initial_query_value
        del st.session_state.query_to_pass_to_chat  # Clear after use
    return initial_query_value


def render_research_settings_panel() -> dict:
    """Render settings expander, return selected values."""
    with st.expander("⚙️ Response Settings", expanded=False):
        # First row - main settings
        settings_cols = st.columns(4)

        with settings_cols[0]:
            response_length = st.select_slider(
                "Response Length (words)",
                options=[250, 500, 1000, 3000],
                value=250,
                format_func=lambda x: f"~{x} words",
            )
        with settings_cols[1]:
            max_sources = st.select_slider(
                "Maximum Sources per Agent",
                options=[1, 5, 15, 30, 50],
                value=15,
            )
        with settings_cols[2]:
            max_agents = st.select_slider(
                "Research Agents",
                options=[1, 2, 3, 4, 5],
                value=1,
                help="Number of specialized agents to deploy for parallel research. More agents = more comprehensive but slower.",
            )
        with settings_cols[3]:
            llm_model = st.selectbox(
                "LLM Model",
                options=["gpt-4.1-mini", "gpt-4.1-nano", "gpt-4.1"],
                index=1,
                help="Model to use for research analysis and synthesis.",
            )

        # Second row - sources and options
        sources_cols = st.columns([2, 2])
        
        with sources_cols[0]:
            research_sources = st.multiselect(
                "Research Sources",
                options=["arxiv", "reddit"],
                default=["arxiv", "reddit"],
                help="Select which data sources to search. ArXiv provides academic papers, Reddit provides community discussions.",
            )
        
        with sources_cols[1]:
            show_only_sources = st.checkbox(
                "Show me only the sources",
                help="Skip generating a response and just show the most relevant papers for this query.",
            )

    return {
        "response_length": response_length,
        "max_sources": max_sources,
        "max_agents": max_agents,
        "llm_model": llm_model,
        "research_sources": research_sources,
        "show_only_sources": show_only_sources,
    }


def display_research_results(
    title: str,
    response: str,
    referenced_arxiv_codes: List[str],
    referenced_reddit_codes: List[str],
    additional_arxiv_codes: List[str],
    additional_reddit_codes: List[str],
    papers_df: pd.DataFrame,
):
    """Display research results with paper citations."""
    st.divider()
    st.markdown(f"#### {title}")
    
    # Only show response if not in sources-only mode
    if response != "Sources retrieved successfully. See referenced sources below.":
        st.markdown(response)

    # Use the already separated codes directly
    arxiv_codes = referenced_arxiv_codes

    # Extract Reddit citations with metadata from response text
    reddit_citations_data = extract_reddit_citations_with_metadata(response)
    
    if len(arxiv_codes) > 0 or reddit_citations_data:
        st.divider()

        # View selector for paper display format (only show if there are papers)
        if len(arxiv_codes) > 0:
            display_format = st.radio(
                "Display Format",
                options=["Grid View", "Citation List"],
                horizontal=True,
                label_visibility="collapsed",
                key="papers_display_format",
            )
        else:
            display_format = "Citation List"  # Default when no papers

        # Count sources for header
        total_sources = len(arxiv_codes) + len(reddit_citations_data)
        # Display arXiv papers if present
        if len(arxiv_codes) > 0:
            st.markdown("<h4>Referenced Papers:</h4>", unsafe_allow_html=True)
            # Get referenced papers
            reference_df = papers_df.loc[
                [c for c in arxiv_codes if c in papers_df.index]
            ]
            if not reference_df.empty:
                if display_format == "Grid View":
                    generate_grid_gallery(
                        reference_df,
                        n_cols=5,
                        extra_key="_chat",
                        image_type=st.session_state.global_image_type,
                    )
                else:
                    generate_citations_list(reference_df)

        # Display Reddit citations if present
        if reddit_citations_data:
            if len(arxiv_codes) > 0:  # Add divider only if papers were shown above
                st.divider()
            st.markdown("<h4>Referenced Discussions:</h4>", unsafe_allow_html=True)
            if display_format == "Grid View":
                generate_reddit_grid_gallery(reddit_citations_data)
            else:
                generate_reddit_citations_list(reddit_citations_data)

        # Display additional relevant papers
        valid_relevant_codes = [c for c in additional_arxiv_codes if c in papers_df.index]
        if len(valid_relevant_codes) > 0:
            st.divider()
            st.markdown("<h4>Other Relevant Papers:</h4>", unsafe_allow_html=True)
            # Filter out codes that don't exist in the dataframe to avoid KeyError
            if len(valid_relevant_codes) > 0:
                relevant_df = papers_df.loc[valid_relevant_codes]
                if display_format == "Grid View":
                    generate_grid_gallery(
                        relevant_df,
                        n_cols=5,
                        extra_key="_chat",
                        image_type=st.session_state.global_image_type,
                    )
                else:
                    generate_citations_list(relevant_df)


def get_reddit_citation_metadata(reddit_citations: List[str]) -> List[Dict]:
    """Fetch Reddit post metadata for citation display. Expects format: r/subreddit:post_id"""
    if not reddit_citations:
        return []

    from utils.db import db_utils

    metadata_list = []
    for citation in reddit_citations:
        # Parse subreddit:post_id format (r/ prefix already removed)
        subreddit, reddit_id = citation.split(":", 1)

        # Query database for Reddit post metadata  
        query = """
            SELECT reddit_id, subreddit, title, selftext as content, author, 
                   score, num_comments, post_timestamp as published_date, permalink
            FROM reddit_posts 
            WHERE reddit_id = :reddit_id AND subreddit = :subreddit 
            LIMIT 1
        """
        result = db_utils.execute_read_query(
            query, {"reddit_id": reddit_id, "subreddit": subreddit}
        )

        if not result.empty:
            row = result.iloc[0]
            metadata = {
                "reddit_id": row["reddit_id"],
                "subreddit": row["subreddit"],
                "title": row["title"],
                "content": row.get("content", "") or "",
                "author": row.get("author", "") or "",
                "score": int(row.get("score", 0)),
                "num_comments": int(row.get("num_comments", 0)),
                "published_date": row["published_date"],
                "permalink": row.get("permalink", 
                    f"https://www.reddit.com/r/{row['subreddit']}/comments/{row['reddit_id']}/"),
                "original_citation": citation,
            }
            metadata_list.append(metadata)

    return metadata_list


def extract_reddit_citations_with_metadata(response: str) -> List[Dict]:
    """Extract Reddit citations from response text and fetch their metadata."""
    from utils.app_utils import extract_reddit_codes

    # Extract Reddit citations using existing function
    reddit_codes = extract_reddit_codes(response)

    # Fetch metadata for each citation
    return get_reddit_citation_metadata(reddit_codes)


def generate_reddit_grid_gallery(reddit_citations: List[Dict], n_cols=5) -> None:
    """Create streamlit grid gallery of Reddit cards matching arXiv card style."""
    if not reddit_citations:
        return
    
    n_rows = int(np.ceil(len(reddit_citations) / n_cols))
    for i in range(n_rows):
        cols = st.columns(n_cols)
        for j in range(n_cols):
            if i * n_cols + j < len(reddit_citations):
                citation = reddit_citations[i * n_cols + j]
                
                # Extract Reddit information
                title = citation["title"]
                subreddit = citation["subreddit"]
                author = citation["author"]
                reddit_id = citation["reddit_id"]
                score = citation["score"]
                num_comments = citation["num_comments"]
                permalink = citation["permalink"]
                content = citation.get("content", "")
                
                # Format published date
                published_date = citation["published_date"]
                if published_date:
                    if hasattr(published_date, "strftime"):
                        date_str = published_date.strftime("%b %d, %Y")
                    else:
                        import pandas as pd
                        date_str = pd.to_datetime(published_date).strftime("%b %d, %Y")
                else:
                    date_str = "Unknown date"
                
                # Create content preview for back of card
                content_preview = "No preview available."
                if content and content.strip():
                    clean_content = content.strip()
                    if len(clean_content) > 150:
                        content_preview = clean_content[:150] + "..."
                    else:
                        content_preview = clean_content
                
                # Sanitize for HTML
                safe_title = html_escape(title)
                safe_content = html_escape(content_preview)
                
                with cols[j]:
                    # Reddit card with similar structure to arXiv cards
                    card_html = f"""
                    <div class="flip-card">
                      <div class="flip-card-inner">
                        <div class="flip-card-front" style="background: linear-gradient(135deg, #FF4500 0%, #FF6B35 100%); color: white;">
                          <div style="padding: var(--space-lg); height: 100%; display: flex; flex-direction: column; justify-content: center; align-items: center; text-align: center;">
                            <div style="font-size: 2rem; margin-bottom: var(--space-base);">💬</div>
                            <div style="font-size: var(--font-size-sm); opacity: 0.9; margin-bottom: var(--space-xs);">r/{subreddit}</div>
                            <div class="flip-title" style="color: white;">{safe_title}</div>
                          </div>
                        </div>
                        <div class="flip-card-back">
                          <div class="flip-card-back-content">{safe_content}</div>
                        </div>
                      </div>
                    </div>
                    """
                    st.markdown(card_html, unsafe_allow_html=True)
                    
                    # Metadata below card (similar to arXiv date/star display)
                    upvote_display = "⬆️" if score > 0 else ""
                    metadata_html = f"""
                    <div class="centered" style="text-align: center; font-size: var(--font-size-sm); margin-top: calc(-1 * var(--space-sm)); margin-bottom: var(--space-sm);">
                        <code>{upvote_display} {date_str}</code>
                    </div>
                    """
                    st.markdown(metadata_html, unsafe_allow_html=True)
                    
                    # Clickable link to Reddit post
                    st.markdown(
                        f'<div style="text-align: center; margin-top: var(--space-xs);"><a href="{permalink}" target="_blank" style="color: #FF4500; text-decoration: none; font-size: var(--font-size-xs);">View Discussion ↗</a></div>',
                        unsafe_allow_html=True
                    )


def generate_reddit_citations_list(reddit_citations: List[Dict]) -> None:
    """Generate formatted list of Reddit citations with rich styling."""
    if not reddit_citations:
        return

    for citation in reddit_citations:
        # Extract Reddit information
        title = citation["title"]
        subreddit = citation["subreddit"]
        author = citation["author"]
        reddit_id = citation["reddit_id"]
        score = citation["score"]
        num_comments = citation["num_comments"]
        permalink = citation["permalink"]
        content = citation.get("content", "")

        # Format published date
        published_date = citation["published_date"]
        if published_date:
            if hasattr(published_date, "strftime"):
                date_str = published_date.strftime("%b %d, %Y")
            else:
                import pandas as pd

                date_str = pd.to_datetime(published_date).strftime("%b %d, %Y")
        else:
            date_str = "Unknown date"

        # Create content preview (first 200 characters)
        content_preview = ""
        if content and content.strip():
            # Clean and truncate content
            clean_content = content.strip()
            if len(clean_content) > 200:
                content_preview = clean_content[:200] + "..."
            else:
                content_preview = clean_content

        # Create styled HTML similar to arXiv citations but with Reddit theme
        citation_html = f"""
        <div style="margin: var(--space-xl) 0; padding: var(--space-xl); border-radius: var(--radius-base); border-left: 4px solid #FF4500;">
            <div style="margin-bottom: var(--space-base);">
                <a href="{permalink}" target="_blank" style="color: #FF4500; text-decoration: none; font-size: var(--font-size-lg); font-weight: bold;">
                    {title}
                </a>
            </div>
            <div style="color: var(--text-color, #666); font-size: var(--font-size-sm); margin-bottom: var(--space-sm);">
                u/{author}
            </div>
            <div style="display: flex; gap: var(--space-base); margin-top: var(--space-sm); font-size: var(--font-size-sm); flex-wrap: wrap;">
                <span style="background-color: rgba(255, 69, 0, 0.05); padding: var(--space-xs) var(--space-sm); border-radius: var(--radius-sm);">
                    📅 {date_str}
                </span>
                <span style="background-color: rgba(255, 69, 0, 0.05); padding: var(--space-xs) var(--space-sm); border-radius: var(--radius-sm);">
                    r/{subreddit}
                </span>
                <span style="background-color: rgba(255, 69, 0, 0.05); padding: var(--space-xs) var(--space-sm); border-radius: var(--radius-sm);">
                    ⬆️ {score} upvotes
                </span>
                <span style="background-color: rgba(255, 69, 0, 0.05); padding: var(--space-xs) var(--space-sm); border-radius: var(--radius-sm);">
                    💬 {num_comments} comments
                </span>
                <a href="{permalink}" target="_blank" style="text-decoration: none;">
                    <span style="background-color: rgba(255, 69, 0, 0.05); padding: var(--space-xs) var(--space-sm); border-radius: var(--radius-sm);">
                        <span style="color: #FF4500;">🔗</span> r/{subreddit}:{reddit_id} <span style="font-size: var(--font-size-xs);">↗</span>
                    </span>
                </a>
            </div>"""

        # Add content preview if available
        if content_preview:
            citation_html += f"""
            <div style="margin-top: var(--space-base); font-style: italic; color: var(--text-color, #666); font-size: var(--font-size-sm);">
                {content_preview}
            </div>"""
        
        citation_html += """
        </div>
        """

        st.markdown(citation_html, unsafe_allow_html=True)


