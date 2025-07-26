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
        horizontal=True
    )
    st.session_state.global_image_type = "artwork" if image_preference == "🎨 Art" else "first_page"

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
                image_url = f"https://arxiv-art.s3.us-west-2.amazonaws.com/{paper_code}.png"
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
        "🤖 Ask GPT Maestro",  # Enhanced title following current style
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
            st.markdown(
                "<div style='margin: 0.5em 0; padding: 0.8em; background: rgba(179, 27, 27, 0.03); border-radius: 8px; border-left: 3px solid rgba(179, 27, 27, 0.2);'>",
                unsafe_allow_html=True,
            )

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
            st.caption("💡 *Quick questions to get started*")
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
                    paper_question, paper_code, model="gpt-4.1-nano"
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
        if not pd.isna(paper["takeaway_title"]):
            st.markdown(f"#### {paper['takeaway_title']}")
        st.markdown(paper["takeaway_example"])

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
            similar_codes = pd.Series(papers_df.loc[paper_code]["similar_docs"])
            if pd.isna(similar_codes).any():
                st.write("Not available yet. Check back soon!")
            else:
                similar_codes = [d for d in similar_codes if d in papers_df.index]
                if len(similar_codes) > 5:
                    similar_codes = np.random.choice(similar_codes, 5, replace=False)
                similar_df = papers_df.loc[similar_codes]
                generate_grid_gallery(
                    similar_df, extra_key=f"_sim_{paper_code}", n_cols=5, image_type=st.session_state.global_image_type
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
                    image_url = f"https://arxiv-art.s3.us-west-2.amazonaws.com/{paper_code}.png"

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
    """Generate a formatted list of paper citations with rich styling."""
    for _, paper in df.iterrows():
        # Extract paper information
        title = paper["title"].replace("\n", "")
        authors = paper["authors"]
        paper_url = paper["url"]
        paper_code = paper["arxiv_code"]
        publish_date = pd.to_datetime(paper["published"]).strftime("%b %d, %Y")
        citation_count = int(paper.get("citation_count", 0))
        influential_count = int(paper.get("influential_citation_count", 0))
        punchline = paper.get("punchline", "")

        # Build HTML components separately
        star_badge = " ⭐️" if influential_count > 0 else ""
        citation_text = f"citation{'s' if citation_count != 1 else ''}"
        punchline_div = (
            f'<div style="margin-top: 12px; font-style: italic; color: var(--text-color, #666);">{punchline}</div>'
            if punchline
            else ""
        )

        citation_html = f"""
        <div style="margin: var(--space-xl) 0; padding: var(--space-xl); border-radius: var(--radius-base); border-left: 4px solid var(--arxiv-red);">
            <div style="margin-bottom: var(--space-base);">
                <span onclick="parent.postMessage({{cmd: 'streamlit:setComponentValue', args: {{value: '{paper_code}', dataType: 'str', key: 'arxiv_code'}}}}, '*')" style="color: var(--arxiv-red); text-decoration: none; font-size: var(--font-size-lg); font-weight: bold; cursor: pointer;">{title}</span>{star_badge}
            </div>
            <div style="color: var(--text-color, #666); font-size: var(--font-size-sm); margin-bottom: var(--space-sm);">
                {authors}
            </div>
            <div style="display: flex; gap: var(--space-base); margin-top: var(--space-sm); font-size: var(--font-size-sm);">
                <span style="background-color: rgba(179, 27, 27, 0.05); padding: var(--space-xs) var(--space-sm); border-radius: var(--radius-sm);">📅 {publish_date}</span>
                <span style="background-color: rgba(179, 27, 27, 0.05); padding: var(--space-xs) var(--space-sm); border-radius: var(--radius-sm);">📊 {citation_count} {citation_text}</span>
                <a href="{paper_url}" target="_blank" style="text-decoration: none;">
                    <span style="background-color: rgba(179, 27, 27, 0.05); padding: var(--space-xs) var(--space-sm); border-radius: var(--radius-sm);">
                        <span style="color: var(--arxiv-red);">📄</span> arXiv:{paper_code} <span style="font-size: var(--font-size-xs);">↗</span>
                    </span>
                </a>
            </div>
            {punchline_div}
        </div>
        """

        st.markdown(citation_html, unsafe_allow_html=True)

        # Hidden button to handle tab switching after state is set
        if paper_code == st.session_state.arxiv_code:
            click_tab(3)
            # st.session_state.pop("arxiv_code", None)  # Clear it after use


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


def display_paper_details_fragment(paper_code: str):
    """Displays the paper details card or an error message within a fragment."""
    st.session_state.details_canvas = st.session_state.details_canvas.empty()
    with st.session_state.details_canvas:
        if len(paper_code) > 0:
            # Access full papers data frame from session state
            if "papers" in st.session_state:
                full_papers_df = st.session_state.papers
                if paper_code in full_papers_df.index:
                    paper = full_papers_df.loc[paper_code].to_dict()
                    create_paper_card(paper, mode="open", name="_focus")
                else:
                    st.error("Paper not found.")
            else:
                # Handle case where papers haven't loaded yet (might happen on initial load)
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
        star = "⭐ " if row.get("influential_citation_count", 0) > 0 else ""
        return f"{star}{title}"

    # Create enhanced card layout
    for rank, (_, paper) in enumerate(display_df.iterrows(), 1):
        paper_code = paper["arxiv_code"]
        title = format_title(paper)
        metric_value = int(paper.get(metric_col, 0))
        punchline = paper.get("punchline", "")
        authors = paper.get("authors", "")
        paper_url = paper.get("url", "")

        # Truncate long author lists
        if len(authors) > 60:
            authors = authors[:60] + "..."

        # Image URL with fallback
        if st.session_state.global_image_type == "first_page":
            image_url = f"https://arxiv-first-page.s3.us-east-1.amazonaws.com/{paper_code}.png"
        else:
            image_url = f"https://arxiv-art.s3.amazonaws.com/{paper_code}.png"

        # Use columns to place buttons to the right of the card
        card_col, button_col = st.columns([10, 1])

        with card_col:
            # Create the enhanced card HTML
            card_html = f"""
            <div class="trending-card">
                <div class="trending-rank">{rank}</div>
                <div class="trending-header">
                    <div class="trending-image">
                        <img src="{image_url}" alt="{html_escape(title)}" 
                             onerror="this.style.display='none'; this.parentElement.style.backgroundColor='var(--secondary-background-color, #f0f0f0)'; this.parentElement.innerHTML='<div style=\\'display:flex;align-items:center;justify-content:center;height:100%;font-size:0.7em;color:var(--text-color,#999);\\'>No Image</div>';">
                    </div>
                    <div class="trending-content">
                        <div class="trending-title">
                            <a href="{paper_url}" target="_blank">{html_escape(title)}</a>
                        </div>
                        {f'<div class="trending-punchline">{html_escape(punchline)}</div>' if punchline and pd.notna(punchline) else ''}
                    </div>
                </div>
                <div class="trending-metadata">
                    <div class="trending-authors">{html_escape(authors)}</div>
                    <div class="trending-metric">
                        <span class="trending-metric-icon">{'📈' if metric_name == 'Likes' else '📊'}</span>
                        <span>{metric_value:,} {metric_name}</span>
                    </div>
                </div>
            </div>
            """
            st.markdown(card_html, unsafe_allow_html=True)

        # Buttons stacked vertically
        with button_col:
            # Vertical alignment hack for the button
            st.markdown(
                "<div style='height: calc(3 * var(--space-base));'></div>",
                unsafe_allow_html=True,
            )
            if st.button(
                "🔍", key=f"details_btn_{paper_code}_{extra_key}", help="Details"
            ):
                st.session_state.arxiv_code = paper_code
                click_tab(3)

            # Tweet toggle button (if enabled and tweets exist) - below the arrow button
            if (
                show_tweets_toggle
                and "tweets" in paper
                and paper["tweets"]
                and len(paper["tweets"]) > 0
            ):
                tweet_count = paper.get("tweet_count", len(paper["tweets"]))
                tweet_toggle_key = f"show_tweets_{paper_code}_{extra_key}"
                current_active_key = st.session_state.get("active_tweet_panel", None)

                if st.button(
                    f"🐦",
                    key=tweet_toggle_key,
                    help=f"View {tweet_count} tweet(s) for this paper",
                ):
                    # Toggle tweet display - only one panel open at a time
                    if current_active_key == tweet_toggle_key:
                        # Close the currently open panel
                        st.session_state["active_tweet_panel"] = None
                    else:
                        # Close any other panel and open this one
                        st.session_state["active_tweet_panel"] = tweet_toggle_key
                    st.rerun(scope="fragment")

        # Display tweets if expanded
        if (
            show_tweets_toggle
            and "tweets" in paper
            and paper["tweets"]
            and st.session_state.get("active_tweet_panel")
            == f"show_tweets_{paper_code}_{extra_key}"
        ):

            st.markdown(
                "<div style='margin-left: var(--space-base); margin-top: var(--space-sm);'>",
                unsafe_allow_html=True,
            )

            # Limit to top 8 tweets by likes and ensure they're properly parsed
            tweets_data = (
                paper["tweets"][:8] if len(paper["tweets"]) > 8 else paper["tweets"]
            )

            for tweet in tweets_data:
                display_individual_tweet(tweet, compact=True)

            st.markdown("</div>", unsafe_allow_html=True)

    # Summary information
    if len(df) > n:
        st.markdown(
            f'<div class="trending-summary">Showing top {n} of {len(df)} papers</div>',
            unsafe_allow_html=True,
        )


@st.fragment
def create_featured_paper_card(paper: Dict) -> None:
    """Display the weekly highlighted paper using a unified card design."""

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
        image_url = f"https://arxiv-first-page.s3.us-east-1.amazonaws.com/{paper_code}.png"
    else:
        image_url = f"https://arxiv-art.s3.us-west-2.amazonaws.com/{paper_code}.png"
    paper_url = paper.get("url", f"https://arxiv.org/abs/{paper_code}")

    # HTML-escape potentially unsafe content
    safe_title = html_escape(title)
    safe_punchline = html_escape(punchline)

    # Build HTML card using shared design tokens
    card_html = f"""
    <div class="featured-card">
        <div class="featured-image">
            <img src=\"{image_url}\" alt=\"{safe_title}\"
                 onerror=\"this.style.display='none'; this.parentElement.style.backgroundColor='var(--secondary-background-color, #f0f0f0)'; this.parentElement.innerHTML='<div style=\\'display:flex;align-items:center;justify-content:center;height:100%;font-size:0.7em;color:var(--text-color,#999);\\'>No Image</div>';\">
        </div>
        <div class="featured-content">
            <div class="featured-title"><a href=\"{paper_url}\" target=\"_blank\">{safe_title}</a></div>
            <div class="featured-punchline">{safe_punchline}</div>
        </div>
    </div>
    """

    st.markdown(card_html, unsafe_allow_html=True)

    btn_cols = st.columns([2, 2, 2])
    with btn_cols[1]:
        if st.button(
            "Read More", key=f"featured_details_{paper_code}", use_container_width=True
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
                    topic = topic_full[:30] + "..." if len(topic_full) > 30 else topic_full

        with cols[col_idx]:
            # Create a container with padding and subtle border
            with st.container():
                # Build topic HTML safely
                topic_html = ""
                if topic and topic_full:
                    topic_html = f"<span class='fact-topic' title='{topic_full}'>{topic}</span>"
                
                st.markdown(
                    f"""<div class="fact-card">
                    <div class="fact-content">{fact['fact']}</div>
                    <div class="fact-metadata">{topic_html}
                    <div class="fact-paper-link">
                        <a href="https://arxiv.org/abs/{fact['arxiv_code']}" target="_blank">
                            {fact['paper_title'][:75] + ('...' if len(fact['paper_title']) > 75 else '')}
                        </a>
                        </div>
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
                        '</div>'
                        f'<div style="flex: 1; font-size: 0.75rem; color: var(--text-color); opacity: 0.6; line-height: 1.3; padding-top: 2px; font-style: italic;">{earlier_count} earlier activities...</div>'
                        '</div>'
                    )
                    st.markdown(truncation_entry, unsafe_allow_html=True)
            
                # Timeline entries
                for i, activity in enumerate(activity_log):
                    # Clean up the activity message for display
                    clean_activity = (
                        activity.replace("🎯 ", "").replace("🤖 ", "").replace("📝 ", "")
                    )
                    if len(clean_activity) > 200:
                        clean_activity = clean_activity[:197] + "..."
                    
                    # Add minimal log-style prefix: step number from total count
                    step_number = total_activities - len(activity_log) + i + 1
                    clean_activity = f"#{step_number} {clean_activity}"
                    
                    # Determine activity type and icon
                    if "Phase" in activity:
                        icon = "🎯"
                        color = "#b31b1b"
                    elif "Agent" in activity:
                        icon = "🤖"
                        color = "#2196f3"
                    elif "complete" in activity.lower():
                        icon = "✅"
                        color = "#4caf50"
                    else:
                        icon = "📝"
                        color = "#9c27b0"
                    
                    # Calculate timeline position (newest at top)
                    is_latest = i == len(activity_log) - 1
                    
                    # Build timeline entry components separately
                    connector_line = '' if i == len(activity_log) - 1 else '<div style="width: 2px; height: 16px; background: rgba(179, 27, 27, 0.2); margin-top: 4px;"></div>'
                    opacity = '1' if is_latest else '0.8'
                    
                    # Simple timeline entry without nested quotes issues
                    timeline_entry = (
                        '<div style="display: flex; margin-bottom: 0rem;">'
                        '<div style="display: flex; flex-direction: column; align-items: center; margin-right: 0.75rem; min-width: 20px;">'
                        f'<div style="width: 20px; height: 20px; background: {color}; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 0.7rem;">{icon}</div>'
                        f'{connector_line}'
                        '</div>'
                        f'<div style="flex: 1; font-size: 0.8rem; color: var(--text-color); opacity: {opacity}; line-height: 1.3; padding-top: 2px;">{clean_activity}</div>'
                        '</div>'
                    )
                    st.caption(timeline_entry, unsafe_allow_html=True)
        
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
        del st.session_state.query_to_pass_to_chat  # Clear after use
    return initial_query_value


def render_research_settings_panel() -> dict:
    """Render settings expander, return selected values."""
    with st.expander("⚙️ Response Settings", expanded=False):
        settings_cols = st.columns(3)

        with settings_cols[0]:
            response_length = st.select_slider(
                "Response Length (words)",
                options=[250, 500, 1000, 3000],
                value=250,
                format_func=lambda x: f"~{x} words",
            )
        with settings_cols[1]:
            max_sources = st.select_slider(
                "Maximum Sources",
                options=[1, 5, 15, 30, 50],
                value=15,
            )
        with settings_cols[2]:
            max_agents = st.select_slider(
                "Research Agents",
                options=[1, 2, 3, 4, 5],
                value=3,
                help="Number of specialized agents to deploy for parallel research. More agents = more comprehensive but slower.",
            )

        show_only_sources = st.checkbox(
            "Show me only the sources",
            help="Skip generating a response and just show the most relevant papers for this query.",
        )

    return {
        "response_length": response_length,
        "max_sources": max_sources,
        "max_agents": max_agents,
        "show_only_sources": show_only_sources,
    }


def display_research_results(
    title: str,
    response: str,
    referenced_codes: List[str],
    relevant_codes: List[str],
    papers_df: pd.DataFrame,
):
    """Display research results with paper citations."""
    st.divider()
    st.markdown(f"#### {title}")
    st.markdown(response)

    if len(referenced_codes) > 0:
        st.divider()

        # View selector for paper display format
        display_format = st.radio(
            "Display Format",
            options=["Grid View", "Citation List"],
            horizontal=True,
            label_visibility="collapsed",
            key="papers_display_format",
        )

        st.markdown("<h4>Referenced Papers:</h4>", unsafe_allow_html=True)
        # Get referenced papers
        reference_df = papers_df.loc[[c for c in referenced_codes if c in papers_df.index]]
        if display_format == "Grid View":
            generate_grid_gallery(reference_df, n_cols=5, extra_key="_chat", image_type=st.session_state.global_image_type)
        else:
            generate_citations_list(reference_df)

        if len(relevant_codes) > 0:
            st.divider()
            st.markdown("<h4>Other Relevant Papers:</h4>", unsafe_allow_html=True)
            # Filter out codes that don't exist in the dataframe to avoid KeyError
            valid_relevant_codes = [c for c in relevant_codes if c in papers_df.index]
            if len(valid_relevant_codes) > 0:
                relevant_df = papers_df.loc[valid_relevant_codes]
                if display_format == "Grid View":
                    generate_grid_gallery(relevant_df, n_cols=5, extra_key="_chat", image_type=st.session_state.global_image_type)
                else:
                    generate_citations_list(relevant_df)


def inject_flip_card_css():
    """Injects CSS for the flip card effect."""
    # Flip card styles are now applied globally via apply_complete_app_styles()
