import streamlit as st

def apply_arxiv_theme():
    """Apply the arXiv-themed styling to the Streamlit app."""
    st.markdown("""
        <style>
            /* Base theme colors */
            :root {
                --arxiv-red: #b31b1b;
                --arxiv-red-light: #c93232;
                --text-light: #fafafa;
                --text-dark: #000000;
            }

            /* Buttons */
            .stButton button {
                background-color: var(--arxiv-red) !important;
                color: var(--text-light) !important;
                border: none !important;
            }

            /* Links */
            a {
                color: var(--arxiv-red) !important;
                text-decoration: none;
            }
            a:hover {
                color: var(--arxiv-red-light) !important;
                text-decoration: underline;
            }

            /* Sliders and progress bars */
            .stSlider [aria-valuemax] {
                background-color: var(--arxiv-red) !important;
            }

            .stProgress > div > div > div > div {
                background-color: var(--arxiv-red) !important;
            }

            /* Tabs */
            .stTabs [data-baseweb="tab-list"] button[role="tab"][aria-selected="true"] {
                color: var(--arxiv-red) !important;
                border-bottom-color: var(--arxiv-red) !important;
            }
        </style>

        <script>
            // Function to set background color based on theme
            function setBackgroundColor() {
                const isDark = window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches;
                const containers = document.querySelectorAll('[data-testid="stAppViewContainer"], [data-testid="stSidebarContent"]');
                containers.forEach(container => {
                    container.style.backgroundColor = isDark ? '#0e1117' : '#ffffff';
                });
            }

            // Run on load
            setBackgroundColor();

            // Watch for theme changes
            window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', setBackgroundColor);
        </script>
    """, unsafe_allow_html=True)

def apply_custom_fonts():
    """Apply custom font styling."""
    st.markdown("""
        <style>
            @import 'https://fonts.googleapis.com/css2?family=Orbitron&display=swap';
            .pixel-font {
                font-family: 'Orbitron', sans-serif;
                font-size: 32px;
                margin-bottom: 1rem;
            }
        </style>
    """, unsafe_allow_html=True)

def apply_centered_style():
    """Apply centered styling for elements."""
    st.markdown("""
        <style>
            .centered {
                display: flex;
                justify-content: center;
                margin-bottom: 10px;
            }
        </style>
    """, unsafe_allow_html=True)