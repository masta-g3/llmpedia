import streamlit as st

# =============================================================================
# DESIGN TOKENS & THEME CONFIGURATION
# =============================================================================

def get_css_variables():
    """Define all CSS custom properties (design tokens) in one place."""
    return """
        :root {
            /* Brand Colors */
            --arxiv-red: #b31b1b;
            --arxiv-red-light: #c93232;
            --arxiv-red-dark: #8f1414;
            
            /* NEW – Surface tokens for subtle gradients */
            --surface-light: #ffffff;
            --surface-light-alt: #fafbfc;   /* just ~2-3% darker */
            --surface-dark: #0E1117;
            --surface-dark-alt: #13151b;    /* very subtle lift */
            
            /* Typography */
            --font-family-base: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
            --font-family-mono: 'Monaco', 'Menlo', 'Ubuntu Mono', monospace;
            --font-family-display: 'Orbitron', sans-serif;
            
            /* Font Sizes */
            --font-size-xs: 0.875rem;
            --font-size-sm: 0.95rem;
            --font-size-base: 1rem;
            --font-size-lg: 1.1rem;
            --font-size-xl: 1.4rem;
            --font-size-2xl: 1.7rem;
            --font-size-3xl: 2rem;
            
            /* Spacing Scale */
            --space-xs: 0.25rem;
            --space-sm: 0.5rem;
            --space-base: 1rem;
            --space-lg: 1.5rem;
            --space-xl: 2rem;
            --space-2xl: 3rem;
            
            /* Border Radius */
            --radius-sm: 4px;
            --radius-base: 8px;
            --radius-lg: 12px;
            --radius-full: 50%;
            
            /* Shadows */
            --shadow-sm: 0 2px 4px rgba(0, 0, 0, 0.04);
            --shadow-base: 0 4px 12px rgba(0, 0, 0, 0.08);
            --shadow-lg: 0 8px 25px rgba(0, 0, 0, 0.12);
            --shadow-color: rgba(179, 27, 27, 0.12);
            
            /* Transitions */
            --transition-fast: 0.15s ease;
            --transition-base: 0.3s ease;
            --transition-slow: 0.6s ease;
            
            /* Z-index Scale */
            --z-dropdown: 1000;
            --z-modal: 1050;
            --z-tooltip: 1100;
        }
    """

def get_base_component_styles():
    """Base styles for reusable component patterns."""
    return """
        /* =============================================================================
           CARD COMPONENTS
           ============================================================================= */
        
        .card-base {
            background: linear-gradient(180deg, var(--surface-light) 0%, var(--surface-light-alt) 100%);
            border: 1px solid rgba(179, 27, 27, 0.08);
            border-radius: var(--radius-lg);
            padding: var(--space-base);
            margin-bottom: var(--space-base);
            transition: all var(--transition-base);
            position: relative;
            overflow: hidden;
            box-shadow: var(--shadow-sm);
        }
        
        .card-base:hover {
            transform: translateY(-2px);
            box-shadow: var(--shadow-lg);
            border-color: rgba(179, 27, 27, 0.2);
        }
        
        .card-base::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 3px;
            background: linear-gradient(90deg, var(--arxiv-red) 0%, var(--arxiv-red-light) 100%);
            opacity: 0;
            transition: opacity var(--transition-base);
        }
        
        .card-base:hover::before {
            opacity: 1;
        }
        
        /* =============================================================================
           BUTTON COMPONENTS
           ============================================================================= */
        
        .btn-primary {
            background: linear-gradient(135deg, var(--arxiv-red) 0%, var(--arxiv-red-light) 100%);
            color: white;
            border: none;
            padding: var(--space-sm) var(--space-base);
            border-radius: var(--radius-base);
            font-size: var(--font-size-sm);
            font-weight: 500;
            cursor: pointer;
            transition: all var(--transition-fast);
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        
        .btn-primary:hover {
            transform: translateY(-1px);
            box-shadow: 0 4px 12px rgba(179, 27, 27, 0.3);
        }
        
        .btn-secondary {
            background: transparent;
            color: var(--arxiv-red);
            border: 1px solid var(--arxiv-red);
            padding: var(--space-sm) var(--space-base);
            border-radius: var(--radius-base);
            font-size: var(--font-size-sm);
            font-weight: 500;
            cursor: pointer;
            transition: all var(--transition-fast);
        }
        
        .btn-secondary:hover {
            background: var(--arxiv-red);
            color: white;
        }
        
        /* =============================================================================
           TYPOGRAPHY COMPONENTS
           ============================================================================= */
        
        .heading-primary {
            font-family: var(--font-family-display);
            font-size: var(--font-size-3xl);
            font-weight: 600;
            margin-bottom: var(--space-lg);
            color: var(--text-color);
        }
        
        .heading-secondary {
            font-size: var(--font-size-xl);
            font-weight: 600;
            margin-bottom: var(--space-base);
            color: var(--text-color);
        }
        
        .text-muted {
            color: var(--text-color, #666);
            opacity: 0.8;
        }
        
        .text-small {
            font-size: var(--font-size-sm);
        }
        
        .pixel-font {
            font-family: var(--font-family-display);
            font-size: var(--font-size-3xl);
            margin-bottom: var(--space-base);
        }
        
        /* =============================================================================
           LAYOUT UTILITIES
           ============================================================================= */
        
        .container-centered {
            display: flex;
            justify-content: center;
            margin-bottom: var(--space-base);
        }
        
        .centered {
            display: flex;
            justify-content: center;
            margin-bottom: var(--space-sm);
        }
        
        .flex-between {
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        
        .flex-center {
            display: flex;
            align-items: center;
            justify-content: center;
        }
        
        .grid-auto-fit {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: var(--space-base);
        }
        
        /* =============================================================================
           METRIC COMPONENTS
           ============================================================================= */
        
        .metric-enhanced {
            background: linear-gradient(180deg, var(--surface-light) 0%, var(--surface-light-alt) 100%);
            border: 1px solid rgba(179, 27, 27, 0.06);
            border-radius: var(--radius-base);
            padding: var(--space-base);
            transition: all var(--transition-fast);
            position: relative;
            overflow: hidden;
        }
        
        .metric-enhanced:hover {
            border-color: rgba(179, 27, 27, 0.12);
            box-shadow: var(--shadow-sm);
        }
        
        .metric-enhanced::before {
            content: '';
            position: absolute;
            bottom: 0;
            left: 0;
            right: 0;
            height: 2px;
            background: linear-gradient(90deg, var(--arxiv-red) 0%, var(--arxiv-red-light) 100%);
            opacity: 0;
            transition: opacity var(--transition-fast);
        }
        
        .metric-enhanced:hover::before {
            opacity: 1;
        }
        
        /* =============================================================================
           RESPONSIVE BREAKPOINTS
           ============================================================================= */
        
        @media (max-width: 768px) {
            .card-base {
                padding: var(--space-sm);
                margin-bottom: var(--space-sm);
            }
            
            .grid-auto-fit {
                grid-template-columns: 1fr;
                gap: var(--space-sm);
            }
            
            .metric-enhanced {
                padding: var(--space-sm);
            }
        }
        
        /* =============================================================================
           DARK MODE ADAPTATIONS
           ============================================================================= */
        
        @media (prefers-color-scheme: dark) {
            .card-base {
                background: linear-gradient(180deg, var(--surface-dark) 0%, var(--surface-dark-alt) 100%);
                border-color: rgba(179, 27, 27, 0.15);
                box-shadow: 0 2px 8px rgba(0, 0, 0, 0.2);
            }
            
            .card-base:hover {
                box-shadow: 0 8px 25px rgba(179, 27, 27, 0.25);
                border-color: rgba(179, 27, 27, 0.3);
            }
            
            .heading-primary,
            .heading-secondary {
                color: var(--text-color, #FAFAFA);
            }
            
            .text-muted {
                color: var(--text-color, #CCCCCC);
            }
        }
    """

# =============================================================================
# COMPONENT-SPECIFIC STYLE GENERATORS
# =============================================================================

def get_flip_card_styles():
    """Generate CSS for flip card components."""
    return """
        .flip-card {
            background-color: transparent;
            width: 100%;
            height: 450px;
            perspective: 1000px;
            margin-bottom: var(--space-base);
        }
        
        .flip-card-inner {
            position: relative;
            width: 100%;
            height: 100%;
            text-align: center;
            transition: transform var(--transition-slow);
            transform-style: preserve-3d;
        }
        
        .flip-card:hover .flip-card-inner {
            transform: rotateY(180deg);
        }
        
        .flip-card-front,
        .flip-card-back {
            position: absolute;
            width: 100%;
            height: 100%;
            -webkit-backface-visibility: hidden;
            backface-visibility: hidden;
            border-radius: var(--radius-base);
            overflow: hidden;
            display: flex;
            flex-direction: column;
            justify-content: space-between;
        }
        
        .flip-card-front {
            background-color: var(--secondary-background-color, #fafafa);
        }
        
        .flip-card-back {
            background-color: var(--background-color, #fff);
            color: var(--text-color, #333);
            transform: rotateY(180deg);
            padding: var(--space-base);
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
        }
        
        .flip-card-front img {
            width: 100%;
            height: 80%;
            object-fit: cover;
        }
        
        .flip-title {
            font-weight: 600;
            font-size: var(--font-size-base);
            color: var(--arxiv-red);
            padding: var(--space-sm);
            text-align: center;
            height: 20%;
            display: flex;
            align-items: center;
            justify-content: center;
            overflow: hidden;
            text-overflow: ellipsis;
        }
        
        .flip-card-back-content {
            font-size: var(--font-size-sm);
            line-height: 1.4;
            margin-bottom: var(--space-base);
            max-height: 80%;
            overflow-y: auto;
            color: var(--text-color, #333);
        }
        
        .flip-card-image-error-text {
            font-size: var(--font-size-sm);
            color: var(--text-color, #555555);
            opacity: 0.7;
            padding: var(--space-base);
            box-sizing: border-box;
            width: 100%;
            height: 100%;
            display: flex;
            align-items: center;
            justify-content: center;
        }
        
        @media (prefers-color-scheme: dark) {
            .flip-card-front {
                background-color: var(--secondary-background-color, #262730);
            }
            
            .flip-card-back {
                background-color: var(--background-color, #0E1117);
                color: var(--text-color, #FAFAFA);
            }
            
            .flip-card-back-content {
                color: var(--text-color, #FAFAFA);
            }
            
            .flip-card-image-error-text {
                color: var(--text-color, #AAAAAA);
            }
        }
    """

def generate_table_styles():
    """Generate CSS for table components."""
    return """
        .paper-header {
            display: flex;
            gap: var(--space-sm);
            font-weight: 600;
            border-bottom: 2px solid var(--secondary-background-color, rgba(179, 27, 27, 0.3));
            padding-bottom: var(--space-sm);
            margin-bottom: var(--space-sm);
            font-size: var(--font-size-base);
        }
        
        .paper-row {
            border-bottom: 1px solid var(--secondary-background-color, rgba(128, 128, 128, 0.2));
            padding: var(--space-base) 0;
            margin-bottom: var(--space-xs);
            transition: background-color var(--transition-fast);
        }
        
        .paper-row:nth-child(odd) {
            background-color: var(--secondary-background-color, rgba(128, 128, 128, 0.04));
        }
        
        .paper-row:hover {
            background-color: var(--secondary-background-color, rgba(179, 27, 27, 0.06));
            border-radius: var(--radius-sm);
        }
        
        .title-link {
            font-weight: 600;
            text-decoration: none;
            display: inline-block;
            margin-bottom: 2px;
            line-height: 1.3;
            color: var(--arxiv-red);
        }
        
        .title-link:hover {
            text-decoration: underline;
            color: var(--arxiv-red-light);
        }
        
        .paper-cell {
            padding: 2px 0;
            line-height: 1.4;
        }
        
        @media (prefers-color-scheme: dark) {
            .paper-row:nth-child(odd) {
                background-color: var(--secondary-background-color, rgba(128, 128, 128, 0.08));
            }
            
            .paper-row:hover {
                background-color: var(--secondary-background-color, rgba(179, 27, 27, 0.14));
            }
            
            .paper-header {
                border-bottom-color: var(--secondary-background-color, rgba(179, 27, 27, 0.4));
            }
        }
    """

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def generate_metric_badge_style(metric_type: str = "default"):
    """Generate CSS for metric badges with different styles."""
    color_map = {
        "citations": "var(--arxiv-red)",
        "trending": "#28a745",
        "likes": "#1da1f2",
        "default": "var(--arxiv-red)"
    }
    
    color = color_map.get(metric_type, color_map["default"])
    
    return f"""
        .metric-badge-{metric_type} {{
            display: inline-flex;
            align-items: center;
            gap: var(--space-xs);
            background: rgba({color.replace('var(--arxiv-red)', '179, 27, 27')}, 0.1);
            padding: var(--space-xs) var(--space-sm);
            border-radius: var(--radius-full);
            font-size: var(--font-size-sm);
            font-weight: 600;
            color: {color};
            white-space: nowrap;
        }}
    """

def apply_design_system():
    """Apply the complete design system to the Streamlit app."""
    css = f"""
        <style>
            {get_css_variables()}
            {get_base_component_styles()}
        </style>
    """
    st.markdown(css, unsafe_allow_html=True)

def apply_complete_app_styles():
    """Single point of truth for all app styling - Master function."""
    css = f"""
        <style>
            @import 'https://fonts.googleapis.com/css2?family=Orbitron&display=swap';
            
            {get_css_variables()}
            {get_base_component_styles()}
            {get_flip_card_styles()}
            {generate_table_styles()}
            {get_advanced_trending_card_styles()}
            {get_individual_tweet_card_styles()}
            {get_interesting_facts_styles()}
            {get_tweet_timeline_styles()}
            {get_featured_card_styles()}
            {get_trending_panel_styles()}
            {get_sidebar_footer_styles()}
            {get_markdown_viewer_styles()}
            {get_streamlit_overrides()}
        </style>
        
        <script>
            // Function to set background color based on theme
            function setBackgroundColor() {{
                const isDark = window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches;
                const containers = document.querySelectorAll('[data-testid="stAppViewContainer"], [data-testid="stSidebarContent"]');
                containers.forEach(container => {{
                    container.style.backgroundColor = isDark ? '#0e1117' : '#ffffff';
                }});
            }}

            // Run on load
            setBackgroundColor();

            // Watch for theme changes
            window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', setBackgroundColor);
        </script>
    """
    st.markdown(css, unsafe_allow_html=True)

# =============================================================================
# EXTRACTED COMPONENT STYLES (FROM APP.PY AND STREAMLIT_UTILS.PY)
# =============================================================================

def get_trending_panel_styles():
    """Generate CSS for trending panel headers (extracted from app.py)."""
    return """
        .trending-panel-header {
            background: linear-gradient(180deg, var(--surface-light) 0%, var(--surface-light-alt) 100%);
            padding: var(--space-base) var(--space-lg);
            border-radius: var(--radius-lg);
            border: 1px solid rgba(179, 27, 27, 0.08);
            margin-bottom: var(--space-lg);
            position: relative;
            overflow: hidden;
        }
        
        .trending-panel-header::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 3px;
            background: linear-gradient(90deg, var(--arxiv-red) 0%, var(--arxiv-red-light) 100%);
        }
        
        .trending-panel-title {
            font-size: var(--font-size-xl);
            font-weight: 600;
            color: var(--text-color, #333);
            margin: 0 0 var(--space-sm) 0;
            display: flex;
            align-items: center;
            gap: var(--space-sm);
        }
        
        .trending-panel-subtitle {
            font-size: var(--font-size-sm);
            color: var(--text-color, #666);
            margin: 0;
            opacity: 0.8;
        }
        
        @media (prefers-color-scheme: dark) {
            .trending-panel-header {
                background: linear-gradient(180deg, var(--surface-dark) 0%, var(--surface-dark-alt) 100%);
                border-color: rgba(179, 27, 27, 0.15);
            }
            
            .trending-panel-title {
                color: var(--text-color, #FAFAFA);
            }
            
            .trending-panel-subtitle {
                color: var(--text-color, #CCCCCC);
            }
        }
    """

def get_sidebar_footer_styles():
    """Generate CSS for sidebar footer styling (extracted from app.py)."""
    return """
        .reportview-container .main footer {
            visibility: hidden;
        }
        
        .llmp-sidebar-footer {
            position: fixed;
            bottom: 0;
            width: 0%;
            text-align: center;
            color: var(--text-color, #888);
            font-size: var(--font-size-xs);
        }
        
        .llmp-sidebar-footer a {
            color: inherit;
            text-decoration: none;
            transition: color var(--transition-fast);
        }
        
        .llmp-sidebar-footer a:hover {
            color: var(--arxiv-red);
        }
        
        .llmp-acknowledgment {
            font-size: var(--font-size-xs);
            font-style: italic;
            text-align: center;
            position: relative;
            top: var(--space-lg);
            color: var(--text-color, #888);
        }
    """

def get_markdown_viewer_styles():
    """Generate CSS for markdown content viewer (extracted from streamlit_utils.py)."""
    return """
        .markdown-body {
            box-sizing: border-box;
            min-width: 200px;
            max-width: 100%;
            margin: 0 auto;
            padding: var(--space-base);
            font-family: var(--font-family-base);
        }
        
        .markdown-body img {
            max-width: 100%;
            height: auto;
            display: block;
            margin: var(--space-lg) auto;
            border-radius: var(--radius-sm);
        }
        
        .markdown-body pre {
            background-color: var(--secondary-background-color, #f6f8fa);
            border-radius: var(--radius-base);
            padding: var(--space-base);
            overflow: auto;
        }
        
        .markdown-body code {
            background-color: rgba(175, 184, 193, 0.2);
            padding: 0.2em 0.4em;
            border-radius: var(--radius-base);
        }
        
        .markdown-body pre code {
            background-color: transparent;
            padding: 0;
        }
        
        @media (prefers-color-scheme: dark) {
            .markdown-body pre {
                background-color: var(--secondary-background-color, #262730);
            }
            
            .markdown-body code {
                background-color: rgba(128, 128, 128, 0.2);
            }
        }
    """

def get_streamlit_overrides():
    """Generate CSS overrides for Streamlit default components."""
    return """
        /* Streamlit Button Overrides */
        .stButton button {
            background-color: var(--arxiv-red) !important;
            color: white !important;
            border: none !important;
            border-radius: var(--radius-base) !important;
            transition: all var(--transition-fast) !important;
        }
        
        .stButton button:hover {
            background-color: var(--arxiv-red-light) !important;
            transform: translateY(-1px) !important;
        }

        /* Streamlit Link Overrides */
        a {
            color: var(--arxiv-red) !important;
            text-decoration: none;
            transition: color var(--transition-fast);
        }
        
        a:hover {
            color: var(--arxiv-red-light) !important;
            text-decoration: underline;
        }

        /* Streamlit Slider Overrides */
        .stSlider [aria-valuemax] {
            background-color: var(--arxiv-red) !important;
        }

        /* Streamlit Progress Bar Overrides */
        .stProgress > div > div > div > div {
            background-color: var(--arxiv-red) !important;
        }

        /* Streamlit Tabs Overrides */
        .stTabs [data-baseweb="tab-list"] button[role="tab"][aria-selected="true"] {
            color: var(--arxiv-red) !important;
            border-bottom-color: var(--arxiv-red) !important;
        }

        /* Streamlit Metric Overrides */
        .stMetric {
            background: linear-gradient(180deg, var(--surface-light) 0%, var(--surface-light-alt) 100%);
            border: 1px solid rgba(179, 27, 27, 0.06);
            border-radius: var(--radius-base);
            padding: var(--space-base);
            transition: all var(--transition-fast);
            position: relative;
            overflow: hidden;
        }
        
        .stMetric:hover {
            border-color: rgba(179, 27, 27, 0.12);
            box-shadow: var(--shadow-sm);
        }
        
        .stMetric::before {
            content: '';
            position: absolute;
            bottom: 0;
            left: 0;
            right: 0;
            height: 2px;
            background: linear-gradient(90deg, var(--arxiv-red) 0%, var(--arxiv-red-light) 100%);
            opacity: 0;
            transition: opacity var(--transition-fast);
        }
        
        .stMetric:hover::before {
            opacity: 1;
        }
        
        @media (prefers-color-scheme: dark) {
            .stMetric {
                background: linear-gradient(180deg, var(--surface-dark) 0%, var(--surface-dark-alt) 100%);
                border-color: rgba(179, 27, 27, 0.15);
                box-shadow: 0 2px 8px rgba(0, 0, 0, 0.2);
            }
            
            .stMetric:hover {
                border-color: rgba(179, 27, 27, 0.3);
                box-shadow: 0 8px 25px rgba(179, 27, 27, 0.25);
            }
        }
    """

# =============================================================================
# LEGACY FUNCTIONS (MAINTAINED FOR COMPATIBILITY)
# =============================================================================

def apply_arxiv_theme():
    """Apply the arXiv-themed styling to the Streamlit app."""
    st.markdown(f"""
        <style>
            {get_css_variables()}

            /* Buttons */
            .stButton button {{
                background-color: var(--arxiv-red) !important;
                color: white !important;
                border: none !important;
            }}

            /* Links */
            a {{
                color: var(--arxiv-red) !important;
                text-decoration: none;
            }}
            a:hover {{
                color: var(--arxiv-red-light) !important;
                text-decoration: underline;
            }}

            /* Sliders and progress bars */
            .stSlider [aria-valuemax] {{
                background-color: var(--arxiv-red) !important;
            }}

            .stProgress > div > div > div > div {{
                background-color: var(--arxiv-red) !important;
            }}

            /* Tabs */
            .stTabs [data-baseweb="tab-list"] button[role="tab"][aria-selected="true"] {{
                color: var(--arxiv-red) !important;
                border-bottom-color: var(--arxiv-red) !important;
            }}
        </style>

        <script>
            // Function to set background color based on theme
            function setBackgroundColor() {{
                const isDark = window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches;
                const containers = document.querySelectorAll('[data-testid="stAppViewContainer"], [data-testid="stSidebarContent"]');
                containers.forEach(container => {{
                    container.style.backgroundColor = isDark ? '#0e1117' : '#ffffff';
                }});
            }}

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
                font-family: var(--font-family-display);
                font-size: var(--font-size-3xl);
                margin-bottom: var(--space-base);
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
                margin-bottom: var(--space-base);
            }
        </style>
    """, unsafe_allow_html=True)

# =============================================================================
# COMPONENT-SPECIFIC STYLE INJECTORS
# =============================================================================

def inject_flip_card_css():
    """Inject CSS for flip card components."""
    st.markdown(f"""
        <style>
            {get_flip_card_styles()}
        </style>
    """, unsafe_allow_html=True)

def inject_table_css():
    """Inject CSS for table components."""
    st.markdown(f"""
        <style>
            {generate_table_styles()}
        </style>
    """, unsafe_allow_html=True)

def get_advanced_trending_card_styles():
    """Generate comprehensive CSS for trending card components."""
    return """
        .trending-container {
            padding: 0;
            margin: 0;
        }
        
        .trending-card {
            background: linear-gradient(180deg, var(--surface-light) 0%, var(--surface-light-alt) 100%);
            border: 1px solid rgba(179, 27, 27, 0.08);
            border-radius: var(--radius-lg);
            padding: var(--space-base);
            margin-bottom: var(--space-base);
            transition: all var(--transition-base);
            position: relative;
            overflow: hidden;
            box-shadow: var(--shadow-sm);
        }
        
        .trending-card:hover {
            transform: translateY(-2px);
            box-shadow: var(--shadow-lg);
            border-color: rgba(179, 27, 27, 0.2);
        }
        
        .trending-card::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 3px;
            background: linear-gradient(90deg, var(--arxiv-red) 0%, var(--arxiv-red-light) 100%);
            opacity: 0;
            transition: opacity var(--transition-base);
        }
        
        .trending-card:hover::before {
            opacity: 1;
        }
        
        .trending-header {
            display: flex;
            align-items: flex-start;
            gap: var(--space-base);
            margin-bottom: var(--space-base);
        }
        
        .trending-image {
            flex-shrink: 0;
            width: 80px;
            height: 80px;
            border-radius: var(--radius-base);
            overflow: hidden;
            background: var(--secondary-background-color, #f0f0f0);
            position: relative;
        }
        
        .trending-image img {
            width: 100%;
            height: 100%;
            object-fit: cover;
            transition: transform var(--transition-base);
        }
        
        .trending-card:hover .trending-image img {
            transform: scale(1.05);
        }
        
        .trending-content {
            flex: 1;
            min-width: 0;
        }
        
        .trending-title {
            font-size: var(--font-size-base);
            font-weight: 600;
            line-height: 1.3;
            margin: 0 0 var(--space-xs) 0;
            color: var(--text-color, #333);
        }
        
        .trending-title a {
            color: var(--arxiv-red);
            text-decoration: none;
            transition: color var(--transition-fast);
        }
        
        .trending-title a:hover {
            color: var(--arxiv-red-light);
            text-decoration: underline;
        }
        
        .trending-punchline {
            font-size: var(--font-size-sm);
            color: var(--text-color, #666);
            line-height: 1.4;
            margin: 0 0 var(--space-sm) 0;
            font-style: italic;
            opacity: 0.9;
        }
        
        .trending-metadata {
            display: flex;
            align-items: center;
            gap: var(--space-base);
            margin-top: var(--space-base);
            padding-top: var(--space-sm);
            border-top: 1px solid rgba(128, 128, 128, 0.1);
            flex-wrap: wrap;
        }
        
        .trending-authors {
            font-size: var(--font-size-xs);
            color: var(--text-color, #888);
            flex: 1;
            min-width: 0;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }
        
        .trending-metric {
            display: flex;
            align-items: center;
            gap: var(--space-xs);
            background: var(--secondary-background-color, rgba(179, 27, 27, 0.05));
            padding: var(--space-xs) var(--space-sm);
            border-radius: var(--radius-full);
            font-size: var(--font-size-sm);
            font-weight: 600;
            color: var(--arxiv-red);
            white-space: nowrap;
        }
        
        .trending-metric-icon {
            font-size: 0.9em;
            opacity: 0.8;
        }
        
        .trending-actions {
            display: flex;
            justify-content: flex-end;
            margin-top: var(--space-sm);
        }
        
        .trending-read-btn {
            background: linear-gradient(135deg, var(--arxiv-red) 0%, var(--arxiv-red-light) 100%);
            color: white;
            border: none;
            padding: var(--space-sm) var(--space-base);
            border-radius: var(--radius-full);
            font-size: var(--font-size-sm);
            font-weight: 500;
            cursor: pointer;
            transition: all var(--transition-fast);
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        
        .trending-read-btn:hover {
            transform: translateY(-1px);
            box-shadow: 0 4px 12px rgba(179, 27, 27, 0.3);
        }
        
        .trending-rank {
            position: absolute;
            top: var(--space-base);
            right: var(--space-base);
            background: linear-gradient(135deg, var(--arxiv-red) 0%, var(--arxiv-red-light) 100%);
            color: white;
            width: 24px;
            height: 24px;
            border-radius: var(--radius-full);
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: var(--font-size-xs);
            font-weight: bold;
        }
        
        .trending-summary {
            margin-top: var(--space-base);
            padding-top: var(--space-base);
            border-top: 1px solid rgba(128, 128, 128, 0.1);
            text-align: center;
            font-size: var(--font-size-xs);
            color: var(--text-color, #888);
            font-style: italic;
        }

        /* Dark mode adaptations */
        @media (prefers-color-scheme: dark) {
            .trending-card {
                background: linear-gradient(180deg, var(--surface-dark) 0%, var(--surface-dark-alt) 100%);
                border-color: rgba(179, 27, 27, 0.15);
                box-shadow: 0 2px 8px rgba(0, 0, 0, 0.2);
            }
            
            .trending-card:hover {
                box-shadow: 0 8px 25px rgba(179, 27, 27, 0.25);
                border-color: rgba(179, 27, 27, 0.3);
            }
            
            .trending-image {
                background: var(--secondary-background-color, #262730);
            }
            
            .trending-title {
                color: var(--text-color, #FAFAFA);
            }
            
            .trending-punchline {
                color: var(--text-color, #CCCCCC);
            }
            
            .trending-authors {
                color: var(--text-color, #AAAAAA);
            }
            
            .trending-metric {
                background: rgba(179, 27, 27, 0.15);
                color: var(--arxiv-red-light);
            }
            
            .trending-summary {
                color: var(--text-color, #AAAAAA);
            }
            
            .stMetric {
                background: linear-gradient(180deg, var(--surface-dark) 0%, var(--surface-dark-alt) 100%);
                border-color: rgba(179, 27, 27, 0.08);
            }
            
            .stMetric:hover {
                border-color: rgba(179, 27, 27, 0.15);
                box-shadow: 0 2px 8px rgba(179, 27, 27, 0.15);
            }
        }
        
        /* Mobile responsiveness */
        @media (max-width: 768px) {
            .trending-card {
                padding: var(--space-sm);
                margin-bottom: var(--space-sm);
            }
            
            .trending-header {
                gap: var(--space-sm);
            }
            
            .trending-image {
                width: 60px;
                height: 60px;
            }
            
            .trending-title {
                font-size: var(--font-size-sm);
            }
            
            .trending-metadata {
                gap: var(--space-sm);
                flex-direction: column;
                align-items: flex-start;
            }
            
            .trending-authors {
                white-space: normal;
                overflow: visible;
                text-overflow: unset;
            }
        }
    """

def get_interesting_facts_styles():
    """Generate CSS for interesting facts components."""
    return """
        .fact-card {
            padding: var(--space-base);
            margin-bottom: var(--space-base);
            background: linear-gradient(180deg, var(--surface-light) 0%, var(--surface-light-alt) 100%);
            border-left: 3px solid var(--arxiv-red);
            border-radius: var(--radius-sm);
            transition: all var(--transition-fast);
        }
        
        .fact-card:hover {
            transform: translateY(-1px);
            box-shadow: var(--shadow-base);
        }
        
        .fact-content {
            font-size: var(--font-size-base);
            margin-bottom: var(--space-sm);
            line-height: 1.5;
        }
        
        .fact-metadata {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-top: var(--space-base);
            gap: var(--space-sm);
        }
        
        .fact-topic {
            background-color: var(--secondary-background-color, rgba(128, 128, 128, 0.1));
            padding: var(--space-xs) var(--space-sm);
            border-radius: var(--radius-lg);
            font-size: var(--font-size-xs);
            white-space: nowrap;
        }
        
        .fact-paper-link {
            font-size: var(--font-size-xs);
            font-style: italic;
            color: var(--text-color, #888);
            margin: 0;
            text-align: right;
            flex-grow: 1;
        }
        
        .fact-paper-link a {
            text-decoration: none;
            color: var(--arxiv-red);
            transition: color var(--transition-fast);
        }
        
        .fact-paper-link a:hover {
            color: var(--arxiv-red-light);
            text-decoration: underline;
        }
        
        @media (prefers-color-scheme: dark) {
            .fact-card {
                background: linear-gradient(180deg, var(--surface-dark) 0%, rgba(19, 21, 27, 0.72) 100%);
                border-left-color: var(--arxiv-red);
            }
            
            .fact-topic {
                background-color: var(--secondary-background-color, rgba(128, 128, 128, 0.2));
            }
            
            .fact-paper-link {
                color: var(--text-color, #CCCCCC);
            }
        }
    """


def get_tweet_timeline_styles():
    """Generate CSS for X.com discussions timeline carousel."""
    return """
        .tweet-timeline-container {
            position: relative;
            margin-bottom: var(--space-lg);
        }
        
        .tweet-timeline-header {
            background: linear-gradient(180deg, var(--surface-light) 0%, var(--surface-light-alt) 100%);
            padding: var(--space-base) var(--space-lg);
            border-radius: var(--radius-lg);
            border: 1px solid rgba(179, 27, 27, 0.08);
            margin-bottom: var(--space-lg);
            position: relative;
            overflow: hidden;
        }
        
        .tweet-timeline-header::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 3px;
            background: linear-gradient(90deg, var(--arxiv-red) 0%, var(--arxiv-red-light) 100%);
        }
        
        .tweet-timeline-title {
            font-size: var(--font-size-xl);
            font-weight: 600;
            color: var(--text-color, #333);
            margin: 0 0 var(--space-sm) 0;
            display: flex;
            align-items: center;
            gap: var(--space-sm);
        }
        
        .tweet-timeline-subtitle {
            font-size: var(--font-size-sm);
            color: var(--text-color, #666);
            margin: 0;
            opacity: 0.8;
        }
        
        .tweet-carousel {
            display: flex;
            gap: var(--space-base);
            overflow-x: auto;
            scroll-behavior: smooth;
            padding: var(--space-sm) 0 var(--space-lg) 0;
            position: relative;
            scrollbar-width: thin;
            scrollbar-color: rgba(179, 27, 27, 0.3) transparent;
            /* Ensure horizontal layout */
            flex-direction: row;
            align-items: stretch;
        }
        
        .tweet-carousel::-webkit-scrollbar {
            height: 6px;
        }
        
        .tweet-carousel::-webkit-scrollbar-track {
            background: rgba(179, 27, 27, 0.05);
            border-radius: var(--radius-full);
        }
        
        .tweet-carousel::-webkit-scrollbar-thumb {
            background: rgba(179, 27, 27, 0.3);
            border-radius: var(--radius-full);
            transition: background var(--transition-fast);
        }
        
        .tweet-carousel::-webkit-scrollbar-thumb:hover {
            background: rgba(179, 27, 27, 0.5);
        }
        
        .tweet-card {
            flex: 0 0 320px;
            background: linear-gradient(180deg, var(--surface-light) 0%, var(--surface-light-alt) 100%);
            border: 1px solid rgba(179, 27, 27, 0.08);
            border-radius: var(--radius-lg);
            padding: var(--space-lg);
            position: relative;
            overflow: hidden;
            transition: all var(--transition-base);
            cursor: default;
            min-height: 220px;
            max-height: 350px;
            display: flex;
            flex-direction: column;
            /* Ensure cards don't shrink below min width */
            flex-shrink: 0;
        }
        
        .tweet-card:hover {
            transform: translateY(-2px);
            box-shadow: var(--shadow-lg);
            border-color: rgba(179, 27, 27, 0.2);
        }
        
        .tweet-card::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            width: 4px;
            height: 100%;
            background: linear-gradient(180deg, var(--arxiv-red) 0%, var(--arxiv-red-light) 100%);
            opacity: 0;
            transition: opacity var(--transition-base);
        }
        
        .tweet-card:hover::before {
            opacity: 1;
        }
        
        .tweet-timestamp {
            display: flex;
            align-items: center;
            gap: var(--space-xs);
            font-size: var(--font-size-xs);
            color: var(--arxiv-red);
            font-weight: 600;
            margin-bottom: var(--space-base);
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        
        .tweet-content {
            font-size: var(--font-size-sm);
            line-height: 1.5;
            color: var(--text-color, #333);
            flex-grow: 1;
            overflow-y: auto;
            padding-right: var(--space-xs);
            scrollbar-width: thin;
            scrollbar-color: rgba(179, 27, 27, 0.2) transparent;
        }
        
        .tweet-content::-webkit-scrollbar {
            width: 4px;
        }
        
        .tweet-content::-webkit-scrollbar-track {
            background: transparent;
        }
        
        .tweet-content::-webkit-scrollbar-thumb {
            background: rgba(179, 27, 27, 0.2);
            border-radius: var(--radius-full);
        }
        
        .tweet-content::-webkit-scrollbar-thumb:hover {
            background: rgba(179, 27, 27, 0.4);
        }
        
        .tweet-timeline-footer {
            text-align: center;
            font-size: var(--font-size-xs);
            color: var(--text-color, #888);
            margin-top: var(--space-base);
            font-style: italic;
        }
        
        @media (prefers-color-scheme: dark) {
            .tweet-timeline-header {
                background: linear-gradient(180deg, var(--surface-dark) 0%, var(--surface-dark-alt) 100%);
                border-color: rgba(179, 27, 27, 0.15);
            }
            
            .tweet-timeline-title {
                color: var(--text-color, #FAFAFA);
            }
            
            .tweet-timeline-subtitle {
                color: var(--text-color, #CCCCCC);
            }
            
            .tweet-card {
                background: linear-gradient(180deg, var(--surface-dark) 0%, var(--surface-dark-alt) 100%);
                border-color: rgba(179, 27, 27, 0.15);
            }
            
            .tweet-card:hover {
                box-shadow: 0 8px 25px rgba(179, 27, 27, 0.25);
                border-color: rgba(179, 27, 27, 0.3);
            }
            
            .tweet-content {
                color: var(--text-color, #FAFAFA);
            }
            
            .tweet-timeline-footer {
                color: var(--text-color, #AAAAAA);
            }
        }
        
        @media (max-width: 768px) {
            .tweet-card {
                flex: 0 0 250px;
                padding: var(--space-base);
                min-height: 150px;
            }
            
            .tweet-content {
                -webkit-line-clamp: 5;
            }
        }
    """

def get_individual_tweet_card_styles():
    """Generate CSS for individual tweet cards displayed in trending papers."""
    return """
        .individual-tweet-card {
            background: var(--background-color, #ffffff);
            border: 1px solid rgba(179, 27, 27, 0.08);
            border-radius: var(--radius-base);
            padding: var(--space-base);
            margin-bottom: var(--space-sm);
            transition: all var(--transition-fast);
            position: relative;
        }
        
        .individual-tweet-card:hover {
            border-color: rgba(179, 27, 27, 0.15);
            box-shadow: 0 2px 8px rgba(179, 27, 27, 0.1);
        }
        
        .tweet-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: var(--space-sm);
        }
        
        .tweet-author {
            font-size: var(--font-size-sm);
            color: var(--text-color, #333);
        }
        
        .tweet-username {
            color: var(--text-color, #666);
            font-weight: normal;
            margin-left: var(--space-xs);
        }
        
        .tweet-time {
            font-size: var(--font-size-xs);
            color: var(--text-color, #888);
            opacity: 0.8;
        }
        
        .tweet-text {
            font-size: var(--font-size-sm);
            line-height: 1.4;
            color: var(--text-color, #333);
            margin-bottom: var(--space-base);
        }
        
        .tweet-metrics {
            display: flex;
            align-items: center;
            gap: var(--space-base);
            flex-wrap: wrap;
        }
        
        .tweet-metric {
            font-size: var(--font-size-xs);
            color: var(--text-color, #666);
            display: flex;
            align-items: center;
            gap: var(--space-xs);
        }
        
        .tweet-link {
            font-size: var(--font-size-xs);
            color: var(--arxiv-red);
            text-decoration: none;
            font-weight: 500;
            margin-left: auto;
            transition: color var(--transition-fast);
        }
        
        .tweet-link:hover {
            color: var(--arxiv-red-light);
            text-decoration: underline;
        }
        
        @media (prefers-color-scheme: dark) {
            .individual-tweet-card {
                background: var(--background-color, #0E1117);
                border-color: rgba(179, 27, 27, 0.15);
            }
            
            .individual-tweet-card:hover {
                border-color: rgba(179, 27, 27, 0.25);
                box-shadow: 0 2px 8px rgba(179, 27, 27, 0.2);
            }
            
            .tweet-author {
                color: var(--text-color, #FAFAFA);
            }
            
            .tweet-username {
                color: var(--text-color, #CCCCCC);
            }
            
            .tweet-time {
                color: var(--text-color, #AAAAAA);
            }
            
            .tweet-text {
                color: var(--text-color, #FAFAFA);
            }
            
            .tweet-metric {
                color: var(--text-color, #CCCCCC);
            }
        }
        
        @media (max-width: 768px) {
            .individual-tweet-card {
                padding: var(--space-sm);
            }
            
            .tweet-header {
                flex-direction: column;
                align-items: flex-start;
                gap: var(--space-xs);
            }
            
            .tweet-metrics {
                flex-wrap: wrap;
                gap: var(--space-sm);
            }
        }
    """

def inject_trending_card_css():
    """Inject CSS for trending card components."""
    st.markdown(f"""
        <style>
            {get_advanced_trending_card_styles()}
        </style>
    """, unsafe_allow_html=True)

def inject_interesting_facts_css():
    """Inject CSS for interesting facts components."""
    st.markdown(f"""
        <style>
            {get_interesting_facts_styles()}
        </style>
    """, unsafe_allow_html=True)

def get_featured_card_styles():
    """Generate CSS for featured paper card, aligning with trending design."""
    return """
        .featured-card {
            background: linear-gradient(180deg, var(--surface-light) 0%, var(--surface-light-alt) 100%);
            border: 1px solid rgba(179, 27, 27, 0.08);
            border-radius: var(--radius-lg);
            box-shadow: var(--shadow-sm);
            overflow: hidden;
            transition: all var(--transition-base);
            position: relative;
            max-width: 450px;
            margin: 0 auto var(--space-lg) auto;
            display: flex;
            flex-direction: column;
        }

        .featured-card:hover {
            transform: translateY(-2px);
            box-shadow: var(--shadow-lg);
            border-color: rgba(179, 27, 27, 0.2);
        }

        .featured-card::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 3px;
            background: linear-gradient(90deg, var(--arxiv-red) 0%, var(--arxiv-red-light) 100%);
        }

        .featured-image {
            width: 100%;
            height: 280px;
            background: var(--secondary-background-color, #f0f0f0);
        }

        .featured-image img {
            width: 100%;
            height: 100%;
            object-fit: cover;
        }

        .featured-content {
            padding: var(--space-base) var(--space-lg);
            display: flex;
            flex-direction: column;
            gap: var(--space-base);
        }

        .featured-title {
            font-size: var(--font-size-xl);
            font-weight: 600;
            line-height: 1.3;
            color: var(--text-color, #333);
        }

        .featured-title a {
            color: var(--arxiv-red);
            text-decoration: none;
            transition: color var(--transition-fast);
        }

        .featured-title a:hover {
            color: var(--arxiv-red-light);
            text-decoration: underline;
        }

        .featured-punchline {
            font-size: var(--font-size-sm);
            line-height: 1.5;
            color: var(--text-color, #666);
            overflow: hidden;
            display: -webkit-box;
            -webkit-line-clamp: 6; /* Show up to 6 lines */
            -webkit-box-orient: vertical;
        }

        @media (prefers-color-scheme: dark) {
            .featured-card {
                background: linear-gradient(180deg, var(--surface-dark) 0%, var(--surface-dark-alt) 100%);
                border-color: rgba(179, 27, 27, 0.15);
                box-shadow: 0 2px 8px rgba(0, 0, 0, 0.2);
            }

            .featured-title {
                color: var(--text-color, #FAFAFA);
            }

            .featured-punchline {
                color: var(--text-color, #CCCCCC);
            }
        }

        @media (max-width: 768px) {
            .featured-card {
                max-width: 100%;
            }

            .featured-image {
                height: 200px;
            }

            .featured-title {
                font-size: var(--font-size-lg);
            }
        }
    """