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
            
            /* Shadows - More subtle and elegant */
            --shadow-sm: 0 1px 3px rgba(0, 0, 0, 0.04);
            --shadow-base: 0 2px 8px rgba(0, 0, 0, 0.06);
            --shadow-lg: 0 4px 16px rgba(0, 0, 0, 0.08);
            --shadow-color: rgba(179, 27, 27, 0.08);
            
            /* Dark mode shadows - neutral and subtle */
            --shadow-dark-sm: 0 1px 3px rgba(0, 0, 0, 0.3);
            --shadow-dark-base: 0 2px 8px rgba(0, 0, 0, 0.4);
            --shadow-dark-lg: 0 4px 16px rgba(0, 0, 0, 0.5);
            --shadow-dark-accent: 0 2px 12px rgba(179, 27, 27, 0.12);
            
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
            transform: translateY(-1px);
            box-shadow: var(--shadow-base);
            border-color: rgba(179, 27, 27, 0.12);
        }
        
        .card-base::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 2px;
            background: linear-gradient(90deg, var(--arxiv-red) 0%, var(--arxiv-red-light) 100%);
            opacity: 0;
            transition: opacity var(--transition-base);
        }
        
        .card-base:hover::before {
            opacity: 0.6;
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
        
        .heading-display {
            font-family: var(--font-family-display);
            font-size: var(--font-size-3xl);
            font-weight: 700;
            letter-spacing: -0.02em;
            line-height: 1.1;
            margin-bottom: var(--space-lg);
            text-rendering: optimizeLegibility;
        }
        
        .heading-geometric {
            font-family: var(--font-family-display);
            font-weight: 600;
            letter-spacing: -0.01em;
            line-height: 1.2;
        }
        
        .text-monospace {
            font-family: var(--font-family-mono);
            font-weight: 500;
            letter-spacing: -0.01em;
        }
        
        .text-precise {
            line-height: 1.4;
            letter-spacing: 0.01em;
        }
        
        /* =============================================================================
           LAYOUT UTILITIES
           ============================================================================= */
        
        .centered {
            display: flex;
            justify-content: center;
            margin-bottom: var(--space-base);
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
        
        /* Enhanced Content Grid */
        .content-grid {
            display: grid;
            grid-template-columns: 2fr 1px 1fr;
            gap: var(--space-xl);
            align-items: start;
            margin: var(--space-xl) 0;
        }
        
        .content-grid-separator {
            width: 1px;
            background: linear-gradient(180deg, transparent 0%, rgba(179, 27, 27, 0.1) 20%, rgba(179, 27, 27, 0.2) 50%, rgba(179, 27, 27, 0.1) 80%, transparent 100%);
            min-height: 200px;
            justify-self: center;
        }
        
        .content-panel {
            position: relative;
        }
        
        .content-panel-main {
            padding-right: var(--space-lg);
        }
        
        .content-panel-sidebar {
            padding-left: var(--space-lg);
        }
        
        /* =============================================================================
           MOBILE LAYOUT & UTILITIES (768px and below)
           ============================================================================= */
        
        @media (max-width: 768px) {
            /* Layout grid system */
            .content-grid {
                grid-template-columns: 1fr;
                gap: var(--space-lg);
                margin: var(--space-lg) 0;
            }
            
            .content-grid-separator {
                display: none;
            }
            
            .content-panel-main,
            .content-panel-sidebar {
                padding: 0;
            }
            
            /* Mobile research portal */
            .research-portal {
                padding: var(--space-lg);
                margin: var(--space-base) 0;
            }
            
            .research-suggestions {
                flex-direction: column;
                gap: var(--space-xs);
            }
            
            .suggestion-chip {
                text-align: center;
                padding: var(--space-sm) var(--space-base);
                font-size: var(--font-size-base);
                min-height: 44px;
                display: flex;
                align-items: center;
                justify-content: center;
            }
            
            /* Mobile discovery elements */
            .discovery-actions {
                flex-direction: column;
                gap: var(--space-xs);
                opacity: 1;
            }
            
            .action-btn {
                padding: var(--space-sm) var(--space-base);
                min-height: 44px;
                justify-content: center;
                font-size: var(--font-size-sm);
            }
            
            /* Mobile fact cards */
            .fact-card {
                padding: var(--space-sm);
                margin-bottom: var(--space-sm);
            }
            
            /* Base component mobile styles */
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
            border-color: rgba(179, 27, 27, 0.08);
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
            opacity: 0.5;
        }
        
        /* =============================================================================
           RESPONSIVE BREAKPOINTS
           ============================================================================= */
        
        
        /* =============================================================================
           DARK MODE ADAPTATIONS - Subtle and Elegant
           ============================================================================= */
        
        @media (prefers-color-scheme: dark) {
            .card-base {
                background: linear-gradient(180deg, var(--surface-dark) 0%, var(--surface-dark-alt) 100%);
                border-color: rgba(179, 27, 27, 0.15);
                box-shadow: 0 2px 8px rgba(0, 0, 0, 0.2);
            }
            
            .card-base:hover {
                box-shadow: var(--shadow-dark-accent);
                border-color: rgba(179, 27, 27, 0.15);
            }
            
            .card-base:hover::before {
                opacity: 0.4;
            }
            
            .metric-enhanced {
                background: linear-gradient(135deg, var(--background-color, #0E1117) 0%, var(--secondary-background-color, #1a1c23) 100%);
                border-color: rgba(179, 27, 27, 0.06);
            }
            
            .metric-enhanced:hover {
                border-color: rgba(179, 27, 27, 0.12);
                box-shadow: var(--shadow-dark-sm);
            }
            
            .metric-enhanced:hover::before {
                opacity: 0.3;
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
            {get_geometric_dividers()}
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
            border: 1px solid rgba(179, 27, 27, 0.06);
            margin-bottom: var(--space-lg);
            position: relative;
            overflow: hidden;
            box-shadow: var(--shadow-sm);
        }
        
        .trending-panel-header::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 2px;
            background: linear-gradient(90deg, var(--arxiv-red) 0%, var(--arxiv-red-light) 100%);
        }
        
        .trending-panel-title {
            font-family: var(--font-family-display);
            font-size: var(--font-size-xl);
            font-weight: 600;
            color: var(--text-color, #333);
            margin: 0 0 var(--space-sm) 0;
            display: flex;
            align-items: center;
            gap: var(--space-sm);
            letter-spacing: -0.01em;
            line-height: 1.2;
        }
        
        .trending-panel-subtitle {
            font-size: var(--font-size-sm);
            color: var(--text-color, #666);
            margin: 0;
            opacity: 0.75;
            line-height: 1.4;
            letter-spacing: 0.01em;
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

def get_geometric_dividers():
    """Generate CSS for geometric section dividers with arXiv red gradients."""
    return """
        /* Geometric Dividers */
        .geometric-divider {
            height: 2px;
            background: linear-gradient(90deg, transparent 0%, var(--arxiv-red) 20%, var(--arxiv-red-light) 50%, var(--arxiv-red) 80%, transparent 100%);
            margin: var(--space-xl) 0;
            opacity: 0.6;
            transition: opacity 0.2s linear;
        }
        
        .geometric-divider:hover {
            opacity: 0.8;
        }
        
        .section-spacer {
            margin: var(--space-2xl) 0;
        }
        
        /* Enhanced section containers */
        .main-section {
            position: relative;
            margin-bottom: var(--space-2xl);
        }
        
        .main-section::after {
            content: '';
            position: absolute;
            bottom: calc(-1 * var(--space-xl));
            left: 0;
            right: 0;
            height: 1px;
            background: linear-gradient(90deg, transparent 0%, rgba(179, 27, 27, 0.1) 20%, rgba(179, 27, 27, 0.2) 50%, rgba(179, 27, 27, 0.1) 80%, transparent 100%);
        }
        
        .main-section:last-child::after {
            display: none;
        }
        
        /* Enhanced Research Portal */
        .research-portal {
            background: linear-gradient(135deg, rgba(179, 27, 27, 0.02) 0%, rgba(179, 27, 27, 0.04) 100%);
            border: 1px solid rgba(179, 27, 27, 0.08);
            border-radius: var(--radius-lg);
            padding: var(--space-xl);
            margin: var(--space-lg) 0;
            position: relative;
            overflow: hidden;
        }
        
        .research-portal::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 2px;
            background: linear-gradient(90deg, var(--arxiv-red) 0%, var(--arxiv-red-light) 50%, var(--arxiv-red) 100%);
            opacity: 0.8;
        }
        
        .research-suggestions {
            display: flex;
            flex-wrap: wrap;
            gap: var(--space-sm);
            margin: var(--space-base) 0;
        }
        
        .suggestion-chip {
            background: rgba(179, 27, 27, 0.08);
            border: 1px solid rgba(179, 27, 27, 0.15);
            border-radius: var(--radius-sm);
            padding: var(--space-xs) var(--space-sm);
            font-size: var(--font-size-sm);
            color: var(--arxiv-red);
            cursor: pointer;
            transition: all 0.2s linear;
            font-family: var(--font-family-base);
            font-weight: 500;
        }
        
        .suggestion-chip:hover {
            background: rgba(179, 27, 27, 0.12);
            border-color: rgba(179, 27, 27, 0.25);
            transform: translateY(-1px);
        }
        
        @media (prefers-color-scheme: dark) {
            .geometric-divider {
                background: linear-gradient(90deg, transparent 0%, var(--arxiv-red) 20%, var(--arxiv-red-light) 50%, var(--arxiv-red) 80%, transparent 100%);
                opacity: 0.5;
            }
            
            .main-section::after {
                background: linear-gradient(90deg, transparent 0%, rgba(179, 27, 27, 0.15) 20%, rgba(179, 27, 27, 0.3) 50%, rgba(179, 27, 27, 0.15) 80%, transparent 100%);
            }
            
            .research-portal {
                background: linear-gradient(135deg, rgba(179, 27, 27, 0.04) 0%, rgba(179, 27, 27, 0.08) 100%);
                border-color: rgba(179, 27, 27, 0.15);
            }
            
            .suggestion-chip {
                background: rgba(179, 27, 27, 0.12);
                border-color: rgba(179, 27, 27, 0.2);
            }
            
            .suggestion-chip:hover {
                background: rgba(179, 27, 27, 0.18);
                border-color: rgba(179, 27, 27, 0.3);
            }
        }
        
        /* Interactive Discovery Elements */
        .discovery-card {
            position: relative;
            transition: all 0.2s linear;
        }
        
        .discovery-card:hover {
            transform: translateY(-1px);
        }
        
        .discovery-actions {
            display: flex;
            gap: var(--space-sm);
            margin-top: var(--space-base);
            opacity: 0;
            transition: opacity 0.2s linear;
        }
        
        .discovery-card:hover .discovery-actions {
            opacity: 1;
        }
        
        .action-btn {
            background: rgba(179, 27, 27, 0.08);
            border: 1px solid rgba(179, 27, 27, 0.15);
            border-radius: var(--radius-sm);
            padding: var(--space-xs) var(--space-sm);
            font-size: var(--font-size-xs);
            color: var(--arxiv-red);
            cursor: pointer;
            transition: all 0.2s linear;
            text-decoration: none;
            display: inline-flex;
            align-items: center;
            gap: var(--space-xs);
            font-weight: 500;
        }
        
        .action-btn:hover {
            background: rgba(179, 27, 27, 0.12);
            border-color: rgba(179, 27, 27, 0.25);
            transform: translateY(-1px);
        }
        
        .preview-tooltip {
            position: absolute;
            top: -10px;
            left: 0;
            right: 0;
            background: var(--surface-light);
            border: 1px solid rgba(179, 27, 27, 0.15);
            border-radius: var(--radius-base);
            padding: var(--space-sm);
            font-size: var(--font-size-sm);
            opacity: 0;
            pointer-events: none;
            transition: opacity 0.2s linear;
            z-index: 1000;
            box-shadow: var(--shadow-lg);
        }
        
        .discovery-card:hover .preview-tooltip {
            opacity: 1;
        }
        
        @media (prefers-color-scheme: dark) {
            .action-btn {
                background: rgba(179, 27, 27, 0.12);
                border-color: rgba(179, 27, 27, 0.2);
            }
            
            .action-btn:hover {
                background: rgba(179, 27, 27, 0.18);
                border-color: rgba(179, 27, 27, 0.3);
            }
            
            .preview-tooltip {
                background: var(--surface-dark);
                border-color: rgba(179, 27, 27, 0.2);
            }
        }
    """

def get_streamlit_overrides():
    """Generate CSS overrides for Streamlit default components."""
    return """
        /* Streamlit Button Overrides */
        /* Primary buttons (default and type="primary") */
        .stButton button[kind="primary"],
        .stButton button[data-testid="baseButton-primary"],
        .stButton button:not([kind]):not([data-testid]):not([class*="tertiary"]):not([class*="secondary"]) {
            background-color: var(--arxiv-red) !important;
            color: white !important;
            border: none !important;
            border-radius: var(--radius-base) !important;
            transition: all var(--transition-fast) !important;
        }
        
        .stButton button[kind="primary"]:hover,
        .stButton button[data-testid="baseButton-primary"]:hover,
        .stButton button:not([kind]):not([data-testid]):not([class*="tertiary"]):not([class*="secondary"]):hover {
            background-color: var(--arxiv-red-light) !important;
            transform: translateY(-1px) !important;
        }
        
        /* Secondary buttons */
        .stButton button[kind="secondary"],
        .stButton button[data-testid="baseButton-secondary"] {
            background: transparent !important;
            color: var(--arxiv-red) !important;
            border: 1px solid var(--arxiv-red) !important;
            border-radius: var(--radius-base) !important;
            transition: all var(--transition-fast) !important;
        }
        
        .stButton button[kind="secondary"]:hover,
        .stButton button[data-testid="baseButton-secondary"]:hover {
            background: var(--arxiv-red) !important;
            color: white !important;
        }
        
        /* Tertiary buttons - subtle border for better clickability affordance */
        .stButton button[kind="tertiary"],
        .stButton button[data-testid="baseButton-tertiary"],
        .stButton button[class*="tertiary"] {
            background: transparent !important;
            color: var(--arxiv-red) !important;
            border: 1px solid rgba(179, 27, 27, 0.12) !important;
            border-radius: var(--radius-base) !important;
            transition: all var(--transition-fast) !important;
            padding: var(--space-sm) var(--space-base) !important;
            box-shadow: none !important;
        }
        
        .stButton button[kind="tertiary"]:hover,
        .stButton button[data-testid="baseButton-tertiary"]:hover,
        .stButton button[class*="tertiary"]:hover {
            background: rgba(179, 27, 27, 0.08) !important;
            border-color: rgba(179, 27, 27, 0.25) !important;
            color: var(--arxiv-red-light) !important;
            text-decoration: underline;
            box-shadow: none !important;
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

        /* Streamlit Metric Overrides - Pixel-Art Precision */
        .stMetric {
            background: linear-gradient(180deg, var(--surface-light) 0%, var(--surface-light-alt) 100%);
            border: 1px solid rgba(179, 27, 27, 0.08);
            border-radius: var(--radius-sm);
            padding: var(--space-base);
            transition: all 0.2s linear;
            position: relative;
            overflow: hidden;
            box-shadow: var(--shadow-sm);
        }
        
        .stMetric:hover {
            border-color: rgba(179, 27, 27, 0.15);
            box-shadow: var(--shadow-base);
            transform: translateY(-2px);
        }
        
        .stMetric::before {
            content: '';
            position: absolute;
            bottom: 0;
            left: 0;
            right: 0;
            height: 3px;
            background: linear-gradient(90deg, var(--arxiv-red) 0%, var(--arxiv-red-light) 100%);
            opacity: 0;
            transition: opacity 0.2s linear;
        }
        
        .stMetric:hover::before {
            opacity: 0.7;
        }
        
        /* Metric connecting lines - geometric precision */
        .stMetric::after {
            content: '';
            position: absolute;
            top: 50%;
            right: -1px;
            width: 1px;
            height: 60%;
            background: rgba(179, 27, 27, 0.1);
            transform: translateY(-50%);
            transition: opacity 0.2s linear;
        }
        
        .stMetric:last-child::after {
            display: none;
        }
        
        /* Enhanced metric value styling */
        .stMetric [data-testid="metric-container"] > div:first-child {
            font-family: var(--font-family-mono);
            font-weight: 600;
            letter-spacing: -0.02em;
        }
        
        @media (prefers-color-scheme: dark) {
            .stMetric {
                background: linear-gradient(180deg, var(--surface-dark) 0%, var(--surface-dark-alt) 100%);
                border-color: rgba(179, 27, 27, 0.15);
                box-shadow: 0 2px 8px rgba(0, 0, 0, 0.2);
            }
            
            .stMetric:hover {
                border-color: rgba(179, 27, 27, 0.25);
                box-shadow: var(--shadow-dark-base);
            }
            
            .stMetric:hover::before {
                opacity: 0.5;
            }
            
            .stMetric::after {
                background: rgba(179, 27, 27, 0.2);
            }
        }
    """


# =============================================================================
# COMPONENT-SPECIFIC STYLES
# =============================================================================

def get_advanced_trending_card_styles():
    """Generate comprehensive CSS for trending card components."""
    return """
        /* Apply trending card styling to Streamlit containers */
        .stContainer > div[data-testid="column"] {
            position: relative;
        }
        
        .trending-card {
            background: linear-gradient(180deg, var(--surface-light) 0%, var(--surface-light-alt) 100%);
            border: 1px solid rgba(179, 27, 27, 0.08);
            border-radius: var(--radius-lg);
            padding: var(--space-lg);
            margin-bottom: var(--space-lg);
            transition: all 0.3s cubic-bezier(0.25, 0.46, 0.45, 0.94);
            position: relative;
            overflow: hidden;
            box-shadow: 
                0 1px 3px rgba(0, 0, 0, 0.12),
                0 1px 2px rgba(0, 0, 0, 0.24);
            backdrop-filter: blur(10px);
        }
        
        /* Ensure Streamlit components inside trending cards inherit proper styling */
        .trending-card .stButton {
            margin-bottom: var(--space-xs);
        }
        
        .trending-card .stImage {
            margin-bottom: 0;
        }
        
        /* Tighter spacing for card content */
        .trending-card .stColumns {
            gap: var(--space-base) !important;
        }
        
        .trending-card .stColumn {
            padding: 0 !important;
        }
        
        .trending-card:hover {
            transform: translateY(-4px) scale(1.02);
            box-shadow: 
                0 14px 28px rgba(0, 0, 0, 0.25),
                0 10px 10px rgba(0, 0, 0, 0.22),
                0 0 0 1px rgba(179, 27, 27, 0.1);
            border-color: rgba(179, 27, 27, 0.15);
        }
        
        .trending-card::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 2px;
            background: linear-gradient(90deg, var(--arxiv-red) 0%, var(--arxiv-red-light) 100%);
            opacity: 0;
            transition: opacity var(--transition-base);
        }
        
        .trending-card:hover::before {
            opacity: 1;
            background: linear-gradient(90deg, var(--arxiv-red) 0%, var(--arxiv-red-light) 50%, var(--arxiv-red) 100%);
            height: 3px;
        }
        
        
        .trending-header {
            display: flex;
            align-items: flex-start;
            gap: var(--space-base);
            margin-bottom: var(--space-base);
        }
        
        .trending-image {
            flex-shrink: 0;
            width: 90px;
            height: 90px;
            border-radius: var(--radius-lg);
            overflow: hidden;
            background: var(--secondary-background-color, #f0f0f0);
            position: relative;
            box-shadow: 
                0 4px 8px rgba(0, 0, 0, 0.12),
                0 2px 4px rgba(0, 0, 0, 0.08);
            transition: all 0.3s cubic-bezier(0.25, 0.46, 0.45, 0.94);
        }
        
        .trending-image::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: linear-gradient(45deg, transparent 0%, rgba(179, 27, 27, 0.02) 100%);
            z-index: 1;
            pointer-events: none;
        }
        
        .trending-image img {
            width: 100%;
            height: 100%;
            object-fit: cover;
            transition: all 0.4s cubic-bezier(0.25, 0.46, 0.45, 0.94);
            filter: brightness(0.95) contrast(1.05);
        }
        
        .trending-card:hover .trending-image {
            transform: scale(1.05) rotate(1deg);
            box-shadow: 
                0 8px 16px rgba(0, 0, 0, 0.2),
                0 4px 8px rgba(179, 27, 27, 0.15);
        }
        
        .trending-card:hover .trending-image img {
            transform: scale(1.1);
            filter: brightness(1.02) contrast(1.1) saturate(1.1);
        }
        
        .trending-content {
            flex: 1;
            min-width: 0;
        }
        
        .trending-title {
            font-family: var(--font-family-display);
            font-size: var(--font-size-lg);
            font-weight: 600;
            line-height: 1.25;
            margin: 0 0 var(--space-sm) 0;
            color: var(--text-color, #333);
            letter-spacing: -0.01em;
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
        
        
        /* Card Content System */
        .card-content {
            display: flex;
            flex-direction: column;
            gap: var(--space-sm);
            position: relative;
        }
        
        .card-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: var(--space-sm);
            gap: var(--space-base);
        }
        
        .badge-left,
        .badge-right {
            font-family: var(--font-family-mono);
            font-size: var(--font-size-xs);
            font-weight: 500;
            color: var(--arxiv-red);
            background: rgba(179, 27, 27, 0.04);
            border: 1px solid rgba(179, 27, 27, 0.08);
            border-radius: var(--radius-sm);
            padding: calc(var(--space-xs) * 0.75) var(--space-sm);
            display: inline-flex;
            align-items: center;
            gap: var(--space-xs);
            transition: all var(--transition-fast);
            letter-spacing: -0.01em;
            position: relative;
            overflow: hidden;
            cursor: default;
            user-select: none;
            opacity: 0.9;
        }
        
        .badge-left::before,
        .badge-right::before {
            content: '';
            position: absolute;
            top: 0;
            left: -100%;
            width: 100%;
            height: 100%;
            background: linear-gradient(90deg, transparent 0%, rgba(179, 27, 27, 0.06) 50%, transparent 100%);
            transition: left var(--transition-base);
        }
        
        .badge-left:hover,
        .badge-right:hover {
            background: rgba(179, 27, 27, 0.06);
            border-color: rgba(179, 27, 27, 0.12);
            opacity: 1;
            transform: translateY(-0.5px);
        }
        
        .badge-left:hover::before,
        .badge-right:hover::before {
            left: 100%;
        }
        
        .trending-punchline {
            font-size: var(--font-size-base);
            color: var(--text-color, #666);
            line-height: 1.5;
            margin: 0 0 var(--space-base) 0;
            opacity: 0.85;
            font-weight: 400;
            letter-spacing: 0.01em;
        }
        
        .trending-metadata {
            display: flex;
            align-items: center;
            gap: var(--space-base);
            margin-top: var(--space-base);
            padding-top: var(--space-sm);
            border-top: 1px solid rgba(128, 128, 128, 0.08);
            flex-wrap: wrap;
        }
        
        .authors {
            flex: 1;
            min-width: 0;
            font-size: var(--font-size-xs);
            color: var(--text-color, #666);
            font-weight: 400;
            opacity: 0.8;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
            transition: opacity var(--transition-fast);
        }
        
        .authors:hover {
            opacity: 1;
        }
        
        .trending-authors {
            font-size: var(--font-size-sm);
            color: var(--text-color, #777);
            flex: 1;
            min-width: 0;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
            font-weight: 400;
        }
        
        .trending-metric {
            display: flex;
            align-items: center;
            gap: var(--space-xs);
            font-size: var(--font-size-base);
            font-weight: 600;
            color: var(--arxiv-red);
            white-space: nowrap;
            font-family: var(--font-family-mono);
            background: rgba(179, 27, 27, 0.06);
            padding: var(--space-xs) var(--space-sm);
            border-radius: var(--radius-full);
            border: 1px solid rgba(179, 27, 27, 0.1);
        }
        
        /* New cleaner metadata layout */
        .trending-metadata-row {
            margin-top: var(--space-base);
            padding: var(--space-sm) 0;
            border-top: 1px solid rgba(128, 128, 128, 0.08);
        }
        
        .trending-authors-metrics {
            display: flex;
            justify-content: space-between;
            align-items: center;
            font-size: var(--font-size-sm);
            color: var(--text-color, #666);
            gap: var(--space-lg);
            padding-top: var(--space-sm);
            border-top: 1px solid rgba(179, 27, 27, 0.06);
        }
        
        .authors-section {
            flex: 1;
            min-width: 0;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
            font-weight: 400;
            opacity: 0.8;
        }
        
        .metrics-section {
            flex-shrink: 0;
            font-weight: 600;
            font-family: var(--font-family-mono);
            color: var(--arxiv-red);
            background: rgba(179, 27, 27, 0.08);
            padding: var(--space-xs) var(--space-sm);
            border-radius: var(--radius-sm);
            font-size: var(--font-size-xs);
            letter-spacing: -0.01em;
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
            border-top: 1px solid rgba(128, 128, 128, 0.08);
            text-align: center;
            font-size: var(--font-size-xs);
            color: var(--text-color, #888);
            font-style: italic;
        }

        /* Dark mode adaptations - Subtle and elegant */
        @media (prefers-color-scheme: dark) {
            .trending-card {
                background: linear-gradient(180deg, var(--surface-dark) 0%, var(--surface-dark-alt) 100%);
                border-color: rgba(179, 27, 27, 0.15);
                box-shadow: 
                    0 1px 3px rgba(0, 0, 0, 0.4),
                    0 1px 2px rgba(0, 0, 0, 0.6);
                backdrop-filter: blur(20px);
            }
            
            .trending-card:hover {
                box-shadow: 
                    0 14px 28px rgba(0, 0, 0, 0.4),
                    0 10px 10px rgba(0, 0, 0, 0.3),
                    0 0 0 1px rgba(179, 27, 27, 0.2);
                border-color: rgba(179, 27, 27, 0.25);
            }
            
            .trending-card:hover::before {
                opacity: 0.8;
            }
            
            .trending-image {
                background: var(--secondary-background-color, #1a1c23);
                box-shadow: 
                    0 4px 8px rgba(0, 0, 0, 0.3),
                    0 2px 4px rgba(0, 0, 0, 0.2);
            }
            
            .trending-card:hover .trending-image {
                box-shadow: 
                    0 8px 16px rgba(0, 0, 0, 0.4),
                    0 4px 8px rgba(179, 27, 27, 0.3);
            }
            
            .trending-title {
                color: var(--text-color, #FAFAFA);
            }
            
            
            .badge-left,
            .badge-right {
                background: rgba(179, 27, 27, 0.08);
                border-color: rgba(179, 27, 27, 0.15);
                color: var(--arxiv-red-light);
                opacity: 0.85;
            }
            
            .badge-left:hover,
            .badge-right:hover {
                background: rgba(179, 27, 27, 0.12);
                border-color: rgba(179, 27, 27, 0.2);
                opacity: 1;
            }
            
            .badge-left::before,
            .badge-right::before {
                background: linear-gradient(90deg, transparent 0%, rgba(179, 27, 27, 0.12) 50%, transparent 100%);
            }
            
            .trending-punchline {
                color: var(--text-color, #CCCCCC);
                opacity: 0.85;
            }
            
            .trending-metadata {
                border-top-color: rgba(179, 27, 27, 0.1);
            }
            
            .authors {
                color: var(--text-color, #AAAAAA);
                opacity: 0.75;
            }
            
            .authors:hover {
                opacity: 0.95;
            }
            
            
            .trending-punchline {
                color: var(--text-color, #CCCCCC);
                opacity: 0.9;
            }
            
            .trending-authors {
                color: var(--text-color, #AAAAAA);
            }
            
            .trending-metric {
                background: rgba(179, 27, 27, 0.15);
                border-color: rgba(179, 27, 27, 0.2);
                color: var(--arxiv-red-light);
            }
            
            .metrics-section {
                background: rgba(179, 27, 27, 0.15);
                color: var(--arxiv-red-light);
            }
            
            .trending-authors-metrics {
                border-top-color: rgba(179, 27, 27, 0.1);
            }
            
            .trending-summary {
                color: var(--text-color, #AAAAAA);
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
            box-shadow: var(--shadow-sm);
        }
        
        .fact-card:hover {
            transform: translateY(-1px);
            box-shadow: var(--shadow-base);
            border-left-color: var(--arxiv-red-light);
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
            background-color: var(--secondary-background-color, rgba(128, 128, 128, 0.08));
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
                box-shadow: var(--shadow-dark-sm);
            }
            
            .fact-card:hover {
                box-shadow: var(--shadow-dark-base);
                border-left-color: var(--arxiv-red-light);
            }
            
            .fact-topic {
                background-color: var(--secondary-background-color, rgba(128, 128, 128, 0.15));
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
            border: 1px solid rgba(179, 27, 27, 0.06);
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
            height: 2px;
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
            scrollbar-color: rgba(179, 27, 27, 0.2) transparent;
            /* Ensure horizontal layout */
            flex-direction: row;
            align-items: stretch;
        }
        
        .tweet-carousel::-webkit-scrollbar {
            height: 6px;
        }
        
        .tweet-carousel::-webkit-scrollbar-track {
            background: rgba(179, 27, 27, 0.04);
            border-radius: var(--radius-full);
        }
        
        .tweet-carousel::-webkit-scrollbar-thumb {
            background: rgba(179, 27, 27, 0.2);
            border-radius: var(--radius-full);
            transition: background var(--transition-fast);
        }
        
        .tweet-carousel::-webkit-scrollbar-thumb:hover {
            background: rgba(179, 27, 27, 0.3);
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
            box-shadow: var(--shadow-sm);
        }
        
        .tweet-card:hover {
            transform: translateY(-1px);
            box-shadow: var(--shadow-base);
            border-color: rgba(179, 27, 27, 0.12);
        }
        
        .tweet-card::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            width: 2px;
            height: 100%;
            background: linear-gradient(180deg, var(--arxiv-red) 0%, var(--arxiv-red-light) 100%);
            opacity: 0;
            transition: opacity var(--transition-base);
        }
        
        .tweet-card:hover::before {
            opacity: 0.6;
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
            scrollbar-color: rgba(179, 27, 27, 0.15) transparent;
        }
        
        .tweet-content::-webkit-scrollbar {
            width: 4px;
        }
        
        .tweet-content::-webkit-scrollbar-track {
            background: transparent;
        }
        
        .tweet-content::-webkit-scrollbar-thumb {
            background: rgba(179, 27, 27, 0.15);
            border-radius: var(--radius-full);
        }
        
        .tweet-content::-webkit-scrollbar-thumb:hover {
            background: rgba(179, 27, 27, 0.25);
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
                box-shadow: var(--shadow-dark-accent);
                border-color: rgba(179, 27, 27, 0.15);
            }
            
            .tweet-card:hover::before {
                opacity: 0.4;
            }
            
            .tweet-content {
                color: var(--text-color, #FAFAFA);
                scrollbar-color: rgba(179, 27, 27, 0.2) transparent;
            }
            
            .tweet-content::-webkit-scrollbar-thumb {
                background: rgba(179, 27, 27, 0.2);
            }
            
            .tweet-content::-webkit-scrollbar-thumb:hover {
                background: rgba(179, 27, 27, 0.3);
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
            transform: translateY(-1px);
            box-shadow: var(--shadow-base);
            border-color: rgba(179, 27, 27, 0.12);
        }

        .featured-card::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 2px;
            background: linear-gradient(90deg, var(--arxiv-red) 0%, var(--arxiv-red-light) 100%);
            opacity: 0;
            transition: opacity var(--transition-base);
        }

        .featured-card:hover::before {
            opacity: 0.6;
        }

        .featured-image {
            width: 100%;
            height: 280px;
            background: var(--secondary-background-color, #f0f0f0);
            overflow: hidden;
        }

        .featured-image img {
            width: 100%;
            height: 100%;
            object-fit: cover;
            transition: transform var(--transition-base);
        }

        .featured-card:hover .featured-image img {
            transform: scale(1.02);
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
            }

            .featured-card:hover::before {
                opacity: 0.4;
            }

            .featured-image {
                background: var(--secondary-background-color, #1a1c23);
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
                margin-bottom: var(--space-lg);
            }

            .featured-image {
                height: 200px;
            }

            .featured-title {
                font-size: var(--font-size-lg);
            }
        }
    """