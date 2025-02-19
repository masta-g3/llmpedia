import utils.db.db_utils as db_utils
import utils.db.paper_db as paper_db

html_template = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <script src="https://unpkg.com/react@17.0.2/umd/react.production.min.js"></script>
    <script src="https://unpkg.com/react-dom@17.0.2/umd/react-dom.production.min.js"></script>
    <script src="https://unpkg.com/prop-types/prop-types.min.js"></script>
    <script src="https://unpkg.com/recharts/umd/Recharts.js"></script>
    <script src="https://cdn.tailwindcss.com"></script>
<style>
    :root {
        --primary-color: #b31b1b;
        --primary-light: #fff1f1;
        --text-on-primary: white;
        --background: #ffffff;
        --text-color: #000000;
    }

    @media (prefers-color-scheme: dark) {
        :root {
            --primary-color: #b31b1b;
            --primary-light: #3d0909;
            --text-on-primary: white;
            --background: #0e1117;
            --text-color: #fafafa;
        }
    }

    body, html {
        margin: 0;
        padding: 0;
        font-family: Arial, sans-serif;
        background-color: var(--background);
        color: var(--text-color);
        border-radius: 10px;
    }
    
    p {
        font-size: 0.9em;
    }

    .card {
        border-radius: 8px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        margin-bottom: 16px;
        overflow: hidden;
        background-color: var(--primary-light);
    }

    .card-header {
        padding: 16px;
        font-weight: bold;
        font-size: 1.2em;
        border-bottom: 1px solid var(--primary-color);
        background-color: var(--primary-color);
        color: var(--text-on-primary);
    }

    .card-content {
        padding: 16px;
        background-color: var(--primary-color);
        color: var(--text-on-primary);
    }

    #summary {
        position: fixed;
        top: 0px;
        left: 0;
        right: 0;
        margin: auto;
        width: 100%;
        z-index: 100;
        background-color: var(--primary-color);
        color: var(--text-on-primary);
        padding: 10px;
        border-radius: 10px;
        text-align: center;
        box-shadow: 0 2px 2px rgba(0, 0, 0, 0.2);
        text-shadow: 0 1px 2px rgba(0, 0, 0, 0.25);
        font-weight: bold;
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
</style>
</head>
<body>
    <div id="summary">
        {summary}
    </div>
    <div id="root"></div>
    <script>
        // Card component
        const Card = ({{ children, className, style }}) => (
            React.createElement('div', {{ className: `card ${{className}}`, style: {{ ...style, backgroundColor: 'var(--primary-light)', borderRadius: '8px', boxShadow: '0 4px 8px rgba(0, 0, 0, 0.1)', overflow: 'hidden', transition: 'all 0.3s ease-in-out' }} }}, children)
        );

        const CardContent = ({{ children, className }}) => (
            React.createElement('div', {{ className: `card-content ${{className}}`, style: {{ padding: '16px', color: 'var(--text-color)' }} }}, children)
        );

        // Tabs components
        const Tabs = ({{ defaultValue, children, className }}) => {{
            const [activeTab, setActiveTab] = React.useState(defaultValue);

            const handleClick = (value) => {{
                setActiveTab(value);
            }};

            return React.createElement('div', {{ className }},
                React.createElement(TabsList, {{ activeTab, handleClick, className: "tabs-header" }}, 
                    React.Children.map(children, child => child.type === TabsTrigger ? React.cloneElement(child, {{ activeTab, handleClick }}) : null)
                ),
                React.Children.map(children, child => child.type === TabsContent ? React.cloneElement(child, {{ activeTab }}) : null)
            );
        }};

        const TabsList = ({{ children, activeTab, handleClick, className }}) => (
            React.createElement('div', {{
                className: `tabs-list ${{className}}`,
                style: {{
                    width: '100%',
                    display: 'flex',
                    justifyContent: 'space-around',
                    padding: '8px 0',
                    background: 'var(--background)',
                    borderBottom: '3px solid var(--primary-color)',
                    zIndex: '101'
                }}
            }},
            React.Children.map(children, child =>
                React.cloneElement(child, {{ activeTab, handleClick }})
            ))
        );

        const TabsTrigger = ({{ value, children, activeTab, handleClick, className }}) => (
            React.createElement('button', {{
                className: `tabs-trigger ${{className}} ${{activeTab === value ? 'active' : ''}}`,
                onClick: () => handleClick(value),
                style: {{ padding: '8px 16px', cursor: 'pointer', borderBottom: activeTab === value ? '2px solid var(--primary-color)' : 'none', transition: 'all 0.3s ease-in-out', color: 'var(--text-color)' }}
            }}, children)
        );

        const TabsContent = ({{ value, children, activeTab, className }}) => (
            React.createElement('div', {{
                className: `tabs-content ${{className}}`,
                style: {{ display: activeTab === value ? 'block' : 'none', padding: '16px', color: 'var(--text-color)'}}
            }}, children)
        );
        {script}

        // Calculate margin-top
        const summaryText = document.getElementById('summary').innerText;
        const charsPerLine = 75;
        const numberOfLines = Math.ceil(summaryText.length / charsPerLine);
        const baseMargin = 110;
        const additionalMarginPerTwoLines = 40;
        let marginTop = baseMargin;
        if (numberOfLines > 4) {{
            marginTop += Math.floor((numberOfLines - 4) / 2) * additionalMarginPerTwoLines;
        }}        
        document.getElementById('root').style.marginTop = `${{marginTop}}px`;
    </script>
</body>
</html>"""


def generate_data_card_html(arxiv_code: str):
    """Generate HTML for a data card."""
    title_map = db_utils.get_arxiv_title_dict()
    title = title_map.get(arxiv_code, "")
    script = paper_db.get_arxiv_dashboard_script(arxiv_code, "script_content")
    summary = paper_db.get_arxiv_dashboard_script(arxiv_code, "summary")
    if not script:
        html_card = None
    else:
        html_card = html_template.format(title=title, summary=summary, script=script)
    return html_card