import utils.db as db

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
    body, html {{
        margin: 0;
        padding: 0;
        font-family: Arial, sans-serif;
        background-color: #FFF5E6;
        border-radius: 10px;
    }}
    p {{
        font-size: 0.9em;
        }}
    #header {{
        position: fixed;
        top: 0;
        width: 100%;
        z-index: 100;
        background-color: #FF8C00;
        color: white;
        padding: 20px;
        text-align: center;
        font-size: 2em;
    }}
    #summary {{
    position: fixed;
    top: 0px; /* Adjust this value based on the height of your header */
    left: 0;
    right: 0;
    margin: auto;
    width: 100%;
    z-index: 100;
    background-color: #FFA500;
    color: white;
    padding: 10px;
    border-radius: 10px;
    text-align: center;
    box-shadow: 0 2px 2px rgba(0, 0, 0, 0.2);
    text-shadow: 0 1px 2px rgba(0, 0, 0, 0.25);
    font-weight: bold;
    border: 1px solid rgba(255, 255, 255, 0.2);
    }}
    #root {{
        padding: 20px;
        margin-top: 110px; /* Adjust this value based on the combined height of your summary */
    }}
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
            React.createElement('div', {{ className: `card ${{className}}`, style: {{ ...style, backgroundColor: '#FFF8E1', borderRadius: '8px', boxShadow: '0 4px 8px rgba(0, 0, 0, 0.1)', overflow: 'hidden', transition: 'all 0.3s ease-in-out' }} }}, children)
        );

        const CardContent = ({{ children, className }}) => (
            React.createElement('div', {{ className: `card-content ${{className}}`, style: {{ padding: '16px', color: '#333333' }} }}, children)
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
                    // position: 'fixed', // Make it fixed at the top
                    width: '100%',
                    display: 'flex',
                    justifyContent: 'space-around',
                    padding: '8px 0',
                    background: '#FFF8E1',
                    borderBottom: '3px solid #FF8C00',
                    zIndex: '101', // Ensure it's above other content, adjust as necessary
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
                style: {{ padding: '8px 16px', cursor: 'pointer', borderBottom: activeTab === value ? '2px solid #FF8C00' : 'none', transition: 'all 0.3s ease-in-out' }}
            }}, children)
        );

        const TabsContent = ({{ value, children, activeTab, className }}) => (
            React.createElement('div', {{
                className: `tabs-content ${{className}}`,
                style: {{ display: activeTab === value ? 'block' : 'none', padding: '16px'}}
            }}, children)
        );
        {script}

        // Calculate margin-top
        const summaryText = document.getElementById('summary').innerText;

        const charsPerLine = 75;
        const numberOfLines = Math.ceil(summaryText.length / charsPerLine);

        const baseMargin = 110; // Base margin for 4 lines
        const additionalMarginPerTwoLines = 40; // Additional margin for every 2 lines above 4
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
    title_map = db.get_arxiv_title_dict()
    title = title_map.get(arxiv_code, "")
    script = db.get_arxiv_dashboard_script(arxiv_code, "script_content")
    summary = db.get_arxiv_dashboard_script(arxiv_code, "summary")
    if not script:
        html_card = None
    else:
        html_card = html_template.format(title=title, summary=summary, script=script)
    return html_card