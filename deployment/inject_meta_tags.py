import os
import shutil
import streamlit
from bs4 import BeautifulSoup

def inject_meta_tags():
    ## Dynamically find the Streamlit package path
    try:
        streamlit_path = streamlit.__path__[0]
        index_path = os.path.join(streamlit_path, 'static', 'index.html')
        if not os.path.exists(index_path):
            print(f"Error: Streamlit index.html not found at expected path: {index_path}")
            return
    except Exception as e:
        print(f"Error finding Streamlit path: {e}")
        return

    ## Create a backup
    backup_path = index_path + '.bak'
    try:
        shutil.copy2(index_path, backup_path)
        print(f"Backup created at: {backup_path}")
    except Exception as e:
        print(f"Error creating backup: {e}")
        ## Continue even if backup fails, but log it

    ## Define Twitter meta tags
    twitter_tags = [
        {'name': 'twitter:card', 'content': 'summary_large_image'},
        {'name': 'twitter:site', 'content': '@GPTMaestro'},
        {'name': 'twitter:creator', 'content': '@GPTMaestro'},
        {'name': 'twitter:title', 'content': 'LLMpedia - The Illustrated Large Language Model Encyclopedia'},
        {'name': 'twitter:description', 'content': 'Discover LLM research visually: AI-generated summaries, interactive visualizations, and weekly reviews of cutting-edge AI papers.'},
        {'name': 'twitter:image', 'content': 'https://raw.githubusercontent.com/masta-g3/llmpedia/refs/heads/main/logo3.png'}
    ]

    try:
        ## Read the original index.html
        with open(index_path, 'r', encoding='utf-8') as file:
            html_content = file.read()

        ## Parse the HTML
        soup = BeautifulSoup(html_content, 'html.parser')

        ## Check if head tag exists
        if not soup.head:
            print("Error: <head> tag not found in index.html")
            return

        ## Remove existing Twitter tags to avoid duplicates (optional but good practice)
        for tag in soup.head.find_all('meta', attrs={'name': lambda x: x and x.startswith('twitter:')}):
            tag.decompose()
            print(f"Removed existing tag: {tag}")

        ## Add new meta tags to the head
        for tag_attrs in twitter_tags:
            new_tag = soup.new_tag('meta')
            for key, value in tag_attrs.items():
                new_tag[key] = value
            soup.head.append(new_tag)
            ## Add a newline for readability in the HTML source
            soup.head.append('\n') 

        ## Save the modified HTML
        with open(index_path, 'w', encoding='utf-8') as file:
            file.write(str(soup))

        print("Successfully injected Twitter meta tags into Streamlit index.html.")

    except Exception as e:
        print(f"Error processing index.html: {e}")
        ## Optionally, restore from backup if something went wrong
        # if os.path.exists(backup_path):
        #     try:
        #         shutil.copy2(backup_path, index_path)
        #         print("Restored index.html from backup.")
        #     except Exception as restore_e:
        #         print(f"Error restoring from backup: {restore_e}")

if __name__ == "__main__":
    inject_meta_tags() 