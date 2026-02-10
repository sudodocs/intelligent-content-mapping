import streamlit as st
import pandas as pd
import requests
from bs4 import BeautifulSoup
import io
import re
import time
import google.generativeai as genai
from openai import OpenAI
from huggingface_hub import InferenceClient
from collections import defaultdict

# Page Config
st.set_page_config(
    page_title="SudoDocs Content Taxonomy Mapper",
    page_icon="https://sudodocs.com/favicon.ico",
    layout="wide"
)
# --- UTILITY FUNCTIONS (Back-end Logic) ---

# A set of common English stop words to filter from the final keyword list
STOP_WORDS = set([
    "a", "an", "the", "and", "or", "but", "if", "in", "on", "at", "to", "for", "with", "by", "of",
    "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "do", "does", "did",
    "will", "can", "should", "would", "could", "its", "it", "i", "me", "my", "we", "our", "you", "your"
])
# A set of overly generic standalone concepts to filter from the final keyword list
GENERIC_CONCEPTS_TO_REMOVE = {"version", "memory", "alation", "db", "database", "navigation bar"}
# A set of common command-line utilities to filter from the final keyword list
COMMON_COMMANDS_TO_REMOVE = {"sudo", "rpm", "sh", "bash", "chmod", "chown", "df", "dpkg"}

def get_deployment_type_prompt(content):
    return f"""
    Analyze the following 'Page Content'. Your task is to determine the correct deployment type.
    **Instructions**:
    - The deployment type must be one of these three options: "Alation Cloud Service", "Customer Managed", or "Alation Cloud Service, Customer Managed".
    - Your response MUST ONLY be the single most appropriate option from that list.
    - If you cannot determine the type, respond with an empty string.
    **Page Content**: --- {content[:4000]} ---
    **Your Response (choose one from the list)**:
    """

def get_mapping_prompt(content, column_name, options_list, url=None):
    additional_instructions = ""
    if column_name == 'User Role':
        additional_instructions = """- If your analysis suggests both "Steward" and "Composer" are relevant, you MUST also include "Server Admin" and "Catalog Admin" in your response."""
    elif column_name == 'Topics':
        auth_instruction = """- If the content discusses user authentication (e.g., SAML, SSO, SCIM, LDAP, login procedures), you MUST include "User Accounts" as one of the topics."""
        url_instruction = ""
        if url and "installconfig/Update/" in url:
            url_instruction = """\n- The URL for this page contains 'installconfig/Update/'. Therefore, you MUST include "Customer Managed Server Update" as one of the topics in your response."""
        additional_instructions = auth_instruction + url_instruction
    return f"""
    Analyze the following 'Page Content'. Your task is to select the most relevant term(s) from the provided 'Options List' that accurately describe the content.
    **Instructions**:
    - If the column is 'Functional Area', select only the single best option.
    - For other columns, you can select multiple options if they are all relevant.
    {additional_instructions}
    - Your response MUST ONLY contain terms from the 'Options List'.
    - Separate multiple terms with a comma.
    - If no terms from the list are relevant, respond with an empty string.
    **Page Content**: --- {content[:4000]} ---
    **Column to Map**: {column_name}
    **Options List**: {options_list}
    **Your Response**:
    """

def get_holistic_keywords_prompt(title, prose, titles, is_lean=False):
    titles_str = ", ".join(titles)
    keyword_count_instruction = "a definitive list of 10-20" if not is_lean else "a concise list of up to 7"
    return f"""
    Act as a senior technical architect and SEO specialist. Your task is to generate {keyword_count_instruction} highly relevant, technical keywords for a documentation page to ensure it is discoverable by the right audience.
    First, analyze the provided Page Title, Section Titles, and Prose Content to understand the page's core topic. Then, leveraging your deep knowledge of enterprise software, cloud technologies, and data management, generate a list of specific, technical keywords.
    **Context**:
    - **Page Title**: {title}
    - **Section Titles**: {titles_str}
    - **Prose Content**: --- {prose[:3000]} ---
    **Keyword Generation Instructions**:
    1.  **Core Concepts**: Include the main technologies and features explicitly mentioned.
    2.  **Knowledge-Based Expansion**: Include essential related technologies, protocols, or standards that an expert on this topic would expect to find (e.g., for a page on 'SAML', you might include 'IdP', 'SP', 'Assertion'). This is critical.
    3.  **Specificity is Key**: Prefer specific, multi-word phrases over single, generic words (e.g., "SAML Configuration" is better than "Configuration").
    4.  **Strict Exclusion List**: You MUST NOT include any of the following: Common English words, vague internal identifiers, example hostnames, overly generic terms, standalone command-line utilities, file paths or generic script names.
    **Your Response (a single, comma-separated list of keywords)**:
    """

def get_disambiguation_prompt(ambiguous_keyword, page_titles):
    titles_str = "\n".join([f"- {title}" for title in page_titles])
    return f"""
    The following technical keyword is ambiguous because it is associated with multiple distinct pages: "{ambiguous_keyword}".
    Your task is to analyze the page titles below and suggest a more specific, unique keyword for each page to help differentiate them in search results.
    **Page Titles Associated with "{ambiguous_keyword}"**: --- {titles_str} ---
    **Instructions**:
    - For each page title, provide one more specific keyword or phrase.
    - Format your response as a list, with each line containing the original page title followed by "::", and then your new suggested keyword.
    - Example: `Original Page Title::Suggested Differentiating Keyword`
    **Your Response**:
    """

def call_ai_provider(prompt, api_key, provider, hf_model_id=None):
    response_text = ""
    try:
        if provider == "Google Gemini":
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel('gemini-2.5-flash')
            response = model.generate_content(prompt)
            response_text = response.text
        elif provider == "OpenAI (GPT-4)":
            client = OpenAI(api_key=api_key)
            response = client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": prompt}])
            response_text = response.choices[0].message.content
        elif provider == "Hugging Face":
            client = InferenceClient(token=api_key)
            response = client.text_generation(prompt, model=hf_model_id, max_new_tokens=256)
            response_text = response
    except Exception as e:
        st.warning(f"AI API call failed: {e}")
        return ""
    sanitized_text = response_text.strip().replace('\n', ', ')
    sanitized_text = sanitized_text.replace('"', '').replace("'", "")
    sanitized_text = re.sub(r'\s*,\s*', ', ', sanitized_text)
    sanitized_text = re.sub(r'(, )+', ', ', sanitized_text)
    return sanitized_text.strip(' ,')

def enrich_data_with_ai(dataframe, user_roles, topics, functional_areas, api_key, provider, hf_model_id=None):
    df_to_process = dataframe.copy()
    if 'Deployment Type' not in df_to_process.columns: df_to_process['Deployment Type'] = ''
    if 'Functional Area' not in df_to_process.columns: df_to_process['Functional Area'] = ''
    if 'Keywords' not in df_to_process.columns: df_to_process['Keywords'] = ''
    total_rows = len(df_to_process)
    pb = st.progress(0, f"Starting AI enrichment for {total_rows} rows...")
    for index, row in df_to_process.iterrows():
        pb.progress((index + 1) / total_rows, f"Processing row {index + 1}/{total_rows}...")
        content, url = row['Page Content'], row['Page URL']
        if pd.isna(row['Deployment Type']) or row['Deployment Type'] == '':
            prompt = get_deployment_type_prompt(content)
            df_to_process.loc[index, 'Deployment Type'] = call_ai_provider(prompt, api_key, provider, hf_model_id)
            time.sleep(1)
        if pd.isna(row['User Role']) or row['User Role'] == '':
            prompt = get_mapping_prompt(content, 'User Role', user_roles)
            ai_suggested_roles = call_ai_provider(prompt, api_key, provider, hf_model_id)
            df_to_process.loc[index, 'User Role'] = apply_role_hierarchy(ai_suggested_roles)
            time.sleep(1)
        if pd.isna(row['Topics']) or row['Topics'] == '':
            prompt = get_mapping_prompt(content, 'Topics', topics, url=url)
            ai_suggested_topics = call_ai_provider(prompt, api_key, provider, hf_model_id)
            temp_row_for_topics = row.copy()
            temp_row_for_topics['Topics'] = ai_suggested_topics
            df_to_process.loc[index, 'Topics'] = augment_topics(temp_row_for_topics)
            time.sleep(1)
        if pd.isna(row['Functional Area']) or row['Functional Area'] == '':
            prompt = get_mapping_prompt(content, 'Functional Area', functional_areas)
            ai_response = call_ai_provider(prompt, api_key, provider, hf_model_id)
            df_to_process.loc[index, 'Functional Area'] = ai_response.split(',')[0].strip() if ',' in ai_response else ai_response
            time.sleep(1)
        page_title, prose_content = row['Page Title'], row['Page Content']
        section_titles = row['Section Titles'].split(',') if pd.notna(row['Section Titles']) else []
        is_lean_content = len(prose_content.split()) < 150
        prompt = get_holistic_keywords_prompt(page_title, prose_content, section_titles, is_lean=is_lean_content)
        ai_keywords = call_ai_provider(prompt, api_key, provider, hf_model_id)
        all_keywords = [k.strip() for k in ai_keywords.split(',') if k.strip()]
        url_keys = re.findall(r'(?i)(V\s?R?\d+)\b', url)
        all_keywords.extend(url_keys)
        df_to_process.loc[index, 'Keywords'] = ", ".join(clean_and_filter_keywords(all_keywords, df_to_process.loc[index], df_to_process))
    return df_to_process

def clean_and_filter_keywords(keywords_list, current_row, dataframe):
    existing_metadata_terms = set()
    for col in ['Deployment Type', 'User Role', 'Topics', 'Functional Area']:
        if col in dataframe.columns and pd.notna(current_row.get(col)):
            terms = [term.strip().lower() for term in current_row[col].split(',') if term.strip()]
            existing_metadata_terms.update(terms)
    unique_keywords_cased, seen_keywords_normalized = [], set()
    for keyword in keywords_list:
        normalized_keyword = keyword.lower().replace(" ", "")
        if normalized_keyword not in seen_keywords_normalized:
            unique_keywords_cased.append(keyword)
            seen_keywords_normalized.add(normalized_keyword)
    sorted_by_len_desc = sorted(unique_keywords_cased, key=len, reverse=True)
    subset_filtered_keywords = []
    for longer_kw in sorted_by_len_desc:
        is_subset_of_existing = False
        for existing_kw in subset_filtered_keywords:
            if re.search(r'\b' + re.escape(longer_kw) + r'\b', existing_kw, re.IGNORECASE):
                is_subset_of_existing = True
                break
        if not is_subset_of_existing:
            subset_filtered_keywords.append(longer_kw)
    vague_identifier_pattern = re.compile(r'^[a-zA-Z]+-\d+$')
    command_flag_pattern = re.compile(r'^--?[a-zA-Z0-9-]+$')
    final_keywords = []
    for kw in subset_filtered_keywords:
        kw_lower = kw.lower()
        if (kw_lower not in STOP_WORDS and kw_lower not in GENERIC_CONCEPTS_TO_REMOVE and
            kw_lower not in COMMON_COMMANDS_TO_REMOVE and kw_lower not in existing_metadata_terms and
            not vague_identifier_pattern.match(kw) and not command_flag_pattern.match(kw) and
            not kw.startswith('.')):
            final_keywords.append(kw)
    return final_keywords

def analyze_and_refine_uniqueness(dataframe, api_key, provider, hf_model_id=None):
    df = dataframe.copy()
    if 'Keywords' not in df.columns: return df, "No keywords to analyze."
    keyword_to_pages = defaultdict(list)
    for index, row in df.iterrows():
        keywords = [k.strip() for k in row['Keywords'].split(',') if k.strip()]
        for kw in keywords: keyword_to_pages[kw].append(row['Page Title'])
    ambiguous_keywords = {kw: pages for kw, pages in keyword_to_pages.items() if len(pages) > 1}
    if not ambiguous_keywords:
        df['Uniqueness Score'] = "100%"
        return df, "All keywords are unique. No refinement needed."
    st.warning(f"Found {len(ambiguous_keywords)} ambiguous keywords. Attempting to refine...")
    total_ambiguous = len(ambiguous_keywords)
    pb_disambiguation = st.progress(0, f"Disambiguating {total_ambiguous} keywords...")
    refined_keywords_map = defaultdict(list)
    for i, (kw, titles) in enumerate(ambiguous_keywords.items()):
        pb_disambiguation.progress((i + 1) / total_ambiguous, f"Refining '{kw}' ({i+1}/{total_ambiguous})...")
        prompt = get_disambiguation_prompt(kw, titles)
        response = call_ai_provider(prompt, api_key, provider, hf_model_id)
        time.sleep(1)
        for line in response.split('\n'):
            if '::' in line:
                title, new_keyword = line.split('::', 1)
                refined_keywords_map[title.strip()].append(new_keyword.strip())
    for index, row in df.iterrows():
        if row['Page Title'] in refined_keywords_map:
            current_keywords = [k.strip() for k in row['Keywords'].split(',') if k.strip()]
            new_suggestions = refined_keywords_map[row['Page Title']]
            ambiguous_for_this_page = [kw for kw in current_keywords if kw in ambiguous_keywords]
            updated_keywords = [kw for kw in current_keywords if kw not in ambiguous_for_this_page]
            updated_keywords.extend(new_suggestions)
            df.loc[index, 'Keywords'] = ", ".join(list(dict.fromkeys(updated_keywords)))
    keyword_to_pages_final = defaultdict(list)
    for index, row in df.iterrows():
        keywords = [k.strip() for k in row['Keywords'].split(',') if k.strip()]
        for kw in keywords: keyword_to_pages_final[kw].append(row['Page Title'])
    df['Uniqueness Score'] = df.apply(lambda row: f"{calculate_uniqueness(row, keyword_to_pages_final):.0f}%", axis=1)
    return df, f"Refined {len(ambiguous_keywords)} ambiguous keywords."

def calculate_uniqueness(row, keyword_map):
    keywords = [k.strip() for k in row['Keywords'].split(',') if k.strip()]
    if not keywords: return 0
    unique_count = sum(1 for kw in keywords if len(keyword_map.get(kw, [])) == 1)
    return (unique_count / len(keywords)) * 100

@st.cache_data
def analyze_page_content(url):
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        title = soup.find('title').get_text(strip=True) if soup.find('title') else 'No Title Found'
        return soup, title
    except requests.exceptions.RequestException as e:
        st.warning(f"Could not fetch {url}: {e}")
        return None, "Fetch Error"

def extract_structured_content(soup):
    if not soup: return {'prose': "Content Not Available", 'titles': [], 'code': ""}
    selectors = ['article', 'main', 'div[role="main"]', '#main-content']
    main_content = next((soup.select_one(s) for s in selectors if soup.select_one(s)), soup.body)
    if not main_content: return {'prose': "Main Content Not Found", 'titles': [], 'code': ""}
    for element in main_content.find_all(['nav', 'header', 'footer', 'aside', 'script', 'style']): element.decompose()
    titles = [h.get_text(strip=True) for h in main_content.find_all(['h2', 'h3', 'h4'])]
    code_content = " ".join([code.get_text() for code in main_content.find_all(['pre', 'code'])])
    for element in main_content.find_all(['h2', 'h3', 'h4', 'pre', 'code']): element.decompose()
    prose = main_content.get_text(separator=' ', strip=True)
    return {'prose': prose, 'titles': titles, 'code': code_content}

def find_items_in_text(text, items):
    if not isinstance(text, str): return ""
    found_items = sorted([item for item in items if re.search(r'\b' + re.escape(item) + r'\b', text, re.IGNORECASE)])
    return ", ".join(found_items)

def map_functional_area_from_url(row, functional_areas):
    if "/fde/" in row['Page URL'].lower(): return "Forward Deployed Engineering"
    url_segment_map = {re.sub(r'\s+', '', area): area for area in functional_areas}
    return next((area_name for segment, area_name in url_segment_map.items() if segment in row['Page URL']), "")

def apply_role_hierarchy(roles_str):
    HIERARCHY = ["Viewer", "Explorer", "Steward", "Composer", "Source Admin", "Catalog Admin", "Server Admin"]
    if not isinstance(roles_str, str) or not roles_str.strip(): return ""
    detected_roles = {r.strip() for r in roles_str.split(',') if r.strip()}
    min_index = float('inf')
    for role in detected_roles:
        if role in HIERARCHY: min_index = min(min_index, HIERARCHY.index(role))
    if min_index == float('inf'): return ", ".join(sorted(list(detected_roles)))
    final_roles = detected_roles.union(set(HIERARCHY[min_index:]))
    return ", ".join(sorted(list(final_roles), key=lambda x: HIERARCHY.index(x) if x in HIERARCHY else float('inf')))

def add_topic(current_topics, new_topic):
    if not isinstance(current_topics, str): current_topics = ""
    topics_set = set([t.strip() for t in current_topics.split(',') if t.strip()])
    topics_set.update(new_topic if isinstance(new_topic, list) else [new_topic])
    return ", ".join(sorted(list(topics_set)))

def augment_topics(row):
    topics = row.get('Topics', '')
    if "installconfig/Update/" in row['Page URL']: topics = add_topic(topics, "Customer Managed Server Update")
    if "/fde/" in row['Page URL'].lower():
        connector_topics = ["Data Source Access", "Connector Setup", "Metadata Extraction", "Query Log Ingestion"]
        found = find_items_in_text(row['Page Content'], connector_topics)
        if found: topics = add_topic(topics, [t.strip() for t in found.split(',')])
    if any(keyword in str(row['Page Content']).lower() for keyword in ['saml', 'sso', 'scim', 'ldap']):
        topics = add_topic(topics, "User Accounts")
    return topics

# --- STREAMLIT UI ---

# Function to load and inject local CSS file
def local_css(file_name):
    try:
        with open(file_name) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        st.error(f"CSS file not found. Please make sure '{file_name}' is in the same directory as the script.")

# Apply the custom CSS
local_css("style.css")

# Initialize session state variables
if 'df1' not in st.session_state: st.session_state.df1 = pd.DataFrame()
if 'df2' not in st.session_state: st.session_state.df2 = pd.DataFrame()
if 'df3' not in st.session_state: st.session_state.df3 = pd.DataFrame()
if 'df_final_pre_ai' not in st.session_state: st.session_state.df_final_pre_ai = pd.DataFrame()
if 'df_final' not in st.session_state: st.session_state.df_final = pd.DataFrame()
if 'df_refined' not in st.session_state: st.session_state.df_refined = pd.DataFrame()
if 'user_roles' not in st.session_state: st.session_state.user_roles = None
if 'topics' not in st.session_state: st.session_state.topics = None
if 'functional_areas' not in st.session_state: st.session_state.functional_areas = None

# --- Sidebar for Configuration ---
with st.sidebar:
    st.title("⚙️ Configuration")
    ai_provider = st.selectbox("Choose AI Provider", ["Google Gemini", "OpenAI (GPT-4)", "Hugging Face"])
    api_key_label = "API Key" if ai_provider != "Hugging Face" else "Hugging Face User Access Token"
    api_key = st.text_input(f"Enter your {api_key_label}", type="password")
    hf_model_id = None
    if ai_provider == "Hugging Face":
        hf_model_id = st.text_input("Enter Hugging Face Model ID", help="e.g., mistralai/Mistral-7B-Instruct-v0.2")

    st.markdown("---")
    st.info(
        "**Privacy Notice:** This application does not store any user data. "
        "All processing is done in-session and data is cleared when you close the tab."
    )

# --- Main App Layout ---
st.title("SudoDocs Content Taxonomy Mapper")
st.markdown("A multi-step tool to scrape, map, enrich, and refine web content taxonomy using focused AI tasks.")

tab1, tab2, tab3 = st.tabs(["Step 1: Ingest & Map", "Step 2: Generate Keywords", "Step 3: Edit & Download Results"])

# --- Tab 1: Ingest & Map Data ---
with tab1:
    st.header("Scrape URLs and Content")
    urls_file = st.file_uploader("Upload URLs File (.txt)", key="step1")
    if st.button("🚀 Scrape URLs", type="primary"):
        if urls_file:
            urls = [line.strip() for line in io.StringIO(urls_file.getvalue().decode("utf-8")) if line.strip()]
            results, pb = [], st.progress(0, "Starting...")
            for i, url in enumerate(urls):
                pb.progress((i + 1) / len(urls), f"Processing URL {i+1}/{len(urls)}...")
                soup, title = analyze_page_content(url)
                data = {'Page Title': title, 'Page URL': url}
                if soup:
                    structured_content = extract_structured_content(soup)
                    data['Page Content'] = structured_content['prose']
                    data['Section Titles'] = ",".join(structured_content['titles'])
                else:
                    data.update({'Page Content': 'Fetch Error', 'Section Titles': ''})
                results.append(data)
            st.session_state.df1 = pd.DataFrame(results)
            # Reset subsequent dataframes
            st.session_state.df2, st.session_state.df3, st.session_state.df_final_pre_ai, st.session_state.df_final, st.session_state.df_refined = pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
            st.success("✅ Scrape complete!")
        else:
            st.warning("⚠️ Please upload a URLs file.")

    st.markdown("---")
    st.header("Map User Roles, Topics, and Functional Areas")
    col_roles, col_topics, col_areas = st.columns(3)
    with col_roles:
        roles_file = st.file_uploader("Upload User Roles (.txt)", key="step2", disabled=st.session_state.df1.empty)
        if st.button("🗺️ Map Roles", disabled=st.session_state.df1.empty):
            if roles_file:
                st.session_state.user_roles = [line.strip() for line in io.StringIO(roles_file.getvalue().decode("utf-8")) if line.strip()]
                df = st.session_state.df1.copy()
                df['User Role'] = df['Page Content'].apply(lambda txt: find_items_in_text(txt, st.session_state.user_roles)).apply(apply_role_hierarchy)
                st.session_state.df2 = df
                st.success("Roles mapped!")
            else: st.warning("Upload a roles file.")
    with col_topics:
        topics_file = st.file_uploader("Upload Topics (.txt)", key="step3", disabled=st.session_state.df2.empty)
        if st.button("🏷️ Map Topics", disabled=st.session_state.df2.empty):
            if topics_file:
                st.session_state.topics = [line.strip() for line in io.StringIO(topics_file.getvalue().decode("utf-8")) if line.strip()]
                df = st.session_state.df2.copy()
                df['Topics'] = df.apply(lambda row: find_items_in_text(row['Page Content'], st.session_state.topics), axis=1).apply(lambda topics: augment_topics({'Topics': topics, 'Page URL': '', 'Page Content': ''}))
                st.session_state.df3 = df
                st.success("Topics mapped!")
            else: st.warning("Upload a topics file.")
    with col_areas:
        areas_file = st.file_uploader("Upload Areas (.txt)", key="step4", disabled=st.session_state.df3.empty)
        if st.button("🗺️ Map Functional Areas", disabled=st.session_state.df3.empty):
            if areas_file:
                st.session_state.functional_areas = [line.strip() for line in io.StringIO(areas_file.getvalue().decode("utf-8")) if line.strip()]
                df = st.session_state.df3.copy()
                df['Functional Area'] = df.apply(map_functional_area_from_url, functional_areas=st.session_state.functional_areas, axis=1)
                st.session_state.df_final_pre_ai = df
                st.success("Areas mapped!")
            else: st.warning("Upload an areas file.")

# --- Tab 2: AI Processing ---
with tab2:
    st.header("Enrich Data with AI")
    if st.button("🤖 Fill Blanks & Generate Keywords", disabled=st.session_state.df_final_pre_ai.empty):
        if not api_key: st.warning(f"Please enter your {api_key_label} in the sidebar.")
        else:
            with st.spinner("AI is processing... This may take several minutes."):
                st.session_state.df_final = enrich_data_with_ai(
                    st.session_state.df_final_pre_ai, st.session_state.user_roles, st.session_state.topics,
                    st.session_state.functional_areas, api_key, ai_provider, hf_model_id
                )
            st.success("✅ AI enrichment complete!")
            st.session_state.df_refined = pd.DataFrame() # Reset refined df if re-enriching

    if not st.session_state.df_final.empty:
        csv_step5 = st.session_state.df_final.to_csv(index=False).encode('utf-8-sig')
        st.download_button("📥 Download Enriched Report (Step 5)", csv_step5, "enriched_report.csv", "text/csv")

    st.markdown("---")
    st.header("Refine with Uniqueness Analysis")
    if st.button("🔍 Analyze and Refine Uniqueness", disabled=st.session_state.df_final.empty):
        if not api_key: st.warning(f"Please enter your {api_key_label} in the sidebar.")
        else:
            with st.spinner("Analyzing keyword uniqueness and refining with AI..."):
                df_refined, message = analyze_and_refine_uniqueness(st.session_state.df_final, api_key, ai_provider, hf_model_id)
                st.session_state.df_refined = df_refined
                st.success(f"✅ {message}")

    if not st.session_state.df_refined.empty:
        csv_step6 = st.session_state.df_refined.to_csv(index=False).encode('utf-8-sig')
        st.download_button("📥 Download Refined Report (Step 6)", csv_step6, "refined_report.csv", "text/csv")

# --- Tab 3: Results Editor ---
with tab3:
    st.header("📊 Interactive Results Editor")
    st.info("The table below is interactive. You can make manual edits and download the results directly from this tab.")

    # Determine which dataframe is active
    df_to_show, current_data_key = (st.session_state.df_refined, 'df_refined') if not st.session_state.df_refined.empty else \
                                   (st.session_state.df_final, 'df_final') if not st.session_state.df_final.empty else \
                                   (st.session_state.df_final_pre_ai, 'df_final_pre_ai') if not st.session_state.df_final_pre_ai.empty else \
                                   (st.session_state.df1, 'df1')

    if not df_to_show.empty:
        final_cols = ['Page Title', 'Page URL', 'Deployment Type', 'User Role', 'Functional Area', 'Topics', 'Keywords', 'Uniqueness Score']
        display_cols = [col for col in final_cols if col in df_to_show.columns]
        edited_df = st.data_editor(df_to_show[display_cols], key="data_editor", num_rows="dynamic", height=600, use_container_width=True)
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("💾 Save Manual Edits"):
                st.session_state[current_data_key] = edited_df
                st.success("Your edits have been saved to the session.")
                time.sleep(2)
                st.rerun()
        with col2:
            csv_edited = edited_df.to_csv(index=False).encode('utf-8-sig')
            st.download_button(
                label="📥 Download Edited Data",
                data=csv_edited,
                file_name="edited_report.csv",
                mime="text/csv",
                type="primary"
            )
    else:
        st.write("Complete the 'Scrape URLs' step in the first tab to begin.")
