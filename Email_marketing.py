import streamlit as st
import pandas as pd
import time
import os
from dotenv import load_dotenv
import re
import json
from textblob import Word
# Removed smtplib, ssl, certifi, EmailMessage imports - now in email_utils
import copy

# --- LangChain / Gemini Imports ---
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.runnables import RunnablePassthrough

# --- Local Module Imports ---
try:
    from lib import knowledge_base_handler as kbh # Handles vector store
    from lib.email_utils import send_email_smtp # Handles sending with attachments
except ImportError as import_err:
    st.error(f"Error importing local modules ({import_err}). Make sure 'knowledge_base_handler.py' and 'email_utils.py' are in the same directory.")
    st.stop()


# --- Constants ---
# SMTP constants removed - now inferred from config/email_utils
SEND_DELAY_SECONDS = 2

# --- Application's Internal Field Names ---
APP_FIELD_FIRST_NAME = 'first_name'
APP_FIELD_LAST_NAME = 'last_name'
APP_FIELD_EMAIL = 'email'
APP_FIELD_OCCUPATION = 'occupation'
APP_FIELD_TOPICS = 'topics'
APP_FIELD_MAILABILITY = 'mailability'
APP_FIELD_NOGO = 'nogo'

# --- Knowledge Base Constants ---
VECTORSTORE_PERSIST_DIR = kbh.DEFAULT_PERSIST_DIRECTORY

# ==============================================================================
# Configuration and Setup Functions
# ==============================================================================

def load_and_validate_config():
    """
    Loads environment variables, validates essential ones, displays sender info
    in the sidebar, and returns a configuration dictionary and status flag.
    """
    load_dotenv()
    config = {}
    config_ok = True

    st.sidebar.title("Sender Info")

    # Load and Validate Google API Key
    config['google_api_key'] = os.getenv('GOOGLE_API_KEY')
    if not config['google_api_key']:
        config_ok = False
        st.sidebar.error("🔴 GOOGLE_API_KEY missing.")
    else:
        st.sidebar.write("🔑 Google API Key: Loaded")

    # Load and Validate Email Credentials
    config['sender_email'] = os.getenv('EMAIL_ADDRESS')
    config['sender_password'] = os.getenv('EMAIL_PASSWORD')

    if not config['sender_email']:
        config_ok = False
        st.sidebar.error("🔴 EMAIL_ADDRESS missing.")
    else:
        st.sidebar.write(f"✉️ Email: {config['sender_email']}")

    if not config['sender_password']:
        config_ok = False
        st.sidebar.error("🔴 EMAIL_PASSWORD missing.")

    # Load Sender Name (with default)
    config['sender_name'] = os.getenv('YOUR_NAME', 'Your Name [Default]')
    st.sidebar.write(f"👤 Name: {config['sender_name']}")

    # Add SMTP details (can be overridden by env vars if needed, but use defaults)
    config['smtp_server'] = os.getenv('SMTP_SERVER', 'smtp.gmail.com')
    config['smtp_port'] = int(os.getenv('SMTP_PORT', 465)) # Ensure port is int

    # Display overall status in sidebar
    if config_ok:
        st.sidebar.success("✅ Configuration loaded.")
    else:
        st.sidebar.error("Configuration incomplete. Check `.env` file.")

    return config, config_ok

def initialize_llm_components(config):
    """
    Initializes LangChain/Gemini components (LLM and Embeddings) if the API key is available.
    Returns llm, embeddings_model, campaign_analyzer, email_drafter objects.
    """
    llm = None
    embeddings_model = None
    campaign_analyzer = None
    email_drafter = None

    google_api_key = config.get('google_api_key')
    if not google_api_key:
        st.sidebar.error("🔴 AI components require GOOGLE_API_KEY.")
        return llm, embeddings_model, campaign_analyzer, email_drafter

    try:
        # Initialize LLM
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest",
                                     google_api_key=google_api_key,
                                     temperature=0.6)

        # Initialize Embeddings Model
        embeddings_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001",
                                                        google_api_key=google_api_key)
        st.sidebar.info("✅ Embeddings Model Initialized.")


        # 1. Campaign Analysis Chain
        analysis_prompt_template = ChatPromptTemplate.from_messages(
             [
                ("system", """You are an AI assistant helping to plan an email marketing campaign.
Analyze the following conversation history where the user describes their campaign goals.
Your task is to extract the key campaign parameters:
1.  **campaign_topic**: The central theme or subject of the email campaign (e.g., "New Product Launch", "Upcoming Webinar", "Data Analysis Trends"). Be specific.
2.  **target_audience**: Describe the target audience based on the user's description and potential columns like 'Occupation' or 'Topics'. Be descriptive (e.g., "Software Engineers interested in AI", "Marketing roles focused on SEO", "Anyone interested in Leadership"). Use "Any" if no specific audience is targeted or if it's unclear.
3.  **call_to_action**: What the user wants the recipient to do (e.g., "Schedule a 15-minute demo", "Visit our new landing page", "Register for the free trial"). Default to "Engage further".
4.  **email_tone**: The desired tone of the email (e.g., "Professional and Concise", "Friendly and Engaging", "Urgent and Action-Oriented", "Informative and Educational"). Infer from the user's description. Default to "Professional".
5.  **more_info*: More info the user needs to be included in email (e.g., "Email should be in german", "Provide the contact information like email or phone number"). Capture all the misc info required by the user.

Format your output ONLY as a JSON object with these exact keys: campaign_topic, target_audience, call_to_action, email_tone. Example:
{{
"campaign_topic": "Launch of New AI-Powered SEO Tool",
"target_audience": "Marketing Specialists interested in SEO & AI",
"call_to_action": "Sign up for the waitlist",
"email_tone": "Excited and Informative"
"more_info: "Email should be in german"
}}

Conversation History:
{chat_history}
"""),
                 ("human", "Based on the conversation history above, please provide the JSON output."),
            ]
        )
        campaign_analyzer = analysis_prompt_template | llm | JsonOutputParser()
        st.sidebar.info("✅ Campaign Analyzer Initialized.")

        # 2. Email Drafting Chain
        draft_prompt_template = ChatPromptTemplate.from_messages(
            [
                ("system", """You are an AI assistant drafting personalized marketing emails.
You will receive information about the overall campaign, a specific contact, and relevant context retrieved from a knowledge base.
Generate a concise, polite, and personalized draft email based ONLY on the provided information.
The email should reflect the campaign's topic and tone, and include the call to action.

**Campaign Details:**
* Topic: {campaign_topic}
* Tone: {email_tone}
* Call to Action: {call_to_action}
* Misc : {more_info}

**Retrieved Knowledge Context (Relevant Chunks):**
```
{retrieved_knowledge_context}
```

**Format your output EXACTLY as follows:**
Subject: <Email Subject Line>
---
<Email Body>

**Instructions:**
* Address the contact by their First Name ({first_name}). If First Name is missing, use a polite generic greeting (e.g., "Hello,").
* If available, subtly reference their Occupation ({occupation}) or Topic of Interest ({contact_topic}) ONLY if it directly relates to the campaign topic and retrieved context. Avoid generic mentions.
* Leverage the **Retrieved Knowledge Context** to add specific details, benefits, or relevant points related to the campaign topic. Make the email more informative and valuable based on these retrieved chunks.
* Maintain the specified email tone throughout the message.
* Clearly state the call to action towards the end. Make it easy to understand what the recipient should do next.
* Keep the email relatively brief (around 150-250 words) and professional, unless the tone specifies otherwise.
* End the email with the sender's name: {sender_name}
* Use the {more_info} for drafting the email as per user needs.

Do NOT include any other text, explanation, or preamble before "Subject:" or after the email body. Do not add placeholders like "[Link]" unless the Call to Action explicitly mentions providing a link."""),
                ("human", """Draft an email for this contact:
First Name: {first_name}
Last Name: {last_name}
Occupation: {occupation}
Email: {contact_email}
Topic of Interest: {contact_topic}

Remember the campaign details: Topic='{campaign_topic}', Tone='{email_tone}', Call to Action='{call_to_action}'.
Use the retrieved knowledge context to enrich the email content where appropriate.""")
            ]
        )
        email_drafter = draft_prompt_template | llm | StrOutputParser()
        st.sidebar.info("✅ Email Drafter Initialized.")

    except Exception as e:
        st.sidebar.error(f"🔴 Error initializing AI components: {e}")
        llm, embeddings_model, campaign_analyzer, email_drafter = None, None, None, None

    return llm, embeddings_model, campaign_analyzer, email_drafter

# ==============================================================================
# Data Handling Functions (Contacts Only)
# ==============================================================================

def load_contacts_from_file(uploaded_file):
    """
    Loads contacts from the uploaded file (CSV or Excel).
    Returns a pandas DataFrame and list of columns, or (None, None) if loading fails.
    """
    if uploaded_file is None:
        st.warning("Please upload a contact file.")
        return None, None
    try:
        file_name = uploaded_file.name
        if file_name.endswith('.csv'):
            try:
                contacts_df = pd.read_csv(uploaded_file, encoding='utf-8')
            except UnicodeDecodeError:
                contacts_df = pd.read_csv(uploaded_file, encoding='latin1')
        elif file_name.endswith(('.xls', '.xlsx')):
             contacts_df = pd.read_excel(uploaded_file, engine='openpyxl')
        else:
            st.error("Unsupported file type. Please upload a CSV or Excel (.xlsx, .xls) file.")
            return None, None

        contacts_df = contacts_df.fillna('')
        contacts_df = contacts_df.astype(str)
        columns = contacts_df.columns.tolist()
        st.sidebar.success(f"Loaded {len(contacts_df)} contacts from '{file_name}'.")
        return contacts_df, columns

    except Exception as e:
        st.error(f"An error occurred reading the contact file: {e}")
        return None, None

def filter_contacts_based_on_analysis(df, campaign_details, column_mapping):
    """
    Filters the contact DataFrame based on mailability and campaign analysis results,
    using the user-defined column mapping.
    Returns a list of contact dictionaries.
    """
    if df is None or campaign_details is None or not column_mapping:
        st.warning("Cannot filter contacts: DataFrame, campaign details, or column mapping missing.")
        return []

    df_filtered = df.copy()

    mailability_col = column_mapping.get(APP_FIELD_MAILABILITY)
    mailability_yes_value = column_mapping.get('mailability_yes_value', 'yes')
    nogo_col = column_mapping.get(APP_FIELD_NOGO)
    occupation_col = column_mapping.get(APP_FIELD_OCCUPATION)
    topics_col = column_mapping.get(APP_FIELD_TOPICS)

    if not mailability_col:
        st.error("Mailability column mapping is missing. Cannot filter contacts.")
        return []

    try:
        df_filtered[mailability_col] = df_filtered[mailability_col].astype(str).str.strip().str.lower()
        df_filtered = df_filtered[df_filtered[mailability_col] == mailability_yes_value.lower()]
        if nogo_col:
            df_filtered[nogo_col] = df_filtered[nogo_col].astype(str).str.strip()
            df_filtered = df_filtered[df_filtered[nogo_col] == '']
    except KeyError as e:
         st.error(f"Filtering error: Column '{e}' not found. Check mapping.")
         return []
    except Exception as e:
        st.error(f"Error during mailability filtering: {e}")
        return []

    if df_filtered.empty:
        st.warning(f"No contacts found matching mailability rules.")
        return []

    target_audience_desc = campaign_details.get('target_audience', 'Any')
    if target_audience_desc and target_audience_desc.lower() != 'any' and (occupation_col or topics_col):
        extracted_words = re.findall(r'\b\w+\b', target_audience_desc.lower())
        stop_words = {'interested', 'in', 'role', 'anyone', 'and', 'or', 'with', 'the', 'a', 'is', 'are', 'for', 'of', 'an', 'to', 'as', 'who', 'like', 'specialist', 'manager', 'engineer', 'lead', 'director'}
        keywords_to_check = list(set(Word(w).singularize() for w in extracted_words if Word(w).singularize() not in stop_words and len(Word(w).singularize()) > 2))

        if keywords_to_check:
            print(f"Filtering based on keywords: {keywords_to_check}")
            try:
                occupation_mask = pd.Series(False, index=df_filtered.index)
                topics_mask = pd.Series(False, index=df_filtered.index)
                if occupation_col:
                    occupation_str_series = df_filtered[occupation_col].astype(str).str.lower()
                    for keyword in keywords_to_check: occupation_mask |= occupation_str_series.str.contains(keyword, regex=False)
                if topics_col:
                    topics_str_series = df_filtered[topics_col].astype(str).str.lower()
                    for keyword in keywords_to_check: topics_mask |= topics_str_series.str.contains(keyword, regex=False)
                combined_mask = occupation_mask | topics_mask
                df_filtered = df_filtered[combined_mask]
            except Exception as e:
                st.error(f"Error during keyword filtering: {e}")
                st.warning("Proceeding without keyword filtering.")
        else:
            st.warning("Could not extract keywords from target audience. Showing all mailable contacts.")
    elif target_audience_desc and target_audience_desc.lower() != 'any':
         st.warning("Target audience specified, but Occupation/Topics columns not mapped.")

    if df_filtered.empty:
         st.warning(f"No contacts matched target audience: '{target_audience_desc}'.")

    return df_filtered.to_dict('records')

def prepare_download_data(filtered_contacts, drafts, send_status):
    """ Prepares data for CSV download. """
    if not filtered_contacts: return pd.DataFrame()
    report_data = copy.deepcopy(filtered_contacts)
    for idx, contact_row in enumerate(report_data):
        status = send_status.get(idx, 'pending')
        status = 'Pending/Not Sent' if status == 'pending' else status
        subject = drafts.get(idx, {}).get('subject', 'N/A')
        contact_row['Send Status'] = status
        contact_row['Email Subject Drafted'] = subject
    return pd.DataFrame(report_data)

# ==============================================================================
# Core Logic Functions (Email Generation)
# ==============================================================================

def parse_llm_output(llm_response):
    """Parses the LLM response string for Subject and Body."""
    subject_match = re.search(r"Subject:\s*(.*)", llm_response, re.IGNORECASE)
    body_match = re.search(r"---\s*(.*)", llm_response, re.DOTALL | re.IGNORECASE)
    subject = subject_match.group(1).strip() if subject_match else "Following Up"
    body = body_match.group(1).strip() if body_match else llm_response.strip()
    if not subject_match or not body_match:
         st.warning(f"Could not reliably parse Subject/Body markers. Using best guess.")
         # Fallback logic
         if "Subject:" in llm_response and "---" not in llm_response:
             parts = llm_response.split("Subject:", 1)
             subject = parts[1].split('\n', 1)[0].strip()
             body = parts[1].split('\n', 1)[1].strip() if '\n' in parts[1] else parts[1].strip()
         elif "---" in llm_response:
             parts = llm_response.split("---", 1)
             subject = "Following Up"; body = parts[1].strip()
    return subject, body

def generate_single_draft(contact_info, campaign_details, vectorstore, email_drafter, sender_name, column_mapping):
    """ Generates a single email draft using vector store context. """
    if not email_drafter: return "Error: Agent Not Ready", "AI component unavailable."
    if not campaign_details: return "Error: Missing Details", "Campaign details missing."
    if not column_mapping: return "Error: Missing Mapping", "Column mapping missing."

    first_name = contact_info.get(column_mapping.get(APP_FIELD_FIRST_NAME, ''), '')
    last_name = contact_info.get(column_mapping.get(APP_FIELD_LAST_NAME, ''), '')
    email = contact_info.get(column_mapping.get(APP_FIELD_EMAIL, ''), '')
    occupation = contact_info.get(column_mapping.get(APP_FIELD_OCCUPATION, ''), '')
    topics = contact_info.get(column_mapping.get(APP_FIELD_TOPICS, ''), '')

    retrieved_context = "No specific knowledge context retrieved."
    more_info = ""
    query = campaign_details.get('campaign_topic', '')
    if vectorstore and query:
        retrieved_context = kbh.get_relevant_context(query, vectorstore, k=4)
    elif not vectorstore: print("Vector store not available.")
    elif not query: print("Campaign topic missing for KB query.")

    prompt_input = {
        'first_name': first_name or "[Not Available]",
        'last_name': last_name or "[Not Available]",
        'occupation': occupation or "N/A",
        'contact_email': email,
        'contact_topic': topics or "N/A",
        'campaign_topic': campaign_details.get('campaign_topic', 'Follow Up'),
        'email_tone': campaign_details.get('email_tone', 'Professional'),
        'call_to_action': campaign_details.get('call_to_action', 'Engage further'),
        'sender_name': sender_name,
        'more_info': campaign_details.get('more_info', ''),
        'retrieved_knowledge_context': retrieved_context
    }

    try:
        llm_response = email_drafter.invoke(prompt_input)
        subject, body = parse_llm_output(llm_response)
        return subject, body
    except Exception as e:
        st.error(f"Error calling Gemini API for {first_name or email}: {e}")
        print(f"Gemini API Error for contact {email}: {e}")
        return f"Error Generating Draft", f"AI model error: {e}"

# send_email_smtp moved to email_utils.py

# ==============================================================================
# Helper Functions
# ==============================================================================

def format_chat_history(history):
    """Formats chat history for the analysis prompt."""
    formatted_history = ""
    for msg in history:
        role = "User" if msg["role"] == "user" else "Assistant"
        content = str(msg['content']).replace("{", "{{").replace("}", "}}")
        formatted_history += f"{role}: {content}\n"
    return formatted_history.strip()

def initialize_session_state():
    """Initializes Streamlit session state variables."""
    defaults = {
        'app_stage': "upload_contacts",
        'uploaded_contact_file_state': None,
        'all_contacts_df': None,
        'csv_columns': [],
        'column_mapping': {},
        'mapping_complete': False,
        'knowledge_base_ready': os.path.exists(VECTORSTORE_PERSIST_DIR),
        'uploaded_knowledge_files_data': {}, # NEW: Store filename:bytes
        'messages': [{
            "role": "assistant",
            "content": """Hi! Let's plan your email campaign. Please tell me about:
\n1.  **Campaign Topic:** What is the main subject?
\n2.  **Target Audience:** Who should receive this?
\n3.  **Call to Action:** What should they do next?
\n(I'll use the persisted knowledge base if available!)"""
        }],
        'campaign_details': None,
        'analysis_error': False,
        'filtered_contacts_list': [],
        'drafts': {},
        'send_status': {},
        'selected_for_send': set(),
        'attachment_selections': {}, # NEW: Store idx:filename selection
        'config_ok': False,
        'llm_components_ready': False,
        'embeddings_model': None
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

def reset_session_state_for_new_campaign():
    """Resets session state for a new campaign, keeping config and AI components."""
    # Keep Config and LLM/Embeddings
    keys_to_keep = [
        'config', 'config_ok', 'llm', 'embeddings_model',
        'campaign_analyzer', 'email_drafter', 'llm_components_ready',
        'knowledge_base_ready', # Keep KB status
        'uploaded_knowledge_files_data' # Keep KB file data for attachments
    ]
    kept_state = {k: st.session_state.get(k) for k in keys_to_keep}

    # Clear all other keys
    all_keys = list(st.session_state.keys())
    for key in all_keys:
        if key not in keys_to_keep:
            # Special handling for widget keys if needed
            if key.startswith("subject_edit_") or key.startswith("body_edit_") or key.startswith("select_") or key.startswith("attach_"):
                 pass # Let Streamlit manage widget state clearing if possible
            else:
                 try:
                     del st.session_state[key]
                 except KeyError:
                     pass # Ignore if key already deleted

    # Re-initialize defaults (will skip kept keys)
    initialize_session_state()

    # Restore kept state
    for key, value in kept_state.items():
        st.session_state[key] = value

def update_selection_set(idx):
    """ Callback to update selected_for_send set based on checkbox state. """
    checkbox_key = f"select_{idx}"
    if st.session_state.get(checkbox_key, False):
        st.session_state.selected_for_send.add(idx)
    else:
        st.session_state.selected_for_send.discard(idx)

# ==============================================================================
# Streamlit UI Rendering Functions
# ==============================================================================

def render_upload_contacts_stage():
    """ Renders UI for uploading contact file. """
    st.header("Step 1: Upload Contact File")
    uploaded_file = st.file_uploader("Choose a CSV or Excel file", type=['csv', 'xlsx', 'xls'], key="contact_uploader")

    if uploaded_file is not None and uploaded_file != st.session_state.uploaded_contact_file_state:
        st.session_state.uploaded_contact_file_state = uploaded_file
        with st.spinner("Processing contact file..."):
            df, cols = load_contacts_from_file(uploaded_file)
        if df is not None and cols:
            st.session_state.all_contacts_df = df; st.session_state.csv_columns = cols
            st.session_state.app_stage = "map_columns"; st.session_state.mapping_complete = False
            st.session_state.column_mapping = {}; st.rerun()
        else: st.session_state.uploaded_contact_file_state = None

    elif st.session_state.all_contacts_df is not None:
         st.info(f"Using: {st.session_state.uploaded_contact_file_state.name}. Re-upload to change.")
         if st.button("Proceed with Current File"): st.session_state.app_stage = "map_columns"; st.rerun()

def render_map_columns_stage():
    """ Renders UI for mapping CSV columns. """
    st.header("Step 2: Map Contact Columns")
    if st.session_state.all_contacts_df is None:
        st.error("Contact data not loaded."); st.stop()

    st.write("Map columns from your file to the required/optional fields:")
    st.dataframe(st.session_state.all_contacts_df.head(), use_container_width=True)
    st.markdown("---")

    cols = st.session_state.csv_columns
    mapping = st.session_state.column_mapping.copy()

    st.subheader("Required Fields")
    likely_email = next((c for c in cols if 'email' in c.lower()), cols[0] if cols else '')
    likely_mailability = next((c for c in cols if 'mailable' in c.lower() or 'opt' in c.lower()), cols[1] if len(cols) > 1 else '')
    mapping[APP_FIELD_EMAIL] = st.selectbox("Email Address Column:", cols, key="map_email", index=cols.index(mapping.get(APP_FIELD_EMAIL, likely_email)) if mapping.get(APP_FIELD_EMAIL, likely_email) in cols else 0)
    mapping[APP_FIELD_MAILABILITY] = st.selectbox("Mailability Column (Can Send?):", cols, key="map_mailability", index=cols.index(mapping.get(APP_FIELD_MAILABILITY, likely_mailability)) if mapping.get(APP_FIELD_MAILABILITY, likely_mailability) in cols else 0)
    mapping['mailability_yes_value'] = st.text_input("Value meaning 'Yes' (case-insensitive):", value=mapping.get('mailability_yes_value', "Yes"), key="map_mailability_yes")

    st.subheader("Recommended & Optional Fields")
    opt_cols = [""] + cols
    likely_fn = next((c for c in cols if 'first' in c.lower()), '')
    likely_ln = next((c for c in cols if 'last' in c.lower()), '')
    likely_occ = next((c for c in cols if 'job' in c.lower() or 'title' in c.lower() or 'occupation' in c.lower()), '')
    likely_top = next((c for c in cols if 'interest' in c.lower() or 'topic' in c.lower()), '')
    likely_no = next((c for c in cols if 'nogo' in c.lower() or 'dne' in c.lower()), '')

    mapping[APP_FIELD_FIRST_NAME] = st.selectbox("First Name:", opt_cols, key="map_fn", index=opt_cols.index(mapping.get(APP_FIELD_FIRST_NAME, likely_fn)))
    mapping[APP_FIELD_LAST_NAME] = st.selectbox("Last Name:", opt_cols, key="map_ln", index=opt_cols.index(mapping.get(APP_FIELD_LAST_NAME, likely_ln)))
    mapping[APP_FIELD_OCCUPATION] = st.selectbox("Occupation/Title:", opt_cols, key="map_occ", index=opt_cols.index(mapping.get(APP_FIELD_OCCUPATION, likely_occ)))
    mapping[APP_FIELD_TOPICS] = st.selectbox("Topics/Interests:", opt_cols, key="map_top", index=opt_cols.index(mapping.get(APP_FIELD_TOPICS, likely_top)))
    mapping[APP_FIELD_NOGO] = st.selectbox("'Do Not Email' Column:", opt_cols, key="map_no", index=opt_cols.index(mapping.get(APP_FIELD_NOGO, likely_no)))
    st.caption("If 'Do Not Email' column selected, contacts are excluded if it's *not* empty.")

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Confirm Mapping", type="primary"):
            if not mapping.get(APP_FIELD_EMAIL) or not mapping.get(APP_FIELD_MAILABILITY): st.error("Email and Mailability columns are required.")
            elif not mapping.get('mailability_yes_value'): st.error("Mailability 'Yes' value is required.")
            else:
                final_mapping = {k: v for k, v in mapping.items() if v}
                st.session_state.column_mapping = final_mapping
                st.session_state.mapping_complete = True
                st.session_state.app_stage = "upload_knowledge"
                st.success("Mapping confirmed!"); time.sleep(1); st.rerun()
    with col2:
        if st.button("Go Back to Upload"):
            st.session_state.app_stage = "upload_contacts"; st.session_state.all_contacts_df = None
            st.session_state.csv_columns = []; st.session_state.uploaded_contact_file_state = None
            st.session_state.mapping_complete = False; st.session_state.column_mapping = {}; st.rerun()

def render_upload_knowledge_stage():
    """ Renders UI for uploading knowledge files and building vector store. """
    st.header("Step 3: Build/Update Knowledge Base (Optional)")
    st.write("Upload **Text (.txt)**, **Markdown (.md)**, or **PDF (.pdf)** files to build or replace the local knowledge base.")
    st.caption("Uploading new files **replaces** the existing knowledge base.")

    embeddings_model = st.session_state.get('embeddings_model')
    if not embeddings_model: st.error("Embeddings model not initialized. Check config."); st.stop()

    uploaded_knowledge_files = st.file_uploader(
        "Choose knowledge files", type=['txt', 'md', 'pdf'],
        key="knowledge_uploader", accept_multiple_files=True
    )

    # Store uploaded file data temporarily when files are uploaded
    if uploaded_knowledge_files:
        current_files_data = {f.name: f.getvalue() for f in uploaded_knowledge_files}
        # Only update if the set of files has changed to avoid unnecessary updates
        if current_files_data.keys() != st.session_state.uploaded_knowledge_files_data.keys():
             st.session_state.uploaded_knowledge_files_data = current_files_data
             st.info(f"{len(uploaded_knowledge_files)} file(s) staged for processing.")

    process_button_disabled = not st.session_state.uploaded_knowledge_files_data
    if st.button("Build/Update Knowledge Base", disabled=process_button_disabled):
        # Use the staged file data for processing
        files_to_process = st.session_state.uploaded_knowledge_files_data
        if files_to_process:
            # Convert stored bytes back to UploadedFile-like objects for the handler
            # This is a bit of a workaround as the handler expects file-like objects
            from io import BytesIO
            staged_file_objects = []
            for name, data in files_to_process.items():
                 file_obj = BytesIO(data)
                 file_obj.name = name # Add name attribute expected by handler
                 staged_file_objects.append(file_obj)

            with st.spinner("Processing knowledge files..."):
                docs = kbh.load_multiple_documents(staged_file_objects) # Pass file-like objects
                if docs:
                    chunks = kbh.split_documents(docs)
                    if chunks:
                        success = kbh.create_and_persist_vectorstore(chunks, embeddings_model, VECTORSTORE_PERSIST_DIR)
                        if success:
                            st.session_state.knowledge_base_ready = True
                            st.success(f"Knowledge base built/updated from {len(files_to_process)} file(s)!")
                            # Keep the data in uploaded_knowledge_files_data for attachment options
                        else: st.session_state.knowledge_base_ready = False; st.error("Failed to create/save KB.")
                    else: st.session_state.knowledge_base_ready = False; st.error("Failed to split documents.")
                else: st.session_state.knowledge_base_ready = False; st.error("Failed to load documents.")
            st.rerun() # Update status display

    st.markdown("---")
    if st.session_state.get('knowledge_base_ready'):
        st.info(f"✅ Knowledge base ready ('{VECTORSTORE_PERSIST_DIR}').")
        # Display files currently considered part of the knowledge base (for attachment selection reference)
        if st.session_state.get('uploaded_knowledge_files_data'):
             st.write("Files available for attachment (from last KB build):")
             # Use an expander to avoid clutter
             with st.expander("Available Attachment Files"):
                 st.json(list(st.session_state.uploaded_knowledge_files_data.keys()))
    else:
        st.warning("🟡 Knowledge base not ready or doesn't exist.")

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Proceed to Campaign Description", type="primary"):
            st.session_state.app_stage = "chat"; st.rerun()
    with col2:
         if st.button("Go Back to Column Mapping"):
            st.session_state.app_stage = "map_columns"; st.rerun()

def render_chat_stage(config_ok, llm_components_ready):
    """ Renders UI for campaign description chat. """
    st.header("Step 4: Describe Your Campaign")
    if st.session_state.get('knowledge_base_ready'): st.info("ℹ️ Knowledge base available for context.")
    else: st.info("ℹ️ No knowledge base available.")

    for msg in st.session_state.messages: st.chat_message(msg["role"]).write(msg["content"])

    if prompt := st.chat_input("Your campaign details..."):
        if not config_ok or not llm_components_ready:
            st.error("Config or AI components not ready.")
        else:
            st.session_state.messages.append({"role": "user", "content": prompt})
            st.chat_message("user").write(prompt)
            st.session_state.app_stage = "analyze_button"; st.rerun()

def render_analyze_button_stage():
    """ Renders button to trigger campaign analysis. """
    st.header("Step 4: Describe Your Campaign")
    for msg in st.session_state.messages: st.chat_message(msg["role"]).write(msg["content"])
    st.markdown("---")
    ai_ready = st.session_state.config_ok and st.session_state.llm_components_ready
    if not ai_ready: st.warning("AI components not ready. Cannot analyze.")
    if st.button("Analyze Campaign Description", type="primary", disabled=not ai_ready):
        st.session_state.app_stage = "analyze_processing"; st.rerun()

def render_analyze_processing_stage(campaign_analyzer):
    """ Handles AI analysis and filtering. """
    st.header("Step 5: Analyzing Campaign & Filtering Contacts...")
    st.session_state.analysis_error = False
    if not campaign_analyzer:
         st.error("Campaign Analyzer not ready."); st.session_state.analysis_error = True
         # Fallback filtering
         if st.session_state.all_contacts_df is not None and st.session_state.mapping_complete:
             st.warning("AI Analyzer failed. Filtering on mailability rules only.")
             st.session_state.filtered_contacts_list = filter_contacts_based_on_analysis(
                 st.session_state.all_contacts_df, {'target_audience': 'Any'}, st.session_state.column_mapping
             )
         else: st.session_state.filtered_contacts_list = []
         st.session_state.app_stage = "confirm"; time.sleep(1); st.rerun(); return

    with st.spinner("AI analyzing & filtering contacts..."):
        try:
            chat_history = format_chat_history(st.session_state.messages)
            analysis_result = campaign_analyzer.invoke({"chat_history": chat_history})
            if not isinstance(analysis_result, dict) or not all(k in analysis_result for k in ['campaign_topic', 'target_audience', 'call_to_action', 'email_tone']):
                 raise ValueError("AI analysis returned unexpected structure.")
            st.session_state.campaign_details = analysis_result
            st.session_state.filtered_contacts_list = filter_contacts_based_on_analysis(
                st.session_state.all_contacts_df, analysis_result, st.session_state.column_mapping
            )
            st.session_state.app_stage = "confirm"
        except Exception as e:
            st.error(f"Error during analysis/filtering: {e}"); st.session_state.analysis_error = True
            st.warning("Filtering on mailability rules only due to error.")
            st.session_state.filtered_contacts_list = filter_contacts_based_on_analysis(
                 st.session_state.all_contacts_df, {'target_audience': 'Any'}, st.session_state.column_mapping
             )
            st.session_state.app_stage = "confirm"
    st.rerun()

def render_confirm_stage():
    """ Renders UI for confirming filtered recipient list. """
    st.header("Step 6: Confirm Recipients")
    column_mapping = st.session_state.get('column_mapping', {})

    if st.session_state.analysis_error: st.warning("Analysis failed. Showing contacts filtered by mailability rules.")
    elif st.session_state.campaign_details:
        st.subheader("AI Campaign Analysis Results:")
        col1, col2 = st.columns(2); details = st.session_state.campaign_details
        with col1:
            st.markdown(f"**Topic:** {details.get('campaign_topic', 'N/A')}")
            st.markdown(f"**CTA:** {details.get('call_to_action', 'N/A')}")
        with col2:
            st.markdown(f"**Audience:** {details.get('target_audience', 'N/A')}")
            st.markdown(f"**Tone:** {details.get('email_tone', 'N/A')}")
        st.markdown("---")
    else: st.warning("Campaign details unavailable.")

    st.subheader(f"Proposed Recipients ({len(st.session_state.filtered_contacts_list)}):")
    if not st.session_state.filtered_contacts_list:
        st.warning("No contacts match the criteria.")
        if st.button("Go Back to Chat"): st.session_state.app_stage = "chat"; st.session_state.campaign_details = None; st.session_state.analysis_error = False; st.rerun()
        if st.button("Adjust Mapping"): st.session_state.app_stage = "map_columns"; st.rerun()
    else:
        cols_priority = [APP_FIELD_FIRST_NAME, APP_FIELD_LAST_NAME, APP_FIELD_EMAIL, APP_FIELD_OCCUPATION, APP_FIELD_TOPICS]
        mapped_cols = [column_mapping.get(f) for f in cols_priority if column_mapping.get(f)]
        email_col = column_mapping.get(APP_FIELD_EMAIL)
        if not email_col: st.error("Email column not mapped."); st.stop()
        if email_col not in mapped_cols: mapped_cols.insert(0, email_col)

        try:
            df_display = pd.DataFrame(st.session_state.filtered_contacts_list)
            valid_cols = [c for c in mapped_cols if c in df_display.columns]
            if not valid_cols: raise KeyError("Mapped display columns not in data.")
            df_display = df_display[valid_cols]
            rev_map = {v: k for k, v in column_mapping.items()}
            disp_map = {csv_col: rev_map.get(csv_col, csv_col).replace('_', ' ').title() for csv_col in valid_cols}
            df_display = df_display.rename(columns=disp_map)
            st.dataframe(df_display, use_container_width=True)
        except Exception as e: st.error(f"Error preparing display table: {e}"); return

        col1, col2 = st.columns(2)
        with col1:
            drafting_ready = st.session_state.config_ok and st.session_state.llm_components_ready
            tooltip = "" if drafting_ready else "AI components not ready."
            if st.button(f"Confirm & Proceed to Drafts ({len(st.session_state.filtered_contacts_list)})", type="primary", disabled=not drafting_ready, help=tooltip):
                st.session_state.app_stage = "draft"; st.session_state.drafts = {}
                st.session_state.send_status = {}; st.session_state.selected_for_send = set()
                st.session_state.attachment_selections = {} # Reset attachment selections
                st.rerun()
        with col2:
            if st.button("Cancel & Revise Campaign"):
                st.session_state.app_stage = "chat"; st.session_state.campaign_details = None
                st.session_state.filtered_contacts_list = []; st.session_state.analysis_error = False; st.rerun()

def render_draft_stage(email_drafter, config):
    """ Renders UI for generating email drafts. """
    st.header("Step 7: Generate Email Drafts")

    column_mapping = st.session_state.get('column_mapping', {})
    embeddings_model = st.session_state.get('embeddings_model')
    if not column_mapping or not embeddings_model: st.error("Mapping or embeddings model missing."); st.stop()

    st.markdown(f"Generating drafts for **{len(st.session_state.filtered_contacts_list)}** recipients.")
    if st.session_state.campaign_details:
        col1, col2 = st.columns(2); details = st.session_state.campaign_details
        with col1: st.markdown(f"**Topic:** {details.get('campaign_topic', 'N/A')}\n\n**CTA:** {details.get('call_to_action', 'N/A')}")
        with col2: st.markdown(f"**Audience:** {details.get('target_audience', 'N/A')}\n\n**Tone:** {details.get('email_tone', 'N/A')}")
    else: st.warning("Campaign details missing. Using generic settings.")

    vectorstore = None
    if st.session_state.get('knowledge_base_ready'):
        with st.spinner("Loading knowledge base..."): vectorstore = kbh.load_vectorstore(embeddings_model, VECTORSTORE_PERSIST_DIR)
        if vectorstore: st.info("✅ Knowledge base loaded.")
        else: st.warning("🟡 Failed to load knowledge base."); st.session_state.knowledge_base_ready = False
    else: st.info("ℹ️ No knowledge base available.")
    st.markdown("---")

    drafting_ready = st.session_state.config_ok and st.session_state.llm_components_ready and email_drafter is not None

    if st.button("✨ Generate Drafts", disabled=not drafting_ready or not st.session_state.filtered_contacts_list):
        if not st.session_state.campaign_details and not st.session_state.analysis_error: st.error("Campaign details missing."); return
        if not st.session_state.filtered_contacts_list: st.warning("No contacts confirmed."); return

        total_contacts = len(st.session_state.filtered_contacts_list)
        prog_bar = st.progress(0, text="Initializing..."); status_txt = st.empty()
        st.session_state.drafts = {}; st.session_state.send_status = {}
        st.session_state.selected_for_send = set(); st.session_state.attachment_selections = {}

        sender_name = config.get('sender_name', 'Your Name')
        details_to_use = st.session_state.campaign_details or {'campaign_topic': 'Follow Up', 'email_tone': 'Professional', 'call_to_action': 'Engage'}
        if not st.session_state.campaign_details: status_txt.warning("Using default campaign details.")

        errors = 0
        for i, contact in enumerate(st.session_state.filtered_contacts_list):
            fn_col = column_mapping.get(APP_FIELD_FIRST_NAME); e_col = column_mapping.get(APP_FIELD_EMAIL)
            contact_id = contact.get(fn_col) or contact.get(e_col) or f'Contact {i+1}'
            status_txt.text(f"Generating draft for {contact_id} ({i+1}/{total_contacts})...")

            subject, body = generate_single_draft(contact, details_to_use, vectorstore, email_drafter, sender_name, column_mapping)
            st.session_state.drafts[i] = {'subject': subject, 'body': body}
            st.session_state.send_status[i] = 'pending'
            if "Error Generating Draft" in subject: errors += 1; st.session_state.send_status[i] = 'failed'
            prog_bar.progress((i + 1) / total_contacts, text=f"Generated draft for {contact_id} ({i+1}/{total_contacts})")

        final_msg = f"✅ Generated drafts for {total_contacts - errors}/{total_contacts} contacts."
        if errors > 0: final_msg += f" ({errors} errors occurred)."; status_txt.error(final_msg)
        else: status_txt.success(final_msg)
        prog_bar.empty(); st.session_state.app_stage = "send"; time.sleep(1); st.rerun()

    elif not drafting_ready: st.warning("Cannot generate drafts. Check config/AI status.")
    elif not st.session_state.filtered_contacts_list: st.warning("No contacts confirmed.")

def render_send_stage(config):
    """ Renders UI for reviewing/sending emails with attachments. """
    st.header("Step 8: Review Drafts, Select Attachments & Send")

    column_mapping = st.session_state.get('column_mapping', {})
    if not column_mapping or not st.session_state.drafts: st.error("Mapping or drafts missing."); st.stop()
    email_col = column_mapping.get(APP_FIELD_EMAIL)
    first_name_col = column_mapping.get(APP_FIELD_FIRST_NAME)
    if not email_col: st.error("Email column mapping missing."); st.stop()

    # --- Attachment Options ---
    # Use filenames from the stored data which persists across KB builds until next build
    available_attachments = ["None"] + list(st.session_state.get('uploaded_knowledge_files_data', {}).keys())
    if len(available_attachments) > 1:
        st.info("Select attachments for individual emails below (optional).")
    else:
        st.info("No knowledge files were uploaded/processed, so no attachments are available.")

    # --- Bulk Actions ---
    st.subheader("Bulk Send Actions")
    col_b1, col_b2, col_b3 = st.columns(3)
    pending_indices = {idx for idx, status in st.session_state.send_status.items() if status == 'pending'}
    num_pending = len(pending_indices)

    with col_b1:
        if st.button(f"Select All Pending ({num_pending})", disabled=num_pending==0):
            st.session_state.selected_for_send = pending_indices.copy(); st.rerun()
    with col_b2:
        if st.button("Unselect All", disabled=not st.session_state.selected_for_send):
            st.session_state.selected_for_send = set(); st.rerun()
    with col_b3:
        indices_to_send = sorted(list(st.session_state.selected_for_send.intersection(pending_indices)))
        num_to_send = len(indices_to_send)
        send_disabled = num_to_send == 0 or not st.session_state.config_ok
        tooltip = "" if st.session_state.config_ok else "Email sending disabled."

        if st.button(f"🚀 Send to Selected ({num_to_send})", disabled=send_disabled, help=tooltip):
            if not indices_to_send: st.warning("No valid pending contacts selected.")
            else:
                ph = st.empty(); prog = st.progress(0); total = len(indices_to_send)
                sent, failed = 0, 0; ph.info(f"Starting bulk send for {total} emails...")

                for i, idx in enumerate(indices_to_send):
                    if idx < len(st.session_state.filtered_contacts_list) and st.session_state.send_status.get(idx) == 'pending':
                        contact = st.session_state.filtered_contacts_list[idx]
                        recipient = contact.get(email_col)
                        name = contact.get(first_name_col)
                        contact_id = name or recipient or f'Contact {idx+1}'
                        subj = st.session_state.get(f"subject_edit_{idx}", st.session_state.drafts.get(idx, {}).get('subject', 'Error'))
                        body = st.session_state.get(f"body_edit_{idx}", st.session_state.drafts.get(idx, {}).get('body', 'Error'))
                        # Get attachment selection for this index from the dedicated state dict
                        selected_att_name = st.session_state.attachment_selections.get(idx, "None") # Default to None
                        att_bytes = None
                        if selected_att_name != "None":
                            att_bytes = st.session_state.uploaded_knowledge_files_data.get(selected_att_name)
                            if not att_bytes:
                                 st.warning(f"Attachment '{selected_att_name}' data not found for {contact_id}. Sending without attachment.")

                        if "Error" in subj or "Error" in body:
                             st.warning(f"Skipped {contact_id}: Draft content error (Index {idx}).")
                             st.session_state.send_status[idx] = 'failed'; failed += 1
                        elif recipient:
                            ph.info(f"Sending to {contact_id} ({i+1}/{total})...")
                            success = send_email_smtp(recipient, subj, body, config,
                                                      attachment_filename=selected_att_name if att_bytes else None,
                                                      attachment_bytes=att_bytes)
                            if success: st.session_state.send_status[idx] = 'sent'; sent += 1
                            else: st.session_state.send_status[idx] = 'failed'; failed += 1
                        else:
                             st.warning(f"Skipped {contact_id}: No valid email."); failed += 1
                             st.session_state.send_status[idx] = 'failed'

                        prog.progress((i + 1) / total)
                        if i < total - 1: time.sleep(SEND_DELAY_SECONDS)

                final_msg = f"Bulk send complete. Sent: {sent}, Failed/Skipped: {failed}"
                if failed > 0: ph.warning(final_msg)
                else: ph.success(final_msg)
                st.session_state.selected_for_send = set()
                time.sleep(2); st.rerun()

    st.markdown("---")

    # --- Display Drafts ---
    st.subheader("Review/Edit Individual Drafts & Select Attachments")
    num_displayed, num_failed_gen = 0, 0
    for idx, status in st.session_state.send_status.items():
        if status == 'failed' and idx in st.session_state.drafts and "Error Generating Draft" in st.session_state.drafts[idx]['subject']:
            num_failed_gen += 1
    if num_failed_gen > 0: st.warning(f"{num_failed_gen} drafts failed generation.")

    sorted_indices = sorted(st.session_state.drafts.keys())
    for idx in sorted_indices:
        draft_info = st.session_state.drafts[idx]
        if idx < len(st.session_state.filtered_contacts_list):
            num_displayed += 1
            contact = st.session_state.filtered_contacts_list[idx]
            recipient = contact.get(email_col)
            name = contact.get(first_name_col)
            contact_id = name or recipient or f'Contact {idx+1}'
            status = st.session_state.send_status.get(idx, 'pending')
            is_pending = (status == 'pending')
            gen_failed = ("Error Generating Draft" in draft_info['subject'])

            row_cols = st.columns([0.1, 0.9]) # Checkbox | Expander
            with row_cols[0]:
                 st.checkbox("Select", key=f"select_{idx}", value=(idx in st.session_state.selected_for_send),
                             on_change=update_selection_set, args=(idx,),
                             disabled=(not is_pending or gen_failed), label_visibility="collapsed")
            with row_cols[1]:
                status_color = "green" if status == 'sent' else ("red" if status == 'failed' else "orange")
                title = f"{idx+1}. To: **{contact_id}** ({recipient or 'No Email!'}) - Status: :{status_color}[{status.upper()}]"
                with st.expander(title, expanded=False):
                    sub_col, att_col = st.columns([0.7, 0.3])
                    with sub_col:
                        subj = st.text_input("Subject", draft_info['subject'], key=f"subject_edit_{idx}", disabled=(not is_pending or gen_failed))
                    with att_col:
                        # Attachment Selector - Use the dedicated state dict
                        selected_att = st.selectbox(
                            "Attachment:", options=available_attachments,
                            key=f"attach_{idx}", # Widget key
                            index=available_attachments.index(st.session_state.attachment_selections.get(idx, "None")), # Set default from state
                            disabled=(not is_pending or gen_failed or len(available_attachments) <= 1),
                            label_visibility="collapsed"
                        )
                        # Update the attachment selection state when the widget changes
                        st.session_state.attachment_selections[idx] = selected_att

                    body_txt = st.text_area("Body", draft_info['body'], height=200, key=f"body_edit_{idx}", disabled=(not is_pending or gen_failed))

                    send_one_disabled = not is_pending or gen_failed or not st.session_state.config_ok or not recipient
                    tooltip = "" # Build tooltip message
                    if gen_failed: tooltip = "Draft generation failed."
                    elif not is_pending: tooltip = f"Status is {status}."
                    elif not st.session_state.config_ok: tooltip = "Config error."
                    elif not recipient: tooltip = "Invalid recipient."

                    if st.button(f"🚀 Send Only This Email", key=f"send_one_{idx}", disabled=send_one_disabled, help=tooltip):
                        ph = st.empty(); ph.info(f"Sending to {recipient}...")
                        # Get attachment details for this specific email from state
                        att_name = st.session_state.attachment_selections.get(idx, "None")
                        att_bytes = None
                        if att_name != "None":
                            att_bytes = st.session_state.uploaded_knowledge_files_data.get(att_name)
                            if not att_bytes:
                                 st.warning(f"Attachment '{att_name}' data not found. Sending without.")

                        success = send_email_smtp(recipient, subj, body_txt, config,
                                                  attachment_filename=att_name if att_bytes else None,
                                                  attachment_bytes=att_bytes)
                        if success:
                            st.session_state.send_status[idx] = 'sent'
                            st.session_state.selected_for_send.discard(idx)
                            ph.success(f"Email sent to {recipient}!")
                        else:
                            st.session_state.send_status[idx] = 'failed'
                            ph.error(f"Failed to send to {recipient}.")
                        time.sleep(1); st.rerun()
        else: st.warning(f"Data inconsistency: Draft index {idx} out of bounds.")
    if num_displayed == 0 and st.session_state.drafts: st.warning("Could not display drafts.")

    # --- Final Actions ---
    st.markdown("---"); st.subheader("Campaign Finish Actions")
    col_f1, col_f2 = st.columns(2)
    with col_f1:
        if st.session_state.filtered_contacts_list and st.session_state.drafts:
            df_dl = prepare_download_data(st.session_state.filtered_contacts_list, st.session_state.drafts, st.session_state.send_status)
            if not df_dl.empty:
                try:
                    csv_data = df_dl.to_csv(index=False).encode('utf-8')
                    st.download_button("Download Send Report (CSV)", csv_data, campaign_analyzer+'email_report.csv', 'text/csv', key='dl_btn')
                except Exception as e: st.error(f"Failed to generate download: {e}")
            else: st.info("No data available to download.")
        else: st.info("No contacts/drafts processed.")
    with col_f2:
        if st.button("Start New Campaign (Reset)", key='reset_btn'):
            reset_session_state_for_new_campaign()
            st.success("Campaign reset."); time.sleep(1); st.rerun()

# ==============================================================================
# Main Application Logic
# ==============================================================================

def main():
    """Main function to run the Streamlit application."""
    st.set_page_config(layout="wide", page_title="AI Email Campaign Assistant")
    st.title("📧 AI Email Campaign Assistant (Attachments)")
    st.markdown("Upload contacts & knowledge, map columns, describe campaign, analyze, confirm, generate drafts, select attachments, review, and send!")

    # --- Initialize State ---
    initialize_session_state()

    # --- Load Config and Check Status ---
    if 'config' not in st.session_state:
        config, config_ok = load_and_validate_config()
        st.session_state.config = config
        st.session_state.config_ok = config_ok
    else:
        config = st.session_state.config; config_ok = st.session_state.config_ok
        load_and_validate_config() # Re-render sidebar

    # --- Initialize LLM & Embeddings Components ---
    if config_ok and not st.session_state.get('llm_components_ready', False):
        llm, embeddings, analyzer, drafter = initialize_llm_components(config)
        if all([llm, embeddings, analyzer, drafter]):
            st.session_state.llm = llm; st.session_state.embeddings_model = embeddings
            st.session_state.campaign_analyzer = analyzer; st.session_state.email_drafter = drafter
            st.session_state.llm_components_ready = True
        else:
            st.session_state.llm_components_ready = False
            st.session_state.llm = None; st.session_state.embeddings_model = None
            st.session_state.campaign_analyzer = None; st.session_state.email_drafter = None
    elif not config_ok:
         st.session_state.llm_components_ready = False
         st.session_state.llm = None; st.session_state.embeddings_model = None
         st.session_state.campaign_analyzer = None; st.session_state.email_drafter = None

    # --- Stage-Based UI Rendering ---
    app_stage = st.session_state.app_stage
    if not config_ok and app_stage not in ["upload_contacts", "map_columns", "upload_knowledge"]:
         st.error("Email/API Configuration incomplete. AI features & email sending disabled.")

    try:
        if app_stage == "upload_contacts": render_upload_contacts_stage()
        elif app_stage == "map_columns":
            if st.session_state.all_contacts_df is not None: render_map_columns_stage()
            else: st.warning("No contact data."); st.session_state.app_stage = "upload_contacts"; time.sleep(1); st.rerun()
        elif app_stage == "upload_knowledge":
            if st.session_state.mapping_complete: render_upload_knowledge_stage()
            else: st.warning("Mapping incomplete."); st.session_state.app_stage = "map_columns"; time.sleep(1); st.rerun()
        elif app_stage == "chat":
            if st.session_state.mapping_complete: render_chat_stage(config_ok, st.session_state.llm_components_ready)
            else: st.warning("Mapping incomplete."); st.session_state.app_stage = "map_columns"; time.sleep(1); st.rerun()
        elif app_stage == "analyze_button":
             if st.session_state.mapping_complete: render_analyze_button_stage()
             else: st.warning("Mapping incomplete."); st.session_state.app_stage = "map_columns"; time.sleep(1); st.rerun()
        elif app_stage == "analyze_processing":
             if st.session_state.mapping_complete: render_analyze_processing_stage(st.session_state.get('campaign_analyzer'))
             else: st.warning("Mapping incomplete."); st.session_state.app_stage = "map_columns"; time.sleep(1); st.rerun()
        elif app_stage == "confirm":
             if st.session_state.mapping_complete: render_confirm_stage()
             else: st.warning("Mapping incomplete."); st.session_state.app_stage = "map_columns"; time.sleep(1); st.rerun()
        elif app_stage == "draft":
             if st.session_state.mapping_complete: render_draft_stage(st.session_state.get('email_drafter'), config)
             else: st.warning("Mapping incomplete."); st.session_state.app_stage = "map_columns"; time.sleep(1); st.rerun()
        elif app_stage == "send":
             if st.session_state.mapping_complete and st.session_state.get('drafts'): render_send_stage(config)
             elif not st.session_state.mapping_complete: st.warning("Mapping incomplete."); st.session_state.app_stage = "map_columns"; time.sleep(1); st.rerun()
             else: st.warning("Drafts not generated."); st.session_state.app_stage = "draft"; time.sleep(1); st.rerun()
    except Exception as main_e:
         st.error(f"An unexpected error occurred: {main_e}")
         st.error("Attempting to reset. Please try starting a new campaign.")
         import traceback; print("--- Main Application Error ---"); traceback.print_exc(); print("---")
         reset_session_state_for_new_campaign(); time.sleep(2); st.rerun()

if __name__ == "__main__":
    required_libs = ['PyPDF2', 'langchain', 'langchain_community', 'langchain_google_genai', 'faiss', 'textblob']
    missing_libs = []
    for lib in required_libs:
        try: __import__(lib.split('-')[0])
        except ImportError: missing_libs.append(lib)
    if missing_libs:
        install_cmd = [f"pip install {lib}" + ("-cpu" if lib == "faiss" else "") for lib in missing_libs]
        st.error(f"Missing libraries: {', '.join(missing_libs)}. Install them:\n" + "\n".join(install_cmd)); st.stop()
    main()
