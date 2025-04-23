import streamlit as st
import pandas as pd
import time
import os
from dotenv import load_dotenv
import re
import json
from textblob import Word
import ssl
import smtplib
import certifi
from email.message import EmailMessage
import copy # Import copy for deep copying lists
# Removed PyPDF2 import, now handled by the handler
# Removed io import, now handled by the handler

# --- LangChain / Gemini Imports ---
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings # Added Embeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.runnables import RunnablePassthrough
# Removed LangChain document/loader/splitter imports, now handled by the handler

# --- Local Knowledge Base Handler Import ---
# Ensure knowledge_base_handler.py is in the same directory
try:
    from lib import knowledge_base_handler as kbh # Import the new handler file
except ImportError:
    st.error("Error: knowledge_base_handler.py not found. Make sure it's in the same directory.")
    st.stop()


# --- Constants ---
SMTP_SERVER = 'smtp.gmail.com' # Example for Gmail
SMTP_PORT = 465
SEND_DELAY_SECONDS = 2

# --- Application's Internal Field Names ---
APP_FIELD_FIRST_NAME = 'first_name'
APP_FIELD_LAST_NAME = 'last_name'
APP_FIELD_EMAIL = 'email'
APP_FIELD_OCCUPATION = 'occupation'
APP_FIELD_TOPICS = 'topics'
APP_FIELD_MAILABILITY = 'mailability' # Column indicating if email can be sent
APP_FIELD_NOGO = 'nogo'             # Column indicating if contact should NOT be emailed

# --- Knowledge Base Constants ---
VECTORSTORE_PERSIST_DIR = kbh.DEFAULT_PERSIST_DIRECTORY # Use directory from handler

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
        st.sidebar.write("🔑 Google API Key: Loaded") # Indicate loaded

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
    # else: # Don't show password status explicitly
        # st.sidebar.write("🔒 Email Password: Loaded")

    # Load Sender Name (with default)
    config['sender_name'] = os.getenv('YOUR_NAME', 'Your Name [Default]')
    st.sidebar.write(f"👤 Name: {config['sender_name']}")


    # Add other config if needed
    config['smtp_server'] = SMTP_SERVER
    config['smtp_port'] = SMTP_PORT

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
    embeddings_model = None # NEW: Initialize embeddings model here
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

        # NEW: Initialize Embeddings Model
        embeddings_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001", # Recommended model for embeddings
                                                        google_api_key=google_api_key)
        st.sidebar.info("✅ Embeddings Model Initialized.")


        # 1. Campaign Analysis Chain (Remains the same)
        analysis_prompt_template = ChatPromptTemplate.from_messages(
             [
                ("system", """You are an AI assistant helping to plan an email marketing campaign.
Analyze the following conversation history where the user describes their campaign goals.
Your task is to extract the key campaign parameters:
1.  **campaign_topic**: The central theme or subject of the email campaign (e.g., "New Product Launch", "Upcoming Webinar", "Data Analysis Trends"). Be specific.
2.  **target_audience**: Describe the target audience based on the user's description and potential columns like 'Occupation' or 'Topics'. Be descriptive (e.g., "Software Engineers interested in AI", "Marketing roles focused on SEO", "Anyone interested in Leadership"). Use "Any" if no specific audience is targeted or if it's unclear.
3.  **call_to_action**: What the user wants the recipient to do (e.g., "Schedule a 15-minute demo", "Visit our new landing page", "Register for the free trial"). Default to "Engage further".
4.  **email_tone**: The desired tone of the email (e.g., "Professional and Concise", "Friendly and Engaging", "Urgent and Action-Oriented", "Informative and Educational"). Infer from the user's description. Default to "Professional".

Format your output ONLY as a JSON object with these exact keys: campaign_topic, target_audience, call_to_action, email_tone. Example:
{{
"campaign_topic": "Launch of New AI-Powered SEO Tool",
"target_audience": "Marketing Specialists interested in SEO & AI",
"call_to_action": "Sign up for the waitlist",
"email_tone": "Excited and Informative"
}}

Conversation History:
{chat_history}
"""),
                 ("human", "Based on the conversation history above, please provide the JSON output."),
            ]
        )
        campaign_analyzer = analysis_prompt_template | llm | JsonOutputParser()
        st.sidebar.info("✅ Campaign Analyzer Initialized.")

        # 2. Email Drafting Chain (Prompt updated slightly for retrieved context)
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

Do NOT include any other text, explanation, or preamble before "Subject:" or after the email body. Do not add placeholders like "[Link]" unless the Call to Action explicitly mentions providing a link."""),
                # MODIFIED: Input variable name changed
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
        # MODIFIED: Input variable name changed in RunnablePassthrough if needed later, but StrOutputParser is fine
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
            # Try reading with common encodings
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
        # Convert all columns to string to avoid potential type issues later
        contacts_df = contacts_df.astype(str)
        columns = contacts_df.columns.tolist()
        st.sidebar.success(f"Loaded {len(contacts_df)} contacts from '{file_name}'.")
        return contacts_df, columns

    except Exception as e:
        st.error(f"An error occurred reading the contact file: {e}")
        return None, None

# load_knowledge_content is now in knowledge_base_handler.py

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

    # --- Get mapped column names ---
    mailability_col = column_mapping.get(APP_FIELD_MAILABILITY)
    mailability_yes_value = column_mapping.get('mailability_yes_value', 'yes') # Default to 'yes'
    nogo_col = column_mapping.get(APP_FIELD_NOGO) # Optional
    occupation_col = column_mapping.get(APP_FIELD_OCCUPATION) # Optional
    topics_col = column_mapping.get(APP_FIELD_TOPICS) # Optional

    # --- 1. Basic Mailability Filter ---
    if not mailability_col:
        st.error("Mailability column mapping is missing. Cannot filter contacts.")
        return []

    try:
        # Ensure comparison is case-insensitive and handles potential type issues
        df_filtered[mailability_col] = df_filtered[mailability_col].astype(str).str.strip().str.lower()
        df_filtered = df_filtered[df_filtered[mailability_col] == mailability_yes_value.lower()]

        # Apply No-Go filter if the column is mapped
        if nogo_col:
            # Keep rows where the No-Go column is empty (or NaN, already handled by fillna/astype(str))
            df_filtered[nogo_col] = df_filtered[nogo_col].astype(str).str.strip()
            df_filtered = df_filtered[df_filtered[nogo_col] == '']

    except KeyError as e:
         st.error(f"Filtering error: Column '{e}' not found in the uploaded file. Please check your mapping.")
         return []
    except Exception as e:
        st.error(f"Error during basic mailability filtering: {e}")
        return []


    if df_filtered.empty:
        st.warning(f"No contacts found where '{mailability_col}' is '{mailability_yes_value}'" + (f" and '{nogo_col}' is empty." if nogo_col else "."))
        return []

    # --- 2. Target Audience Filter ---
    target_audience_desc = campaign_details.get('target_audience', 'Any')
    if target_audience_desc and target_audience_desc.lower() != 'any' and (occupation_col or topics_col):
        # Simple keyword extraction and singularization
        extracted_words = re.findall(r'\b\w+\b', target_audience_desc.lower())
        # More comprehensive stop words list
        stop_words = {'interested', 'in', 'role', 'anyone', 'and', 'or', 'with', 'the', 'a', 'is', 'are', 'for', 'of', 'an', 'to', 'as', 'who', 'like', 'specialist', 'manager', 'engineer', 'lead', 'director'}
        keywords_to_check = [
            Word(word).singularize()
            for word in extracted_words
            if Word(word).singularize() not in stop_words and len(Word(word).singularize()) > 2 # Avoid short/common words
        ]
        keywords_to_check = list(set(keywords_to_check)) # Remove duplicates

        if keywords_to_check:
            print(f"Filtering based on keywords: {keywords_to_check}") # Debug print
            try:
                # Create boolean masks for occupation and topics
                occupation_mask = pd.Series(False, index=df_filtered.index)
                topics_mask = pd.Series(False, index=df_filtered.index)

                if occupation_col:
                    occupation_str_series = df_filtered[occupation_col].astype(str).str.lower()
                    for keyword in keywords_to_check:
                         occupation_mask |= occupation_str_series.str.contains(keyword, regex=False) # Use contains for substring matching

                if topics_col:
                    topics_str_series = df_filtered[topics_col].astype(str).str.lower()
                    for keyword in keywords_to_check:
                        topics_mask |= topics_str_series.str.contains(keyword, regex=False)

                # Combine masks: keep row if keyword found in either occupation OR topics
                combined_mask = occupation_mask | topics_mask
                df_filtered = df_filtered[combined_mask]

            except KeyError as e:
                st.error(f"Keyword filtering error: Column '{e}' not found. Please check mapping.")
                st.warning("Proceeding without keyword filtering due to error.")
            except Exception as e:
                st.error(f"Error during keyword filtering: {e}")
                st.warning("Proceeding without keyword filtering due to error.")
        else:
            st.warning("Could not extract specific keywords from target audience description for filtering. Showing all mailable contacts.")
    elif target_audience_desc and target_audience_desc.lower() != 'any':
         st.warning("Target audience description provided, but 'Occupation' or 'Topics' columns not mapped. Cannot perform keyword filtering.")


    if df_filtered.empty:
         st.warning(f"No contacts matched the target audience description: '{target_audience_desc}' after initial filtering.")

    # Convert the final filtered DataFrame to list of dicts
    return df_filtered.to_dict('records')

def prepare_download_data(filtered_contacts, drafts, send_status):
    """
    Prepares data for CSV download, adding status and subject columns.
    Args:
        filtered_contacts (list): List of contact dictionaries (filtered list).
        drafts (dict): Dictionary of drafts {index: {'subject': str, 'body': str}}.
        send_status (dict): Dictionary of send statuses {index: 'sent'/'failed'/'pending'}.

    Returns:
        pandas.DataFrame: DataFrame ready for download.
    """
    if not filtered_contacts:
        return pd.DataFrame() # Return empty DataFrame if no contacts

    report_data = copy.deepcopy(filtered_contacts)

    for idx, contact_row in enumerate(report_data):
        status = send_status.get(idx, 'pending')
        if status == 'pending':
            status = 'Pending/Not Sent'

        subject = drafts.get(idx, {}).get('subject', 'N/A')

        contact_row['Send Status'] = status
        contact_row['Email Subject Drafted'] = subject

    report_df = pd.DataFrame(report_data)
    return report_df


# ==============================================================================
# Core Logic Functions (Email Generation & Sending)
# ==============================================================================

def parse_llm_output(llm_response):
    """Parses the LLM response string for Subject and Body."""
    subject_match = re.search(r"Subject:\s*(.*)", llm_response, re.IGNORECASE)
    body_match = re.search(r"---\s*(.*)", llm_response, re.DOTALL | re.IGNORECASE)
    subject = subject_match.group(1).strip() if subject_match else "Following Up"
    body = body_match.group(1).strip() if body_match else llm_response.strip() # Strip whitespace from body
    if not subject_match or not body_match:
         st.warning(f"Could not reliably parse Subject/Body markers from LLM draft output. Using best guess.")
         # Fallback logic if markers aren't found
         if "Subject:" in llm_response and "---" not in llm_response:
             parts = llm_response.split("Subject:", 1)
             subject = parts[1].split('\n', 1)[0].strip()
             body = parts[1].split('\n', 1)[1].strip() if '\n' in parts[1] else parts[1].strip()
         elif "---" in llm_response:
             parts = llm_response.split("---", 1)
             subject = "Following Up" # Assign default subject
             body = parts[1].strip()

    return subject, body

# MODIFIED: Takes vectorstore instead of raw context string
def generate_single_draft(contact_info, campaign_details, vectorstore, email_drafter, sender_name, column_mapping):
    """
    Generates a single email draft using the provided LLM chain, column mapping,
    and retrieves context from the vector store.
    Returns subject and body strings.
    """
    if not email_drafter:
        st.error("Email drafter agent not initialized.")
        return "Error: Agent Not Ready", "Could not generate draft because the AI component is not available."
    if not campaign_details:
        st.error("Campaign details are missing.")
        return "Error: Missing Details", "Could not generate draft because campaign details are missing."
    if not column_mapping:
        st.error("Column mapping is missing.")
        return "Error: Missing Mapping", "Could not generate draft because column mapping is missing."

    # --- Get data from contact_info using the mapping ---
    first_name = contact_info.get(column_mapping.get(APP_FIELD_FIRST_NAME, ''), '')
    last_name = contact_info.get(column_mapping.get(APP_FIELD_LAST_NAME, ''), '')
    email = contact_info.get(column_mapping.get(APP_FIELD_EMAIL, ''), '')
    occupation = contact_info.get(column_mapping.get(APP_FIELD_OCCUPATION, ''), '')
    topics = contact_info.get(column_mapping.get(APP_FIELD_TOPICS, ''), '')

    # --- Retrieve relevant context from Vector Store ---
    retrieved_context = "No specific knowledge context retrieved." # Default
    query = campaign_details.get('campaign_topic', '')
    if vectorstore and query:
        # Use the handler function to get context
        retrieved_context = kbh.get_relevant_context(query, vectorstore, k=4) # Get top 4 chunks
    elif not vectorstore:
        print("Vector store not available for context retrieval.")
    elif not query:
         print("Campaign topic missing, cannot query vector store effectively.")


    # --- Prepare input for the LLM prompt ---
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
        'retrieved_knowledge_context': retrieved_context # Pass retrieved context
    }

    try:
        llm_response = email_drafter.invoke(prompt_input)
        subject, body = parse_llm_output(llm_response)
        return subject, body
    except Exception as e:
        st.error(f"Error calling Gemini API for {first_name or email}: {e}")
        print(f"Gemini API Error for contact {email}: {e}")
        return f"Error Generating Draft", f"An error occurred while contacting the AI model: {e}"

def send_email_smtp(recipient, subject, body, config):
    """
    Connects to SMTP server using config details and sends the email.
    Returns True if successful, False otherwise.
    """
    sender_email = config.get('sender_email')
    sender_password = config.get('sender_password')
    smtp_server = config.get('smtp_server')
    smtp_port = config.get('smtp_port')

    if not sender_email or not sender_password:
        st.error("Sender email credentials not configured.")
        return False
    # Basic email format validation
    if not recipient or not re.match(r"[^@]+@[^@]+\.[^@]+", recipient):
        st.error(f"Invalid recipient email format: {recipient}")
        return False
    if not subject:
        st.warning("Sending email with an empty subject.")
        subject = "(No Subject)"
    if not body:
        st.warning(f"Sending email with empty body to {recipient}.")
        body = "(Empty Body)" # Avoid sending completely empty content

    message = EmailMessage()
    message['From'] = f"{config.get('sender_name', sender_email)} <{sender_email}>" # Add sender name
    message['To'] = recipient
    message['Subject'] = subject
    message.set_content(body) # Assumes plain text

    # Use certifi's CA bundle for SSL context
    context = ssl.create_default_context(cafile=certifi.where())

    try:
        with smtplib.SMTP_SSL(smtp_server, smtp_port, context=context) as server:
            server.login(sender_email, sender_password)
            server.send_message(message)
        print(f"Successfully sent email to {recipient}") # Log success
        return True
    except smtplib.SMTPAuthenticationError:
        st.error("SMTP Authentication Failed. Check EMAIL_ADDRESS/EMAIL_PASSWORD. For Gmail/Google Workspace, ensure 'Less Secure App Access' is ON or use an App Password.")
        print("SMTP Authentication Error.") # Log error
        return False
    except smtplib.SMTPRecipientsRefused as e:
         st.error(f"Recipient refused for {recipient}: {e.recipients}")
         print(f"SMTP Recipient Refused for {recipient}: {e.recipients}") # Log error
         return False
    except Exception as e:
        st.error(f"Error sending email to {recipient}: {e}")
        print(f"Generic SMTP Error for {recipient}: {e}") # Log error
        return False

# ==============================================================================
# Helper Functions
# ==============================================================================

def format_chat_history(history):
    """Formats chat history for the analysis prompt."""
    formatted_history = ""
    for msg in history:
        role = "User" if msg["role"] == "user" else "Assistant"
        # Basic sanitization: remove potential prompt injection markers
        content = str(msg['content']).replace("{", "{{").replace("}", "}}")
        formatted_history += f"{role}: {content}\n"
    return formatted_history.strip()

def initialize_session_state():
    """Initializes Streamlit session state variables if they don't exist."""
    # --- Application Flow and Data ---
    if 'app_stage' not in st.session_state:
        st.session_state.app_stage = "upload_contacts" # Start at contact file upload
    if 'uploaded_contact_file_state' not in st.session_state:
         st.session_state.uploaded_contact_file_state = None # Track uploaded contact file
    if 'all_contacts_df' not in st.session_state:
        st.session_state.all_contacts_df = None # Loaded DataFrame
    if 'csv_columns' not in st.session_state:
        st.session_state.csv_columns = [] # Columns from uploaded file
    if 'column_mapping' not in st.session_state:
        st.session_state.column_mapping = {} # User's column mapping
    if 'mapping_complete' not in st.session_state:
        st.session_state.mapping_complete = False # Flag if mapping is done

    # --- Knowledge Base ---
    # Removed uploaded_knowledge_file_state and knowledge_content
    if 'knowledge_base_ready' not in st.session_state:
        # Check if the persistent directory exists as an initial check
        st.session_state.knowledge_base_ready = os.path.exists(VECTORSTORE_PERSIST_DIR)

    # --- Campaign Planning ---
    if 'messages' not in st.session_state:
        st.session_state.messages = [{
            "role": "assistant",
            "content": """Hi! Let's plan your email campaign. Please tell me about:
\n1.  **Campaign Topic:** What is the main subject of the email? (e.g., New product, webinar invite, special offer)
\n2.  **Target Audience:** Who should receive this email? Describe them (e.g., 'Software Engineers interested in AI', 'Marketing Managers', 'Anyone interested in leadership') - I'll use the 'Occupation' and 'Topics' columns you mapped.
\n3.  **Call to Action:** What do you want the recipients to do after reading? (e.g., 'Schedule a meeting', 'Visit website', 'Register now')
\n(I'll use the persisted knowledge base if available!)""" # MODIFIED Message
        }]
    if 'campaign_details' not in st.session_state:
        st.session_state.campaign_details = None
    if 'analysis_error' not in st.session_state:
        st.session_state.analysis_error = False

    # --- Email Generation & Sending ---
    if 'filtered_contacts_list' not in st.session_state:
        st.session_state.filtered_contacts_list = []
    if 'drafts' not in st.session_state:
        st.session_state.drafts = {} # {index_in_filtered_list: {'subject': str, 'body': str}}
    if 'send_status' not in st.session_state:
        st.session_state.send_status = {} # {index_in_filtered_list: 'pending'/'sent'/'failed'}
    if 'selected_for_send' not in st.session_state:
        st.session_state.selected_for_send = set() # Store indices selected for bulk send

    # --- Configuration & Setup Status ---
    if 'config_ok' not in st.session_state:
        st.session_state.config_ok = False
    if 'llm_components_ready' not in st.session_state:
        st.session_state.llm_components_ready = False
    if 'embeddings_model' not in st.session_state: # NEW state for embeddings model
         st.session_state.embeddings_model = None

def reset_session_state_for_new_campaign():
    """Resets session state variables for starting a new campaign, keeping config and LLM components."""
    # Keep Config and LLM/Embeddings
    current_config = st.session_state.get('config', {})
    current_config_ok = st.session_state.get('config_ok', False)
    current_llm = st.session_state.get('llm')
    current_embeddings = st.session_state.get('embeddings_model') # Keep embeddings model
    current_analyzer = st.session_state.get('campaign_analyzer')
    current_drafter = st.session_state.get('email_drafter')
    current_llm_ready = st.session_state.get('llm_components_ready', False)
    # Keep knowledge_base_ready status - user might want to reuse existing KB
    current_kb_ready = st.session_state.get('knowledge_base_ready', False)


    # Clear most session state keys
    keys_to_reset = [
        'app_stage', 'uploaded_contact_file_state', 'all_contacts_df', 'csv_columns',
        'column_mapping', 'mapping_complete', 'messages', 'campaign_details',
        'analysis_error', 'filtered_contacts_list', 'drafts', 'send_status',
        'selected_for_send'
        # Don't reset knowledge_base_ready here
    ]
    # Also clear widget states related to drafts and selection
    for key in list(st.session_state.keys()):
         if key.startswith("subject_edit_") or key.startswith("body_edit_") or key.startswith("select_"):
             del st.session_state[key]

    for key in keys_to_reset:
        if key in st.session_state:
            del st.session_state[key]


    # Re-initialize state (will preserve kb_ready if it was True)
    initialize_session_state()

    # Restore config and LLM components
    st.session_state.config = current_config
    st.session_state.config_ok = current_config_ok
    st.session_state.llm = current_llm
    st.session_state.embeddings_model = current_embeddings
    st.session_state.campaign_analyzer = current_analyzer
    st.session_state.email_drafter = current_drafter
    st.session_state.llm_components_ready = current_llm_ready
    st.session_state.knowledge_base_ready = current_kb_ready # Restore KB ready status

# NEW: Callback function for individual checkboxes
def update_selection_set(idx):
    """
    Callback to update the selected_for_send set when an individual checkbox changes.
    Reads the current value of the checkbox from st.session_state using its key.
    """
    checkbox_key = f"select_{idx}"
    if st.session_state.get(checkbox_key, False): # Check if the widget is checked
        st.session_state.selected_for_send.add(idx)
    else:
        st.session_state.selected_for_send.discard(idx)


# ==============================================================================
# Streamlit UI Rendering Functions (Organized by Stage)
# ==============================================================================

def render_upload_contacts_stage():
    """Renders the UI for uploading the contact file."""
    st.header("Step 1: Upload Contact File")
    uploaded_file = st.file_uploader("Choose a CSV or Excel file", type=['csv', 'xlsx', 'xls'], key="contact_uploader")

    # Process file only if it's newly uploaded or hasn't been processed yet
    if uploaded_file is not None and uploaded_file != st.session_state.uploaded_contact_file_state:
        st.session_state.uploaded_contact_file_state = uploaded_file # Store the uploaded file object
        with st.spinner("Processing contact file..."):
            df, cols = load_contacts_from_file(uploaded_file)
        if df is not None and cols:
            st.session_state.all_contacts_df = df
            st.session_state.csv_columns = cols
            st.session_state.app_stage = "map_columns" # Proceed to mapping
            st.session_state.mapping_complete = False # Reset mapping status
            st.session_state.column_mapping = {} # Clear previous mapping
            st.rerun()
        else:
            # Error handled in load_contacts_from_file
            st.session_state.uploaded_contact_file_state = None # Reset if loading failed

    elif st.session_state.all_contacts_df is not None and st.session_state.csv_columns:
         st.info(f"Using previously uploaded contact file: {st.session_state.uploaded_contact_file_state.name}. Re-upload to change.")
         if st.button("Proceed with Current Contact File"):
              st.session_state.app_stage = "map_columns"
              st.rerun()


def render_map_columns_stage():
    """Renders the UI for mapping CSV columns to application fields."""
    st.header("Step 2: Map Contact Columns")

    if st.session_state.all_contacts_df is None or not st.session_state.csv_columns:
        st.error("Contact data not loaded. Please go back to upload.")
        if st.button("Go Back to Upload Contacts"):
            st.session_state.app_stage = "upload_contacts"
            st.rerun()
        return

    st.write("Please map the columns from your contact file to the required fields:")
    st.dataframe(st.session_state.all_contacts_df.head(), use_container_width=True)
    st.markdown("---")

    cols = st.session_state.csv_columns
    mapping = st.session_state.column_mapping.copy() # Load existing mapping for defaults

    # --- Required Mappings ---
    st.subheader("Required Fields")
    # Try to find likely default columns
    likely_email = next((c for c in cols if 'email' in c.lower()), cols[0] if cols else '')
    likely_mailability = next((c for c in cols if 'mailable' in c.lower() or 'opt_in' in c.lower() or 'consent' in c.lower()), cols[1] if len(cols) > 1 else (cols[0] if cols else ''))

    mapping[APP_FIELD_EMAIL] = st.selectbox(f"Email Address Column:", cols, key="map_email", index=cols.index(mapping.get(APP_FIELD_EMAIL, likely_email)) if mapping.get(APP_FIELD_EMAIL, likely_email) in cols else 0)
    mapping[APP_FIELD_MAILABILITY] = st.selectbox(f"Column Indicating Email Can Be Sent:", cols, key="map_mailability", index=cols.index(mapping.get(APP_FIELD_MAILABILITY, likely_mailability)) if mapping.get(APP_FIELD_MAILABILITY, likely_mailability) in cols else 0)
    mapping['mailability_yes_value'] = st.text_input("Value in the above column meaning 'Yes' (case-insensitive):", value=mapping.get('mailability_yes_value', "Yes"), key="map_mailability_yes")

    # --- Optional Mappings ---
    st.subheader("Recommended Fields")
    optional_cols_with_none = [""] + cols # Use empty string for 'None'
    likely_first_name = next((c for c in cols if 'first' in c.lower() and 'name' in c.lower()), '')
    likely_last_name = next((c for c in cols if 'last' in c.lower() and 'name' in c.lower()), '')

    default_first_name = mapping.get(APP_FIELD_FIRST_NAME, likely_first_name)
    default_last_name = mapping.get(APP_FIELD_LAST_NAME, likely_last_name)

    mapping[APP_FIELD_FIRST_NAME] = st.selectbox(f"First Name Column:", optional_cols_with_none, key="map_first_name", index=optional_cols_with_none.index(default_first_name) if default_first_name in optional_cols_with_none else 0)
    mapping[APP_FIELD_LAST_NAME] = st.selectbox(f"Last Name Column:", optional_cols_with_none, key="map_last_name", index=optional_cols_with_none.index(default_last_name) if default_last_name in optional_cols_with_none else 0)


    st.subheader("Optional Fields for Personalization/Filtering")
    likely_occupation = next((c for c in cols if 'occupation' in c.lower() or 'job' in c.lower() or 'title' in c.lower()), '')
    likely_topics = next((c for c in cols if 'topic' in c.lower() or 'interest' in c.lower() or 'tag' in c.lower()), '')
    likely_nogo = next((c for c in cols if 'nogo' in c.lower() or 'dne' in c.lower() or 'opt_out' in c.lower()), '')

    default_occupation = mapping.get(APP_FIELD_OCCUPATION, likely_occupation)
    default_topics = mapping.get(APP_FIELD_TOPICS, likely_topics)
    default_nogo = mapping.get(APP_FIELD_NOGO, likely_nogo)

    mapping[APP_FIELD_OCCUPATION] = st.selectbox(f"Occupation/Job Title Column:", optional_cols_with_none, key="map_occupation", index=optional_cols_with_none.index(default_occupation) if default_occupation in optional_cols_with_none else 0)
    mapping[APP_FIELD_TOPICS] = st.selectbox(f"Topics/Interests Column:", optional_cols_with_none, key="map_topics", index=optional_cols_with_none.index(default_topics) if default_topics in optional_cols_with_none else 0)
    mapping[APP_FIELD_NOGO] = st.selectbox(f"Column Indicating 'Do Not Email' (No-Go):", optional_cols_with_none, key="map_nogo", index=optional_cols_with_none.index(default_nogo) if default_nogo in optional_cols_with_none else 0)
    st.caption("If a 'No-Go' column is selected, contacts will be excluded if this column is *not* empty.")


    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Confirm Mapping", type="primary"):
            # Basic validation
            if not mapping.get(APP_FIELD_EMAIL) or not mapping.get(APP_FIELD_MAILABILITY):
                st.error("Please select columns for Email and Mailability.")
            elif not mapping.get('mailability_yes_value'):
                 st.error("Please enter the value indicating 'Yes' for mailability.")
            else:
                # Remove empty string selections before saving
                final_mapping = {k: v for k, v in mapping.items() if v} # Check for truthiness (non-empty string)
                st.session_state.column_mapping = final_mapping
                st.session_state.mapping_complete = True
                st.session_state.app_stage = "upload_knowledge" # Proceed to knowledge upload
                st.success("Column mapping confirmed!")
                time.sleep(1)
                st.rerun()
    with col2:
        if st.button("Go Back to Upload Contacts"):
            st.session_state.app_stage = "upload_contacts"
            # Clear potentially loaded data if going back
            st.session_state.all_contacts_df = None
            st.session_state.csv_columns = []
            st.session_state.uploaded_contact_file_state = None
            st.session_state.mapping_complete = False
            st.session_state.column_mapping = {}
            st.rerun()

# MODIFIED: Stage for building/updating the knowledge base vector store
def render_upload_knowledge_stage():
    """Renders the UI for uploading knowledge files and building the vector store."""
    st.header("Step 3: Build/Update Knowledge Base (Optional)")
    st.write("Upload **Text (.txt)**, **Markdown (.md)**, or **PDF (.pdf)** files to build or replace the local knowledge base used for email drafting.")
    st.caption("Note: Uploading new files here will **replace** the existing knowledge base.")

    # Check if embeddings model is ready
    embeddings_model = st.session_state.get('embeddings_model')
    if not embeddings_model:
         st.error("Embeddings model is not initialized. Please check your GOOGLE_API_KEY configuration.")
         st.stop() # Stop rendering this stage if embeddings aren't ready

    # Allow uploading multiple files
    uploaded_knowledge_files = st.file_uploader(
        "Choose Text, Markdown, or PDF files",
        type=['txt', 'md', 'pdf'],
        key="knowledge_uploader",
        accept_multiple_files=True # Allow multiple files
    )

    # Button to trigger processing
    if st.button("Build/Update Knowledge Base from Uploaded Files", disabled=not uploaded_knowledge_files):
        if uploaded_knowledge_files:
            with st.spinner("Processing knowledge files... This may take a moment."):
                # 1. Load documents from all uploaded files
                docs = kbh.load_multiple_documents(uploaded_knowledge_files)

                if docs:
                    # 2. Split documents into chunks
                    chunks = kbh.split_documents(docs)

                    if chunks:
                        # 3. Create and persist the vector store (overwrites existing)
                        success = kbh.create_and_persist_vectorstore(chunks, embeddings_model, VECTORSTORE_PERSIST_DIR)
                        if success:
                            st.session_state.knowledge_base_ready = True
                            st.success(f"Knowledge base built/updated successfully from {len(uploaded_knowledge_files)} file(s)!")
                        else:
                            st.session_state.knowledge_base_ready = False
                            st.error("Failed to create or save the knowledge base.")
                    else:
                        st.session_state.knowledge_base_ready = False
                        st.error("Failed to split documents into chunks.")
                else:
                    st.session_state.knowledge_base_ready = False
                    st.error("Failed to load any documents from the uploaded files.")
            # Rerun to update status display after processing
            st.rerun()

    st.markdown("---")
    # Display current knowledge base status
    if st.session_state.get('knowledge_base_ready', False):
        st.info(f"✅ Knowledge base is ready (stored in '{VECTORSTORE_PERSIST_DIR}'). It will be used for drafting emails.")
        # Optionally add a button to clear the knowledge base?
        # if st.button("Clear Knowledge Base"):
        #     # Logic to remove the persist directory
        #     pass
    else:
        st.warning("🟡 Knowledge base is not ready or does not exist. Emails will be drafted without knowledge context.")

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        # Always allow proceeding, KB is optional
        if st.button("Proceed to Campaign Description", type="primary"):
            st.session_state.app_stage = "chat"
            st.rerun()
    with col2:
         if st.button("Go Back to Column Mapping"):
            st.session_state.app_stage = "map_columns"
            st.rerun()


def render_chat_stage(config_ok, llm_components_ready):
    """Renders the UI for the campaign description chat stage."""
    st.header("Step 4: Describe Your Campaign")

    # Display knowledge base status
    if st.session_state.get('knowledge_base_ready'):
        st.info("ℹ️ A knowledge base is available and will be used for context during drafting.")
    else:
        st.info("ℹ️ No knowledge base available. Drafting will rely solely on campaign description.")

    # Display chat messages
    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    # Get user input
    if prompt := st.chat_input("Your campaign details..."):
        if not config_ok or not llm_components_ready:
            st.error("Configuration or AI components not ready. Cannot process chat.")
            if not config_ok: st.error("Please ensure Email/API keys are set in the .env file.")
            if not llm_components_ready: st.error("AI components failed to initialize. Check API key and model access.")
        else:
            st.session_state.messages.append({"role": "user", "content": prompt})
            st.chat_message("user").write(prompt)
            st.session_state.app_stage = "analyze_button" # Set stage for button
            st.rerun()

def render_analyze_button_stage():
    """Renders the button to trigger campaign analysis."""
    st.header("Step 4: Describe Your Campaign")
    # Display chat messages again
    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    st.markdown("---")
    # Disable button if AI components aren't ready
    ai_ready = st.session_state.config_ok and st.session_state.llm_components_ready
    if not ai_ready:
         st.warning("AI components not ready. Cannot analyze campaign. Please check configuration.")

    if st.button("Analyze Campaign Description", type="primary", disabled=not ai_ready):
        st.session_state.app_stage = "analyze_processing"
        st.rerun()

def render_analyze_processing_stage(campaign_analyzer):
    """Handles the AI analysis process and updates the state."""
    st.header("Step 5: Analyzing Campaign & Filtering Contacts...")
    st.session_state.analysis_error = False # Reset error flag
    if not campaign_analyzer:
         st.error("Campaign Analyzer component is not ready. Cannot analyze.")
         st.session_state.analysis_error = True
         # Fallback: Use basic mailability filter only
         if st.session_state.all_contacts_df is not None and st.session_state.mapping_complete:
             st.warning("AI Analyzer failed. Filtering contacts based on 'Mailability' and 'No-Go' columns only.")
             mailability_col = st.session_state.column_mapping.get(APP_FIELD_MAILABILITY)
             mailability_yes_value = st.session_state.column_mapping.get('mailability_yes_value', 'yes')
             nogo_col = st.session_state.column_mapping.get(APP_FIELD_NOGO)
             df = st.session_state.all_contacts_df
             try:
                df_fallback = df.copy()
                df_fallback[mailability_col] = df_fallback[mailability_col].astype(str).str.strip().str.lower()
                df_fallback = df_fallback[df_fallback[mailability_col] == mailability_yes_value.lower()]
                if nogo_col:
                    df_fallback[nogo_col] = df_fallback[nogo_col].astype(str).str.strip()
                    df_fallback = df_fallback[df_fallback[nogo_col] == '']
                st.session_state.filtered_contacts_list = df_fallback.to_dict('records')
             except Exception as fallback_e:
                 st.error(f"Fallback filtering failed: {fallback_e}")
                 st.session_state.filtered_contacts_list = []
         else:
              st.session_state.filtered_contacts_list = []
         st.session_state.app_stage = "confirm"
         time.sleep(1) # Pause to show message
         st.rerun()
         return # Exit the function

    with st.spinner("AI is analyzing the campaign description & filtering contacts..."):
        try:
            chat_history_formatted = format_chat_history(st.session_state.messages)
            # Invoke the analyzer chain
            analysis_result = campaign_analyzer.invoke({"chat_history": chat_history_formatted})

            # Validate analysis result structure
            if not isinstance(analysis_result, dict) or not all(k in analysis_result for k in ['campaign_topic', 'target_audience', 'call_to_action', 'email_tone']):
                 raise ValueError("AI analysis did not return the expected JSON structure.")

            st.session_state.campaign_details = analysis_result # Store the dict

            # Filter contacts based on the analysis AND the mapping
            st.session_state.filtered_contacts_list = filter_contacts_based_on_analysis(
                st.session_state.all_contacts_df,
                st.session_state.campaign_details,
                st.session_state.column_mapping # Pass the mapping
            )
            st.session_state.app_stage = "confirm"

        except (json.JSONDecodeError, ValueError) as json_e:
             st.error(f"Error processing AI analysis result: {json_e}")
             st.warning("Could not parse campaign details from AI. Filtering contacts based on 'Mailability' and 'No-Go' columns only.")
             st.session_state.analysis_error = True
             # Fallback logic (same as above)
             if st.session_state.all_contacts_df is not None and st.session_state.mapping_complete:
                 mailability_col = st.session_state.column_mapping.get(APP_FIELD_MAILABILITY)
                 mailability_yes_value = st.session_state.column_mapping.get('mailability_yes_value', 'yes')
                 nogo_col = st.session_state.column_mapping.get(APP_FIELD_NOGO)
                 df = st.session_state.all_contacts_df
                 try:
                    df_fallback = df.copy()
                    df_fallback[mailability_col] = df_fallback[mailability_col].astype(str).str.strip().str.lower()
                    df_fallback = df_fallback[df_fallback[mailability_col] == mailability_yes_value.lower()]
                    if nogo_col:
                        df_fallback[nogo_col] = df_fallback[nogo_col].astype(str).str.strip()
                        df_fallback = df_fallback[df_fallback[nogo_col] == '']
                    st.session_state.filtered_contacts_list = df_fallback.to_dict('records')
                 except Exception as fallback_e:
                     st.error(f"Fallback filtering failed: {fallback_e}")
                     st.session_state.filtered_contacts_list = []
             else:
                 st.session_state.filtered_contacts_list = []
             st.session_state.app_stage = "confirm" # Still proceed to confirmation

        except Exception as e:
            st.error(f"Error during campaign analysis or filtering: {e}")
            st.warning("Could not automatically filter contacts based on analysis. Filtering contacts based on 'Mailability' and 'No-Go' columns only.")
            st.session_state.analysis_error = True
            # Fallback logic (same as above)
            if st.session_state.all_contacts_df is not None and st.session_state.mapping_complete:
                 mailability_col = st.session_state.column_mapping.get(APP_FIELD_MAILABILITY)
                 mailability_yes_value = st.session_state.column_mapping.get('mailability_yes_value', 'yes')
                 nogo_col = st.session_state.column_mapping.get(APP_FIELD_NOGO)
                 df = st.session_state.all_contacts_df
                 try:
                    df_fallback = df.copy()
                    df_fallback[mailability_col] = df_fallback[mailability_col].astype(str).str.strip().str.lower()
                    df_fallback = df_fallback[df_fallback[mailability_col] == mailability_yes_value.lower()]
                    if nogo_col:
                        df_fallback[nogo_col] = df_fallback[nogo_col].astype(str).str.strip()
                        df_fallback = df_fallback[df_fallback[nogo_col] == '']
                    st.session_state.filtered_contacts_list = df_fallback.to_dict('records')
                 except Exception as fallback_e:
                     st.error(f"Fallback filtering failed: {fallback_e}")
                     st.session_state.filtered_contacts_list = []
            else:
                 st.session_state.filtered_contacts_list = []
            st.session_state.app_stage = "confirm" # Still proceed to confirmation

    st.rerun() # Move to confirmation stage

def render_confirm_stage():
    """Renders the UI for confirming the filtered recipient list."""
    st.header("Step 6: Confirm Recipients")

    column_mapping = st.session_state.get('column_mapping', {})

    # Display Analysis Results or Warnings
    if st.session_state.analysis_error:
        st.warning("Analysis failed or couldn't extract clear criteria. Showing contacts filtered only by 'Mailability' and 'No-Go' rules.")
    elif st.session_state.campaign_details:
        st.subheader("AI Campaign Analysis Results:")
        # Use columns for better layout
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"**Campaign Topic:**")
            st.markdown(f"{st.session_state.campaign_details.get('campaign_topic', 'N/A')}")
            st.markdown(f"**Call to Action:**")
            st.markdown(f"{st.session_state.campaign_details.get('call_to_action', 'N/A')}")
        with col2:
            st.markdown(f"**Target Audience (Interpretation):**")
            st.markdown(f"{st.session_state.campaign_details.get('target_audience', 'N/A')}")
            st.markdown(f"**Email Tone:**")
            st.markdown(f"{st.session_state.campaign_details.get('email_tone', 'N/A')}")
        st.markdown("---")
    else:
        # This case should ideally not happen if analysis_error is handled, but good fallback
        st.warning("Campaign details not available. Filtering may be based on mailability rules only.")


    st.subheader(f"Proposed Recipients ({len(st.session_state.filtered_contacts_list)}):")
    if not st.session_state.filtered_contacts_list:
        st.warning("No contacts match the criteria based on the mailability rules and campaign analysis.")
        if st.button("Go Back to Chat"):
            st.session_state.app_stage = "chat"
            # Clear analysis results if going back
            st.session_state.campaign_details = None
            st.session_state.analysis_error = False
            st.rerun()
        if st.button("Adjust Column Mapping"):
             st.session_state.app_stage = "map_columns"
             st.rerun()

    else:
        # --- Determine columns to display based on mapping ---
        # Prioritize showing essential + personalization columns if mapped
        cols_priority = [APP_FIELD_FIRST_NAME, APP_FIELD_LAST_NAME, APP_FIELD_EMAIL, APP_FIELD_OCCUPATION, APP_FIELD_TOPICS]
        mapped_cols_for_display = [column_mapping.get(field) for field in cols_priority if column_mapping.get(field)]

        # Ensure email is always included if mapped, even if not in priority list somehow
        email_col_name = column_mapping.get(APP_FIELD_EMAIL)
        if not email_col_name:
             st.error("Critical Error: Email column is not mapped. Cannot proceed.")
             if st.button("Go Back to Column Mapping"):
                 st.session_state.app_stage = "map_columns"
                 st.rerun()
             return
        if email_col_name not in mapped_cols_for_display:
             mapped_cols_for_display.insert(0, email_col_name) # Add email to the beginning if missing

        try:
            filtered_df_display = pd.DataFrame(st.session_state.filtered_contacts_list)
            # Ensure only existing columns are selected from the dataframe
            valid_display_cols = [col for col in mapped_cols_for_display if col in filtered_df_display.columns]
            if not valid_display_cols:
                 raise KeyError("None of the mapped display columns exist in the filtered data.")

            # Create the display dataframe with only valid columns
            filtered_df_display = filtered_df_display[valid_display_cols]

            # Rename columns for better readability
            # Create reverse mapping from CSV col name back to internal App Field name
            reverse_rename_map = {v: k for k, v in column_mapping.items()}
            # Generate display names (e.g., 'First Name') from internal names
            display_rename_map = {
                csv_col: reverse_rename_map.get(csv_col, csv_col).replace('_', ' ').title()
                for csv_col in valid_display_cols
            }
            filtered_df_display = filtered_df_display.rename(columns=display_rename_map)

            st.dataframe(filtered_df_display, use_container_width=True)
        except KeyError as e:
             st.error(f"Display error: Column '{e}' (expected based on mapping) not found in filtered data. This might indicate an issue during filtering or mapping.")
             st.write("Filtered Data Sample (first 2 rows):", st.session_state.filtered_contacts_list[:2]) # Show sample raw data
             if st.button("Go Back to Column Mapping"):
                 st.session_state.app_stage = "map_columns"
                 st.rerun()
             return
        except Exception as e:
             st.error(f"An unexpected error occurred while preparing the display table: {e}")
             return


        col1_confirm, col2_confirm = st.columns(2)
        with col1_confirm:
            # Disable button if AI components aren't ready for the next step
            drafting_ready = st.session_state.config_ok and st.session_state.llm_components_ready
            tooltip_msg = "" if drafting_ready else "AI components not ready. Check config."

            if st.button(f"Confirm and Proceed to Drafts ({len(st.session_state.filtered_contacts_list)})", type="primary", disabled=not drafting_ready, help=tooltip_msg):
                st.session_state.app_stage = "draft"
                st.session_state.drafts = {}
                st.session_state.send_status = {}
                st.session_state.selected_for_send = set() # Ensure selection is clear
                st.rerun()
        with col2_confirm:
            if st.button("Cancel and Revise Campaign"):
                st.session_state.app_stage = "chat"
                st.session_state.campaign_details = None
                st.session_state.filtered_contacts_list = []
                st.session_state.analysis_error = False # Reset error flag
                st.rerun()

def render_draft_stage(email_drafter, config):
    """Renders the UI for generating email drafts."""
    st.header("Step 7: Generate Email Drafts")

    column_mapping = st.session_state.get('column_mapping', {})
    embeddings_model = st.session_state.get('embeddings_model') # Get embeddings model

    if not column_mapping or not st.session_state.mapping_complete:
        st.error("Column mapping is not complete. Please go back and map columns.")
        if st.button("Go to Column Mapping"): st.session_state.app_stage = "map_columns"; st.rerun()
        return
    if not embeddings_model:
         st.error("Embeddings model not available. Cannot load knowledge base. Check config.")
         if st.button("Go Back to Config/Restart"): st.session_state.app_stage = "upload_contacts"; st.rerun() # Go back to start
         return

    st.markdown(f"Generating drafts for the **{len(st.session_state.filtered_contacts_list)}** confirmed recipients based on:")
    if st.session_state.campaign_details:
        # Display campaign details (layout improved)
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"**Campaign Topic:** {st.session_state.campaign_details.get('campaign_topic', 'N/A')}")
            st.markdown(f"**Call to Action:** {st.session_state.campaign_details.get('call_to_action', 'N/A')}")
        with col2:
            st.markdown(f"**Target Audience:** {st.session_state.campaign_details.get('target_audience', 'N/A')}")
            st.markdown(f"**Email Tone:** {st.session_state.campaign_details.get('email_tone', 'N/A')}")
    else:
        st.warning("Campaign details missing (analysis might have failed). Using generic settings.")

    # --- Load Vector Store ---
    vectorstore = None
    if st.session_state.get('knowledge_base_ready'):
        with st.spinner("Loading knowledge base..."):
            vectorstore = kbh.load_vectorstore(embeddings_model, VECTORSTORE_PERSIST_DIR)
        if vectorstore:
            st.info("✅ Knowledge base loaded successfully. Context will be retrieved for drafts.")
        else:
            st.warning("🟡 Failed to load knowledge base. Drafting without retrieved context.")
            st.session_state.knowledge_base_ready = False # Mark as not ready if loading failed
    else:
        st.info("ℹ️ No knowledge base available. Drafting without retrieved context.")
    st.markdown("---")

    # Check if components needed for drafting are ready
    drafting_ready = st.session_state.config_ok and st.session_state.llm_components_ready and email_drafter is not None

    if st.button("✨ Generate Drafts", disabled=not drafting_ready or not st.session_state.filtered_contacts_list):
        if not st.session_state.campaign_details and not st.session_state.analysis_error:
             st.error("Cannot generate drafts without campaign details. Please go back and analyze the campaign.")
        elif not st.session_state.filtered_contacts_list:
             st.warning("No contacts confirmed to generate drafts for.")
        else:
            total_contacts = len(st.session_state.filtered_contacts_list)
            progress_bar = st.progress(0, text="Initializing draft generation...")
            status_text = st.empty()
            st.session_state.drafts = {} # Reset drafts
            st.session_state.send_status = {} # Reset status
            st.session_state.selected_for_send = set() # Reset selection

            sender_name = config.get('sender_name', 'Your Name [Default]')
            # Use campaign details if available, otherwise use defaults
            campaign_details_to_use = st.session_state.campaign_details or {
                'campaign_topic': 'Follow Up', 'email_tone': 'Professional', 'call_to_action': 'Engage further'
            }
            if not st.session_state.campaign_details:
                 status_text.warning("Using default campaign details as analysis results were not available.")

            # Loop through the CONFIRMED FILTERED contacts
            generation_errors = 0
            for i, contact in enumerate(st.session_state.filtered_contacts_list):
                first_name_col = column_mapping.get(APP_FIELD_FIRST_NAME)
                email_col = column_mapping.get(APP_FIELD_EMAIL)
                contact_identifier = contact.get(first_name_col) or contact.get(email_col) or f'Contact {i+1}'
                status_text.text(f"Generating draft for {contact_identifier} ({i+1}/{total_contacts})...")

                # Generate draft for the current contact, passing the loaded vectorstore
                subject, body = generate_single_draft(
                    contact,
                    campaign_details_to_use,
                    vectorstore, # Pass the loaded vector store (or None)
                    email_drafter,
                    sender_name,
                    column_mapping
                )

                # Store draft and status
                st.session_state.drafts[i] = {'subject': subject, 'body': body}
                st.session_state.send_status[i] = 'pending'
                if "Error Generating Draft" in subject:
                    generation_errors += 1
                    st.session_state.send_status[i] = 'failed' # Mark as failed

                progress_bar.progress((i + 1) / total_contacts, text=f"Generated draft for {contact_identifier} ({i+1}/{total_contacts})")
                # time.sleep(0.05) # Slightly faster UI update

            final_message = f"✅ Generated drafts for {total_contacts - generation_errors} / {total_contacts} confirmed contacts."
            if generation_errors > 0:
                 final_message += f" ({generation_errors} errors occurred - check logs/warnings)."
                 status_text.error(final_message)
            else:
                 status_text.success(final_message)

            progress_bar.empty()
            st.session_state.app_stage = "send" # Move to send stage
            time.sleep(2 if generation_errors > 0 else 1)
            st.rerun()
    elif not drafting_ready:
         st.warning("Cannot generate drafts. Check configuration and AI component status in the sidebar.")
    elif not st.session_state.filtered_contacts_list:
         st.warning("No contacts confirmed to generate drafts for.")


def render_send_stage(config):
    """Renders the UI for reviewing and sending emails with bulk actions."""
    st.header("Step 8: Review Drafts and Send Emails")

    # --- Initial Checks ---
    column_mapping = st.session_state.get('column_mapping', {})
    if not column_mapping or not st.session_state.mapping_complete:
        st.error("Column mapping is not complete."); st.stop()
    if not st.session_state.drafts:
         st.warning("No drafts generated yet or generation failed."); st.stop()
    email_col = column_mapping.get(APP_FIELD_EMAIL)
    first_name_col = column_mapping.get(APP_FIELD_FIRST_NAME)
    if not email_col: st.error("Email column mapping is missing."); st.stop()

    # --- Bulk Action Controls ---
    st.subheader("Bulk Send Actions")
    col_bulk1, col_bulk2, col_bulk3 = st.columns(3)

    pending_indices = {idx for idx, status in st.session_state.send_status.items() if status == 'pending'}
    num_pending = len(pending_indices)

    with col_bulk1:
        if st.button(f"Select All Pending ({num_pending})", disabled=num_pending==0):
            st.session_state.selected_for_send = pending_indices.copy(); st.rerun()
    with col_bulk2:
        if st.button("Unselect All", disabled=not st.session_state.selected_for_send):
            st.session_state.selected_for_send = set(); st.rerun()
    with col_bulk3:
        indices_to_send_now = sorted(list(st.session_state.selected_for_send.intersection(pending_indices)))
        num_to_send = len(indices_to_send_now)
        send_selected_disabled = num_to_send == 0 or not st.session_state.config_ok
        tooltip_send_bulk = "" if st.session_state.config_ok else "Email sending disabled. Check config."

        if st.button(f"🚀 Send to Selected ({num_to_send})", disabled=send_selected_disabled, help=tooltip_send_bulk):
            if not indices_to_send_now: st.warning("No valid pending contacts selected.")
            else:
                send_placeholder = st.empty()
                send_progress_bar = st.progress(0)
                total_to_send = len(indices_to_send_now)
                num_sent, num_failed = 0, 0
                send_placeholder.info(f"Starting bulk send for {total_to_send} emails...")

                for i, idx in enumerate(indices_to_send_now):
                    if idx < len(st.session_state.filtered_contacts_list) and st.session_state.send_status.get(idx) == 'pending':
                        contact = st.session_state.filtered_contacts_list[idx]
                        recipient = contact.get(email_col)
                        name = contact.get(first_name_col)
                        contact_id = name or recipient or f'Contact {idx+1}'
                        subject = st.session_state.get(f"subject_edit_{idx}", st.session_state.drafts.get(idx, {}).get('subject', 'Error: Subject Missing'))
                        body = st.session_state.get(f"body_edit_{idx}", st.session_state.drafts.get(idx, {}).get('body', 'Error: Body Missing'))

                        if "Error:" in subject or "Error:" in body:
                             st.warning(f"Skipped {contact_id}: Draft content missing/invalid (Index {idx}).")
                             st.session_state.send_status[idx] = 'failed'; num_failed += 1
                        elif recipient:
                            send_placeholder.info(f"Sending to {contact_id} ({recipient}) ({i+1}/{total_to_send})...")
                            success = send_email_smtp(recipient, subject, body, config)
                            if success: st.session_state.send_status[idx] = 'sent'; num_sent += 1
                            else: st.session_state.send_status[idx] = 'failed'; num_failed += 1
                        else:
                             st.warning(f"Skipped {contact_id}: No valid email address.")
                             st.session_state.send_status[idx] = 'failed'; num_failed += 1

                        send_progress_bar.progress((i + 1) / total_to_send)
                        if i < total_to_send - 1: time.sleep(SEND_DELAY_SECONDS)

                final_msg = f"Bulk send complete. Sent: {num_sent}, Failed/Skipped: {num_failed}"
                if num_failed > 0: send_placeholder.warning(final_msg + ". Check logs/warnings.")
                else: send_placeholder.success(final_msg)
                st.session_state.selected_for_send = set()
                time.sleep(2); st.rerun()

    st.markdown("---")

    # --- Display Drafts with Checkboxes ---
    st.subheader("Review/Edit Individual Drafts")
    num_displayed = 0
    num_failed_gen = sum(1 for idx, status in st.session_state.send_status.items() if status == 'failed' and idx in st.session_state.drafts and "Error Generating Draft" in st.session_state.drafts[idx]['subject'])
    if num_failed_gen > 0: st.warning(f"{num_failed_gen} drafts failed generation and cannot be sent.")

    sorted_draft_indices = sorted(st.session_state.drafts.keys())
    for idx in sorted_draft_indices:
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

            row_cols = st.columns([0.1, 0.9])
            with row_cols[0]:
                 # MODIFIED: Added on_change callback
                 st.checkbox("Select", key=f"select_{idx}",
                             value=(idx in st.session_state.selected_for_send),
                             on_change=update_selection_set, # Call the callback function
                             args=(idx,), # Pass the index to the callback
                             disabled=(not is_pending or gen_failed),
                             label_visibility="collapsed")
            with row_cols[1]:
                status_color = "green" if status == 'sent' else ("red" if status == 'failed' else "orange")
                title = f"{idx+1}. To: **{contact_id}** ({recipient or 'No Email!'}) - Status: :{status_color}[{status.upper()}]"
                with st.expander(title, expanded=False):
                    subj = st.text_input("Subject", draft_info['subject'], key=f"subject_edit_{idx}", disabled=(not is_pending or gen_failed))
                    body_txt = st.text_area("Body", draft_info['body'], height=200, key=f"body_edit_{idx}", disabled=(not is_pending or gen_failed))
                    send_one_disabled = not is_pending or gen_failed or not st.session_state.config_ok or not recipient
                    tooltip = ""
                    if gen_failed: tooltip = "Draft generation failed."
                    elif status != 'pending': tooltip = f"Status is {status}."
                    elif not st.session_state.config_ok: tooltip = "Config error prevents sending."
                    elif not recipient: tooltip = "Invalid/missing recipient email."

                    if st.button(f"🚀 Send Only This Email", key=f"send_one_{idx}", disabled=send_one_disabled, help=tooltip):
                        ph = st.empty(); ph.info(f"Sending to {recipient}...")
                        success = send_email_smtp(recipient, subj, body_txt, config)
                        if success:
                            st.session_state.send_status[idx] = 'sent'
                            st.session_state.selected_for_send.discard(idx) # Ensure it's unselected if sent individually
                            ph.success(f"Email sent successfully to {recipient}!")
                        else:
                            st.session_state.send_status[idx] = 'failed'
                            ph.error(f"Failed to send email to {recipient}.")
                        time.sleep(1); st.rerun()
        else: st.warning(f"Data inconsistency: Draft index {idx} out of bounds.")
    if num_displayed == 0 and st.session_state.drafts: st.warning("Could not display drafts (inconsistency or all failed generation).")

    # --- Final Actions ---
    st.markdown("---"); st.subheader("Campaign Finish Actions")
    col_final1, col_final2 = st.columns(2)
    with col_final1:
        if st.session_state.filtered_contacts_list and st.session_state.drafts:
            df_dl = prepare_download_data(st.session_state.filtered_contacts_list, st.session_state.drafts, st.session_state.send_status)
            if not df_dl.empty:
                try:
                    csv_data = df_dl.to_csv(index=False).encode('utf-8')
                    st.download_button(label="Download Send Report (CSV)", data=csv_data, file_name='email_campaign_report.csv', mime='text/csv', key='dl_report_btn')
                except Exception as e: st.error(f"Failed to generate download file: {e}")
            else: st.info("No data available to download.")
        else: st.info("No contacts/drafts processed to download.")
    with col_final2:
        if st.button("Start New Campaign (Reset)", key='reset_campaign_btn'):
            reset_session_state_for_new_campaign()
            st.success("Campaign reset. Ready for new upload."); time.sleep(1); st.rerun()

# ==============================================================================
# Main Application Logic
# ==============================================================================

def main():
    """Main function to run the Streamlit application."""
    st.set_page_config(layout="wide", page_title="AI Email Campaign Assistant")
    st.title("📧 AI Email Campaign Assistant (Vector Store KB)")
    st.markdown("Upload contacts & knowledge files (optional), map columns, describe campaign, analyze, confirm, generate drafts with KB context, review, and send!")

    # --- Initialize State ---
    initialize_session_state()

    # --- Load Config and Check Status ---
    if 'config' not in st.session_state:
        config, config_ok = load_and_validate_config()
        st.session_state.config = config
        st.session_state.config_ok = config_ok
    else:
        config = st.session_state.config
        config_ok = st.session_state.config_ok
        load_and_validate_config() # Re-render sidebar

    # --- Initialize LLM & Embeddings Components ---
    if config_ok and not st.session_state.get('llm_components_ready', False):
        llm, embeddings_model, campaign_analyzer, email_drafter = initialize_llm_components(config)
        if all([llm, embeddings_model, campaign_analyzer, email_drafter]):
            st.session_state.llm = llm
            st.session_state.embeddings_model = embeddings_model # Store embeddings model
            st.session_state.campaign_analyzer = campaign_analyzer
            st.session_state.email_drafter = email_drafter
            st.session_state.llm_components_ready = True
        else:
            st.session_state.llm_components_ready = False
            # Clear potentially partially initialized components
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
            else: st.warning("No contact data. Redirecting..."); st.session_state.app_stage = "upload_contacts"; time.sleep(1); st.rerun()
        elif app_stage == "upload_knowledge":
            if st.session_state.mapping_complete: render_upload_knowledge_stage()
            else: st.warning("Mapping not complete. Redirecting..."); st.session_state.app_stage = "map_columns"; time.sleep(1); st.rerun()
        elif app_stage == "chat":
            if st.session_state.mapping_complete: render_chat_stage(config_ok, st.session_state.llm_components_ready)
            else: st.warning("Mapping not complete. Redirecting..."); st.session_state.app_stage = "map_columns"; time.sleep(1); st.rerun()
        elif app_stage == "analyze_button":
             if st.session_state.mapping_complete: render_analyze_button_stage()
             else: st.warning("Mapping not complete. Redirecting..."); st.session_state.app_stage = "map_columns"; time.sleep(1); st.rerun()
        elif app_stage == "analyze_processing":
             if st.session_state.mapping_complete: render_analyze_processing_stage(st.session_state.get('campaign_analyzer'))
             else: st.warning("Mapping not complete. Redirecting..."); st.session_state.app_stage = "map_columns"; time.sleep(1); st.rerun()
        elif app_stage == "confirm":
             if st.session_state.mapping_complete: render_confirm_stage()
             else: st.warning("Mapping not complete. Redirecting..."); st.session_state.app_stage = "map_columns"; time.sleep(1); st.rerun()
        elif app_stage == "draft":
             if st.session_state.mapping_complete: render_draft_stage(st.session_state.get('email_drafter'), config)
             else: st.warning("Mapping not complete. Redirecting..."); st.session_state.app_stage = "map_columns"; time.sleep(1); st.rerun()
        elif app_stage == "send":
             # Need mapping and drafts to exist for this stage
             if st.session_state.mapping_complete and st.session_state.get('drafts'): render_send_stage(config)
             elif not st.session_state.mapping_complete: st.warning("Mapping incomplete. Redirecting..."); st.session_state.app_stage = "map_columns"; time.sleep(1); st.rerun()
             else: st.warning("Drafts not generated. Redirecting..."); st.session_state.app_stage = "draft"; time.sleep(1); st.rerun()
    except Exception as main_e:
         st.error(f"An unexpected error occurred: {main_e}")
         st.error("Attempting to reset. Please try starting a new campaign.")
         # Log the full traceback for debugging
         import traceback
         print("--- Main Application Error ---")
         traceback.print_exc()
         print("-----------------------------")
         # Force reset on critical error
         reset_session_state_for_new_campaign()
         time.sleep(2)
         st.rerun()

if __name__ == "__main__":
    # Add basic error handling for library imports
    required_libs = ['PyPDF2', 'langchain', 'langchain_community', 'langchain_google_genai', 'faiss'] # Check for faiss too
    missing_libs = []
    for lib in required_libs:
        try:
            # Attempt to import the base package name (e.g., 'faiss' for 'faiss-cpu')
            base_lib = lib.split('-')[0]
            __import__(base_lib)
        except ImportError:
            missing_libs.append(lib)

    if missing_libs:
        install_commands = [f"pip install {lib}" + ("-cpu" if lib == "faiss" else "") for lib in missing_libs] # Add -cpu for faiss
        st.error(f"Required libraries missing: {', '.join(missing_libs)}. Please install them and restart:\n" + "\n".join(install_commands))
        st.stop()

    main()
