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

# --- LangChain / Gemini Imports ---
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.runnables import RunnablePassthrough

# --- Constants ---
SMTP_SERVER = 'smtp.gmail.com' # Example for Gmail
SMTP_PORT = 465
SEND_DELAY_SECONDS = 5 # Delay between sending emails

# --- Application's Internal Field Names ---
APP_FIELD_FIRST_NAME = 'first_name'
APP_FIELD_LAST_NAME = 'last_name'
APP_FIELD_EMAIL = 'email'
APP_FIELD_OCCUPATION = 'occupation'
APP_FIELD_TOPICS = 'topics'
APP_FIELD_MAILABILITY = 'mailability' # Column indicating if email can be sent
APP_FIELD_NOGO = 'nogo'             # Column indicating if contact should NOT be emailed

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

    st.sidebar.title("Sender Info") # Changed title

    # Load and Validate Google API Key (Validation still happens, but no sidebar message)
    config['google_api_key'] = os.getenv('GOOGLE_API_KEY')
    if not config['google_api_key']:
        # st.sidebar.error("🔴 GOOGLE_API_KEY is missing.") # Removed
        config_ok = False
    # else: # Removed
        # st.sidebar.write("🔑 Google API Key: Loaded") # Removed

    # Load and Validate Email Credentials (Validation still happens, but no sidebar message)
    config['sender_email'] = os.getenv('EMAIL_ADDRESS')
    config['sender_password'] = os.getenv('EMAIL_PASSWORD')

    if not config['sender_email']:
        # st.sidebar.error("🔴 EMAIL_ADDRESS is missing.") # Removed
        config_ok = False
    else:
        # Display Sender Email if available
        st.sidebar.write(f"Email: {config['sender_email']}") # Keep this, simplified label

    if not config['sender_password']:
        # st.sidebar.error("🔴 EMAIL_PASSWORD is missing.") # Removed
        config_ok = False
    # else: # Removed
        # st.sidebar.write("🔒 Email Password: Loaded") # Removed

    # Load Sender Name (with default)
    config['sender_name'] = os.getenv('YOUR_NAME', 'Your Name [Default]')
    # if not os.getenv('YOUR_NAME'): # Removed warning
        # st.sidebar.warning("🟠 YOUR_NAME is missing (using default).") # Removed
    # Display Sender Name
    st.sidebar.write(f"Name: {config['sender_name']}") # Keep this, simplified label


    # Add other config if needed
    config['smtp_server'] = SMTP_SERVER
    config['smtp_port'] = SMTP_PORT

    # Removed overall status messages from sidebar
    # if config_ok:
    #     st.sidebar.success("✅ Configuration loaded successfully.")
    # else:
    #     st.sidebar.error("❌ Please check your `.env` file for missing values.")
    #     # Keep main panel warning if needed, but removed from sidebar
    #     # st.warning("App functionality will be limited due to missing configuration.")

    return config, config_ok

def initialize_llm_components(config):
    """
    Initializes LangChain/Gemini components if the API key is available.
    Returns llm, campaign_analyzer, email_drafter objects.
    """
    llm = None
    campaign_analyzer = None
    email_drafter = None

    # Use st.sidebar for LLM status messages as they indicate readiness
    if not config.get('google_api_key'):
        st.sidebar.error("🔴 AI components require GOOGLE_API_KEY.") # Keep this essential feedback
        return llm, campaign_analyzer, email_drafter

    try:
        # Changed model to gemini-1.5-pro-latest
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest",
                                     google_api_key=config['google_api_key'],
                                     temperature=0.5)

        # 1. Campaign Analysis Chain
        analysis_prompt_template = ChatPromptTemplate.from_messages(
             [
                ("system", """You are an AI assistant helping to plan an email marketing campaign.
Analyze the following conversation history where the user describes their campaign goals.
Your task is to extract the key campaign parameters:
1.  **campaign_topic**: The central theme or subject of the email campaign (e.g., "New Product Launch", "Upcoming Webinar", "Data Analysis Trends").
2.  **target_audience**: Describe the target audience based on the user's description. This might include job roles (like 'Software Engineer', 'Marketing Specialist') or topics of interest (like 'Data Analysis', 'AI & Machine Learning') found in the contact list. Be descriptive (e.g., "Software Engineers interested in AI", "Marketing roles", "Anyone interested in Leadership"). Use "Any" if no specific audience is targeted or if it's unclear.
3.  **call_to_action**: What the user wants the recipient to do (e.g., "Schedule a meeting", "Visit website", "Register for event"). Default to "Engage further".
4.  **email_tone**: The desired tone of the email (e.g., "Professional", "Friendly", "Urgent", "Informative"). Infer from the user's description. Default to "Professional".

Format your output ONLY as a JSON object with these exact keys: campaign_topic, target_audience, call_to_action, email_tone. Example:
{{
"campaign_topic": "AI in Marketing Webinar",
"target_audience": "Marketing Specialists interested in AI & Machine Learning",
"call_to_action": "Register for webinar",
"email_tone": "Informative"
}}

Conversation History:
{chat_history}
"""),
                 ("human", "Based on the conversation history above, please provide the JSON output."),
            ]
        )
        campaign_analyzer = analysis_prompt_template | llm | JsonOutputParser()
        st.sidebar.info("✅ Campaign Analyzer Initialized.") # Keep AI status

        # 2. Email Drafting Chain
        draft_prompt_template = ChatPromptTemplate.from_messages(
            [
                ("system", """You are an AI assistant drafting personalized marketing emails.
You will receive information about the overall campaign and a specific contact.
Generate a concise, polite, and personalized draft email based ONLY on the provided information.
The email should reflect the campaign's topic and tone, and include the call to action.

**Campaign Details:**
* Topic: {campaign_topic}
* Tone: {email_tone}
* Call to Action: {call_to_action}

**Format your output EXACTLY as follows:**
Subject: <Email Subject Line>
---
<Email Body>

**Instructions:**
* Address the contact by their First Name ({first_name}).
* If available, subtly reference their Occupation ({occupation}) or Topic of Interest ({contact_topic}) if relevant to the campaign topic.
* Maintain the specified email tone.
* Include the call to action towards the end.
* Keep the email relatively brief and professional.
* End the email with the sender's name: {sender_name}

Do NOT include any other text, explanation, or preamble before "Subject:" or after the email body."""),
                ("human", """Draft an email for this contact:
First Name: {first_name}
Last Name: {last_name}
Occupation: {occupation}
Email: {contact_email}
Topic of Interest: {contact_topic}

Remember the campaign details: Topic='{campaign_topic}', Tone='{email_tone}', Call to Action='{call_to_action}'.""")
            ]
        )
        email_drafter = draft_prompt_template | llm | StrOutputParser()
        st.sidebar.info("✅ Email Drafter Initialized.") # Keep AI status

    except Exception as e:
        st.sidebar.error(f"🔴 Error initializing AI components: {e}") # Keep AI status
        llm, campaign_analyzer, email_drafter = None, None, None

    return llm, campaign_analyzer, email_drafter

# ==============================================================================
# Data Handling Functions
# ==============================================================================

def load_contacts_from_file(uploaded_file):
    """
    Loads contacts from the uploaded file (CSV or Excel).
    Returns a pandas DataFrame and list of columns, or (None, None) if loading fails.
    """
    if uploaded_file is None:
        st.warning("Please upload a file.")
        return None, None
    try:
        # Determine file type and read accordingly
        file_name = uploaded_file.name
        if file_name.endswith('.csv'):
            contacts_df = pd.read_csv(uploaded_file)
        elif file_name.endswith(('.xls', '.xlsx')):
             # Specify engine='openpyxl' for modern Excel files
             contacts_df = pd.read_excel(uploaded_file, engine='openpyxl')
        else:
            st.error("Unsupported file type. Please upload a CSV or Excel (.xlsx, .xls) file.")
            return None, None

        # Handle potential NaN values gracefully
        contacts_df = contacts_df.fillna('')
        columns = contacts_df.columns.tolist()
        # Keep success message for file loading in sidebar
        st.sidebar.success(f"Loaded {len(contacts_df)} contacts from '{file_name}'.")
        return contacts_df, columns

    except Exception as e:
        st.error(f"An error occurred reading the file: {e}")
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
        df_filtered = df_filtered[df_filtered[mailability_col].astype(str).str.strip().str.lower() == mailability_yes_value.lower()]

        # Apply No-Go filter if the column is mapped
        if nogo_col:
            # Keep rows where the No-Go column is empty (or NaN, already handled by fillna)
            df_filtered = df_filtered[df_filtered[nogo_col].astype(str).str.strip() == '']

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
        keywords_to_check = [
            Word(word).singularize()
            for word in extracted_words
            if Word(word).singularize() not in ['interested', 'in', 'role', 'anyone', 'and', 'or', 'with', 'the', 'a', 'is', 'are', 'for'] # Extended stop words
        ]
        keywords_to_check = list(set(keywords_to_check)) # Remove duplicates

        if keywords_to_check:
            print(f"Filtering based on keywords: {keywords_to_check}") # Debug print
            try:
                # Apply filtering row-wise, checking mapped Occupation and Topics columns if they exist
                def check_row(row):
                    match = False
                    if occupation_col:
                        occupation_str = str(row.get(occupation_col, '')).lower()
                        if any(keyword in occupation_str for keyword in keywords_to_check):
                            match = True
                    if not match and topics_col: # Only check topics if occupation didn't match
                         topic_str = str(row.get(topics_col, '')).lower()
                         if any(keyword in topic_str for keyword in keywords_to_check):
                             match = True
                    return match

                df_filtered = df_filtered[df_filtered.apply(check_row, axis=1)]

            except Exception as e:
                st.error(f"Error during keyword filtering: {e}")
                st.warning("Proceeding without keyword filtering due to error.")
        else:
            st.warning("Could not extract specific keywords from target audience description. Showing all mailable contacts.")
    elif target_audience_desc and target_audience_desc.lower() != 'any':
         st.warning("Target audience description provided, but 'Occupation' or 'Topics' columns not mapped. Cannot perform keyword filtering.")


    if df_filtered.empty:
         st.warning(f"No contacts matched the target audience description: '{target_audience_desc}' after initial filtering.")

    # Convert the final filtered DataFrame to list of dicts
    # Important: Convert to dict *after* all filtering is done
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

    # Create a deep copy to avoid modifying the original list in session state
    report_data = copy.deepcopy(filtered_contacts)

    for idx, contact_row in enumerate(report_data):
        # Get status, defaulting to 'Pending/Not Sent' if not found or still 'pending'
        status = send_status.get(idx, 'pending')
        if status == 'pending':
            status = 'Pending/Not Sent' # Clarify status for download

        # Get subject, defaulting to 'N/A' if draft doesn't exist for the index
        subject = drafts.get(idx, {}).get('subject', 'N/A')

        # Add new columns to the dictionary for this row
        contact_row['Send Status'] = status
        contact_row['Email Subject Drafted'] = subject

    # Convert the list of enhanced dictionaries to a DataFrame
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
    body = body_match.group(1).strip() if body_match else llm_response
    if not subject_match or not body_match:
         st.warning(f"Could not reliably parse Subject/Body markers from LLM draft output.")
    return subject, body

def generate_single_draft(contact_info, campaign_details, email_drafter, sender_name, column_mapping):
    """
    Generates a single email draft using the provided LLM chain and column mapping.
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
    occupation = contact_info.get(column_mapping.get(APP_FIELD_OCCUPATION, ''), 'N/A') # Provide default if not mapped/present
    topics = contact_info.get(column_mapping.get(APP_FIELD_TOPICS, ''), 'N/A') # Provide default

    # --- Prepare input for the LLM prompt ---
    prompt_input = {
        'first_name': first_name,
        'last_name': last_name,
        'occupation': occupation,
        'contact_email': email,
        'contact_topic': topics,
        'campaign_topic': campaign_details.get('campaign_topic', 'Follow Up'),
        'email_tone': campaign_details.get('email_tone', 'Professional'),
        'call_to_action': campaign_details.get('call_to_action', 'Engage further'),
        'sender_name': sender_name
    }

    try:
        llm_response = email_drafter.invoke(prompt_input)
        subject, body = parse_llm_output(llm_response)
        return subject, body
    except Exception as e:
        st.error(f"Error calling Gemini API for {first_name}: {e}")
        return f"Error Generating Draft", f"An error occurred: {e}"

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
    if not recipient or '@' not in recipient:
        st.error(f"Invalid recipient email: {recipient}")
        return False
    if not subject:
        st.warning("Sending email with an empty subject.")
        subject = "(No Subject)"

    message = EmailMessage()
    message['From'] = sender_email
    message['To'] = recipient
    message['Subject'] = subject
    message.set_content(body) # Assumes plain text

    # Use certifi's CA bundle for SSL context
    context = ssl.create_default_context(cafile=certifi.where())

    try:
        with smtplib.SMTP_SSL(smtp_server, smtp_port, context=context) as server:
            server.login(sender_email, sender_password)
            server.send_message(message)
        return True
    except smtplib.SMTPAuthenticationError:
        st.error("SMTP Auth Failed. Check EMAIL_ADDRESS/EMAIL_PASSWORD (use App Password for Gmail/Google Workspace).")
        return False
    except Exception as e:
        st.error(f"Error sending email to {recipient}: {e}")
        return False

# ==============================================================================
# Helper Functions
# ==============================================================================

def format_chat_history(history):
    """Formats chat history for the analysis prompt."""
    formatted_history = ""
    for msg in history:
        role = "User" if msg["role"] == "user" else "Assistant"
        formatted_history += f"{role}: {msg['content']}\n"
    return formatted_history.strip()

def initialize_session_state():
    """Initializes Streamlit session state variables if they don't exist."""
    # --- Application Flow and Data ---
    if 'app_stage' not in st.session_state:
        st.session_state.app_stage = "upload" # Start at file upload
    if 'uploaded_file_state' not in st.session_state:
         st.session_state.uploaded_file_state = None # Track uploaded file
    if 'all_contacts_df' not in st.session_state:
        st.session_state.all_contacts_df = None # Loaded DataFrame
    if 'csv_columns' not in st.session_state:
        st.session_state.csv_columns = [] # Columns from uploaded file
    if 'column_mapping' not in st.session_state:
        st.session_state.column_mapping = {} # User's column mapping
    if 'mapping_complete' not in st.session_state:
        st.session_state.mapping_complete = False # Flag if mapping is done

    # --- Campaign Planning ---
    if 'messages' not in st.session_state:
        st.session_state.messages = [{
            "role": "assistant",
            "content": """Hi! Let's plan your email campaign. Please tell me about:
\n1.  **Campaign Topic:** What is the main subject of the email? (e.g., New product, webinar invite, special offer)
\n2.  **Target Audience:** Who should receive this email? Describe them (e.g., 'Software Engineers interested in AI', 'Marketing Managers', 'Anyone interested in leadership') - I'll use the 'Occupation' and 'Topics' columns you mapped.
\n3.  **Call to Action:** What do you want the recipients to do after reading? (e.g., 'Schedule a meeting', 'Visit website', 'Register now')"""
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

def reset_session_state_for_new_campaign():
    """Resets session state variables for starting a new campaign, keeping config."""
    current_config = st.session_state.get('config', {})
    current_config_ok = st.session_state.get('config_ok', False)
    current_llm_components = {
        'llm': st.session_state.get('llm'),
        'campaign_analyzer': st.session_state.get('campaign_analyzer'),
        'email_drafter': st.session_state.get('email_drafter'),
        'llm_components_ready': st.session_state.get('llm_components_ready', False)
    }

    # Clear most session state keys
    keys_to_reset = [
        'app_stage', 'uploaded_file_state', 'all_contacts_df', 'csv_columns',
        'column_mapping', 'mapping_complete', 'messages', 'campaign_details',
        'analysis_error', 'filtered_contacts_list', 'drafts', 'send_status',
        'selected_for_send' # Also clear the selection set
    ]
    for key in keys_to_reset:
        if key in st.session_state:
            del st.session_state[key]

    # Re-initialize state
    initialize_session_state()

    # Restore config and LLM components
    st.session_state.config = current_config
    st.session_state.config_ok = current_config_ok
    st.session_state.llm = current_llm_components['llm']
    st.session_state.campaign_analyzer = current_llm_components['campaign_analyzer']
    st.session_state.email_drafter = current_llm_components['email_drafter']
    st.session_state.llm_components_ready = current_llm_components['llm_components_ready']


# ==============================================================================
# Streamlit UI Rendering Functions (Organized by Stage)
# ==============================================================================

def render_upload_stage():
    """Renders the UI for uploading the contact file."""
    st.header("Step 1: Upload Contact File")
    uploaded_file = st.file_uploader("Choose a CSV or Excel file", type=['csv', 'xlsx', 'xls'])

    # Process file only if it's newly uploaded or hasn't been processed yet
    if uploaded_file is not None and uploaded_file != st.session_state.uploaded_file_state:
        st.session_state.uploaded_file_state = uploaded_file # Store the uploaded file object
        df, cols = load_contacts_from_file(uploaded_file)
        if df is not None and cols:
            st.session_state.all_contacts_df = df
            st.session_state.csv_columns = cols
            st.session_state.app_stage = "map_columns"
            st.session_state.mapping_complete = False # Reset mapping status
            st.session_state.column_mapping = {} # Clear previous mapping
            st.rerun()
        else:
            # Error handled in load_contacts_from_file
            st.session_state.uploaded_file_state = None # Reset if loading failed

    elif st.session_state.all_contacts_df is not None and st.session_state.csv_columns:
         st.info(f"Using previously uploaded file: {st.session_state.uploaded_file_state.name}. Re-upload to change.")
         # Add a button to proceed if stuck? Usually rerun handles it.
         if st.button("Proceed with Current File"):
              st.session_state.app_stage = "map_columns"
              st.rerun()


def render_map_columns_stage():
    """Renders the UI for mapping CSV columns to application fields."""
    st.header("Step 2: Map CSV Columns")

    if st.session_state.all_contacts_df is None or not st.session_state.csv_columns:
        st.error("Contact data not loaded. Please go back to upload.")
        if st.button("Go Back to Upload"):
            st.session_state.app_stage = "upload"
            st.rerun()
        return

    st.write("Please map the columns from your file to the required fields:")
    st.dataframe(st.session_state.all_contacts_df.head(), use_container_width=True)
    st.markdown("---")

    cols = st.session_state.csv_columns
    # Use a dictionary to store selections temporarily
    mapping = {}

    # --- Required Mappings ---
    st.subheader("Required Fields")
    mapping[APP_FIELD_EMAIL] = st.selectbox(f"Email Address Column:", cols, key="map_email", index=cols.index(st.session_state.column_mapping.get(APP_FIELD_EMAIL)) if st.session_state.column_mapping.get(APP_FIELD_EMAIL) in cols else 0)
    mapping[APP_FIELD_MAILABILITY] = st.selectbox(f"Column Indicating Email Can Be Sent:", cols, key="map_mailability", index=cols.index(st.session_state.column_mapping.get(APP_FIELD_MAILABILITY)) if st.session_state.column_mapping.get(APP_FIELD_MAILABILITY) in cols else 0)
    mapping['mailability_yes_value'] = st.text_input("Value in the above column meaning 'Yes' (case-insensitive):", value=st.session_state.column_mapping.get('mailability_yes_value', "Yes"), key="map_mailability_yes")

    # --- Optional Mappings ---
    st.subheader("Recommended Fields")
    # Add a "None" option for optional fields
    optional_cols_with_none = [None] + cols
    default_first_name = st.session_state.column_mapping.get(APP_FIELD_FIRST_NAME)
    default_last_name = st.session_state.column_mapping.get(APP_FIELD_LAST_NAME)

    mapping[APP_FIELD_FIRST_NAME] = st.selectbox(f"First Name Column:", optional_cols_with_none, key="map_first_name", index=optional_cols_with_none.index(default_first_name) if default_first_name in optional_cols_with_none else 0)
    mapping[APP_FIELD_LAST_NAME] = st.selectbox(f"Last Name Column:", optional_cols_with_none, key="map_last_name", index=optional_cols_with_none.index(default_last_name) if default_last_name in optional_cols_with_none else 0)


    st.subheader("Optional Fields for Personalization/Filtering")
    default_occupation = st.session_state.column_mapping.get(APP_FIELD_OCCUPATION)
    default_topics = st.session_state.column_mapping.get(APP_FIELD_TOPICS)
    default_nogo = st.session_state.column_mapping.get(APP_FIELD_NOGO)

    mapping[APP_FIELD_OCCUPATION] = st.selectbox(f"Occupation Column:", optional_cols_with_none, key="map_occupation", index=optional_cols_with_none.index(default_occupation) if default_occupation in optional_cols_with_none else 0)
    mapping[APP_FIELD_TOPICS] = st.selectbox(f"Topics/Interests Column:", optional_cols_with_none, key="map_topics", index=optional_cols_with_none.index(default_topics) if default_topics in optional_cols_with_none else 0)
    mapping[APP_FIELD_NOGO] = st.selectbox(f"Column Indicating 'Do Not Email' (No-Go):", optional_cols_with_none, key="map_nogo", index=optional_cols_with_none.index(default_nogo) if default_nogo in optional_cols_with_none else 0)
    st.caption("If a 'No-Go' column is selected, contacts will be excluded if this column is *not* empty.")


    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Confirm Mapping", type="primary"):
            # Basic validation
            if not mapping[APP_FIELD_EMAIL] or not mapping[APP_FIELD_MAILABILITY]:
                st.error("Please select columns for Email and Mailability.")
            elif not mapping['mailability_yes_value']:
                 st.error("Please enter the value indicating 'Yes' for mailability.")
            else:
                # Remove None selections before saving
                final_mapping = {k: v for k, v in mapping.items() if v is not None}
                st.session_state.column_mapping = final_mapping
                st.session_state.mapping_complete = True
                st.session_state.app_stage = "chat" # Proceed to chat
                st.success("Column mapping confirmed!")
                time.sleep(1) # Brief pause to show success
                st.rerun()
    with col2:
        if st.button("Go Back to Upload"):
            st.session_state.app_stage = "upload"
            # Clear potentially loaded data if going back
            st.session_state.all_contacts_df = None
            st.session_state.csv_columns = []
            st.session_state.uploaded_file_state = None
            st.rerun()


def render_chat_stage(config_ok, llm_components_ready):
    """Renders the UI for the campaign description chat stage."""
    st.header("Step 3: Describe Your Campaign")

    # Display chat messages
    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    # Get user input
    if prompt := st.chat_input("Your campaign details..."):
        if not config_ok or not llm_components_ready:
            st.error("Configuration or AI components not ready. Cannot process chat.")
        else:
            st.session_state.messages.append({"role": "user", "content": prompt})
            st.chat_message("user").write(prompt)
            st.session_state.app_stage = "analyze_button" # Set stage for button
            st.rerun()

def render_analyze_button_stage():
    """Renders the button to trigger campaign analysis."""
    st.header("Step 3: Describe Your Campaign")
    # Display chat messages again
    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    st.markdown("---")
    if st.button("Analyze Campaign Description", type="primary"):
        st.session_state.app_stage = "analyze_processing"
        st.rerun()

def render_analyze_processing_stage(campaign_analyzer):
    """Handles the AI analysis process and updates the state."""
    st.header("Step 4: Analyzing Campaign & Filtering Contacts...")
    st.session_state.analysis_error = False # Reset error flag
    if not campaign_analyzer:
         st.error("Campaign Analyzer component is not ready. Cannot analyze.")
         st.session_state.analysis_error = True
         # Fallback: Use basic mailability filter only
         if st.session_state.all_contacts_df is not None and st.session_state.mapping_complete:
             # Apply only the basic mailability filter as a fallback
             mailability_col = st.session_state.column_mapping.get(APP_FIELD_MAILABILITY)
             mailability_yes_value = st.session_state.column_mapping.get('mailability_yes_value', 'yes')
             nogo_col = st.session_state.column_mapping.get(APP_FIELD_NOGO)
             df = st.session_state.all_contacts_df
             try:
                df_fallback = df[df[mailability_col].astype(str).str.strip().str.lower() == mailability_yes_value.lower()]
                if nogo_col:
                    df_fallback = df_fallback[df_fallback[nogo_col].astype(str).str.strip() == '']
                st.session_state.filtered_contacts_list = df_fallback.to_dict('records')
             except Exception as fallback_e:
                 st.error(f"Fallback filtering failed: {fallback_e}")
                 st.session_state.filtered_contacts_list = []
         else:
              st.session_state.filtered_contacts_list = []
         st.session_state.app_stage = "confirm"
         st.rerun()
         return # Exit the function

    with st.spinner("AI is analyzing the campaign description & filtering contacts..."):
        try:
            chat_history_formatted = format_chat_history(st.session_state.messages)
            # Invoke the analyzer chain
            analysis_result = campaign_analyzer.invoke({"chat_history": chat_history_formatted})
            st.session_state.campaign_details = analysis_result # Store the dict

            # Filter contacts based on the analysis AND the mapping
            st.session_state.filtered_contacts_list = filter_contacts_based_on_analysis(
                st.session_state.all_contacts_df,
                st.session_state.campaign_details,
                st.session_state.column_mapping # Pass the mapping
            )
            st.session_state.app_stage = "confirm"

        except Exception as e:
            st.error(f"Error during campaign analysis or filtering: {e}")
            st.warning("Could not automatically filter contacts based on analysis. Please review the list based on mailability rules or revise your description/mapping.")
            st.session_state.analysis_error = True
            # Fallback: use basic mailability filter only
            if st.session_state.all_contacts_df is not None and st.session_state.mapping_complete:
                 # Apply only the basic mailability filter as a fallback
                 mailability_col = st.session_state.column_mapping.get(APP_FIELD_MAILABILITY)
                 mailability_yes_value = st.session_state.column_mapping.get('mailability_yes_value', 'yes')
                 nogo_col = st.session_state.column_mapping.get(APP_FIELD_NOGO)
                 df = st.session_state.all_contacts_df
                 try:
                    df_fallback = df[df[mailability_col].astype(str).str.strip().str.lower() == mailability_yes_value.lower()]
                    if nogo_col:
                        df_fallback = df_fallback[df_fallback[nogo_col].astype(str).str.strip() == '']
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
    st.header("Step 5: Confirm Recipients")

    column_mapping = st.session_state.get('column_mapping', {})

    if st.session_state.analysis_error:
        st.warning("Analysis failed or couldn't extract clear criteria. Showing contacts filtered only by mailability rules.")
    elif st.session_state.campaign_details:
        st.subheader("AI Campaign Analysis Results:")
        st.markdown(f"**Campaign Topic:** {st.session_state.campaign_details.get('campaign_topic', 'N/A')}")
        st.markdown(f"**Target Audience (Interpretation):** {st.session_state.campaign_details.get('target_audience', 'N/A')}")
        st.markdown(f"**Call to Action:** {st.session_state.campaign_details.get('call_to_action', 'N/A')}")
        st.markdown(f"**Email Tone:** {st.session_state.campaign_details.get('email_tone', 'N/A')}")
        st.markdown("---")
    else:
        st.warning("Campaign details not available.")


    st.subheader("Proposed Recipients:")
    if not st.session_state.filtered_contacts_list:
        st.warning("No contacts match the criteria based on the mailability rules and campaign analysis.")
        if st.button("Go Back to Chat"):
            st.session_state.app_stage = "chat"
            st.rerun()
    else:
        # --- Determine columns to display based on mapping ---
        cols_to_display_internal = [APP_FIELD_FIRST_NAME, APP_FIELD_LAST_NAME, APP_FIELD_EMAIL, APP_FIELD_OCCUPATION, APP_FIELD_TOPICS]
        # Get the actual CSV column names from the mapping for display
        mapped_cols_for_display = [column_mapping.get(field) for field in cols_to_display_internal if column_mapping.get(field)]

        if not mapped_cols_for_display:
             st.warning("Essential columns (like Email) not mapped correctly. Cannot display recipients.")
             # Add button to go back to mapping?
             if st.button("Go Back to Column Mapping"):
                 st.session_state.app_stage = "map_columns"
                 st.rerun()
             return

        try:
            # Create DataFrame from the filtered list *using only the mapped columns*
            filtered_df_display = pd.DataFrame(st.session_state.filtered_contacts_list)
            # Select only the columns that were successfully mapped for display
            filtered_df_display = filtered_df_display[mapped_cols_for_display]
            # Rename columns for better readability in the UI
            rename_map = {column_mapping.get(k): k.replace('_', ' ').title() for k in cols_to_display_internal if column_mapping.get(k)}
            filtered_df_display = filtered_df_display.rename(columns=rename_map)

            st.dataframe(filtered_df_display, use_container_width=True)
        except KeyError as e:
             st.error(f"Display error: Column '{e}' (expected based on mapping) not found in filtered data. This might indicate an issue during filtering.")
             st.write("Filtered Data Sample:", st.session_state.filtered_contacts_list[:2]) # Show sample raw data
             if st.button("Go Back to Column Mapping"):
                 st.session_state.app_stage = "map_columns"
                 st.rerun()
             return
        except Exception as e:
             st.error(f"An unexpected error occurred while preparing the display table: {e}")
             return


        col1_confirm, col2_confirm = st.columns(2)
        with col1_confirm:
            if st.button(f"Confirm and Proceed to Drafts ({len(st.session_state.filtered_contacts_list)})", type="primary"):
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
                st.rerun()

def render_draft_stage(email_drafter, config):
    """Renders the UI for generating email drafts."""
    st.header("Step 6: Generate Email Drafts")

    column_mapping = st.session_state.get('column_mapping', {})
    if not column_mapping or not st.session_state.mapping_complete:
        st.error("Column mapping is not complete. Please go back and map columns.")
        if st.button("Go to Column Mapping"):
            st.session_state.app_stage = "map_columns"
            st.rerun()
        return

    st.markdown(f"Generating drafts for the **{len(st.session_state.filtered_contacts_list)}** confirmed recipients based on:")
    if st.session_state.campaign_details:
        st.markdown(f"**Campaign Topic:** {st.session_state.campaign_details.get('campaign_topic', 'N/A')}")
        st.markdown(f"**Call to Action:** {st.session_state.campaign_details.get('call_to_action', 'N/A')}")
        st.markdown(f"**Email Tone:** {st.session_state.campaign_details.get('email_tone', 'N/A')}")
    else:
        st.warning("Campaign details missing (analysis might have failed). Using generic settings.")

    # Check if components needed for drafting are ready
    drafting_ready = st.session_state.config_ok and st.session_state.llm_components_ready and email_drafter is not None

    if st.button("✨ Generate Drafts", disabled=not drafting_ready):
        if not st.session_state.campaign_details:
             st.error("Cannot generate drafts without campaign details.")
        elif not st.session_state.filtered_contacts_list:
             st.warning("No contacts confirmed to generate drafts for.")
        else:
            total_contacts = len(st.session_state.filtered_contacts_list)
            progress_bar = st.progress(0, text="Initializing draft generation...")
            status_text = st.empty()
            st.session_state.drafts = {} # Reset drafts
            st.session_state.send_status = {} # Reset status
            st.session_state.selected_for_send = set() # Reset selection

            # Get sender name once
            sender_name = config.get('sender_name', 'Your Name [Default]')

            # Loop through the CONFIRMED FILTERED contacts
            for i, contact in enumerate(st.session_state.filtered_contacts_list):
                 # Get first name using mapping for progress text
                first_name_col = column_mapping.get(APP_FIELD_FIRST_NAME)
                name_for_status = contact.get(first_name_col, f'Contact {i+1}') if first_name_col else f'Contact {i+1}'
                status_text.text(f"Generating draft for {name_for_status} ({i+1}/{total_contacts})...")

                # Generate draft for the current contact, passing the mapping
                subject, body = generate_single_draft(
                    contact,
                    st.session_state.campaign_details,
                    email_drafter,
                    sender_name,
                    column_mapping # Pass the mapping
                )

                # Use the index 'i' from the filtered list enumeration as the key
                st.session_state.drafts[i] = {'subject': subject, 'body': body}
                st.session_state.send_status[i] = 'pending'
                progress_bar.progress((i + 1) / total_contacts, text=f"Generated draft for {name_for_status} ({i+1}/{total_contacts})")
                time.sleep(0.1) # Small UI delay

            status_text.success(f"✅ Generated drafts for all {total_contacts} confirmed contacts.")
            progress_bar.empty()
            st.session_state.app_stage = "send" # Move to send stage
            st.rerun()
    elif not drafting_ready:
         st.warning("Cannot generate drafts. Check configuration and AI component status in the sidebar.")


def render_send_stage(config):
    """Renders the UI for reviewing and sending emails with bulk actions."""
    st.header("Step 7: Review Drafts and Send Emails")

    # --- Initial Checks ---
    column_mapping = st.session_state.get('column_mapping', {})
    if not column_mapping or not st.session_state.mapping_complete:
        st.error("Column mapping is not complete. Please go back and map columns.")
        if st.button("Go to Column Mapping"):
            st.session_state.app_stage = "map_columns"
            st.rerun()
        return

    if not st.session_state.drafts:
         st.warning("No drafts generated yet. Go back and generate drafts.")
         if st.button("Go Back to Generate Drafts"):
             st.session_state.app_stage = "draft"
             st.rerun()
         return

    email_col = column_mapping.get(APP_FIELD_EMAIL)
    first_name_col = column_mapping.get(APP_FIELD_FIRST_NAME)
    if not email_col:
         st.error("Email column mapping is missing. Cannot proceed.")
         return

    # --- Bulk Action Controls ---
    st.subheader("Bulk Send Actions")
    col_bulk1, col_bulk2, col_bulk3 = st.columns(3)

    # Get indices of currently pending emails
    pending_indices = {idx for idx, status in st.session_state.send_status.items() if status == 'pending'}

    with col_bulk1:
        if st.button("Select All Pending"):
            st.session_state.selected_for_send = pending_indices.copy() # Select all pending
            st.rerun()
    with col_bulk2:
        if st.button("Unselect All"):
            st.session_state.selected_for_send = set() # Clear selection
            st.rerun()
    with col_bulk3:
        # Build the list of indices to send based *only* on currently checked boxes
        indices_checked_now = set()
        for idx in st.session_state.drafts.keys():
             # Check if the widget key exists and is True
             if st.session_state.get(f"select_{idx}", False):
                 indices_checked_now.add(idx)

        # Only consider those that are checked AND are actually pending
        indices_to_send_now = sorted([idx for idx in indices_checked_now if idx in pending_indices])
        num_to_send = len(indices_to_send_now)

        send_selected_disabled = num_to_send == 0 or not st.session_state.config_ok

        if st.button(f"🚀 Send to Selected ({num_to_send})", disabled=send_selected_disabled):
            if not indices_to_send_now: # Double check
                 st.warning("No valid pending contacts selected for sending.")
            else:
                send_placeholder = st.empty()
                send_progress_bar = st.progress(0)
                total_to_send = len(indices_to_send_now)
                num_sent_this_batch = 0
                num_failed_this_batch = 0

                for i, idx in enumerate(indices_to_send_now):
                    # Check index validity and status again just in case
                    if idx < len(st.session_state.filtered_contacts_list) and st.session_state.send_status.get(idx) == 'pending':
                        contact = st.session_state.filtered_contacts_list[idx]
                        recipient = contact.get(email_col, None)
                        name = contact.get(first_name_col, f'Contact {idx+1}') if first_name_col else f'Contact {idx+1}'

                        # Use the current values from text input/area for subject/body
                        subject = st.session_state.get(f"subject_edit_{idx}", st.session_state.drafts[idx]['subject'])
                        body = st.session_state.get(f"body_edit_{idx}", st.session_state.drafts[idx]['body'])

                        send_placeholder.info(f"Sending to {name} ({i+1}/{total_to_send})...")

                        if recipient:
                            success = send_email_smtp(recipient, subject, body, config)
                            if success:
                                st.session_state.send_status[idx] = 'sent'
                                num_sent_this_batch += 1
                            else:
                                st.session_state.send_status[idx] = 'failed'
                                num_failed_this_batch += 1
                        else:
                             st.session_state.send_status[idx] = 'failed' # Mark as failed if no recipient
                             num_failed_this_batch += 1
                             st.warning(f"Skipped {name}: No valid email address found.")

                        send_progress_bar.progress((i + 1) / total_to_send)
                        if i < total_to_send - 1: # Don't sleep after the last one
                            time.sleep(SEND_DELAY_SECONDS)
                    # else: skip if index out of bounds or status not pending

                send_placeholder.success(f"Bulk send complete. Sent: {num_sent_this_batch}, Failed: {num_failed_this_batch}")
                # Clear the main selection set after processing
                st.session_state.selected_for_send = set()
                time.sleep(2) # Pause to show final message
                st.rerun() # Rerun to update statuses and checkboxes

    st.markdown("---")

    # --- Display Drafts with Checkboxes ---
    st.subheader("Review/Edit Individual Drafts")
    num_displayed = 0
    checkbox_states = {} # Store current visual state of checkboxes

    for idx, draft_info in st.session_state.drafts.items():
        # Ensure the index is valid for the filtered list
        if idx < len(st.session_state.filtered_contacts_list):
            num_displayed += 1
            contact = st.session_state.filtered_contacts_list[idx]
            recipient = contact.get(email_col, None)
            name = contact.get(first_name_col, f'Contact {idx+1}') if first_name_col else f'Contact {idx+1}'
            status = st.session_state.send_status.get(idx, 'pending')

            # --- Row for Checkbox and Expander Title ---
            row_cols = st.columns([0.1, 0.9]) # Adjust ratio as needed
            with row_cols[0]:
                 # Checkbox state reflects the session state set (updated by Select/Unselect All)
                 # Use default value from the set, disable if not pending
                 checkbox_states[idx] = st.checkbox("Select", key=f"select_{idx}", value=(idx in st.session_state.selected_for_send), disabled=(status != 'pending'), label_visibility="collapsed")

            with row_cols[1]:
                expander_title = f"{idx+1}. To: {name} ({recipient or 'No Email!'}) - Status: {status.upper()}"
                with st.expander(expander_title, expanded=False): # Default to collapsed
                    # Edit fields - these will be read by the Send buttons if needed
                    edited_subject = st.text_input("Subject", draft_info['subject'], key=f"subject_edit_{idx}")
                    edited_body = st.text_area("Body", draft_info['body'], height=200, key=f"body_edit_{idx}")

                    # --- Individual Send Button ---
                    send_one_button_disabled = status != 'pending' or not st.session_state.config_ok or not recipient
                    tooltip_msg_one = ""
                    if status != 'pending': tooltip_msg_one = f"Status is {status}."
                    elif not st.session_state.config_ok: tooltip_msg_one = "Configuration error prevents sending."
                    elif not recipient: tooltip_msg_one = f"Recipient email address missing or invalid in column '{email_col}'."

                    if st.button(f"🚀 Send Only This Email", key=f"send_one_{idx}", disabled=send_one_button_disabled, help=tooltip_msg_one):
                        send_one_placeholder = st.empty()
                        send_one_placeholder.info(f"Sending to {recipient}...")
                        # Use the potentially edited subject/body for sending
                        success = send_email_smtp(recipient, edited_subject, edited_body, config)
                        if success:
                            st.session_state.send_status[idx] = 'sent'
                            # If sent individually, remove from bulk selection set
                            st.session_state.selected_for_send.discard(idx)
                            send_one_placeholder.success(f"Email sent successfully to {recipient}!")
                        else:
                            st.session_state.send_status[idx] = 'failed'
                            send_one_placeholder.error(f"Failed to send email to {recipient}. Check console/terminal for specific errors.")

                        time.sleep(1) # Short pause
                        st.rerun() # Rerun to update status and potentially disable checkbox/button
        else:
            st.warning(f"Data inconsistency: Draft found for index {idx}, but no corresponding contact in the filtered list.")

    if num_displayed == 0: # Handle case where drafts exist but don't match filtered list
         st.warning("Could not display any drafts. Potential data inconsistency.")


    # --- Final Actions (Download/Reset) ---
    st.markdown("---")
    st.subheader("Campaign Finish Actions")

    col_final1, col_final2 = st.columns(2)

    with col_final1:
        # --- Download Button ---
        if st.session_state.filtered_contacts_list:
            df_download = prepare_download_data(
                st.session_state.filtered_contacts_list,
                st.session_state.drafts,
                st.session_state.send_status
            )
            if not df_download.empty:
                csv_data = df_download.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download Send Report (CSV)",
                    data=csv_data,
                    file_name='email_campaign_report.csv',
                    mime='text/csv',
                    key='download_report_btn'
                )
            else:
                st.info("No data available to download yet.")
        else:
            st.info("No contacts processed in this campaign to download.")

    with col_final2:
        if st.button("Start New Campaign (Reset)", key='reset_campaign_btn'):
            reset_session_state_for_new_campaign()
            st.rerun()


# ==============================================================================
# Main Application Logic
# ==============================================================================

def main():
    """Main function to run the Streamlit application."""
    st.set_page_config(layout="wide")
    st.title("📧 Email Campaign Assistant")
    st.markdown("Upload contacts, map columns, describe your campaign, let AI analyze, confirm, generate drafts, and send!")

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
        load_and_validate_config() # Re-render sidebar status

    # --- Initialize LLM Components ---
    # Re-initialize if config becomes OK or components aren't ready yet
    # This handles the case where config was initially bad but gets fixed
    if config_ok and not st.session_state.get('llm_components_ready', False):
        llm, campaign_analyzer, email_drafter = initialize_llm_components(config)
        st.session_state.llm = llm
        st.session_state.campaign_analyzer = campaign_analyzer
        st.session_state.email_drafter = email_drafter
        # Update readiness flag only after attempting initialization
        st.session_state.llm_components_ready = all([llm, campaign_analyzer, email_drafter])
    elif not config_ok:
         # Ensure components are marked as not ready if config is bad
         st.session_state.llm_components_ready = False


    # --- Stage-Based UI Rendering ---
    app_stage = st.session_state.app_stage

    # Only allow proceeding past upload/mapping if config is OK
    # Display error in main panel if config is bad and user tries to proceed
    if not config_ok and app_stage not in ["upload", "map_columns"]:
         st.error("Email/API Configuration is incomplete. Please check your `.env` file. Functionality is limited.")
         # Prevent moving forward implicitly by disabling buttons or showing error message
         # Buttons in later stages check config_ok anyway

    # --- Render Current Stage ---
    if app_stage == "upload":
        render_upload_stage()
    elif app_stage == "map_columns":
        # Requires DataFrame to be loaded
        if st.session_state.all_contacts_df is not None:
            render_map_columns_stage()
        else:
            st.warning("No contact data loaded. Redirecting to upload.")
            st.session_state.app_stage = "upload"
            time.sleep(1)
            st.rerun()
    elif app_stage == "chat":
         # Requires mapping to be complete
        if st.session_state.mapping_complete:
            render_chat_stage(config_ok, st.session_state.llm_components_ready)
        else:
            st.warning("Column mapping not complete. Redirecting to mapping step.")
            st.session_state.app_stage = "map_columns"
            time.sleep(1)
            st.rerun()
    elif app_stage == "analyze_button":
         if st.session_state.mapping_complete:
             render_analyze_button_stage()
         else: # Should not happen if logic is correct, but as safeguard
              st.warning("Mapping not complete. Redirecting.")
              st.session_state.app_stage = "map_columns"
              time.sleep(1)
              st.rerun()
    elif app_stage == "analyze_processing":
         if st.session_state.mapping_complete:
             render_analyze_processing_stage(st.session_state.get('campaign_analyzer'))
         else: # Safeguard
              st.warning("Mapping not complete. Redirecting.")
              st.session_state.app_stage = "map_columns"
              time.sleep(1)
              st.rerun()
    elif app_stage == "confirm":
         if st.session_state.mapping_complete:
             render_confirm_stage()
         else: # Safeguard
              st.warning("Mapping not complete. Redirecting.")
              st.session_state.app_stage = "map_columns"
              time.sleep(1)
              st.rerun()
    elif app_stage == "draft":
         if st.session_state.mapping_complete:
             render_draft_stage(st.session_state.get('email_drafter'), config)
         else: # Safeguard
              st.warning("Mapping not complete. Redirecting.")
              st.session_state.app_stage = "map_columns"
              time.sleep(1)
              st.rerun()
    elif app_stage == "send":
         if st.session_state.mapping_complete:
             render_send_stage(config)
         else: # Safeguard
              st.warning("Mapping not complete. Redirecting.")
              st.session_state.app_stage = "map_columns"
              time.sleep(1)
              st.rerun()


if __name__ == "__main__":
    main()
