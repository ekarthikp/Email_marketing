import streamlit as st
import pandas as pd
import time
import os
from dotenv import load_dotenv
import re
import json
from textblob import Word

# --- LangChain / Gemini Imports ---
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.runnables import RunnablePassthrough


# --- Email Sending Imports ---
import ssl # Ensure ssl is imported
import smtplib # Ensure smtplib is imported
import certifi 
from email.message import EmailMessage

# --- SET PAGE CONFIG FIRST ---
st.set_page_config(layout="wide")

# --- Load Environment Variables ---
load_dotenv()

# --- Configuration & Constants ---
# Make sure this path is correct relative to where you run the script
CSV_FILE_PATH = 'random_nogo_email_list_with_occupation.csv'
SMTP_SERVER = 'smtp.gmail.com' # Example for Gmail
SMTP_PORT = 465
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
SENDER_EMAIL = os.getenv('EMAIL_ADDRESS')
SENDER_PASSWORD = os.getenv('EMAIL_PASSWORD')
YOUR_NAME = os.getenv('YOUR_NAME', 'Your Name [Default]')
SEND_DELAY_SECONDS = 5 # Delay between sending emails

# --- Helper Function to Safely Get Environment Variables ---
def get_env_variable(var_name, is_critical=True):
    value = os.getenv(var_name)
    if not value and is_critical:
        st.error(f"Error: Critical environment variable '{var_name}' not found. Please set it in your .env file.")
        st.stop()
    return value

# --- Check Essential Configuration on App Load ---
st.sidebar.title("Configuration Status")
config_ok = True
# Check Google API Key
if not GOOGLE_API_KEY:
    st.sidebar.error("🔴 GOOGLE_API_KEY is missing.")
    config_ok = False
else:
    st.sidebar.write("🔑 Google API Key: Loaded")

# Check Email Credentials
if not SENDER_EMAIL:
    st.sidebar.error("🔴 EMAIL_ADDRESS is missing.")
    config_ok = False
else:
     st.sidebar.write(f"✉️ Sender Email: {SENDER_EMAIL}")
if not SENDER_PASSWORD:
    st.sidebar.error("🔴 EMAIL_PASSWORD is missing.")
    config_ok = False
else:
    st.sidebar.write("🔒 Email Password: Loaded")

# Check Sender Name
if not os.getenv('YOUR_NAME'):
     st.sidebar.warning("🟠 YOUR_NAME is missing (using default).")
     YOUR_NAME = "Your Name [Default]"
else:
    st.sidebar.write(f"👤 Sender Name: {YOUR_NAME}")


if config_ok:
    st.sidebar.success("✅ Configuration loaded successfully.")
else:
    st.sidebar.error("❌ Please check your `.env` file for missing values.")
    st.warning("App functionality will be limited due to missing configuration.")

# --- LangChain Setup (Run only if config is ok) ---
llm = None
campaign_analyzer = None
email_drafter = None

if config_ok:
    try:
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", google_api_key=GOOGLE_API_KEY,
                                     temperature=0.5)

        # 1. Campaign Analysis Chain (Corrected and Updated for New Fields)
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
{{  # <-- Escaped curly brace
"campaign_topic": "AI in Marketing Webinar",
"target_audience": "Marketing Specialists interested in AI & Machine Learning",
"call_to_action": "Register for webinar",
"email_tone": "Informative"
}}  # <-- Escaped curly brace

Conversation History:
{chat_history}
"""),
                 ("human", "Based on the conversation history above, please provide the JSON output."),
            ]
        )
        # The rest of the chain definition remains the same
        campaign_analyzer = analysis_prompt_template | llm | JsonOutputParser()
        st.sidebar.info("✅ Campaign Analyzer Initialized.")


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
* Subtly incorporate the campaign topic.
* Maintain the specified email tone.
* Include the call to action towards the end.
* Keep the email relatively brief and professional.
* Replace "[Your Name]" with the actual sender name at the end.

Do NOT include any other text, explanation, or preamble before "Subject:" or after the email body."""),
                ("human", """Draft an email for this contact:
First Name: {first_name}
Last Name: {last_name}
Occupation: {occupation}
Email: {contact_email}
Relevant Topic (from list): {contact_topic}

Remember the campaign details: Topic='{campaign_topic}', Tone='{email_tone}', Call to Action='{call_to_action}'.""")
            ]
        )
        output_parser = StrOutputParser()
        email_drafter = draft_prompt_template | llm | output_parser
        st.sidebar.info("✅ Email Drafter Initialized.")

    except Exception as e:
        st.sidebar.error(f"🔴 Error initializing LangChain/Gemini: {e}")
        config_ok = False # Prevent further operations requiring the agent

# --- Core Application Functions ---

def parse_llm_output(llm_response):
    """Parses the LLM response string for Subject and Body."""
    subject_match = re.search(r"Subject:\s*(.*)", llm_response, re.IGNORECASE)
    body_match = re.search(r"---\s*(.*)", llm_response, re.DOTALL | re.IGNORECASE)
    subject = subject_match.group(1).strip() if subject_match else "Following Up"
    body = body_match.group(1).strip() if body_match else llm_response
    if not subject_match or not body_match:
         st.warning(f"Could not reliably parse Subject/Body markers from LLM draft output.")
    return subject, body

def generate_draft_with_gemini(contact_info, campaign_details):
    """Generates an email draft using the email_drafter chain."""
    if not email_drafter:
        st.error("Email drafter agent not initialized.")
        return "Error", "Agent not ready."

    # Prepare input, merging contact info and campaign details
    prompt_input = {
        'first_name': contact_info.get('First Name', ''),
        'last_name': contact_info.get('Last Name', ''),
        'occupation': contact_info.get('Occupation', 'N/A'),
        'contact_email': contact_info.get('Email', ''),
        'contact_topic': contact_info.get('Topics to Send', 'General'), # Use topic from CSV
        'campaign_topic': campaign_details.get('campaign_topic', 'Follow Up'),
        'email_tone': campaign_details.get('email_tone', 'Professional'),
        'call_to_action': campaign_details.get('call_to_action', 'Engage further')
    }

    try:
        llm_response = email_drafter.invoke(prompt_input)
        subject, body = parse_llm_output(llm_response)
        # Replace placeholder name immediately after generation
        body = body.replace("[Your Name]", YOUR_NAME)
        return subject, body
    except Exception as e:
        st.error(f"Error calling Gemini API for {contact_info.get('First Name', 'N/A')}: {e}")
        return f"Error generating draft", f"An error occurred: {e}"

def send_email(recipient, subject, body):
    """Connects to SMTP server and sends the email."""
    if not SENDER_EMAIL or not SENDER_PASSWORD:
        st.error("Sender email credentials not configured.")
        return False
    if not recipient or '@' not in recipient:
        st.error(f"Invalid recipient email: {recipient}")
        return False
    if not subject:
        st.warning("Sending email with an empty subject.")
        subject = "(No Subject)"

    message = EmailMessage()
    message['From'] = SENDER_EMAIL
    message['To'] = recipient
    message['Subject'] = subject
    message.set_content(body) # Assumes plain text

    # --- Use certifi's CA bundle ---
    context = ssl.create_default_context(cafile=certifi.where())
    # --------------------------------

    try:
        with smtplib.SMTP_SSL(SMTP_SERVER, SMTP_PORT, context=context) as server: # Pass the context here
            server.login(SENDER_EMAIL, SENDER_PASSWORD)
            server.send_message(message)
        return True
    except smtplib.SMTPAuthenticationError:
        st.error("SMTP Auth Failed. Check EMAIL_ADDRESS/EMAIL_PASSWORD (use App Password for Gmail/Google Workspace).")
        return False
    except Exception as e:
        # Catch other potential errors including SSL issues if context didn't fix it
        st.error(f"Error sending email to {recipient}: {e}")
        return False

def format_chat_history(history):
    """Formats chat history for the analysis prompt."""
    formatted_history = ""
    for msg in history:
        role = "User" if msg["role"] == "user" else "Assistant"
        formatted_history += f"{role}: {msg['content']}\n"
    return formatted_history.strip()


# --- Streamlit App UI ---

st.title("📧 Email Campaign Assistant")
st.markdown("Chat about your campaign, let the AI analyze it, confirm contacts, generate personalized drafts, and send!")

# --- Initialize Streamlit Session State ---
if 'app_stage' not in st.session_state:
    st.session_state.app_stage = "chat" # Start at chat stage
    # --- Updated Initial Assistant Message ---
    st.session_state.messages = [{
        "role": "assistant",
        "content": """Hi! Let's plan your email campaign. Please tell me about:
\n1.  **Campaign Topic:** What is the main subject of the email? (e.g., New product, webinar invite, special offer)
\n2.  **Target Audience:** Who should receive this email? Describe them (e.g., 'Software Engineers interested in AI', 'Marketing Managers', 'Anyone interested in leadership')
\n3.  **Call to Action:** What do you want the recipients to do after reading? (e.g., 'Schedule a meeting', 'Visit website', 'Register now')"""
    }]
    # --- End of Updated Message ---
    st.session_state.campaign_details = None
    st.session_state.all_contacts_df = None
    st.session_state.filtered_contacts_list = []
    st.session_state.confirmed_contacts_indices = [] # Store indices of confirmed contacts
    st.session_state.drafts = {} # {index: {'subject': str, 'body': str}}
    st.session_state.send_status = {} # {index: 'pending'/'sent'/'failed'}
    st.session_state.analysis_error = False


# --- Load Contacts Automatically (once) ---
if st.session_state.all_contacts_df is None and config_ok:
    try:
        contacts_df = pd.read_csv(CSV_FILE_PATH)
        # Basic validation
        required_columns = ['First Name', 'Last Name', 'Email', 'Occupation', 'Topics to Send', 'Email to Send', 'No-Go']
        missing_cols = [col for col in required_columns if col not in contacts_df.columns]
        if missing_cols:
             st.error(f"CSV file '{CSV_FILE_PATH}' is missing required columns: {', '.join(missing_cols)}")
             st.stop() # Stop if essential columns are missing

        # Handle potential NaN values gracefully
        contacts_df = contacts_df.fillna('') # Replace NaN with empty strings

        st.session_state.all_contacts_df = contacts_df
        st.sidebar.success(f"Loaded {len(st.session_state.all_contacts_df)} contacts initially.")

    except FileNotFoundError:
        st.error(f"Error: CSV file not found at '{CSV_FILE_PATH}'. Please ensure it's in the correct directory relative to the script.")
        st.stop()
    except Exception as e:
        st.error(f"An error occurred reading the CSV: {e}")
        st.stop()


# --- Main App Logic ---

# == Stage 1: Chat about the Campaign ==
if st.session_state.app_stage == "chat":
    st.header("Step 1: Describe Your Campaign")

    # Display chat messages
    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    # Get user input
    if prompt := st.chat_input("Your campaign details..."):
        if not config_ok or not campaign_analyzer:
            st.error("Configuration or AI Analyzer not ready. Cannot process chat.")
        else:
            st.session_state.messages.append({"role": "user", "content": prompt})
            st.chat_message("user").write(prompt)
            st.session_state.app_stage = "analyze_button" # Set stage for button
            st.rerun()

# == Stage 1.5: Button to Trigger Analysis ==
if st.session_state.app_stage == "analyze_button":
     st.header("Step 1: Describe Your Campaign")
     # Display chat messages again
     for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

     st.markdown("---")
     if st.button("Analyze Campaign Description", type="primary"):
        st.session_state.app_stage = "analyze_processing"
        st.rerun()

# == Stage 2: Analyze Campaign via AI ==
if st.session_state.app_stage == "analyze_processing":
    st.header("Step 2: Analyzing Campaign...")
    st.session_state.analysis_error = False # Reset error flag
    with st.spinner("AI is analyzing the campaign description..."):
        try:
            chat_history_formatted = format_chat_history(st.session_state.messages)
            # Invoke the analyzer chain
            analysis_result = campaign_analyzer.invoke({"chat_history": chat_history_formatted})
            st.session_state.campaign_details = analysis_result # Store the dict

            # --- Filtering Logic Adapted for 'target_audience' ---
            details = st.session_state.campaign_details
            df = st.session_state.all_contacts_df.copy()

            # 1. Basic Mailability Filter (remains the same)
            df_filtered = df[(df['Email to Send'].str.lower() == 'yes') & (df['No-Go'] == '')]

            # 2. Target Audience Filter (Adapted Logic)
            target_audience_desc = details.get('target_audience', 'Any')
            print(target_audience_desc)
            if target_audience_desc and target_audience_desc.lower() != 'any':
                # Simple keyword extraction from AI description
                # You might want to refine this keyword extraction and matching logic
                print()
                extracted_words = re.findall(r'\b\w+\b', target_audience_desc.lower())
                keywords_to_check = [
                        Word(word).singularize() # Apply singularize here
                        for word in extracted_words
                        if Word(word).singularize() not in ['interested', 'in', 'role', 'anyone', 'and', 'or', 'with', 'the', 'a'] # Check singular form against stop words
                 ]   
                # Remove duplicates after singularization
                keywords_to_check1 = list(set(keywords_to_check))
                print(keywords_to_check)
                print(keywords_to_check1)
                  # Output: curtain

                if keywords_to_check: # Only filter if we have keywords
                    print(keywords_to_check)
                    df_filtered = df_filtered[
                        df_filtered.apply(lambda row: any(
                            keyword in str(row['Occupation']).lower() or keyword in str(row['Topics to Send']).lower()
                            for keyword in keywords_to_check
                        ), axis=1)
                    ]
                else:
                    # If description is vague or only contains ignored words, maybe don't filter? Or handle differently.
                    st.warning("Could not extract specific keywords from target audience description. Showing all mailable contacts.")


            st.session_state.filtered_contacts_list = df_filtered.to_dict('records')
            st.session_state.app_stage = "confirm"

        except Exception as e:
            st.error(f"Error during campaign analysis or filtering: {e}")
            st.warning("Could not automatically filter contacts based on analysis. Please review the full list or refine your description.")
            st.session_state.analysis_error = True
            # Fallback: use all contacts allowed to be emailed if analysis fails
            df = st.session_state.all_contacts_df
            df_filtered = df[(df['Email to Send'].str.lower() == 'yes') & (df['No-Go'] == '')]
            st.session_state.filtered_contacts_list = df_filtered.to_dict('records')
            st.session_state.app_stage = "confirm" # Still proceed to confirmation

    st.rerun() # Move to confirmation stage


# == Stage 3: Confirm Recipients ==
if st.session_state.app_stage == "confirm":
    st.header("Step 3: Confirm Recipients")

    if st.session_state.analysis_error:
        st.warning("Analysis failed or couldn't extract clear criteria. Showing all mailable contacts.")
    elif st.session_state.campaign_details:
        st.subheader("AI Campaign Analysis Results:")
        # Display formatted details instead of raw JSON for clarity
        st.markdown(f"**Campaign Topic:** {st.session_state.campaign_details.get('campaign_topic', 'N/A')}")
        st.markdown(f"**Target Audience:** {st.session_state.campaign_details.get('target_audience', 'N/A')}")
        st.markdown(f"**Call to Action:** {st.session_state.campaign_details.get('call_to_action', 'N/A')}")
        st.markdown(f"**Email Tone:** {st.session_state.campaign_details.get('email_tone', 'N/A')}")
        st.markdown("---")

    st.subheader("Proposed Recipients:")
    if not st.session_state.filtered_contacts_list:
        st.warning("No contacts match the criteria based on the campaign analysis and the CSV list.")
        if st.button("Go Back to Chat"):
            st.session_state.app_stage = "chat"
            st.rerun()
    else:
        filtered_df = pd.DataFrame(st.session_state.filtered_contacts_list)
        st.dataframe(filtered_df[['First Name', 'Last Name', 'Email', 'Occupation', 'Topics to Send']], use_container_width=True)

        col1_confirm, col2_confirm = st.columns(2)
        with col1_confirm:
            if st.button(f"Confirm and Proceed to Drafts ({len(filtered_df)})", type="primary"):
                st.session_state.confirmed_contacts_indices = list(filtered_df.index) # Get original indices if needed
                st.session_state.app_stage = "draft"
                # Reset drafts and statuses for the new list
                st.session_state.drafts = {}
                st.session_state.send_status = {}
                st.rerun()
        with col2_confirm:
            if st.button("Cancel and Revise Campaign"):
                st.session_state.app_stage = "chat"
                st.rerun()


# == Stage 4: Generate Drafts ==
if st.session_state.app_stage == "draft":
    st.header("Step 4: Generate Email Drafts")
    st.markdown(f"Generating drafts for the **{len(st.session_state.filtered_contacts_list)}** confirmed recipients based on:")
    if st.session_state.campaign_details:
         # Display formatted details
        st.markdown(f"**Campaign Topic:** {st.session_state.campaign_details.get('campaign_topic', 'N/A')}")
        st.markdown(f"**Target Audience:** {st.session_state.campaign_details.get('target_audience', 'N/A')}")
        st.markdown(f"**Call to Action:** {st.session_state.campaign_details.get('call_to_action', 'N/A')}")
        st.markdown(f"**Email Tone:** {st.session_state.campaign_details.get('email_tone', 'N/A')}")
    else:
        st.warning("Campaign details missing (analysis might have failed). Using generic settings.")

    if st.button("✨ Generate Drafts", disabled=not config_ok or not email_drafter):
        if not st.session_state.campaign_details:
             st.error("Cannot generate drafts without campaign details.")
        else:
            total_contacts = len(st.session_state.filtered_contacts_list)
            if total_contacts == 0:
                st.warning("No contacts confirmed to generate drafts for.")
            else:
                progress_bar = st.progress(0, text="Initializing draft generation...")
                status_text = st.empty()
                st.session_state.drafts = {} # Reset drafts
                st.session_state.send_status = {} # Reset status

                # Loop through the CONFIRMED FILTERED contacts
                for i, contact in enumerate(st.session_state.filtered_contacts_list):
                    name = contact.get('First Name', f'Contact {i+1}')
                    status_text.text(f"Generating draft for {name} ({i+1}/{total_contacts})...")

                    # Pass the specific contact and the campaign details
                    subject, body = generate_draft_with_gemini(contact, st.session_state.campaign_details)

                    # Use the index 'i' from the filtered list enumeration as the key
                    st.session_state.drafts[i] = {'subject': subject, 'body': body}
                    st.session_state.send_status[i] = 'pending'
                    progress_bar.progress((i + 1) / total_contacts, text=f"Generated draft for {name} ({i+1}/{total_contacts})")
                    time.sleep(0.1) # Small UI delay

                status_text.success(f"✅ Generated drafts for all {total_contacts} confirmed contacts.")
                progress_bar.empty()
                st.session_state.app_stage = "send" # Move to send stage
                st.rerun()


# == Stage 5: Review and Send Emails ==
if st.session_state.app_stage == "send":
    st.header("Step 5: Review Drafts and Send Emails")

    if not st.session_state.drafts:
         st.warning("No drafts generated yet. Go back and generate drafts.")
         if st.button("Go Back to Generate Drafts"):
             st.session_state.app_stage = "draft"
             st.rerun()
    else:
        # Display summary metrics
        num_drafts = len(st.session_state.drafts)
        num_sent = sum(1 for status in st.session_state.send_status.values() if status == 'sent')
        num_failed = sum(1 for status in st.session_state.send_status.values() if status == 'failed')
        num_pending = num_drafts - num_sent - num_failed

        col1, col2, col3 = st.columns(3)
        col1.metric("Pending Review/Send", num_pending)
        col2.metric("Emails Sent", num_sent)
        col3.metric("Send Failures", num_failed)
        st.markdown("---")

        # Display each draft for the confirmed contacts
        for idx, draft_info in st.session_state.drafts.items():
            if idx < len(st.session_state.filtered_contacts_list):
                contact = st.session_state.filtered_contacts_list[idx]
                recipient = contact.get('Email', None)
                name = contact.get('First Name', f'Contact {idx+1}')
                status = st.session_state.send_status.get(idx, 'pending')

                expander_title = f"{idx+1}. To: {name} ({recipient or 'No Email!'}) - Status: {status.upper()}"
                with st.expander(expander_title, expanded=(status=='pending')):
                    edited_subject = st.text_input("Subject", draft_info['subject'], key=f"subject_{idx}")
                    edited_body = st.text_area("Body", draft_info['body'], height=200, key=f"body_{idx}")

                    send_button_disabled = status != 'pending' or not config_ok or not recipient
                    tooltip_msg = ""
                    if status != 'pending': tooltip_msg = f"Status is {status}."
                    elif not config_ok: tooltip_msg = "Configuration error prevents sending."
                    elif not recipient: tooltip_msg = "Recipient email address is missing."

                    if st.button(f"🚀 Send Email to {name}", key=f"send_{idx}", disabled=send_button_disabled, help=tooltip_msg):
                        send_placeholder = st.empty()
                        send_placeholder.info(f"Sending to {recipient}...")
                        if send_email(recipient, edited_subject, edited_body):
                            st.session_state.send_status[idx] = 'sent'
                            send_placeholder.success(f"Email sent successfully to {recipient}!")
                        else:
                            st.session_state.send_status[idx] = 'failed'
                            send_placeholder.error(f"Failed to send email to {recipient}.")

                        st.info(f"Pausing for {SEND_DELAY_SECONDS} seconds...")
                        time.sleep(SEND_DELAY_SECONDS)
                        st.rerun() # Update status, metrics, and button states
            else:
                st.warning(f"Mismatch in draft index {idx}. Contact data not found.")


        st.markdown("---")
        if st.button("Start New Campaign (Reset)"):
             # Reset relevant states
            st.session_state.app_stage = "chat"
            st.session_state.messages = [{ # Reset to the initial prompt
                "role": "assistant",
                "content": """Hi! Let's plan your email campaign. Please tell me about:
\n1.  **Campaign Topic:** What is the main subject of the email? (e.g., New product, webinar invite, special offer)
\n2.  **Target Audience:** Who should receive this email? Describe them (e.g., 'Software Engineers interested in AI', 'Marketing Managers', 'Anyone interested in leadership')
\n3.  **Call to Action:** What do you want the recipients to do after reading? (e.g., 'Schedule a meeting', 'Visit website', 'Register now')"""
            }]
            st.session_state.campaign_details = None
            st.session_state.filtered_contacts_list = []
            st.session_state.confirmed_contacts_indices = []
            st.session_state.drafts = {}
            st.session_state.send_status = {}
            st.session_state.analysis_error = False
            st.rerun()

