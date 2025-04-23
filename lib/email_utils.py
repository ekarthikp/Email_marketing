import smtplib
import ssl
import certifi
import re
import mimetypes # For guessing MIME type
from email.message import EmailMessage
import streamlit as st # For st.error/warning

def send_email_smtp(recipient, subject, body, config, attachment_filename=None, attachment_bytes=None):
    """
    Connects to SMTP server using config details, sends the email,
    and optionally attaches a file.

    Args:
        recipient (str): The recipient's email address.
        subject (str): The email subject line.
        body (str): The plain text email body.
        config (dict): Dictionary containing sender configuration
                       (sender_email, sender_password, sender_name, smtp_server, smtp_port).
        attachment_filename (str, optional): The desired filename for the attachment. Defaults to None.
        attachment_bytes (bytes, optional): The raw bytes of the file to attach. Defaults to None.

    Returns:
        bool: True if successful, False otherwise.
    """
    sender_email = config.get('sender_email')
    sender_password = config.get('sender_password')
    sender_name = config.get('sender_name', sender_email) # Default sender name to email if not set
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
    message['From'] = f"{sender_name} <{sender_email}>"
    message['To'] = recipient
    message['Subject'] = subject
    message.set_content(body) # Set the plain text body

    # --- Add Attachment if provided ---
    if attachment_filename and attachment_bytes:
        try:
            # Guess the MIME type
            ctype, encoding = mimetypes.guess_type(attachment_filename)
            if ctype is None or encoding is not None:
                # Use a generic type if guessing fails
                ctype = 'application/octet-stream'
            maintype, subtype = ctype.split('/', 1)

            message.add_attachment(attachment_bytes,
                                   maintype=maintype,
                                   subtype=subtype,
                                   filename=attachment_filename)
            print(f"Attachment '{attachment_filename}' added to email for {recipient}.")
        except Exception as attach_e:
            st.error(f"Failed to attach file '{attachment_filename}' for {recipient}: {attach_e}")
            # Decide if you want to send the email anyway or fail here
            # return False # Uncomment this to prevent sending if attachment fails

    # Use certifi's CA bundle for SSL context
    context = ssl.create_default_context(cafile=certifi.where())

    try:
        with smtplib.SMTP_SSL(smtp_server, smtp_port, context=context) as server:
            server.login(sender_email, sender_password)
            server.send_message(message)
        print(f"Successfully sent email to {recipient}" + (f" with attachment '{attachment_filename}'" if attachment_filename else "")) # Log success
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

