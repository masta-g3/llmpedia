import smtplib
from email import message
import os


def send_email_alert(tweet_content, arxiv_code):
    sender_email = os.getenv("SENDER_EMAIL")
    sender_password = os.getenv("SENDER_EMAIL_PASSWORD")
    receiver_email = os.getenv("RECEIVER_EMAIL")
    
    subject = f"New Tweet Alert for arXiv:{arxiv_code}"
    body = f"""
    A new tweet has been posted:

    {tweet_content}

    arXiv link: https://arxiv.org/abs/{arxiv_code}
    Tweet link: https://twitter.com/search?q=from%3ALLMPedia%20{arxiv_code}&src=typed_query&f=live

    """

    msg = message.EmailMessage()
    msg.set_content(body)
    msg['Subject'] = subject
    msg['From'] = sender_email
    msg['To'] = receiver_email

    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(sender_email, sender_password)
            server.send_message(msg)
    except Exception as e:
        print(f"Error sending email alert: {str(e)}")