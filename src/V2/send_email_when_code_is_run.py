import smtplib
from email.message import EmailMessage
import os


def send_email(sender_email, recipient_email, subject, message):
    msg = EmailMessage()
    msg.set_content(message)
    msg['Subject'] = subject
    msg['From'] = sender_email
    msg['To'] = recipient_email

    # Enter your SMTP server information
    smtp_server = 'smtp.gmail.com'
    smtp_port = 587
    smtp_username = 'guskikala@gmail.com'
    smtp_password = 'KIdmcCnnBPNN123098'
    # smtp_password = smtp_password

    with smtplib.SMTP(smtp_server, smtp_port) as server:
        server.starttls()
        server.login(smtp_username, smtp_password)
        server.send_message(msg)