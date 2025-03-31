import requests

def send_discord_notification(webhook_url, content):
    """
    Sends a notification to a Discord channel via webhook.

    Args:
        webhook_url (str): The Discord webhook URL.
        content (str): The message content to send.
    """ 
    webhook_url = "https://discord.com/api/webhooks/1332390354430984242/HyVY_RCiwqYIcbP_Yc9BHQ8POZC9xQXUv_rxIq08u2Dm3CbAqc4HAcWjqEp66DIdBt4i"
    data = {"content": content}  # The message content

    try:
        response = requests.post(webhook_url, json=data)
        if response.status_code == 204:
            print("Notification sent successfully!")
        else:
            print(f"Failed to send notification. Status code: {response.status_code}, Response: {response.text}")

    except Exception as e:
        print(f"Error sending notification: {e}")