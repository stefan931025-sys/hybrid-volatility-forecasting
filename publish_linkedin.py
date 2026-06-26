import os
import json
import requests

def publish_to_linkedin():
    print("\n[DISTRIBUTION] Initializing LinkedIn Auto-Publishing Module...")
    
    # 1. Retrieve the secure token and your profile ID from GitHub Secrets
    access_token = os.getenv("LINKEDIN_ACCESS_TOKEN")
    
    if not access_token:
        print("[ERROR] Missing LINKEDIN_ACCESS_TOKEN secret. Aborting publish step.")
        return

    # Paths to the generated post
    post_path = "output/linkedin_post.txt"
    if not os.path.exists(post_path):
        print(f"[ERROR] Content artifact not found at '{post_path}'. Ensure the pipeline ran first.")
        return
        
    with open(post_path, "r", encoding="utf-8") as f:
        post_text = f.read()

    # 2. First, fetch your profile URN (Member ID) dynamically using the token
    profile_url = "https://api.linkedin.com/v2/userinfo"
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json",
        "X-Restli-Protocol-Version": "2.0.0"
    }
    
    try:
        profile_response = requests.get(profile_url, headers=headers)
        profile_response.raise_for_status()
        profile_data = profile_response.json()
        author_urn = f"urn:li:person:{profile_data['sub']}"
        print(f"[SUCCESS] Authenticated successfully as profile URN: {author_urn}")
    except Exception as e:
        print(f"[ERROR] Failed to fetch profile details: {e}")
        return

    # 3. Construct the Share API payload (V2 UgcPost Engine)
    api_url = "https://api.linkedin.com/v2/ugcPosts"
    
    payload = {
        "author": author_urn,
        "lifecycleState": "PUBLISHED",
        "specificContent": {
            "com.linkedin.ugc.ShareContent": {
                "shareCommentary": {
                    "text": post_text
                },
                "shareMediaCategory": "NONE"
            }
        },
        "visibility": {
            "com.linkedin.ugc.MemberNetworkVisibility": "PUBLIC"
        }
    }

    # 4. Broadcast to your feed
    print("[POSTING] Transmitting content packet to LinkedIn API...")
    response = requests.post(api_url, headers=headers, json=payload)
    
    if response.status_code in [201, 200]:
        print("🚀 [BROADCAST SUCCESS] Daily market update is live on your LinkedIn feed!")
    else:
        print(f"[ERROR] API Broadcast failed with status code {response.status_code}")
        print(response.text)

if __name__ == "__main__":
    publish_to_linkedin()
