#!/usr/bin/env python3
"""
First-time Firebase setup for friday_close.py
Run once: python3 stock_model/setup_firebase.py
"""
import json
import os
import sys

config_dir = os.path.join(os.path.dirname(__file__), 'config')
config_path = os.path.join(config_dir, 'user_config.json')

print("=== Stock Model — Firebase Setup ===")
print("\nYou need your Firebase UID.")
print("Find it here:")
print("  1. Go to console.firebase.google.com")
print("  2. Select project: stock-model-42fb2")
print("  3. Click Authentication → Users")
print("  4. Find your email row")
print("  5. Copy the UID from that row\n")

uid = input("Paste your Firebase UID: ").strip()
if not uid:
    print("No UID entered. Exiting.")
    sys.exit(1)

os.makedirs(config_dir, exist_ok=True)
with open(config_path, 'w') as f:
    json.dump({"uid": uid}, f, indent=2)

print(f"\n✓ Saved to {config_path}")
print("\nTesting Firestore connection...")

try:
    import firebase_admin
    from firebase_admin import firestore
    if not firebase_admin._apps:
        firebase_admin.initialize_app()
    db = firestore.client()
    doc = db.collection('users').document(uid)\
            .collection('data').document('candidatePool').get()
    if doc.exists:
        items = doc.to_dict().get('items', [])
        print(f"✓ Connected! Found {len(items)} items in candidatePool")
    else:
        print("✓ Connected! candidatePool not yet created (normal for new setup)")
    print("\nSetup complete. You can now run:")
    print("  python3 stock_model/friday_close.py")
except ImportError:
    print("✗ Firebase Admin SDK not installed.")
    print("  Run: pip3 install firebase-admin --break-system-packages")
except Exception as e:
    print(f"✗ Connection failed: {e}")
    print("\nMake sure you are authenticated with Google:")
    print("  gcloud auth application-default login")
    print("Or set GOOGLE_APPLICATION_CREDENTIALS to a service account key")
