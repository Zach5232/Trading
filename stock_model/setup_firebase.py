#!/usr/bin/env python3
"""
First-time Firebase setup for friday_close.py
Run once: python3 stock_model/setup_firebase.py

Re-running reuses your saved UID and just re-verifies the connection.
Use --reset to clear the saved UID and re-enter it:
    python3 stock_model/setup_firebase.py --reset
"""
import json
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_DIR = os.path.join(SCRIPT_DIR, 'config')
KEY_PATH = os.path.join(CONFIG_DIR, 'firebase_service_account.json')
CONFIG_PATH = os.path.join(CONFIG_DIR, 'user_config.json')

print("=== Stock Model — Firebase Setup ===")

if '--reset' in sys.argv:
    if os.path.exists(CONFIG_PATH):
        os.remove(CONFIG_PATH)
        print("✓ Cleared saved config.\n")
    else:
        print("No saved config to clear.\n")

uid = None
if os.path.exists(CONFIG_PATH):
    try:
        with open(CONFIG_PATH) as f:
            uid = json.load(f).get('uid')
    except (json.JSONDecodeError, OSError):
        uid = None

if uid:
    print(f"✓ Using saved UID: {uid[:8]}...")
else:
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

    os.makedirs(CONFIG_DIR, exist_ok=True)
    with open(CONFIG_PATH, 'w') as f:
        json.dump({"uid": uid}, f, indent=2)
    print(f"\n✓ Saved to {CONFIG_PATH}")

print("\nTesting Firestore connection...")

try:
    import firebase_admin
    from firebase_admin import credentials, firestore
    if not firebase_admin._apps:
        try:
            firebase_admin.initialize_app()
            firestore.client()  # ADC is resolved lazily — force it now so failures are caught here
        except Exception:
            for app in list(firebase_admin._apps.values()):
                firebase_admin.delete_app(app)
            print(f"  Looking for service account key at: {KEY_PATH}")
            print(f"  Key exists: {os.path.exists(KEY_PATH)}")
            if os.path.exists(KEY_PATH):
                cred = credentials.Certificate(KEY_PATH)
                firebase_admin.initialize_app(cred)
            else:
                print("No credentials found. Either:")
                print("  1. Install gcloud and run:")
                print("     gcloud auth application-default login")
                print("  2. Or download a service account key from:")
                print("     Firebase Console → Project Settings → Service Accounts")
                print("     Save as: stock_model/config/firebase_service_account.json")
                sys.exit(1)
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
