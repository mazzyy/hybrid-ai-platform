#!/usr/bin/env python3
"""
E-T-A RAG Prototype - CLI Chat Interface

Interactive chat with RAG-powered Q&A, including role switching.

Usage:
    python scripts/chat.py
    python scripts/chat.py --query "What is the thermal rating of CB-2410?"
    python scripts/chat.py --user thomas.mueller
"""
import sys
import os
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from retrieval.rag_engine import RAGEngine
from config.rbac import MOCK_USERS, ROLE_ACCESS


def print_header():
    print()
    print("╔══════════════════════════════════════════════════════════╗")
    print("║    E-T-A RAG Prototype - Knowledge Assistant            ║")
    print("║    Type 'help' for commands, 'quit' to exit             ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print()


def print_users():
    print("\n📋 Available users (for RBAC testing):")
    for uid, info in MOCK_USERS.items():
        role_desc = ROLE_ACCESS.get(info["role"], {}).get("description", "")
        print(f"   {uid:25s} → {info['role']:15s} ({role_desc})")
    print()


def main():
    parser = argparse.ArgumentParser(description="E-T-A RAG Chat")
    parser.add_argument("--query", "-q", type=str, help="Single query (non-interactive)")
    parser.add_argument("--user", "-u", type=str, default="david.braun", help="User ID")
    args = parser.parse_args()

    # Initialize RAG engine
    print("Loading RAG engine...")
    engine = RAGEngine()

    # Set current user
    current_user_id = args.user
    current_user = MOCK_USERS.get(current_user_id, {
        "name": "Unknown User",
        "role": "all_staff",
        "department": "general",
    })

    # Single query mode
    if args.query:
        result = engine.query(
            question=args.query,
            user_role=current_user["role"],
            user_name=current_user["name"],
        )
        print(f"\n🤖 {result['answer']}")
        if result["sources"]:
            print(f"\n📎 Sources: {', '.join(s['file'] for s in result['sources'])}")
        return

    # Interactive mode
    print_header()
    print(f"👤 Current user: {current_user['name']} (role: {current_user['role']})")
    print(f"   Access: {ROLE_ACCESS.get(current_user['role'], {}).get('description', 'Basic')}")
    print()

    while True:
        try:
            query = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not query:
            continue

        if query.lower() == "quit":
            print("Goodbye!")
            break

        if query.lower() == "help":
            print("\nCommands:")
            print("  help         - Show this help")
            print("  users        - List available users for RBAC testing")
            print("  switch <id>  - Switch to a different user")
            print("  whoami       - Show current user info")
            print("  quit         - Exit")
            print()
            continue

        if query.lower() == "users":
            print_users()
            continue

        if query.lower() == "whoami":
            print(f"\n👤 {current_user['name']} ({current_user_id})")
            print(f"   Role: {current_user['role']}")
            print(f"   Access: {ROLE_ACCESS.get(current_user['role'], {}).get('description', '')}")
            print()
            continue

        if query.lower().startswith("switch "):
            new_id = query.split(" ", 1)[1].strip()
            if new_id in MOCK_USERS:
                current_user_id = new_id
                current_user = MOCK_USERS[new_id]
                print(f"\n✅ Switched to {current_user['name']} (role: {current_user['role']})")
                print(f"   Access: {ROLE_ACCESS.get(current_user['role'], {}).get('description', '')}")
            else:
                print(f"\n❌ Unknown user: {new_id}. Type 'users' to see available users.")
            print()
            continue

        # RAG query
        result = engine.query(
            question=query,
            user_role=current_user["role"],
            user_name=current_user["name"],
        )

        print(f"\n🤖 {result['answer']}")

        if result["sources"]:
            print(f"\n📎 Sources:")
            for src in result["sources"]:
                print(f"   - {src['file']} ({src['department']}, {src['doc_type']})")

        print(f"   [{result['chunks_retrieved']} chunks retrieved, role: {result['user_role']}]")
        print()


if __name__ == "__main__":
    main()
