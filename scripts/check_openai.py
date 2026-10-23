"""Check that your OpenAI key, quota, and model work before using the app.

    python scripts/check_openai.py              # full check (one tiny paid request)
    python scripts/check_openai.py --skip-call  # config + model access only (free)
"""

from claims_autopilot.llm import main

if __name__ == "__main__":
    raise SystemExit(main())
