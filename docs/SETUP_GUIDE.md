# Step-by-step setup guide (beginner friendly)

This guide takes you from nothing to the app running with live AI on your own computer.
It takes about 20 minutes the first time. Every command goes in a **terminal**
(Command Prompt on Windows, Terminal on Mac). Type it and press Enter.

You need an API key. Pick one:
- **Free:** a Google Gemini key (no credit card needed for the free tier, but it has daily limits)
- **Paid:** an OpenAI key (a full demo plus evaluation costs a few cents)

---

## Step 1: Install Python (skip if you already have it)

1. Go to https://www.python.org/downloads/ and download the latest Python 3.
2. Run the installer.
   - **Windows:** on the first screen, tick **"Add python.exe to PATH"**, then click Install Now.
   - **Mac:** click through the installer.
3. Check that it worked. Open a terminal and type:
   - Windows: `python --version`
   - Mac: `python3 --version`

   You should see something like `Python 3.12.x`. On a Mac, type `python3` everywhere this guide says `python`.

## Step 2: Download the project

1. Open https://github.com/kanishk33/claims-autopilot
2. Click the green **Code** button, then **Download ZIP**.
3. Unzip it somewhere easy, such as your Desktop. You'll get a folder called `claims-autopilot-main`.

## Step 3: Open a terminal inside the project folder

- **Windows:** open the `claims-autopilot-main` folder in File Explorer, click the address bar at the top, type `cmd`, and press Enter. A Command Prompt opens already inside the folder.
- **Mac:** open Terminal, type `cd ` (with a space after it), drag the `claims-autopilot-main` folder into the Terminal window, and press Enter.

Type `dir` (Windows) or `ls` (Mac). You should see `app.py`, `README.md`, and so on.

## Step 4: Create a virtual environment and install packages

A virtual environment is a private box for this project's packages. Do this once:

**Windows (Command Prompt):**
```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

**Mac:**
```
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

When it works, your prompt starts with `(.venv)`. The install takes 1 to 3 minutes.

## Step 5: Get an API key

### Option A: free Gemini key
1. Go to https://aistudio.google.com/apikey and sign in with a Google account.
2. Click **Create API key** and copy it. It starts with `AIza...`.

### Option B: OpenAI key
1. Go to https://platform.openai.com/api-keys, sign in, and click **Create new secret key**. It starts with `sk-...`.
2. Make sure your account has credit under Settings, then Billing.

Treat the key like a password. Never paste it into code, screenshots, or GitHub.

## Step 6: Put the key in a `.env` file

1. Make your own copy of the settings file:
   - Windows: `copy .env.example .env`
   - Mac: `cp .env.example .env`
2. Open it in a plain text editor:
   - Windows: `notepad .env`
   - Mac: `open -e .env`
3. Change the top lines so they look like **one** of these, then save and close.

**Gemini (free):**
```
OPENAI_API_KEY=AIza...your-gemini-key...
OPENAI_MODEL=gemini-2.5-flash
OPENAI_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai/
```

**OpenAI:**
```
OPENAI_API_KEY=sk-...your-openai-key...
OPENAI_MODEL=gpt-4.1-mini
```

The `.env` file never goes to GitHub, because `.gitignore` blocks it.

## Step 7: Check that the key works

```
python scripts/check_openai.py
```

You want to see three `[PASS]` lines and `All checks passed.` If a line says `[FAIL]`, it tells you what's wrong (see Troubleshooting below).

## Step 8: Run the app

```
streamlit run app.py
```

Your browser opens at http://localhost:8501. Then:
1. **Build Claim:** choose *Example A* in the dropdown and click **Extract claim**. The green bar shows the real model name.
2. **Precheck:** see the LOW risk result. Go back, choose *Example B*, extract it, and check Precheck again for the HIGH risk result.
3. **Denial Resolution:** choose *Sample denial*, click **Analyze denial**, then **Draft letter**.
4. **Analytics:** click **Load synthetic dataset**.

Take screenshots of the Build Claim and Denial Resolution tabs for the README.

To stop the app, click the terminal and press **Ctrl+C**.

## Step 9: Measure the AI's accuracy (for your resume)

```
python scripts/evaluate.py --llm
```

This sends the 28 synthetic claims to the model and compares every field with the correct answer. Results are written to `docs/EVALUATION.md`.

Free Gemini keys have daily request limits. If you see rate-limit failures, run a smaller batch, for example `python scripts/evaluate.py --llm --limit 10`, and note the number of claims in your results.

## Next time

You only need two commands from inside the project folder:
- Windows: `.venv\Scripts\activate`, then `streamlit run app.py`
- Mac: `source .venv/bin/activate`, then `streamlit run app.py`

---

## Troubleshooting

| You see | What it means | Fix |
|---|---|---|
| `'python' is not recognized` | Python isn't on PATH | Reinstall Python with "Add to PATH" ticked, then open a new terminal |
| `.venv\Scripts\activate` cannot be loaded (PowerShell) | PowerShell blocks scripts | Use Command Prompt (`cmd`) instead, as in Step 3 |
| `No module named claims_autopilot` | Packages were installed outside the venv | Activate the venv (prompt shows `(.venv)`), then run `pip install -r requirements.txt` again |
| `OPENAI_API_KEY is not set` | `.env` missing or misnamed | The file must be called exactly `.env`, not `.env.txt` (on Windows, check in Notepad's Save As dialog) |
| `authentication` | Wrong key | Copy the key again with no spaces |
| `insufficient_quota` | OpenAI account has no credit | Add credit, or switch to the free Gemini option |
| `rate_limit` | Too many requests (common on free tiers) | Wait a minute; for evaluation use `--limit` |
| `model_access` | Model name not available to your key | For Gemini try `gemini-2.5-flash`; for OpenAI try `gpt-4o-mini` |
| `bad_request` mentioning `response_format` | Model doesn't support structured output | Use one of the model names above |
