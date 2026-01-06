# AI-Powered HR Assistant

This repository contains an AI-powered HR Assistant web app that exposes a simple Flask-based UI and HTTP API to interact with an LLM-enabled assistant for HR tasks such as:

- Retrieving employee details
- Checking leave balances
- Generating interview questions for job roles
- Looking up company policies

The project provides a UI (served at `/`) and endpoints for chat (`/chat`) and clearing sessions (`/clear`).

**Important:** The application uses an OpenAI-compatible LLM via the code references in `hr_assistant.py`. Do not commit your OpenAI API key into the repository. Use environment variables as described below.

**Contents**
- `hr_assistant.py`: Main app, agents and tools (Flask app)
- `src/`: Package entry with `main.py` to run the app
- `sample/`: Example requests and usage
- `Makefile`: Common tasks (`install`, `run`, `lint`)
- `requirements.txt`: Python package dependencies
- `.gitignore`: Files to ignore in git


## Setup

1. Create and activate a virtual environment (recommended):

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# POSIX
source .venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Set your OpenAI API key as an environment variable (recommended):

- Windows (PowerShell):

```powershell
$env:OPENAI_API_KEY = "sk-..."
```

- macOS / Linux (bash):

```bash
export OPENAI_API_KEY="sk-..."
```

The app reads `OPENAI_API_KEY` from the environment. If it is not set, the app will warn at startup.


## Run

Start the app using the package entry point:

```bash
python src/main.py
```

Open your browser to `http://127.0.0.1:5000` to view the UI, or use the sample curl requests in `sample/`.


## API

- `POST /chat` — JSON body: `{ "message": "Your question here" }`. Returns `{'response': "..."}`.
- `POST /clear` — Clears the session for the requester.


## Security and Notes

- Never commit secrets (API keys) to version control. Use environment variables or secret managers.
- The included `hr_assistant.py` contains mock databases and example tools; replace or extend them to connect to real databases and secure services.


## Contributing

Feel free to open PRs for features, bug fixes, and documentation improvements.


## License

This project is unlicensed. Add a license file if you plan to publish the repository publicly.
