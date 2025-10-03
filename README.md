# Run AWS Bedrock LLM Inference in Google Colab (Step-by-Step, Human-Friendly)

If you want a fast way to try **Amazon Bedrock** models from a notebook, Google Colab is perfect: no local setup, easy package installs, and a built-in place to store secrets. Below is a clean, professional walkthrough—based directly on the notebook  how to send a prompt to **`amazon.nova-lite-v1:0`** and read back the model’s response.

---

## What you’ll build

You’ll create a short Colab notebook that:

1. Installs `boto3` (the AWS SDK for Python).
2. Securely loads your AWS credentials from Colab’s **User data** (secrets).
3. Calls **Bedrock Runtime** with a JSON payload like this:

```json
{
  "inferenceConfig": { "max_new_tokens": 1000 },
  "messages": [
    { "role": "user", "content": [ { "text": "this is where you place your input text" } ] }
  ]
}
```

4. Parses the response and prints the generated text.

---

## Prerequisites (quick checklist)

* **AWS account** with **Bedrock** enabled in your region (e.g., `us-east-1`).
* **Model access** for `amazon.nova-lite-v1:0` (enable it in the Bedrock console).
* **IAM credentials** that can invoke models:

  * `bedrock:InvokeModel` (and typically `bedrock:InvokeModelWithResponseStream` if you stream).
* A **Google Colab** notebook.

> **Tip:** Bedrock access is region-scoped. Make sure your chosen region matches the one where you enabled the model.

---

## 1) Add your AWS credentials to Colab “User data”

In Colab, go to **Tools → User data** and add:

* `AWS_ACCESS_KEY_ID`
* `AWS_SECRET_ACCESS_KEY`
* `AWS_DEFAULT_REGION` (e.g., `us-east-1`)
* *(optional)* `AWS_BEARER_TOKEN_BEDROCK` — If you’re using **temporary** credentials, you can paste the **session token** here. In the notebook below, we map it to `AWS_SESSION_TOKEN` so `boto3` picks it up automatically.

> Never hardcode secrets in notebooks. Colab’s **User data** keeps them out of your code and out of version control.

---

## 2) Install dependencies (once per runtime)

```python
# Install dependencies
import sys, subprocess, importlib

def pip_install(pkg):
    if importlib.util.find_spec(pkg) is None:
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])

for pkg in ["boto3", "botocore"]:
    pip_install(pkg)

print("✔ Dependencies ready")
```

This cell checks for `boto3`/`botocore` and installs them if needed.

---

## 3) Load secrets from Colab and export to environment

```python
from google.colab import userdata
import os

aws_access_key_id     = userdata.get('AWS_ACCESS_KEY_ID')
aws_secret_access_key = userdata.get('AWS_SECRET_ACCESS_KEY')
aws_default_region    = userdata.get('AWS_DEFAULT_REGION')
aws_bearer_token      = userdata.get('AWS_BEARER_TOKEN_BEDROCK')  # optional (use if you're on temporary creds)

missing = [k for k,v in {
    "AWS_ACCESS_KEY_ID": aws_access_key_id,
    "AWS_SECRET_ACCESS_KEY": aws_secret_access_key,
    "AWS_DEFAULT_REGION": aws_default_region,
}.items() if not v]

if missing:
    print("⚠ Missing keys in Colab userdata:", ", ".join(missing))
else:
    print("✔ Found required keys in Colab userdata")

# Export so boto3 can auto-discover them
if aws_access_key_id:     os.environ["AWS_ACCESS_KEY_ID"]     = aws_access_key_id
if aws_secret_access_key: os.environ["AWS_SECRET_ACCESS_KEY"] = aws_secret_access_key
if aws_default_region:    os.environ["AWS_DEFAULT_REGION"]    = aws_default_region

# If you provided a session token under AWS_BEARER_TOKEN_BEDROCK, map it here:
if aws_bearer_token:
    os.environ["AWS_SESSION_TOKEN"] = aws_bearer_token
    print("ℹ Using AWS_BEARER_TOKEN_BEDROCK as AWS_SESSION_TOKEN")
```

**What this does:**

* Reads your secrets from Colab’s store.
* Sets environment variables so `boto3` uses them automatically.
* Gracefully handles optional session tokens (typical when you use temporary STS credentials).

---

## 4) Prepare your model request

In Bedrock, text-generation models follow a consistent shape: an `inferenceConfig` block for generation parameters and a `messages` array for the conversation.

```python
import json

model_id = "amazon.nova-lite-v1:0"

# Your prompt:
user_prompt = "Explain the difference between a star and a planet in three simple points."

payload = {
    "inferenceConfig": {
        "max_new_tokens": 1000
        # You can also include optional parameters if the model supports them, e.g.:
        # "temperature": 0.7,
        # "top_p": 0.9
    },
    "messages": [
        {
            "role": "user",
            "content": [
                {"text": user_prompt}
            ]
        }
    ]
}

body = json.dumps(payload)
print("✅ Inference payload prepared.")
```

---

## 5) Call Bedrock Runtime and parse the response

```python
import boto3, json
from botocore.exceptions import BotoCoreError, ClientError

bedrock = boto3.client("bedrock-runtime")  # region comes from env

try:
    print("\n🚀 Sending request to Bedrock...")
    response = bedrock.invoke_model(
        modelId=model_id,
        contentType="application/json",
        accept="application/json",
        body=body.encode("utf-8")
    )

    # Read and parse the response body
    response_body = json.loads(response["body"].read())

    # For nova-lite, the text is usually here:
    text = response_body["output"]["message"]["content"][0]["text"]

    print("\n" + "="*50)
    print("✅ INFERENCE COMPLETE")
    print("="*50)
    print(f"\n🗣️ USER PROMPT:\n{user_prompt}")
    print("\n🤖 MODEL RESPONSE:")
    print(text)
    print("\n" + "="*50)

except (BotoCoreError, ClientError) as e:
    print("🛑 Invocation error:", e)
except KeyError:
    # If the structure changes or you’re using a different model family,
    # dump the raw JSON to inspect the fields.
    print("⚠ Could not find expected text field. Full JSON:")
    print(json.dumps(response_body, indent=2))
```

**Why this works well:**

* It sends the exact structure Bedrock expects.
* It extracts the assistant’s text from a predictable path for `nova-lite`.
* It falls back to printing raw JSON so you can debug other models or future schema tweaks.

---

## Running the same from a local machine (optional)

If you want to run locally instead of Colab, set environment variables in your shell:

```bash
export AWS_ACCESS_KEY_ID="YOUR_ACCESS_KEY"
export AWS_SECRET_ACCESS_KEY="YOUR_SECRET_KEY"
export AWS_DEFAULT_REGION="us-east-1"
# If using temporary credentials:
# export AWS_SESSION_TOKEN="YOUR_SESSION_TOKEN"
```

Then remove the `google.colab.userdata` part and rely on `boto3`’s default credential chain. The rest of the code stays the same.

---

## Common pitfalls & fixes

* **`AccessDeniedException`**
  You likely don’t have permission to invoke the model, or you haven’t enabled access to `amazon.nova-lite-v1:0` in your region. Fix your IAM policy (e.g., include `bedrock:InvokeModel`) and enable model access in the Bedrock console.

* **`ValidationException` / empty outputs**
  Check your payload keys and types (`inferenceConfig`, `messages`, `role`, `content`, `text`). Typos or wrong types are the usual culprit.

* **“Unable to locate credentials”**
  In Colab, verify you stored `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, and `AWS_DEFAULT_REGION` in **Tools → User data**. Locally, verify your environment variables or AWS profile.

* **Wrong region**
  If the model isn’t enabled in your region, Bedrock won’t accept the call. Make sure `AWS_DEFAULT_REGION` matches where you turned on the model.

---

## Security & cost notes

* **Never** commit credentials to code or notebooks. Use Colab’s secret storage or environment variables.
* Grant the **least privileges** your workflow needs.
* LLM invocations can incur costs. Keep an eye on `max_new_tokens`, and consider truncating long prompts or responses in experiments.

---

## That’s it

You now have a compact, production-style Colab flow for sending prompts to **AWS Bedrock** and reading results—no mystery, no hidden steps. From here you can:

* Add parameters like `temperature`, `top_p`, or safety settings (if supported).
* Wrap the call in a function and iterate on prompts.
* Swap `modelId` to try other Bedrock models you’ve enabled.

Happy building! 🚀
