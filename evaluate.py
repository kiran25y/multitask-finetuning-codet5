from flask import Flask, request, render_template_string
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

app = Flask(__name__)

# ---------- Load base model (you can change to CodeT5 or your checkpoint) ----------
MODEL_NAME = "google/flan-t5-base"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def generate(text, max_new_tokens=128):
    inputs = tokenizer(text, return_tensors="pt", truncation=True).to(device)
    outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# ---------- Single-file HTML + Flask ----------
HTML = """
<!doctype html>
<html>
<head>
  <title>Code Prompt Evaluation</title>
  <style>
    body { font-family: Arial, sans-serif; margin: 20px; }
    textarea { width: 100%; height: 150px; }
    pre { background:#f4f4f4; padding:10px; white-space:pre-wrap; }
    .box { margin-bottom:20px; }
  </style>
</head>
<body>
  <h1>Multi-task Code Prompt Demo (FLAN-T5 Base)</h1>
  <form method="post">
    <div class="box">
      <label>Task:</label>
      <select name="task">
        <option value="summary"  {{ 'selected' if task=='summary'  else '' }}>Code Summary</option>
        <option value="repair"   {{ 'selected' if task=='repair'   else '' }}>Code Repair</option>
        <option value="signature"{{ 'selected' if task=='signature'else '' }}>Signature Generation</option>
      </select>
    </div>
    <div class="box">
      <label>Prompt / Input:</label><br>
      <textarea name="prompt">{{ prompt }}</textarea>
    </div>
    <button type="submit">Run</button>
  </form>

  {% if output %}
  <h2>Result</h2>
  <div class="box">
    <strong>Original Prompt:</strong>
    <pre>{{ prompt }}</pre>
  </div>
  <div class="box">
    <strong>Model Output:</strong>
    <pre>{{ output }}</pre>
  </div>
  {% endif %}
</body>
</html>
"""

@app.route("/", methods=["GET", "POST"])
def index():
    prompt = ""
    output = ""
    task = "summary"
    if request.method == "POST":
        prompt = request.form.get("prompt", "")
        task = request.form.get("task", "summary")

        # Add simple task tokens to look like your training prompts
        if task == "summary":
            full_input = f"<SUMMARY> summarize code:\n{prompt}"
        elif task == "repair":
            full_input = f"<REPAIR> fix the bug in this code:\n{prompt}"
        elif task == "signature":
            full_input = f"<SIGNATURE> infer signature from body:\n{prompt}"
        else:
            full_input = prompt

        output = generate(full_input)

    return render_template_string(HTML, prompt=prompt, output=output, task=task)

if __name__ == "__main__":
    app.run(debug=True)
"""

Save this as `app.py`, then:

