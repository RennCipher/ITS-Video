import json
from pathlib import Path
from llama_cpp import Llama

# =========================
# PATHS
# =========================

DATA_DIR = Path("data")
TOPICS_PATH = DATA_DIR / "processed_topics.json"
MCQ_PATH = DATA_DIR / "mcqs.json"
OUTPUT_PATH = DATA_DIR / "final_report.json"

MODEL_PATH = "models/Phi-3.5-mini-instruct-Q4_K_L.gguf"

# =========================
# LOAD MODEL
# =========================

print("🧠 Loading Phi-3.5-mini-instruct...")

llm = Llama(
    model_path=MODEL_PATH,
    n_ctx=2048,
    n_threads=8,
    temperature=0.3,
    verbose=False
)

print("✅ Model loaded\n")

# =========================
# LOAD DATA
# =========================

with open(TOPICS_PATH, "r", encoding="utf-8") as f:
    topics = json.load(f)

with open(MCQ_PATH, "r", encoding="utf-8") as f:
    mcqs = json.load(f)

# =========================
# BUILD CONTENT SUMMARY
# =========================

all_explanations = "\n".join(
    f"- {t['clean_explanation']}" for t in topics
)

# =========================
# DESCRIPTIVE QUESTIONS
# =========================

descriptive_questions = [
    "Explain the most important concept you learned in this video in your own words.",
    "Choose one concept from the lesson and explain why it is important, using an example."
]

student_answers = []

print("📝 Final Descriptive Evaluation\n")

for q in descriptive_questions:
    print("\n" + "=" * 60)
    print("Question:")
    print(q)
    answer = input("\nYour answer:\n").strip()
    student_answers.append({
        "question": q,
        "answer": answer
    })

# =========================
# EVALUATE DESCRIPTIVE ANSWERS
# =========================

evaluations = []

def evaluation_prompt(question, answer, content):
    return f"""
<|system|>
You are an expert teacher evaluating a student's written answer.
Be constructive and specific.
<|end|>

<|user|>
Question:
{question}

Student answer:
{answer}

Relevant instructional content:
{content}

Evaluate the answer on:
1. Conceptual correctness
2. Completeness
3. Clarity

Then give improvement suggestions.
<|end|>

<|assistant|>
"""

print("\n🧠 Evaluating descriptive answers...\n")

for item in student_answers:
    prompt = evaluation_prompt(
        item["question"],
        item["answer"],
        all_explanations
    )

    output = llm(prompt, max_tokens=500)
    feedback = output["choices"][0]["text"].strip()

    print("Feedback:")
    print(feedback)

    evaluations.append({
        "question": item["question"],
        "student_answer": item["answer"],
        "feedback": feedback
    })

# =========================
# FINAL REPORT GENERATION
# =========================

def report_prompt(topics, mcqs, evaluations):
    topic_list = ", ".join(
        f"Topic {t['topic_id']}" for t in topics
    )

    return f"""
<|system|>
You are an intelligent tutoring system generating a final learning report.
<|end|>

<|user|>
Topics covered:
{topic_list}

MCQ summary:
Total questions: {len(mcqs)}

Descriptive evaluations:
{evaluations}

Generate a final student-facing report with:
- Topics learned
- Strengths
- Weak areas
- Final summary notes
- Study recommendations
<|end|>

<|assistant|>
"""

prompt = report_prompt(topics, mcqs, evaluations)

output = llm(prompt, max_tokens=700)
final_report_text = output["choices"][0]["text"].strip()

# =========================
# SAVE FINAL REPORT
# =========================

final_report = {
    "descriptive_evaluations": evaluations,
    "final_report": final_report_text
}

with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(final_report, f, indent=2, ensure_ascii=False)

print("\n✅ STEP 7 COMPLETE")
print(f"📄 Final report saved to: {OUTPUT_PATH}")



print("\n📊 Generating learner visual analytics...")

from collections import Counter
import matplotlib.pyplot as plt

# =========================
# PATHS
# =========================

DATA_DIR = Path("data")

QUIZ_SCORE_PATH = DATA_DIR / "quiz_first_attempt_scores.json"
DESCRIPTIVE_PATH = DATA_DIR / "final_report.json"

IMAGE_OUTPUT_PATH = DATA_DIR / "learner_visual_analytics.png"
REPORT_OUTPUT_PATH = DATA_DIR / "learner_summary.json"

# =========================
# COLORS (STRICT & MINIMAL)
# =========================

GREEN = "#2E7D32"   # Strength / Mastery
RED = "#C62828"     # Weakness / Needs Reinforcement

# =========================
# LOAD DATA
# =========================

with open(QUIZ_SCORE_PATH, "r", encoding="utf-8") as f:
    quiz_data = json.load(f)

with open(DESCRIPTIVE_PATH, "r", encoding="utf-8") as f:
    descriptive_data = json.load(f)

# =========================
# EXTRACT QUIZ METRICS
# =========================

difficulty_levels = ["easy", "medium", "difficult"]

scores = []
weak_concepts = []

for level in difficulty_levels:
    level_info = quiz_data.get(level, {})
    scores.append(level_info.get("score", 0) * 100)
    weak_concepts.extend(level_info.get("weak_concepts", []))

overall_score = round(sum(scores) / len(scores), 2)

weak_counter = Counter(weak_concepts)

total_concepts = quiz_data.get("total_concepts", len(set(weak_concepts)))
mastered_concepts = total_concepts - len(set(weak_concepts))

# =========================
# VISUAL ANALYTICS
# =========================

plt.figure(figsize=(14, 10))

# -------------------------
# 1️⃣ Performance by Difficulty
# -------------------------
plt.subplot(2, 2, 1)
plt.bar(
    difficulty_levels,
    scores,
    color=[GREEN if s >= 80 else RED for s in scores]
)
plt.ylim(0, 100)
plt.ylabel("Score (%)")
plt.title("Quiz Performance by Difficulty Level")

# -------------------------
# 2️⃣ Concept Mastery Overview
# -------------------------
plt.subplot(2, 2, 2)
plt.bar(
    ["Mastered Concepts", "Needs Reinforcement"],
    [mastered_concepts, len(set(weak_concepts))],
    color=[GREEN, RED]
)
plt.title("Concept Mastery Overview")

# -------------------------
# 3️⃣ Weak Concepts Breakdown
# -------------------------
plt.subplot(2, 2, 3)
if weak_counter:
    plt.barh(
        list(weak_counter.keys()),
        weak_counter.values(),
        color=RED
    )
    plt.title("Concepts Requiring Reinforcement")

# -------------------------
# 4️⃣ Overall Understanding Index
# -------------------------
plt.subplot(2, 2, 4)
plt.bar(
    ["Overall Understanding"],
    [overall_score],
    color=GREEN if overall_score >= 80 else RED
)
plt.ylim(0, 100)
plt.title(f"Overall Understanding: {overall_score}%")

plt.tight_layout()
plt.savefig(IMAGE_OUTPUT_PATH, dpi=200)
plt.close()

# =========================
# FINAL SUMMARY REPORT
# =========================

summary_report = {
    "overall_quiz_score": overall_score,
    "quiz_scores_by_level": dict(zip(difficulty_levels, scores)),
    "strengths": {
        "mastered_concepts_count": mastered_concepts
    },
    "weaknesses": {
        "weak_concepts": list(set(weak_concepts))
    },
    "descriptive_evaluation": descriptive_data["descriptive_evaluations"],
    "final_instructor_feedback": descriptive_data["final_report"],
    "visual_report_path": str(IMAGE_OUTPUT_PATH)
}

with open(REPORT_OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(summary_report, f, indent=2, ensure_ascii=False)

# =========================
# DONE
# =========================

print("✅ STEP 8 COMPLETE")
print(f"🖼️ Visual analytics saved to: {IMAGE_OUTPUT_PATH}")
print(f"📄 Learner summary saved to: {REPORT_OUTPUT_PATH}")
