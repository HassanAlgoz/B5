# HELP

## Coding Cycle

You typically code in this cyclic manner:

1. Write the plan in 5 bullets (no code yet)
2. Implement the smallest piece
3. Run → break → read error → fix
4. Repeat

**When adding features or fixing bugs you commit**:

```sh
git commit -m "feat: render output as markdown"
```

Or 

```sh
git commit -m "fix: use Path from pathlib instead of os.path"
```

**By the end of each day:**

```sh
git push
```


## AI Assistance:

- You may enable the in-editor **Co-pilot** for AI-assisted auto-complete.
- You may ask it to explain concepts or errors.
- You may not ask it to solve the task.

Below is an example help prompt which you may use for maximum learning with AI assistant.

1. Copy
2. Edit: fill in the 7 blanks
3. Send to AI Assistant


```text
I am a student working on a technical assignment. I need you to act as a Socratic Tutor. 

CRITICAL INSTRUCTION: Do not provide the direct solution or corrected code. Instead, analyze my input and ask 1-2 leading questions that help me realize the mistake or the next step on my own.

---
1. CONTEXT:
- Project/Lesson: [Insert Lesson Name]
- Goal: [What are you trying to achieve?]

2. THE PROBLEM:
- Symptoms: [Describe exactly what is happening vs. what you expected]
- Error Message: [Paste the text of the error here, or say 'None']

3. MY EFFORTS:
- Previous Attempts: [What did you already try to fix this?]
- My Theory: [What do you think is causing the issue?]

4. THE CODE:
[Paste your code snippet here]
---

Help me understand the underlying concept rather than just fixing the bug. What should I look at first?
```