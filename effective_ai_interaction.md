# Guidelines for Effective AI Interaction

## 1. Start with Context
Instead of:
```
"Let's optimize X"
```
Better:
```
"I'm looking at optimizing X because we saw Y performance issue before.
Can you first show me how this currently works in the code?"
```

## 2. Request Analysis Before Implementation
Ask for:
- Show code flow
- Explain current design
- Point out ordering/timing dependencies
```
"Before we change anything, can you:
- Show me the code flow
- Explain the current design
- Point out any ordering/timing dependencies"
```

## 3. Share Historical Context
Provide background:
```
"Previously we had issue X because of Y.
Things have changed (like HCA support).
Let's verify if this is still a problem."
```

## 4. Question AI's Assumptions
Challenge the AI:
```
"Before we implement that, can you explain:
- Why you think this change is needed?
- What problem it's solving?
- Whether the problem still exists?"
```

## 5. Guide AI's Focus
Direct the analysis:
```
"Don't implement yet. First show me:
- Where this operation happens
- What happens before/after it
- How it interacts with other operations"
```

## Key Takeaways
1. Force AI to analyze before implementing
2. Help AI understand system context
3. Get AI to expose important details you might have forgotten
4. Avoid wasting time on unnecessary changes

## Remember
The AI is eager to help implement, but needs guidance to analyze first. 
Make it show you the current system behavior before allowing it to suggest changes.

## Today's Example
What went wrong:
- AI jumped to implementation without understanding system flow
- Missed critical ordering (copy after send) that made optimization unnecessary
- Didn't question whether historical problems still existed with new infrastructure

What should have happened:
1. First understand current code flow
2. Notice clever existing design (copy after send)
3. Question whether optimization was still needed
4. Only then consider changes if necessary 