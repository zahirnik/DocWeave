# Critic Prompt

Given the draft answer and the contexts, check for:
- fabrications: numbers or claims not present in contexts
- missing citations when facts are asserted
- verbosity: prefer succinct bullet points

If issues are found, output a **revised** answer; else, return the original.
Keep formatting minimal and preserve citations like [1], [2].
