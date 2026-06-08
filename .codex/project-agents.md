# Project Agents Routing

This file is a project-local routing guide for Codex-style work in this repository.

## Purpose

- Use this file as the first project-specific routing reference when the user asks to follow the project's local agents, Copilot-style agent files, or repository-specific workflows.
- Treat files under `.github/agents/` as the primary project-local agent definitions.
- Keep this file lightweight. Do not duplicate long background knowledge here when it already exists in the agent files.

## Routing Rules

1. If the user explicitly names an agent file, read that file first and follow it unless it conflicts with higher-priority system or developer instructions.
2. If the user asks to use the project's default agent workflow, choose from `.github/agents/` using the task mapping below.
3. If multiple agent files are relevant, prefer one primary agent and use others only as supporting references.
4. When project-local agent guidance conflicts with repository reality, inspect the code/files and follow the repository state.
5. Project-local agent files are repository-scoped context, not global knowledge. Do not carry their assumptions into unrelated repositories.

## Task Mapping

- Code understanding, architecture tracing, training/inference flow explanation, VisionSelector integration details:
  - `.github/agents/code-understand-master.md`

- HF conversion, HF-side inference parity, conversion scripts, RVOS evaluation wiring:
  - `.github/agents/visionselector-sa2va-hf-conversion.agent.md`

- CVPR 2026 paper drafting, method writing, experiment writeups, LaTeX section drafting:
  - `.github/agents/cvpr2026-writing-expert.agent.md`

- Reviewer-style critique, novelty framing, claim sanity check, fusion-writing feedback:
  - `.github/agents/multimodal-fusion-writing-reviewer.agent.md`

## Selection Hints

- If the task is about modifying code, prefer the agent whose scope matches the target files.
- If the task is about explanation only, prefer the most read-oriented agent.
- If the task mixes code and writing, choose the main deliverable first:
  - code change first -> use the engineering/code agent as primary
  - paper text first -> use the writing agent as primary

## Fallback Behavior

- If no project-local agent clearly matches the task, continue with normal Codex reasoning.
- If a referenced agent file is missing, ignore that route and continue with the closest available agent or normal Codex reasoning.
