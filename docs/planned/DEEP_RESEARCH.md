Next-Level Capability: Deep Research Tool for SOTA Integration

1. Introduction: Bridging the Research-Development Gap

The current PyTorch ML Research Agent (PyTorch-MRA) excels at translating declarative configurations into executable PyTorch code and iteratively refining architectures based on performance metrics. However, its exploration space is limited to variations the Planning LLM can conceive internally.

To achieve true architectural innovation, the agent must be able to perform targeted, external research to:

Identify SOTA: Discover the most current and effective techniques for a given task (e.g., "What is the SOTA backbone for image segmentation?").

Extract Novel Concepts: Parse complex academic language (e.g., from arXiv) and distill novel concepts, layers, or activation functions into structured data.

Ground the Plan: Incorporate research findings into the model_config and the overall execution plan, making the agent's decisions evidence-based and informed by the ML community's latest work.

This document outlines the requirements for the Deep Research Tool ($\text{DRT}_{\text{LLM}}$), which will serve as the Planning LLM's primary external knowledge source.

2. The Deep Research Tool ($\text{DRT}_{\text{LLM}}$)

The $\text{DRT}_{\text{LLM}}$ is a specialized tool that uses the Planning LLM (or a dedicated, highly contextual research-focused LLM) in conjunction with targeted search queries to generate actionable research summaries.

2.1 Tool Signature and Function

The $\text{DRT}_{\text{LLM}}$ will be an atomic tool available for the Planning LLM to call before generating the initial model_config or when a refinement plan requires new architectural ideas.

Tool Name: deep_research_tool

Purpose: Executes a focused search over academic sources (e.g., arXiv, ML blogs, key conference proceedings) and synthesizes the findings into a structured JSON output for the Planning LLM.

Tool Inputs (Required):

research_query: A precise, natural language query (e.g., "novel attention mechanisms for CNNs", "transformer block alternatives for time series").

output_format: The desired structure for the synthesized output (e.g., "list of three SOTA architectural ideas for image classification").

Tool Output (Example JSON Structure):

The tool returns a string containing a JSON object, parsed by the Planning LLM for strategic use.

{
  "summary_title": "Recent Advances in Image Classification Backbones (2024)",
  "insights": [
    "SOTA models frequently incorporate techniques like dynamic convolution and spatial attention mechanisms (SAM) instead of standard residual blocks.",
    "Efficient training is achieved using cosine annealing learning rate schedules, often coupled with larger batch sizes (e.g., 2048)."
  ],
  "architectural_suggestions": [
    {
      "name": "Dynamic Convolution Block (DCB)",
      "description": "Convolution kernel parameters are adaptively generated based on the input, enhancing model capacity without deeper networks.",
      "implementation_note": "Requires custom PyTorch module; key components are a lightweight prediction head and weighted aggregation."
    },
    {
      "name": "GhostNet Module",
      "description": "Generates feature maps from cheap operations (identity mapping and linear transformations), reducing computational cost by 50%.",
      "implementation_note": "A sequence of two convolutions: one standard, one depth-wise linear transformation."
    }
  ],
  "sources": [
    {"title": "Paper X: Dynamic Convolution", "url": "[https://arxiv.org/abs/XXXXX](https://arxiv.org/abs/XXXXX)"},
    {"title": "Paper Y: GhostNet", "url": "[https://arxiv.org/abs/YYYYY](https://arxiv.org/abs/YYYYY)"}
  ]
}


2.2 Integration into the Agent Workflow

The $\text{DRT}_{\text{LLM}}$ introduces a critical Research Phase into the Autonomous Research Loop:

Phase

Previous Workflow

New Workflow with $\text{DRT}_{\text{LLM}}$

1. Goal Initiation

User defines the target (e.g., 70% accuracy on CIFAR-10).

User defines the target.

2. Research (NEW)

N/A

Planning LLM calls deep_research_tool to find SOTA architectures relevant to the goal.

3. Plan & Config

Planning LLM generates model_config based on internal knowledge.

Planning LLM generates model_config informed by the $\text{DRT}_{\text{LLM}}$ JSON output, explicitly integrating novel concepts (e.g., using a GhostNet layer).

4. Prototyping

Agent calls pytorch_model_assembler.

Agent calls pytorch_model_assembler.

5. Validation & Analysis

Agent performs quick evaluation and refines the current concept.

Agent analyzes results, and if refinement fails, it can call the $\text{DRT}_{\text{LLM}}$ again with a new, more specific query (e.g., "Why did the GhostNet module fail on this dataset?").

2.3 Success Criteria

The $\text{DRT}_{\text{LLM}}$ will be considered successful if:

Structured Output: It consistently returns a valid JSON object matching the requested schema.

SOTA Adoption: The Planning LLM's generated model_config explicitly incorporates a concept (layer, activation, structure) derived from the $\text{DRT}_{\text{LLM}}$ output in at least $75\%$ of initial research phases.

Performance Uplift: The average performance of the final, best model developed by the agent is statistically higher when the $\text{DRT}_{\text{LLM}}$ is enabled, compared to a baseline where only internal knowledge is used, demonstrating the value of SOTA integration.
