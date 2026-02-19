A curated list of papers, datasets, and resources focusing on Multi-turn interactions, Multimodal Embeddings, Long Reasoning, and Context Compression in Large Language Models (LLMs) and Multimodal LLMs (MLLMs).

## Contents

- [Multiturn LLM/MLLM](#multiturn-llmmllm)
    - [Existing Works](#existing-works)
    - [Datasets](#datasets)
- [Multimodal Embedding](#multimodal-embedding)
- [Long Reasoning Generation](#long-reasoning-generation)
- [Long Prompt Context & Compression](#long-prompt-context--compression)
- [Contributing](#contributing)

---

## Multiturn LLM/MLLM

Research focused on improving model performance in multi-turn dialogues and addressing issues like context loss and intent mismatch.

### Existing Works

* **ConsistentChat: Building Skeleton-Guided Consistent Multi-Turn Dialogues for Large Language Models from Scratch** [📄 Paper](https://arxiv.org/abs/2506.03558)  
    > **Summary:** Proposes a skeleton-guided framework to construct high-quality multi-turn datasets from scratch. The construction method operates in two stages: (1) mapping conversations into 9 predefined intent trajectories to generate a globally coherent sequence of user queries (the "skeleton"), and (2) employing a single-pass Chain-of-Thought (CoT) generation strategy to synthesize all AI responses simultaneously. This prevents context drift and ensures strong cross-turn consistency. 

* **LLMs Get Lost In Multi-Turn Conversations** [📄 Paper](https://arxiv.org/abs/2505.06120)  
    > **Summary:** When converting complicated questions into a multi-turn format, models often fail to answer correctly, leading to a 35% performance degradation.

* **Intent Mismatch Causes LLMs to Get Lost in Multi-Turn Conversation** [📄 Paper](https://arxiv.org/abs/2602.07338v1)  
    > **Summary:** Addresses poor performance in multi-turn conversations by using an external model to interpret user intention and feeding this extra information to the inference model.

* **Taking Notes Brings Focus? Towards Multi-Turn Multimodal Dialogue Learning** [📄 Paper](https://arxiv.org/abs/2503.07002)  
    > **Summary:** Proposes keeping important information during inference to use as context, alleviating the issue of context overload.

### Datasets (Text-Only & Multimodal)

| Dataset | Type/Domain | Link/Source |
| :--- | :--- | :--- |
| **ConsistentChat** | Multi-turn Dialogue | [Hugging Face](https://huggingface.co/datasets/jiawei-ucas/ConsistentChat) |
| **LiC** | Math, Code, Database, Action | `microsoft/lost_in_conversation` |
| **M2Lingual** | Multilingual | - |
| **SkillMix** | Instruction Tuning | `PrincetonPLI/Instruct-SkillMix-SDD` |
| **MMDiag** | Multimodal | *(Unreleased)* |
| **CB-300K** | Chat | [GitHub](https://github.com/sunsmarterjie/ChatterBox?tab=readme-ov-file) |
| **BelleGroup** | Chat (0.8M) | `BelleGroup/multiturn_chat_0.8M` |
| **MMDU** | Multimodal Dialogue | `laolao77/MMDU` |
| **ConvBench** | Benchmark | - |
| **MIRAGE** | Benchmark | `MIRAGE-Benchmark/MIRAGE` |
| **MIMIC-IT** | Instruction Tuning | - |
| **M3T** | Multimodal | *(Unreleased)* |
---

## Multimodal Embedding

Techniques for unifying vision and text representations, specifically for contrastive learning and richer visual understanding.

* **MuCo: Multi-turn Contrastive Learning for Multimodal Embedding Model** [📄 Paper](https://arxiv.org/abs/2602.06393)  
    > **Summary:** Introduces a contrastive learning framework specifically for multimodal embeddings to better capture the relationship between vision and text across multiple dialogue turns.

* **VLM2Vec-V2: Unified Multimodal Embedding Framework** [📄 Paper](https://arxiv.org/abs/2410.05160)  
    > **Summary:** Converts Qwen2VL into a document/video embedder using contrastive learning. It maps temporal data (videos) and complex visual documents (PDFs, websites) into a single, unified vector space.

* **LLM2CLIP: Powerful Language Model Unlocks Richer Visual Representation** [📄 Paper](https://arxiv.org/abs/2411.04997)  
    > **Summary:** Enhances CLIP by replacing its standard text encoder with a more powerful LLM. It uses contrastive loss to differentiate captions and leverages LLM embeddings to provide richer supervision for visual features.

---

## Long Reasoning Generation

Approaches to optimize Chain-of-Thought (CoT) reasoning, reduce computational cost, and maintain visual grounding over long sequences.

* **Dynamic Chain-of-Thought: Towards Adaptive Deep Reasoning** [📄 Paper](https://arxiv.org/abs/2502.10428)  
    > **Summary:** Implements an "Importance Score" for reasoning blocks to prune redundant steps. Uses RL to adaptively set thresholds, keeping a macro summary and micro details to allow models to "shortcut" to an answer.

* **TokenSkip: Controllable Chain-of-Thought Compression in LLMs** [📄 Paper](https://arxiv.org/abs/2502.12067)  
    > **Summary:** Aims to reduce the "reasoning tax" by using a judge model to delete low-importance tokens within reasoning chains. The model is then fine-tuned (SFT) on this compressed reasoning data.

* **Mitigating Visual Forgetting via Take-along Visual Conditioning** [📄 Paper](https://arxiv.org/abs/2503.13360)  
    > **Summary:** Addresses the issue where models "forget" original image features during long-form reasoning. It re-injects visual tokens or summaries into hidden states at specific reasoning intervals.

---

## Long Prompt Context & Compression

Methods for handling long contexts efficiently, including training data selection, token compression, and memory agents.

* **LADM: Long-context Training Data Selection with Attention-based Dependency Measurement** [📄 Paper](https://arxiv.org/abs/2503.02502)  
    > **Summary:** Filters training data by measuring "Contextual Dependency." Uses attention scores to ensure the model learns from "dense," interconnected information rather than unrelated text chunks.

* **Lossless Token Sequence Compression via Meta-Tokens** [📄 Paper](https://arxiv.org/abs/2506.00307)  
    > **Summary:** Uses a "Meta-Encoder" to compress 512 raw tokens into 16 "Meta-Tokens." This shrinks the KV cache and context length while preserving the original semantic and structural essence.

* **MemAgent: Reshaping Long-Context LLM with Multi-Conv RL-based Memory Agent** [📄 Paper](https://arxiv.org/abs/2507.02259)  
    > **Summary:** Employs an RL-trained agent that manages a compact memory. It updates its internal memory state based on what information actually contributes to the correctness of the final answer.

* **Text or Pixels? On the Token Efficiency of Visual Text Inputs in Multimodal LLMs** [📄 Paper](https://arxiv.org/abs/2510.18279)  
    > **Summary:** Investigates rendering long text into high-resolution images to be processed by vision encoders. This "visual text" approach can be more token-efficient than standard text embeddings for certain long-context tasks.

---
