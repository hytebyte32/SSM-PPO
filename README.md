# Architecture Overview

The **SSM-PPO** model implements a custom recurrent neural network block inspired by **state space models (SSMs)** for reinforcement learning. The architecture is designed to capture **temporal dependencies** and **complex dynamics** in sequential data.

---

## Key Components

| Component                | Description |
|--------------------------|-------------|
| **ShiftedSigmoid**       | A learnable sigmoid activation with trainable smoothness, used for gating mechanisms. |
| **RecurrentBlock**       | The core module that manages the recurrent state and transitions. |
| **State Transition Matrix** | Constructed from learnable magnitude and phase parameters, forming a complex-valued matrix for hidden state evolution. |
| **Hidden State Initialization** | Learnable parameters for the initial hidden state, also complex-valued. |
| **Discretization Matrix**| Controls the time discretization for state updates. |
| **Input/Output Filters** | Linear layers that project embeddings into the state space and back. |
| **Output Processing**    | Combines magnitude and phase features from the complex output, embeds them, and compresses back to the embedding size. |
| **State Prediction**     | Predicts the next input from the processed output. |
| **RecurrentNetwork**     | Wraps the RecurrentBlock and manages the forward pass over sequences. |

---

## Checkpointing

Uses **PyTorch checkpointing** for memory efficiency.

---

## Forward Pass

Processes inputs sequentially, updating the hidden state and producing outputs at each step.

---

## Prediction

Maps the processed output to the next-step prediction.

---

## Data Flow

1. **Input Embedding**  
   Raw inputs are embedded into a complex-valued space.

2. **Gating & Discretization**  
   Inputs are gated and modulated by learnable gates and discretization matrices.

3. **State Update**  
   The hidden state is updated using the complex state transition matrix and input filters.

4. **Output Generation**  
   The output is filtered, processed for magnitude and phase, and compressed.

5. **Prediction**  
   The processed output is used to predict the next input or action.

---

## Notable Features

- All core matrices (state transition, hidden state, gates) are **complex-valued and learnable**.
- Gating and discretization are controlled by **custom, learnable activations**.
- Output processing converts complex values to real by explicitly seperating and embedding **magnitude and phase** information.
- Designed for both **full sequence** and **online processing**. Supports **batch operations** and **checkpointing**.
