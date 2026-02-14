# 🎮 Player Behavior Sequence Modeling

## 📌 Overview
This project models player behavior from sequential action logs using deep learning. Rather than treating players as static feature vectors,
we model ordered action sequences to learn behavioral representations using an **LSTM-based architecture**.

The core challenge was not just building a model that predicts actions, but creating an **embedding space** that truly captures a player's behavioral identity.

---

## 🔄 Project Evolution & Design Decisions
This project evolved through multiple structured phases, moving from initial sequence modeling to a refined supervised representation learning approach.

### 🟢 Phase 1 — Synthetic Data & Sequence Pipeline
**What We Built:**
* **Archetype Design:** Designed four behavior archetypes: *Aggressive, Explorer, Casual, and Quitter*.
* **Stochastic Logs:** Generated synthetic player data: `player_id | timestamp | action`.
* **Preprocessing:** Converted logs into padded integer sequences (Reserved `0` for padding) and created masks for LSTM training.

**Why This Mattered:**
Established a controlled, reproducible dataset to test if a model could identify structural differences in playstyles.

### 🔵 Phase 2 — LSTM Sequence Modeling (Self-Supervised)
**The Setup:**
* **Architecture:** Embedding Layer → LSTM Backbone → Next-action Prediction Head.
* **Initial Objective:** Predict the next action in a sequence.

> [!CAUTION]
> **The Insight:** While the model trained stably and loss decreased, the **Adjusted Rand Index (ARI) was ≈ 0.00**. 
> **Conclusion:** The model learned to predict the *next move*, but the embeddings did NOT strongly reflect behavioral identity.

### 🟡 Phase 3 — Objective Redesign (Critical Turning Point)
**Why We Changed Approach:**
Self-supervised next-action prediction does not guarantee behavioral separation. We redesigned the objective to force the embedding space to encode behavioral distinctions.

* **Step 3.1 — Dataset Upgrade:** Modified the generator to include `behavior_type` ground-truth labels.
* **Step 3.2 — Supervised Classification:** Introduced a new head: `LSTM Backbone → Linear Classifier → Behavior Type`.
* **Step 3.3 — The Alignment Debug:** Initially, ARI remained low (~0.02). I identified a
*  **DataLoader shuffling issue** that caused misalignment between embeddings and labels during validation. 

**Result after fix:** **Corrected ARI ≈ 0.53**.

---

## 📊 Results & Validation
The LSTM backbone now successfully captures structural differences in player behavior.

### 🧪 Evaluation Metrics
| Metric | Value |
| :--- | :--- |
| **Classification Accuracy** | ~75% |
| **Adjusted Rand Index (ARI)** | **~0.53** |

### 📈 Visualization
Embeddings reduced to 2D using **PCA**:
* **Learned Clusters (KMeans):** Shows how the model groups players.
* **True Behavior Types:** Shows the actual archetypes.

*The structural similarity between these plots confirms that the model has learned a meaningful behavioral representation.*

---

## 🧠 Key Design Lessons
1. **Objective Selection:** The training objective (Loss Function) determines the utility of the learned representation.
2. **Beyond Visuals:** Quantitative validation (ARI) is necessary; visual PCA plots can be misleading without metric-based grounding.
3. **Pipeline Integrity:** Evaluation alignment is as important as model architecture (The "Shuffle Bug" lesson).
4. **Structured Iteration:** This project reflects a research-oriented mindset—moving from hypothesis to failure, then to a redesigned solution.

---

## 🛠️ Tech Stack
* **Language:** Python
* **Deep Learning:** PyTorch
* **Mathematics & Stats:** NumPy, Scikit-learn
* **Visualization:** Matplotlib

---

## 🚀 How to Run
1. **Activate environment:**
   ```bash
   conda activate pytorch_dl
2. **Run notebooks in order:**

   01_data_exploration.ipynb

   02_sequence_building.ipynb

   03_lstm_model.ipynb

   04_behavior_visualization.ipynb

---
```
Player_Behavior_Sequence_Modeling
|
├── data/       # Synthetic player logs & .npy tensors
├── notebooks/  # Structured experimentation phases
├── src/        # Model definitions (LSTM Backbone + Heads)
└── results/    
```
---
##🏆 Final Takeaway
This project demonstrates end-to-end ML pipeline construction, behavioral sequence modeling, 
and the critical importance of representation validation. It marks a shift from simply "running a model" 
to performing quantitative ML experimentation.
