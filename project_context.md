# Context Brief: AI Agent Handoff

## User Profile & Preferences
* **Strict Language Rule:** The user will communicate in **English**. All project-related deliverables, explanations, code comments, and documentation MUST be generated in **French**.

## Project Overview
* **Topic:** Optimisation de la distribution d'énergie à l'aide de Graph Neural Networks (GNN) et machine learning.
* **Goal:** Compare the performance of classical tabular Machine Learning against Graph Neural Networks (GNN) for a smart grid distribution problem.
* **Problem Type:** Classification. Predicting if a network zone (node) is in a "Normal" (0) or "Congestionné/Critique" (1) state.

## Technical Stack & Modélisation
* **Language:** Python
* **Data Manipulation:** `pandas`, `numpy`
* **Classical ML Baseline:** `XGBoost`
* **Graph Neural Network:** `GCN` (Graph Convolutional Network) using `PyTorch Geometric` (`PyG`).
* **Graph Structure (Hybrid Approach):** * **Nodes:** 370 consumption points (clients).
  * **Edges:** Statistical similarity (e.g., Pearson correlation) between historical consumption profiles.

## Dataset Specifications
* **Dataset:** Electricity Load Diagrams 2011-2014 (UCI Machine Learning Repository).
* **Characteristics:** 370 clients (nodes), recorded every 15 minutes in kW.
* **Preprocessing Requirements:**
  * Re-sample data from 15-minute to 1-hour intervals to smooth out daylight saving time anomalies (March/October).
  * Handle clients with initial zero-values (clients created post-2011).
  * **Target Variable Creation:** Convert the regression dataset into a classification dataset. Calculate a capacity threshold (e.g., the 95th percentile of historical maximum consumption). If consumption exceeds this threshold, label the state as `1` (Congested); otherwise, `0` (Normal).
* **Data Splitting:** strictly chronological to prevent data leakage (e.g., Train: 2011-2013, Validation: 2014 H1, Test: 2014 H2). No random shuffling.

## Current Project Status
* **Completed:** The project formulation is locked. The initial "État d'Avancement" (Progress Report) has been written in LaTeX and submitted to the professor. 
* **Next Immediate Step:** Write the first Python script to load the CSV dataset (`pandas`), handle the semicolon separator (`;`), perform the time-series resampling, and generate the target classification labels.

## Instructions for the AI Agent
1. Read and acknowledge this context.
2. Ensure all technical outputs are in French.
3. Await the user's prompt to begin drafting the data preprocessing script.