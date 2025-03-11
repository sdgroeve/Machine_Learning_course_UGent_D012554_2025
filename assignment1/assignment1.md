## Project Assignment: Donor Splice Site Prediction

**Background and Introduction to Gene Splicing**

Gene splicing is a crucial biological process in which introns (non-coding regions) are removed from pre-mRNA transcripts, and exons (coding regions) are joined together to form mature messenger RNA (mRNA). The accuracy of splicing is essential for producing functional proteins. In eukaryotes, the spliceosome machinery recognizes specific sequences at the exon-intron boundaries, one of which is the donor splice site, typically marked by the nucleotides "GT" at the 5' end of the intron. Predicting these sites accurately from DNA sequence data is important for understanding gene regulation and for applications in genetic research and biotechnology.

**Competition Overview**

In this Kaggle competition, you will develop Machine Learning models to predict donor splice sites from DNA sequence data. Each data point consists of a DNA segment with 200 base pairs (bp) upstream and 200 bp downstream from the canonical "GT" donor site.

**Part 1: Baseline Model**

-   **Objective:** Build a baseline model using a restricted context.
-   **Data Context:** Instead of using the full 400 bp window, limit your input sequence to 6 bp upstream and 6 bp downstream of the "GT" donor site (total 12 bp).
-   **Feature Engineering:** Use one-hot encoding for the nucleotide sequence.
-   **Modeling:** Fit a simple machine learning model (e.g., logistic regression, decision tree, or any baseline classifier) on the one-hot encoded features.
-   **Deliverables:**

-   An executable Jupyter notebook containing your code.
-   A Kaggle submission file containing your predictions.

**Part 2: Advanced Modeling**

-   Enhance your model with extensive feature engineering and model optimization.
-   Experiment with different sequence representations (e.g., k-mer embeddings, physicochemical properties, or other custom features).
-   Try advanced machine learning models (e.g., ensemble methods, neural networks) and optimize hyperparameters.
-   Analyze the importance of features and the performance of your model.

-   **Final Deliverable:** A report summarizing your data analysis and model insights.

**Report Requirements**

-   **Content:** The final report should provide an in-depth analysis of your approach and findings. It should not include any code.
-   **Visuals:** The report should contain no more than one figure and one table that best illustrate your key results or methodology.
-   **Length:** Concise and focused.

----------

**Structure for the** **report:**

1.  **Title**

-   A concise and descriptive title of your project.

2.  **Abstract** (max 150 words)

-   **Purpose:** Briefly describe the objective of the study.
-   **Methods:** Summarize the approach and key methods used.
-   **Results:** Highlight the main findings.
-   **Conclusion:** State the significance of the results.

3.  **Materials and Methods**

-   **Data Description:** Describe the dataset used (e.g., the sequence context around the "GT" site).
-   **Feature Engineering:** Explain the one-hot encoding and any additional features used.
-   **Modeling Approach:** Summarize the baseline model and any advanced techniques applied.
-   **Evaluation Metrics:** List the metrics used to assess model performance.

4.  **Results**

-   **Main Findings:** Summarize key results from your analysis.
-   **Figure:** Include one figure (e.g., ROC curve, feature importance, or a comparison of model performances).
-   **Table:** Provide one table summarizing key performance metrics or model comparisons.

5.  **Discussion**

-   **Interpretation:** Discuss what your results imply about donor splice site prediction.
-   **Strengths and Limitations:** Reflect on the advantages of your approach and potential limitations.
-   **Future Work:** Suggest avenues for further research or model improvements.

## The full report should be no more than 1500 words.
