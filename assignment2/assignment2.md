# Deep Learning in Biomedical Sciences: CNN Training on MedMNIST PathMNIST64

### Overview

In this assignment, you will explore the computational challenges of training convolutional neural networks (CNNs) on the **MedMNIST PathMNIST64** dataset. You will implement and compare two main approaches:

-   **Training a CNN from scratch**
-   **Fine-tuning a pre-trained CNN**

In addition, you will assess the impact of data augmentation on model performance. 

> **Note:** Starter notebook on Kaggle: https://www.kaggle.com/code/svendegroeve/d012554a-pathmnist-assignment

----------

### Assignment Tasks

#### 1. CNN Training from Scratch

-   **Design a Custom CNN:** Develop a CNN architecture specifically for PathMNIST64. In your notebook, explain the choices behind your network design (e.g., number of layers, filter sizes, activation functions).
-   **Training and Evaluation:** Train your custom CNN on the dataset. Record performance metrics such as accuracy, precision, recall, and F1-score along with the training time. Include visualizations like loss and accuracy curves.

#### 2. Fine-Tuning a Pre-Trained CNN

-   **Select a Pre-Trained Model:** Choose a popular model (e.g., ResNet, VGG, MobileNet) that has been pre-trained on a large dataset such as ImageNet.
-   **Adapt the Model:** Modify the final layers to match the number of classes in the PathMNIST64 dataset.
-   **Fine-Tuning Process:** Fine-tune the network on the dataset. Monitor and compare training times and performance metrics with your custom CNN.

#### 3. Data Augmentation Experiment

-   **Implement Data Augmentation:** Integrate common augmentation techniques (e.g., rotations, flips, zooms, shifts) into your training pipeline.
-   **Evaluation:** Evaluate how these augmentations impact the performance of both your custom CNN and the fine-tuned model. Compare metrics and training behaviors with and without augmentation.

----------

### Deliverable: Kaggle Notebook with an Embedded Report

You are required to submit one comprehensive Kaggle Notebook that includes both your code and a brief report. The report should be written using markdown cells and organized into **three sections only**: **Methods, Results, and Discussion**. Use the following notebook structure as a guideline:

#### **Notebook Structure**

1.  **Methods**
    
    -   **Custom CNN Training**
        -   **Architecture Description:** Detail your custom CNN design, including key layers and hyperparameters.
        -   **Training Procedure:** Outline the training settings (e.g., learning rate, batch size, number of epochs) along with relevant code snippets.
    -   **Fine-Tuning a Pre-Trained Model**
        -   **Model Selection and Adaptation:** Explain which pre-trained model you chose and how you modified it for the PathMNIST64 dataset.
        -   **Training Details:** Provide details on the fine-tuning process and training settings.
    -   **Data Augmentation**
        -   **Techniques Implemented:** Describe the augmentation methods you applied.
        -   **Integration:** Show how these augmentations were incorporated into your training pipeline with examples from your code.
2.  **Results**
    
    -   **Performance Metrics:** Present the evaluation metrics (accuracy, precision, recall, F1-score) and training times for both the custom CNN and the fine-tuned model.
    -   **Visualizations:** Include charts such as loss curves, accuracy plots, and any tables or graphs that compare the outcomes with and without data augmentation.
3.  **Discussion**
    
    -   **Analysis:** Compare the performance and training times between the two approaches.
    -   **Data Augmentation Impact:** Discuss the effect of data augmentation on model performance and any trade-offs observed.
    -   **Observations:** Provide insights on the challenges of training on a Kaggle NVidia K80 GPU and suggestions for potential improvements.

----------

### Submission Guidelines

-   **Submission Format:** Share your Kaggle Notebook with your instructor. Ensure the notebook runs from start to finish on Kaggle without errors.
-   **Evaluation Criteria:**
    -   Clear and modular code.
    -   The thoroughness and clarity of the embedded report (Methods, Results, and Discussion).
    -   Depth of analysis when comparing training from scratch versus fine-tuning and the role of data augmentation.
    -   Effective use of visualizations to support your findings.
    -   Overall presentation and reproducibility of results.
