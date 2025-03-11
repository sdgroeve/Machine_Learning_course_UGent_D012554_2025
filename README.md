# Machine & Deep Learning Course Ghent Univeristy (D012554)

Course materials for the course Machine Learning Methods for Biomedical Data (D012554)[https://studiekiezer.ugent.be/2023/studiefiche/en/D012554].
This course is taught to Biomedical Sciences students at the Department of Biomolecular Medicine, Faculty of Medicine and Health Sciences, Ghent University.

The following topics are included:

*   **Machine Learning**: Introduces the fundamental concept where computer programs learn from experience to improve their performance on specific tasks.
*   **Classification**: Covers methods for categorizing data into predefined classes based on their features.
*   **Decision tree**: Explores a supervised learning algorithm that uses a tree-like structure to model decisions and their potential outcomes for classification and regression.
*   **Model evaluation**: Discusses various techniques to assess the performance and generalization ability of machine learning models.
*   **Data normalization**: Explains the process of scaling data to a standard range to prevent features with larger values from dominating model training.
*   **Data leakage**: Highlights the critical issue where information from the test set unintentionally influences the training of a machine learning model.
*   **Data preprocessing**: Encompasses the essential steps involved in cleaning and preparing raw data to make it suitable for machine learning algorithms.
*   **Regression**: Deals with modeling the relationship between a dependent variable and one or more independent variables.
    *   **Linear regression**: Focuses on modeling this relationship using a linear equation.
    *   **Non-linear regression**: Explores techniques to model relationships between variables that are not linear, including polynomial transformations.
*   **Logistic regression**: Introduces a classification algorithm that models the probability of a binary or multi-class outcome.
*   **Model regularization**: Describes methods used to prevent overfitting by adding a penalty to the model's loss function, thereby discouraging overly complex models.
*   **Support vector machine**: Covers a powerful classification algorithm that aims to find the optimal hyperplane that best separates different classes of data.
*   **Ensemble methods**: Discusses techniques that combine the predictions of multiple individual models to improve overall performance and robustness.
*   **Bias and variance**: Explains the two primary sources of error in supervised learning models and the important trade-off between them.
*   **Bagging and boosting**: Introduces two major categories of ensemble learning methods used to reduce variance (bagging) and bias (boosting).
    *   **Bagging: Random Forest**: Details a specific bagging technique that uses an ensemble of decision trees trained on different subsets of the data.
    *   **Boosting**: Explores a family of ensemble methods that sequentially train models, with each new model attempting to correct the errors made by the previous ones.
*   **Deep neural networks**: Introduces neural networks with multiple layers that can learn complex patterns from large amounts of data.
*   **GPU vs CPU**: Compares the architectures and suitability of Graphical Processing Units and Central Processing Units for deep learning tasks, highlighting the parallel processing capabilities of GPUs.
*   **Neural network**: Explains the basic building block of deep learning models, consisting of interconnected neurons that perform weighted sums and apply activation functions.
*   **Non-linearity: activation function**: Describes the role of activation functions in neural networks to introduce non-linearities, enabling the modeling of complex relationships.
*   **Neural network training**: Outlines the process of adjusting the weights and biases of a neural network using forward propagation, loss calculation, backpropagation, and weight updates to minimize the error between predictions and actual targets.
*   **Loss function**: Defines the function used to quantify the error made by a machine learning model, guiding the optimization process during training.
*   **Stochastic gradient descent (SGD)**: Introduces an optimization algorithm that updates model parameters based on the gradient computed from a single random data point, offering computational efficiency and the potential to escape local minima.
*   **Mini batch gradient descent**: Explains a compromise between gradient descent and SGD, where the gradient is computed on small subsets of the data (batches).
*   **Momentum in SGD**: Describes a technique that accelerates SGD by accumulating past gradients to maintain direction and overcome local minima.
*   **Adaptive Learning Rates**: Introduces optimization methods like AdaGrad and Adam that adjust the learning rate for each parameter based on historical gradient information.
*   **Image data**: Introduces the concept of digital images and their representation as numerical data for computer vision tasks.
*   **Computer vision**: Covers the field of enabling computers to "see" and interpret information from digital images and videos for various applications.
*   **What computers see**: Explores how computers process image data, often by treating individual pixels or learned features as input.
*   **Invariances**: Discusses the ability of computer vision models to recognize objects despite variations in their appearance, such as changes in position, scale, or lighting.
*   **Convolutional filter**: Explains the use of small matrices that slide over images to detect local patterns and features.
*   **Convolutional neural networks (CNN)**: Introduces a specialized type of neural network designed for processing grid-like data such as images, using convolutional layers and pooling layers to learn hierarchical features and achieve translation invariance.
*   **Feature map**: Describes the output of a convolutional layer, representing the presence and strength of detected features across the input image.
*   **Pooling**: Explains a downsampling technique used in CNNs to reduce the spatial dimensions of feature maps, contributing to location invariance.
*   **Invariances: data augmentation**: Discusses a regularization technique where artificial training data is generated by applying transformations to existing images (e.g., rotation, scaling) to improve the model's robustness to variations.
*   **ImageNet**: Introduces a large-scale visual database that has played a crucial role in advancing deep learning for computer vision.
*   **ResNet (Residual Network)**: Explains a deep CNN architecture that uses skip connections to address the vanishing gradient problem and enable the training of very deep and effective networks.
*   **Image segmentation: U-Net**: Describes a CNN architecture specifically designed for biomedical image segmentation, using an encoder-decoder structure with skip connections for precise localization.
*   **Object recognition: YOLO (You Only Look Once)**: Introduces a real-time object detection system that predicts bounding boxes and class probabilities in a single forward pass.
*   **Common Objects in Context dataset (COCO)**: Presents a large-scale dataset used for object detection, segmentation, and captioning, serving as a benchmark for computer vision models.
*   **Activation maximization**: Explains a technique to visualize what CNNs have learned by generating input images that maximally activate specific neurons or filters.
*   **Transfer learning**: Discusses a machine learning technique where a model pre-trained on a large dataset is adapted to a new, related task with limited data.
*   **Sequence data**: Introduces the concept of ordered data where the position of elements is important, such as text, speech, or time series.
*   **Sequence modeling**: Covers deep learning approaches designed to learn patterns and dependencies within sequential data.
*   **Recurrent neural network (RNN)**: Explains a type of neural network designed to process sequential data by maintaining an internal hidden state that captures information about past elements in the sequence.
*   **RNN encoding - decoding**: Describes an architecture where an RNN first encodes an input sequence into a fixed-length vector, and another RNN (the decoder) then generates an output sequence based on this encoding.
*   **Learning to pay attention**: Introduces the concept of attention mechanisms, which allow sequence models to weigh the importance of different parts of the input sequence when producing an output.
*   **(Self-)attention**: Explains a specific type of attention mechanism where different positions within the same sequence attend to each other.
*   **Transformer architecture**: Introduces a neural network architecture based entirely on self-attention mechanisms, known for its parallel processing capabilities and effectiveness in capturing long-range dependencies.
*   **Sequence tokenization**: Describes the process of converting text or other sequential data into a sequence of tokens that can be processed by neural networks.
*   **BERT (Bidirectional Encoder Representations from Transformers)**: Introduces a transformer-based model pre-trained on a large corpus of text, designed to produce rich contextual embeddings for words by considering both left and right context.
*   **BERT pre-training**: Outlines the unsupervised tasks used to train BERT, including masked language modeling and next sentence prediction.
*   **BERT Flavours**: Presents different variations of the original BERT model, often fine-tuned or pre-trained on specific types of data (e.g., biomedical text).
*   **GPT (Generative Pre-trained Transformer)**: Introduces a decoder-only transformer model focused on autoregressive text generation.
*   **Input layer: token embeddings**: Explains the initial step in transformer models where tokens are converted into dense vector representations.
*   **Positional encoding**: Describes a technique used in transformers to provide information about the order of tokens in a sequence, as the self-attention mechanism itself is order-agnostic.
*   **Attention head**: Details the core component of the transformer architecture that calculates attention scores to determine the relevance of different tokens in a sequence.
*   **MLP (Feedforward) Layers in the Transformer**: Explains the role of multi-layer perceptron layers applied independently to each token within each transformer layer.
*   **Data: Common Crawl**: Introduces a large, open repository of web crawl data used for training various natural language processing models.
*   **Chat bots**: Mentions the application of these sequence models in conversational AI.
*   **Superposition**: Describes a phenomenon in neural networks where multiple features can be represented in overlapping directions within the weight space.
*   **Few-shot learning**: Briefly touches upon the ability of some large language models to perform tasks with very few examples.
*   **DNA/protein LLMs**: Introduces the application of transformer models to biological sequences like DNA and proteins.
*   **E.g. ESM2 (Evolutionary Scale Modeling v2)**: Provides an example of a transformer-based protein language model.
*   **Vision Transformer (ViT)**: Explains how the transformer architecture can be applied to image data by treating image patches as a sequence of tokens.
*   **Retrieval-Augmented Generation (RAG)**: Introduces a technique that enhances large language models by allowing them to retrieve information from external knowledge sources before generating a response.
*   **Autoencoder**: Describes an unsupervised neural network designed to learn efficient data representations by encoding input into a lower-dimensional space and then decoding it back to the original form.
*   **Variational autoencoder (VAE)**: Introduces a probabilistic version of the autoencoder that learns a distribution over the latent space, enabling generative capabilities and smoother interpolations.
*   **VAE: applications**: Highlights various uses of VAEs, particularly in biomedical data analysis.
*   **VAE: drug discovery**: Mentions the application of VAEs in representing and searching chemical spaces for drug development.
*   **Generative adversarial network (GAN)**: Explains a generative model consisting of two networks, a generator and a discriminator, trained in an adversarial manner to produce realistic synthetic data.
*   **Conditional GAN (cGAN)**: Introduces an extension of GANs that allows for the generation of data conditioned on additional input information.
*   **Diffusion model**: Describes a generative model that learns to reverse a gradual noising process to generate high-quality data from random noise.
*   **Contrastive Language-Image Pretraining (CLIP)**: Explains a model trained to understand the relationship between images and text by learning a shared embedding space.
*   **Data: LAION datasets**: Introduces large-scale open datasets of image-text pairs used for training multimodal models like CLIP and diffusion models.
*   **Zero-shot image classification with CLIP**: Describes the ability of CLIP to classify images into categories not seen during training by leveraging its understanding of semantic relationships between images and text.
*   **Diffusion models: RFdiffusion**: Provides an example of a diffusion model applied to the de novo design of protein structures.

