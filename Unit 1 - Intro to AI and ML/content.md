# Unit 1: Introduction to Artificial Intelligence and Machine Learning

In this unit, you will learn about the fundamentals of artificial intelligence (AI) and machine learning (ML), including supervised and unsupervised learning, reinforcement learning, and common ML algorithms and their applications.

## 1.1 AI and ML Fundamentals

Artificial Intelligence (AI) is the field of computer science that aims to create machines capable of performing tasks that typically require human intelligence. AI systems can process large amounts of data, recognize patterns, make decisions, and improve their performance over time. Machine Learning (ML) is a subset of AI that focuses on enabling machines to learn from data and improve their performance over time. ML algorithms iteratively learn from the data and adapt their models to make accurate predictions or decisions.

### Key Concepts

- **Artificial Intelligence (AI):** The science of creating machines that can perform tasks typically requiring human intelligence. AI systems can process large amounts of data, recognize patterns, make decisions, and improve their performance over time.

- **Machine Learning (ML):** A subset of AI that focuses on enabling machines to learn from data and improve their performance over time. ML algorithms iteratively learn from the data and adapt their models to make accurate predictions or decisions.

### Learning Resources

- [Introduction to AI (Video)](https://www.youtube.com/watch?v=mJeNghZXtMo)
- [Artificial Intelligence: A Modern Approach (Textbook)](http://aima.cs.berkeley.edu/)
- [Machine Learning: An Introduction (Article)](https://towardsdatascience.com/machine-learning-an-introduction-23b84d51e6d0)

## 1.2 Supervised and Unsupervised Learning

Supervised learning is a type of ML where the model learns from labeled data, which includes both input features and the corresponding output labels. The goal is to learn a mapping from input features to output labels so that the model can make accurate predictions on new, unseen data. Examples of supervised learning tasks include classification (predicting discrete labels) and regression (predicting continuous numerical values).

Unsupervised learning, on the other hand, involves learning from unlabeled data, where the model must find patterns and relationships within the data on its own. The goal of unsupervised learning is to discover underlying structure or representations in the data. Examples of unsupervised learning tasks include clustering (grouping similar data points) and dimensionality reduction (reducing the number of features while preserving important information).

### Key Concepts

- **Supervised Learning:** Learning from labeled data, where the model must find the relationship between input features and output labels. Examples of supervised learning tasks include classification (predicting discrete labels) and regression (predicting continuous numerical values).

- **Unsupervised Learning:** Learning from unlabeled data, where the model must find patterns and relationships within the data on its own. Examples of unsupervised learning tasks include clustering (grouping similar data points) and dimensionality reduction (reducing the number of features while preserving important information).

### Learning Resources

- [Supervised vs Unsupervised Learning (Video)](https://www.youtube.com/watch?v=AXDByU3D1hA)
- [Supervised and Unsupervised Learning (Article)](https://towardsdatascience.com/supervised-vs-unsupervised-learning-14f68e32ea8d)

## 1.3 Reinforcement Learning

Reinforcement learning (RL) is another type of ML where an agent learns to make decisions by interacting with an environment. The agent takes actions in the environment and receives feedback in the form of rewards or penalties. The goal of the agent is to maximize the cumulative reward over time. Unlike supervised learning, the agent is not provided with explicit input-output pairs. Instead, it learns from trial and error, adjusting its behavior based on the feedback received.

Reinforcement learning can be modeled as a Markov Decision Process (MDP), which consists of a set of states, a set of actions, transition probabilities between states, and rewards associated with state-action pairs. The agent's objective is to learn a policy, a mapping from states to actions, that maximizes the expected cumulative reward.

### Key Concepts

- **Reinforcement Learning:** A type of ML where an agent learns to make decisions by interacting with an environment and receiving feedback in the form of rewards or penalties. The agent learns from trial and error, adjusting its behavior based on the feedback received.

- **Markov Decision Process (MDP):** A mathematical framework used to model reinforcement learning problems, consisting of a set of states, a set of actions, transition probabilities between states, and rewards associated with state-action pairs.

- **Policy:** A mapping from states to actions that the agent follows to make decisions in an MDP. The objective of reinforcement learning is to learn a policy that maximizes the expected cumulative reward.

### Learning Resources

- [Introduction to Reinforcement Learning (Video)](https://www.youtube.com/watch?v=2pWv7GOvuf0)
- [Deep Reinforcement Learning (Article)](https://lilianweng.github.io/lil-log/2018/02/19/a-long-peek-into-reinforcement-learning.html)

## 1.4 Common ML Algorithms and Applications

There are many ML algorithms available for different types of problems. Some common algorithms include linear regression, logistic regression, decision trees, support vector machines, and k-means clustering.

### Key Concepts

- **Linear Regression:** A regression algorithm used for predicting continuous numerical values. Linear regression assumes a linear relationship between the input features and the output value. It learns the model parameters (weights and bias) that minimize the mean squared error between the predicted values and the true values.

- **Logistic Regression:** A classification algorithm used for predicting binary outcomes. Logistic regression uses the logistic function (sigmoid) to model the probability of a binary output given the input features. It learns the model parameters (weights and bias) that maximize the likelihood of the observed data.

- **Decision Trees:** A hierarchical algorithm used for classification and regression tasks. Decision trees recursively partition the input space based on feature values, creating a tree-like structure where each leaf node represents a decision. The algorithm learns the optimal splits in the feature space that minimize the impurity (e.g., Gini index or entropy) in the resulting partitions.

- **Support Vector Machines:** A classification algorithm used for linearly separable data. Support vector machines find the optimal hyperplane that separates the data into different classes while maximizing the margin between the classes. The margin is defined as the distance between the hyperplane and the closest data points (called support vectors). The algorithm can be extended to non-linearly separable data using kernel functions.

- **k-Means Clustering:** An unsupervised learning algorithm used for partitioning data into k distinct clusters based on similarity. The algorithm initializes k centroids randomly and iteratively updates them by minimizing the sum of squared distances between each data point and its nearest centroid. The algorithm converges when the centroids no longer change significantly.

### Learning Resources

- [Linear Regression (Video)](https://www.youtube.com/watch?v=zPG4NjIkCjc)
- [Logistic Regression (Video)](https://www.youtube.com/watch?v=yIYKR4

