## AWS (Amazon Web Services)

Amazon Web Services (AWS) is a collection of cloud computing services that allow users to build, deploy, and manage applications and infrastructure on the cloud. AWS provides a wide range of services, including compute, storage, databases, machine learning, and analytics.

## Amazon SageMaker

Amazon SageMaker is a fully managed service within AWS that provides tools and resources to build, train, and deploy machine learning models. SageMaker makes it easier to develop, train, and deploy models by providing a platform to streamline these processes.

### SageMaker Notebooks

SageMaker Notebooks are JupyterLab-based environments for writing code, exploring data, and running machine learning experiments. Notebooks provide a convenient and interactive environment for developing machine learning models using familiar tools like Python, R, and TensorFlow.

### SageMaker Training Jobs

SageMaker Training Jobs are used to train machine learning models on SageMaker using the data you provide. You can configure a training job by specifying the type of machine learning algorithm, the input data, and the hyperparameters for the algorithm. SageMaker takes care of provisioning the necessary compute resources and managing the training process.

### SageMaker Endpoints

SageMaker Endpoints allow you to deploy your trained machine learning models for real-time inference. You can create an endpoint by specifying the trained model, the compute resources required, and the instance type. SageMaker provisions the necessary resources and hosts your model for you, making it accessible through an API.

## Knowledge needed to use AWS and SageMaker for programming

### Basic AWS concepts

Understanding the core AWS concepts like Regions, Availability Zones, and IAM Roles will help you navigate and use AWS services effectively.

#### Regions

- Core concept: A region is a geographically separate area where AWS data centers are located.
- Explanation: AWS regions consist of multiple Availability Zones and provide low latency, fault tolerance, and data redundancy.
- Pros: Better performance, fault tolerance, and compliance with data regulations.
- Cons: Services and resources might have different availability and pricing across regions.

#### Availability Zones

- Core concept: Availability Zones (AZs) are isolated locations within a region, each with independent power, cooling, and networking infrastructure.
- Explanation: AZs provide high availability and fault tolerance for AWS services by allowing users to distribute resources and applications across multiple AZs.
- Pros: Improved application performance, fault tolerance, and disaster recovery.
- Cons: Increased management complexity when distributing resources across AZs.

#### IAM Roles

- Core concept: Identity and Access Management (IAM) Roles are a way to grant permissions to AWS services and resources without using access keys.
- Explanation: IAM Roles can be assigned to AWS services or resources, allowing them to assume the role and gain specific permissions defined in the role's policy.
- Pros: Enhanced security, easier management of permissions, and support for temporary credentials.
- Cons: Additional complexity in managing and assigning roles.

### Python programming

Familiarity with Python programming, including basic syntax, functions, and libraries, is essential for using SageMaker Notebooks and writing machine learning code.

- Core concept: Python is a high-level, versatile programming language widely used in various domains, including machine learning and data science.
- Explanation: Python's simple syntax, rich ecosystem, and extensive library support make it ideal for developing machine learning applications, especially with SageMaker Notebooks.
- Pros: Easy to learn, extensive library support, and a large community.
- Cons: Slower execution speed compared to some other languages like C++ or Java.

### JupyterLab

Using JupyterLab (or Jupyter Notebooks) is helpful for working with SageMaker Notebooks and creating, editing, and running code cells interactively.

- Core concept: JupyterLab is an interactive web-based environment for developing, documenting, and executing code, often used for data exploration and machine learning.
- Explanation: JupyterLab provides a convenient interface for creating, editing, and running code cells, enabling users to iteratively develop and debug their code. It supports various programming languages, including Python, R, and Julia, through the use of kernels.
- Pros: Interactive development, support for multiple languages, and a rich ecosystem of extensions.
- Cons: Resource-intensive, limited support for real-time collaboration, and potential issues with version control.

### Machine learning concepts

Knowledge of machine learning concepts, such as training, validation, and testing datasets, model evaluation metrics, and hyperparameter tuning, will help you make the most of SageMaker's capabilities.

### Deep learning frameworks

Familiarity with deep learning frameworks like TensorFlow, PyTorch, or MXNet is beneficial for using SageMaker's built-in algorithms or developing custom models.

By gaining knowledge in these areas and exploring AWS and SageMaker documentation, you will be better equipped to use SageMaker for your machine learning projects.
