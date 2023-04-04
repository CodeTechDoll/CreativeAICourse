## IAM Roles for Amazon SageMaker

### Core concept: IAM roles

- An IAM (Identity and Access Management) role is a set of permissions that define what actions are allowed and denied by an AWS service or user.
- IAM roles help manage access to AWS resources securely and follow the principle of least privilege, which means granting only the permissions necessary to accomplish tasks.

### Step 1: Create an IAM role for SageMaker

- Implementation:
  1. Open the IAM console in the AWS Management Console.
  2. In the left navigation pane, click "Roles."
  3. Click "Create role" button.
- Why: To create a new role that will grant necessary permissions for SageMaker to access other AWS services.
- Pros: Allows fine-grained control over access to AWS resources.
- Cons: Requires an understanding of AWS concepts and permissions.

### Step 2: Choose a trusted entity

- Implementation:
  1. In the "Create role" page, under "Select type of trusted entity," choose "AWS service."
  2. In the "Choose a use case" section, find and click on "SageMaker."
  3. Click "Next: Permissions" button.
- Why: Specifies that the role will be used by SageMaker service.
- Pros: Ensures that only SageMaker service can assume the role.
- Cons: None.

### Step 3: Attach policies to the IAM role

- Implementation:
  1. On the "Attach permissions policies" page, search for "AmazonS3FullAccess" and select the checkbox next to it.
  2. Optionally, you can search for and attach other policies depending on your project's requirements (e.g., access to other AWS services).
  3. Click "Next: Tags" button.
- Why: Grants SageMaker the necessary permissions to access your S3 buckets for reading and writing data.
- Pros: Ensures that SageMaker can read and write data to and from S3 for model training and deployment.
- Cons: May grant overly broad permissions if not carefully reviewed.

### Step 4: Add tags (optional)

- Implementation:
  1. Optionally, you can add key-value pairs to tag your role for better organization and tracking.
  2. Click "Next: Review" button.
- Why: Tags help in organizing and managing AWS resources.
- Pros: Improves manageability and tracking of resources.
- Cons: None.

### Step 5: Review and create the IAM role

- Implementation:
  1. On the "Review" page, enter a unique name for the IAM role.
  2. Review the role's permissions to ensure it has the necessary access.
  3. Click "Create role" button.
- Why: Finalizes the creation of the IAM role with the specified permissions.
- Pros: Creates a role that can be used by SageMaker to access necessary AWS resources.
- Cons: None.

Once you have created the IAM role, you can use it when setting up your Amazon SageMaker notebook instance.