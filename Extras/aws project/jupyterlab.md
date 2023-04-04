### Step-by-Step Guide to JupyterLab

1. **Launch JupyterLab**
   - Core concept: Start a JupyterLab session in your web browser to access the interactive development environment.
   - Example: In AWS SageMaker, click "Open JupyterLab" on your notebook instance to launch JupyterLab.

2. **Create a new notebook**
   - Core concept: Notebooks are interactive documents containing live code, equations, visualizations, and narrative text.
   - Example: Click the "+" icon in the JupyterLab Launcher, and then select the desired kernel (e.g., Python 3) to create a new notebook.

3. **Understand code cells and markdown cells**
   - Core concept: Notebooks consist of cells, which can be either code cells (for writing and executing code) or markdown cells (for adding formatted text, images, or equations).
   - Example: To add a code cell, click the "+" icon in the notebook toolbar. To add a markdown cell, click the dropdown menu in the toolbar (which defaults to "Code") and change it to "Markdown."

4. **Write and run code**
   - Core concept: In code cells, you can write and execute code interactively.
   - Example: Type `print("Hello, World!")` in a code cell and press `Shift + Enter` to run the cell and display the output.

5. **Edit markdown cells**
   - Core concept: Use markdown cells to add narrative text, images, equations, or other non-code content.
   - Example: In a markdown cell, type `# This is a header` and press `Shift + Enter` to render the formatted header.

6. **Save and checkpoint your notebook**
   - Core concept: Regularly save your work and create checkpoints to track different versions of your notebook.
   - Example: Click the floppy disk icon in the toolbar or press `Ctrl + S` to save your notebook. To create a checkpoint, go to the "File" menu and click "Save Notebook with New Checkpoint."

7. **Manage files and folders**
   - Core concept: Use the JupyterLab file browser to organize and manage your notebooks, scripts, and other files.
   - Example: In the file browser (left sidebar), right-click to create new folders, rename files, or move items to the trash.

8. **Configure Git integration**
   - Core concept: Integrate JupyterLab with your Git repository to track changes and collaborate on your projects.
   - Example: You mentioned that your sandbox is already connected to a Git repo called "sagemaker-sandbox." Use the built-in Git extension in JupyterLab to commit changes, create branches, and sync your work with the remote repository.

9. **Install extensions**
   - Core concept: Customize your JupyterLab environment with various extensions that add new features or improve existing functionality.
   - Example: To install the `jupyterlab-lsp` extension for code autocompletion, open a terminal in JupyterLab and run the following commands:

     ```
     pip install jupyterlab-lsp
     jupyter labextension install @krassowski/jupyterlab-lsp
     ```

   - Restart JupyterLab for the changes to take effect.

## Kernels in JupyterLab

A kernel is a computing engine that executes the code contained in a Jupyter Notebook. In JupyterLab, kernels are responsible for running code cells in your notebook, console, or other interactive environments. When you create a new notebook or console, you need to choose a kernel that matches the programming language you want to use.

In the JupyterLab Launcher, you'll see the following categories:

1. **Notebook**
   - Notebooks are interactive documents that contain code cells, markdown cells, and output cells. They allow you to write, edit, and run code interactively while also providing narrative context and visualizations.
   - When you create a new notebook, you need to choose a kernel. This determines the programming language and environment for your code cells. Examples of kernels include Python 3, R, and Julia.

2. **Console**
   - Consoles are interactive command-line interfaces for kernels. They are similar to IPython or the standard Python REPL (Read-Eval-Print Loop). You can write and execute code line by line, making consoles useful for quick experimentation or debugging.
   - Like notebooks, consoles require a kernel to execute code. Choose a kernel that matches the programming language you want to use.

3. **Other**
   - This category includes various other tools and utilities available in JupyterLab. Some examples are:
     - Terminal: Provides access to a command-line interface within JupyterLab. Useful for managing files, installing packages, or running shell commands.
     - Text File: Opens a simple text editor for creating or editing plain text files.
     - Markdown File: Opens a markdown editor for creating or editing markdown files.

When choosing a kernel, remember that it determines the programming language and environment for your notebook or console. For example, if you select the Python 3 kernel, you can write and execute Python code in your code cells. Kernels also manage the state of your code execution, such as variable values and imported libraries, ensuring that your notebook or console session maintains a consistent state throughout its lifetime.

