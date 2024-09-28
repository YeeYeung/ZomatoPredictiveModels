# Zomato Predictive Models

This repository contains the source code, dependencies, and instructions for the Zomato predictive modeling project. The project involves predicting restaurant ratings and binary classification of restaurant types based on various features.

## Project Structure

- **Part_A_Jupyter.ipynb**: Jupyter notebook containing data cleaning, exploratory data analysis, and feature engineering steps.
- **Part_B_Jupyter.ipynb**: Jupyter notebook containing predictive modeling tasks, including regression and classification.
- **Dockerfile**: Configuration file to build the Docker image.
- **requirements.txt**: List of dependencies required for the project.
  
## Setup Instructions

### 1. Clone the Repository

To get a copy of the project up and running on your local machine, clone this GitHub repository:

```bash
git clone https://github.com/YeeYeung/ZomatoPredictiveModels.git
cd ZomatoPredictiveModels
```

### 2. Install Dependencies

Make sure you have Python 3.12 or later installed. Install the necessary dependencies using the following command:

```bash
pip install -r requirements.txt
```

### 3. Running the Jupyter Notebooks

To run the notebooks (`Part_A_Jupyter.ipynb` and `Part_B_Jupyter.ipynb`), start JupyterLab or Jupyter Notebook:

```bash
jupyter lab
```

or

```bash
jupyter notebook
```

Navigate to the notebooks and run the cells to reproduce the analysis and model results.

### 4. Running with Docker

You can also run the project using Docker, which will set up all the necessary dependencies in a container. Follow the steps below:

1. **Build the Docker Image**

   In the project directory, build the Docker image using the provided `Dockerfile`:

   ```bash
   docker build -t yeeyeung/zomato_predictive_model .
   ```

2. **Run the Docker Container**

   Once the image is built, you can run the container and access the Jupyter notebooks:

   ```bash
   docker run -it -p 8888:8888 yeeyeung/zomato_predictive_model
   ```

   Open your browser and go to the URL provided in the terminal (e.g., `http://127.0.0.1:8888`), then you can interact with the notebooks.

### 5. Docker Image on Docker Hub

The Docker image for this project is available on Docker Hub. You can pull the image and run it directly without building it locally:

```bash
docker pull yeeyeung/zomato_predictive_model
docker run -it -p 8888:8888 yeeyeung/zomato_predictive_model
```

[Link to Docker Hub Image](https://hub.docker.com/r/yeeyeung/zomato_predictive_model)

### 6. GitHub Deployment

This repository is continuously updated. You can contribute by following the Git flow:

```bash
# Add changes to the repository
git add .

# Commit changes
git commit -m "Your commit message"

# Push to GitHub
git push origin master
```
