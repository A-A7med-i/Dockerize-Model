# Dockerize-Model

## Project Overview

This project is a deep learning solution for classifying fresh and rotten fruits (peaches, pomegranates, strawberries). It is designed for end-to-end reproducibility, modularity, and easy deployment using Docker.

## Features

- Image classification for multiple fruit types (fresh/rotten)
- Modular codebase (data, models, API, utilities)
- REST API with FastAPI for model inference
- Dockerized for consistent deployment
- Logging

## Project Structure

```
Dockerize-Model
├── LICENSE
├── README.md
├── requirements.txt
├── setup.py
├── data/
│   ├── processed/
│   └── raw/
│       └── images/
├── logs/
│   └── logging.log
├── models/
│   └── checkpoints/
├── notebooks/
│   ├── EDA.ipynb
│   └── experiments.ipynb
├── src/
│   ├── __init__.py
│   ├── api/
│   │   ├── main.py
│   │   ├── endpoints.py
│   │   ├── process.py
│   │   └── schemas.py
│   ├── constants/
│   │   ├── __init__.py
│   │   └── constants.py
│   ├── data/
│   │   └── implement_data.py
│   ├── models/
│   │   └── model.py
│   ├── processing/
│   │   └── processor.py
│   ├── utils/
│   │   └── helper.py
│   └── visualization/
│       └── plot.py
├── template/
│   ├── __init__.py
│   ├── main.py
│   ├── project_setup.py
│   └── project_structure.py
├── Dockerfile
```

## Getting Started

### Prerequisites

- Docker (recommended for deployment)
- Python 3.12 or later (for local development)

### Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/A-A7med-i/Dockerize-Model.git
   cd Dockerize-Model
   ```

2. Install Python dependencies (for local development):

   ```bash
   pip install -r requirements.txt
   ```

### Configuration

- Edit `src/constant/constant.py` to set data/model paths and parameters as needed.

### Running the Project

#### Using Docker (Recommended)

1. Build the Docker image:

   ```bash
   docker build -t dockerize-model .
   ```

2. Run the Docker container:

   ```bash
   docker run -it -p 5000:5000 dockerize-model
   ```

   The API will be available at `http://localhost:5000`.

#### Local Development

1. Start the FastAPI server:

   ```bash
   python src/api/main.py
   ```

2. Access the API docs at `http://localhost:5000/docs`

## API Usage

- The FastAPI server exposes endpoints for image prediction.

## Notebooks

- Use the `notebooks/` directory for EDA and experiments.

## License

This repository is licensed under the [MIT License](LICENSE).
