# Dockerize-Model

## Project Overview

This project focuses on building a deep learning model for classifying fresh and rotten fruits, including peaches, pomegranates, and strawberries. The project is structured to facilitate easy deployment using Docker.

## Project Structure

```
Dockerize-Model
├── LICENSE
├── README.md
├── requirements.txt
├── setup.py
├── config/
│   └── config.yml
├── data/
│   ├── processed/
│   └── raw/
│       └── images/
├── src/
│   ├── __init__.py
│   ├── constants/
│   │   ├── __init__.py
│   │   └── constants.py
├── template/
│   ├── __init__.py
│   ├── main.py
│   ├── project_setup.py
│   └── project_structure.py
```

## Getting Started

### Prerequisites

- Docker installed on your system
- Python 3.12 or later

### Installation

1. Clone the repository:

   ```bash
   git clone <repository-url>
   cd Dockerize-Model
   ```

2. Install the required Python packages:

   ```bash
   pip install -r requirements.txt
   ```

### Configuration

Update the `config/config.yml` file to set the paths for you.

### Running the Project

1. Build the Docker image:

   ```bash
   docker build -t dockerize-model .
   ```

2. Run the Docker container:

   ```bash
   docker run -it dockerize-model
   ```

## License

This project is licensed under the terms of the LICENSE file.
