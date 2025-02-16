# Computational Intelligence Projects

Welcome to the **Computational Intelligence Projects** repository! This collection showcases various projects developed as part of the Computational AI university course, each focusing on different aspects of computational intelligence.

## Table of Contents

- [Project Overview](#project-overview)
- [Project Details](#project-details)
- [Installation and Usage](#installation-and-usage)
- [License](#license)

## Project Overview

This repository contains multiple projects that explore various computational intelligence techniques, including neural networks and genetic algorithms. Each project is organized into its own directory with detailed documentation and source code.

## Project Details

### Project 1: [Genetic]

**Description:**  
This project provides a solution to the optimal route problem for a package delivery company using the Genetic Algorithm. It also simulates a dynamic city environment, including factors like changing traffic conditions.

**Features:**
- Fully object-oriented implementation, including Car and Package data classes, as well as Distance Matrix and Traffic System classes.
- Dynamically reassigns packages mid-transportation if a more optimal route is available.
- Implements multiple mutation methods.
- Uses Tournament Selection and Single-Point Crossover.
- Applies penalties for general delays, priority package delays, and exceeding capacity limits.

**Directory:** `[./Genetic]`(./Genetic)

## Installation and Usage

To run any of these projects locally:

1.**Clone the repository:**
   ```bash
   git clone https://github.com/abolfazmz81/Computational-intelligence.git
   cd Computational-intelligence
   ```

2.**Navigate to the project directory:**
  ```bash
  cd [ProjectDirectory]
  ```
Replace [ProjectDirectory] with the specific project folder.

3.**Install dependencies:** 
Ensure you have Python installed. Install the required packages using:
  ```bash
  pip install -r requirements.txt
  ```
Each project directory contains its own requirements.txt file with the necessary dependencies(except knn).

4.**Run the project:** 
Execute the main script for the project:
  ```bash
  python main.py
  ```

## License
This repository is licensed under the MIT License. See the [LICENSE](./LICENSE) file for more information.

