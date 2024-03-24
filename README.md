# Speaker2Vec

## Overview

This code presents "Speaker2Vec", a Natural Language Processing (NLP) methodology for vectorizing speaker features in online forums. The project addresses the gap in user representation in digital discourse analysis.
This approach, relevant to our NLP course, demonstrates the practical application of course concepts in analyzing and understanding complex communication patterns online, showcasing the potential of combining diverse NLP techniques to enhance discourse analysis.
The results of our evaluation demonstrate that Speaker2Vec successfully captures and quantifies the nuances of user engagement and communication styles, validating its effectiveness in the targeted task.

## Authors

- Daniel Samira
- Yuval Gorodissky
- Lior Mishutin
- Uri Zlotkin

## Project Structure

- `main.py`: Main script that orchestrates the vectorization process.
- `preprocess.py`: Contains preprocessing functions for feature extraction.
- `preprocessEmbeddings.py`: Includes functions for text embedding preprocessing.
- `annotated_trees_101.csv`: Dataset comprising 101 annotated "Change My View" (CMV) threads.

## Prerequisites

Before running Speaker2Vec, ensure you have the following installed:
- Python 3.8 or higher
- Required Python libraries: `numpy`, `pandas`, `torch`, `transformers`

## Installation

1. Clone the Speaker2Vec repository:
   ```
   git clone http://github.com/DanielSamira96/Speaker2Vec.git
   ```
2. Navigate to the Speaker2Vec directory:
   ```
   cd Speaker2Vec
   ```
3. Install the required Python libraries:
   ```
   pip install -r requirements.txt
   ```

## How to Use

To use Speaker2Vec, follow these steps:

1. Prepare your dataset. Ensure it is in the correct format as the provided `annotated_trees_101.csv`.
2. Run `main.py` to start the vectorization process:
   ```
   python main.py
   ```
   
This script will preprocess your data, perform feature extraction and text embedding, and output the vectorized speaker features.

## Contact

For further information, please contact the authors at their respective email addresses:

- Daniel Samira: samirada@post.bgu.ac.il
- Yuval Gorodissky: yuvalgor@post.bgu.ac.il
- Lior Mishutin: mishutin@post.bgu.ac.il
- Uri Zlotkin: urizlo@post.bgu.ac.il
