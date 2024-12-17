
# **Fine-Tuning Large Language Models with Curriculum Learning**

This repository demonstrates fine-tuning a large language model (LLM) for question-answering tasks using **LoRA** (Low-Rank Adaptation), **4-bit quantization**, and **curriculum learning**. The workflow includes dataset preparation, fine-tuning, and inference, with results saved for further analysis.

---

## **Table of Contents**
1. [Overview](#overview)
2. [Features](#features)
3. [Dependencies](#dependencies)
4. [Setup Instructions](#setup-instructions)
5. [Workflow](#workflow)
6. [Dataset Details](#dataset-details)
7. [Usage](#usage)
8. [Project Structure](#project-structure)
9. [Results](#results)
10. [Contributing](#contributing)
11. [License](#license)

---

## **Overview**

This project fine-tunes the **Mistral-7B** model for a multiple-choice question-answering task. Using:
- **4-bit quantization** for efficient memory usage.
- **LoRA** for low-resource fine-tuning.
- **Curriculum Learning**: Organizes data by difficulty, calculated using simple heuristics.
- **Dynamic Difficulty Calculation**: Adds weights based on question length and complexity.

---

## **Features**
- **Memory Efficiency**: Supports 4-bit quantization using **BitsAndBytes** (bnb).
- **Curriculum Learning**: Splits the dataset into learning stages based on question difficulty.
- **Dynamic Difficulty Scoring**: Calculates question difficulty heuristically.
- **Fine-Tuning with LoRA**: Reduces trainable parameters while maintaining performance.
- **Inference Pipeline**: Generates answers for test questions and extracts selected options.
- **Results Export**: Saves results in a CSV file for further analysis.

---

## **Dependencies**

Install the required libraries:

```bash
pip install peft accelerate bitsandbytes datasets
pip install -U git+https://github.com/huggingface/transformers.git
pip install -U git+https://github.com/huggingface/peft.git
```

---

## **Setup Instructions**

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your_username/your_repo_name.git
   cd your_repo_name
   ```

2. **Prepare the dataset**:
   Place the training and test datasets in the root directory:
   - `Hackathon_KB_updated.csv` (Training dataset)
   - `Hackathon_Question_set_sample.csv` (Test dataset)

3. **Run the main script**:
   ```bash
   python main.py
   ```

---

## **Workflow**

### 1. **Dataset Preparation**
   - **`prepare_data`**: Formats the input into structured prompts for causal language modeling.

### 2. **Dynamic Difficulty Scoring**
   - **`calculate_question_difficulty`**: Assigns a difficulty score based on:
     - Question length.
     - Presence of complex keywords (e.g., "analyze", "compare").
     - Technical terms (e.g., "algorithm", "methodology").

### 3. **Fine-Tuning**
   - Fine-tunes the **Mistral-7B** model using LoRA and curriculum learning.
   - Divides the dataset into stages based on calculated difficulty scores.

### 4. **Inference**
   - Processes the test set to generate answers using `generate_answer`.
   - Extracts the most probable option using `extract_selected_option`.

### 5. **Result Export**
   - Saves answers and generated text to `answers.csv`.

---

## **Dataset Details**

The dataset contains **14,000 records** of multiple-choice questions with options and answers:

| Column Name   | Description                        | Example                                  |
|---------------|------------------------------------|------------------------------------------|
| `Number`      | Unique question identifier.        | `1`                                      |
| `prompt`      | The question text.                 | "What is the capital of France?"         |
| `A`, `B`, `C`, `D`, `E` | Multiple-choice options. | "A) Paris", "B) London", "C) Berlin", etc. |
| `answer`      | The correct answer.                | `A`                                      |

### **Example Record**:
```
Question: What is the capital of France?
A) Paris
B) London
C) Berlin
D) Madrid
E) Rome
Answer: A
```

---

## **Usage**

### **Run the Full Pipeline**
Execute the main script to fine-tune the model and generate answers:
```bash
python main.py
```

### **Custom Inference**
Use the fine-tuned model for generating answers:
```python
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load fine-tuned model
model_name = "fine_tuned_model"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Generate an answer
question = "What is the capital of France?"
generated_answer = generate_answer(question, model, tokenizer)
print("Generated Answer:", generated_answer)
```

---

## **Project Structure**

```
├── main.py                        # Main pipeline: fine-tuning, inference, and results export
├── setup_model()                  # Model setup with 4-bit quantization and LoRA
├── prepare_data()                 # Dataset preparation for training
├── calculate_question_difficulty  # Heuristic difficulty scoring
├── fine_tune_model()              # Fine-tuning loop with curriculum learning
├── generate_answer()              # Generates an answer for input questions
├── extract_selected_option()      # Extracts the correct answer option
├── Hackathon_KB_updated.csv       # Training dataset
├── Hackathon_Question_set_sample.csv  # Test dataset
└── answers.csv                    # Final output file
```

---

## **Results**

Once the pipeline runs, results are saved in **answers.csv**:

| Number | Answer | Generated_Text                       |
|--------|--------|--------------------------------------|
| 1      | A      | Paris is the capital of France.      |
| 2      | B      | London is the capital of the UK.     |

The distribution of answers is printed for quick evaluation:
```
Distribution of answers:
A    10
B     8
C     5
D     3
N/A   2
```

---

## **Contributing**

Contributions are welcome! If you encounter a bug, want to request a feature, or improve the code, feel free to open an issue or submit a pull request.

---

## **License**

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## **Acknowledgements**
- [Hugging Face Transformers](https://huggingface.co/)
- [BitsAndBytes](https://github.com/TimDettmers/bitsandbytes)
- LoRA implementation via [PEFT](https://github.com/huggingface/peft)
