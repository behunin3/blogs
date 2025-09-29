# AI Evaluations

Have you ever wondered how well different AI models perform on various tasks? Congratulations! You've just found an amazing, out-of-this-world resource that walks you through how to evaluate AI models effectively.

## Why do we need to evaluate AI?
When we train a model, we need to figure out a way to evaluate its performance so that we can know how it's doing. 

## Common Evaluation Metrics
There are many ways in which we can evaluate AI. The most common ways are 
- Accuracy
- Precision
- Recall
- F1 Score. 

All four of these metrics relate to the confusion matrix. Please refer to the following:

|               | **Predicted Positive** | **Predicted Negative** |
| :------------ | :-------------------: | :--------------------: |
| **Actual Positive** | True Positive (TP)    | False Negative (FN)    |
| **Actual Negative** | False Positive (FP)   | True Negative (TN)     |

## Evaluation Calculations
Let's take a moment to see how each of the evaluation metrics use the confusion matrix to evaluate an AI's performance.

### Accuracy
Accuracy is concerned with how many total correct guesses the AI made. It is calculated by counting how many correct predictions, either positively or negatively, were made proportional to the whole confusion matrix. Here's the calculation:
$$
\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}
$$
Accuracy is great if you purely want to know how many correct predictions are being made.

### Precision
Sometimes, however, a false positive can be extremely detrimental. For example, maybe a false positive medical diagnosis would be very costly in terms of money, but also in terms of emotional duress in the patient. Precision then is concerned by trying to help the false positive rate be lower. Precision measures the proportion of positive classifications compared to everything the AI predicts as positive.

$$
\text{Precision} = \frac{TP}{TP + FP}
$$

### Recall
Recall in some ways, is the opposite of precision. Instead of trying to minimize the false positive rate, recall tries to minimize the false negative rate. Recall measures the actual positives correctly identified.

$$
\text{Recall} = \frac{TP}{TP + FN}
$$

### F1 Scores
The challenge with recall and precision is that each ignores a piece of the confusion matrix. Thus we need to find a way to balance the two. F1 scores are a great way to balance precision and recall.

$$
\text{F1} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision}+\text{Recall}}
$$
## Walkthrough example
## Takeaway 


