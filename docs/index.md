# AI Evaluations

Have you ever wondered how well different AI models perform on various tasks?  
Good news—you’ve just found a clear, approachable guide to understanding how to evaluate AI models effectively.

## Why Evaluate AI?
Training an AI model is only half the battle. To know whether it’s performing well, we need reliable ways to measure its success and failure. That’s where evaluation metrics come in.

## Common Evaluation Metrics
There are many ways to measure an AI’s performance, but four metrics are used most often:

- **Accuracy**
- **Precision**
- **Recall**
- **F1 Score**

All of these metrics are based on the *confusion matrix*, shown below:

|                  | **Predicted Positive** | **Predicted Negative** |
| :--------------- | :--------------------: | :--------------------: |
| **Actual Positive** | True Positive (TP)    | False Negative (FN)    |
| **Actual Negative** | False Positive (FP)   | True Negative (TN)     |

## Evaluation Calculations
Let’s walk through how each metric uses the confusion matrix to measure model performance.

### Accuracy
Accuracy answers a simple question: *What fraction of predictions were correct overall?*

$$
\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}
$$

Accuracy works well when errors are equally costly, but it can be misleading if false positives or false negatives matter more in your context.

---

### Precision
Precision focuses on the quality of positive predictions. It asks: *Of everything the model predicted as positive, how many were actually correct?*

$$
\text{Precision} = \frac{TP}{TP + FP}
$$

This is especially important in scenarios where false positives carry a high cost—for example, diagnosing a serious illness when the patient is actually healthy.

---

### Recall
Recall flips the question: *Of all the actual positives, how many did the model successfully identify?*

$$
\text{Recall} = \frac{TP}{TP + FN}
$$

High recall is crucial when missing a positive case would be dangerous or costly—such as failing to detect fraud or overlooking a disease.

---

### F1 Score
Precision and recall often pull in opposite directions. The **F1 Score** balances the two by taking their harmonic mean:

$$
\text{F1} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}
$$

The F1 Score is most useful when you need a single number that captures the trade-off between precision and recall.

---
<details>
<summary>Walkthrough Example: Breast Cancer Dataset</summary>

Let's walk through an example together of how to use these metrics. We are going to train a simple model on the Breast Cancer Wisconsin dataset. This dataset is an example of binary classification, with a slightly imbalanced dataset and will thus be a great example to showcase the differences in evaluation metrics. Create a jupyter notebook and follow along.

First we are going to get all our imports.

```python
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
```

After that, we need to load our data

```python
data = load_breast_cancer()
X, y = data.data, data.target # y=0 malignant, y=1 benign
```

We then need to split our data into a training and testing set. Luckily, sklearn provides a function that will handle this for us. Make sure to include `stratify=y` as an argument to preserve class balances in the dataset.

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
```

Let's then train a model and make some predictions

```python
model = LogisticRegression(max_iter=500)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
```

We then need to extract our confusion matrix using the following:

```python
cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()
```

We can then calculate accuracy, precision, recall, and f1 scores using the equations from above.

```python
# Accuracy
accuracy = (tp + tn) / (tp + tn + fp + fn)

# Precision (positive = class 1, benign in this dataset)
precision = tp / (tp + fp)

# Recall (a.k.a sensitivity, true positive rate)
recall = tp / (tp + fn)

# F1 Score
f1 = 2 * (precision * recall) / (precision + recall)

# Print our metrics
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
```

After we run everything, you should see something similar to the following:
- Accuracy: 0.9415
- Precision: 0.9292
- Recall: 0.9813
- F1 Score: 0.9545

As you can see, different evaluation metrics don't perform the same, and hopefully you can see the importance of choosing the right metric based on your needs.
</details>

## Conclusion

Different evaluation metrics highlight different aspects of an AI model's performance. **Accuracy** tells you how often the model is correct overall, **precision** emphasizes the reliability of positive predictions, **recall** captures how well the model identifies all true positives, and **F1 Score** balances precision and recall into a single number.

Choosing the right metric—or a combination of metrics—depends on the problem you’re solving and the cost of different types of errors. For example, in medical diagnoses, missing a positive case may be far more critical than accidentally predicting a false positive.

Now that you’ve seen how to calculate and interpret these metrics, try applying them to a different dataset or model. Experimenting with datasets that are more imbalanced or noisy will help you see why some metrics are more informative than others and deepen your understanding of model evaluation in real-world scenarios.


