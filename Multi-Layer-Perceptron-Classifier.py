
# Multi-Layer Perceptron Classifier



import os
import warnings
import numpy as np
import matplotlib.pyplot as plt
import cv2
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
from timeit import default_timer as timer
from tqdm import tqdm
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings('ignore', category=UserWarning)

LABELS = ['Bulbasaur', 'Meowth', 'Mew', 'Pidgeot', 'Pikachu', 'Snorlax', 'Squirtle', 'Venusaur', 'Wartortle', 'Zubat']


def preprocess_image(path_to_image, img_size=150):
    """
    Read and resize an input image
    :param path_to_image: path of image file
    :param img_size: image size
    :return: image as a Numpy array
    """
    img = cv2.imread(path_to_image, cv2.IMREAD_COLOR)  # Read image in
    img = cv2.resize(img, (img_size, img_size))  # type: ignore # Resize image
    return np.array(img)


def extract_color_histogram(dataset, hist_size=6):
    """
    Extract colour histogram features from a dataset of images
    :param dataset: dataset of images
    :param hist_size: histogram size for each dimension
    :return: colour histograms
    """
    col_hist = []
    for img in dataset:
        hist = cv2.calcHist([img], [0, 1, 2], None, (hist_size, hist_size, hist_size), [0, 256, 0, 256, 0, 256])
        col_hist.append(cv2.normalize(hist, None, 0, 1, cv2.NORM_MINMAX).flatten()) # type: ignore
    return np.array(col_hist)


def load_dataset(base_path='Dataset\\PokemonData'):
    X = []
    Y = []
    for i in range(0, len(LABELS)):
        current_size = len(X)
        for img in tqdm(os.listdir(base_path + os.sep + LABELS[i])):
            X.append(preprocess_image(base_path + os.sep + LABELS[i] + '/' + img))
            Y.append(LABELS[i])
        print(f'Loaded {len(X) - current_size} {LABELS[i]} images')
    return X, Y

# Function to Find Most Confused Pair of Pokemon
def find_most_confused_pair(confusion_matrix, labels):
    """
    Find the pair of classes that are most frequently confused
    """
    max_confusion = 0
    confused_pair = None

    # Check off-diagonal elements
    for i in range(len(labels)):
        for j in range(len(labels)):
            if i != j and confusion_matrix[i, j] > max_confusion: # Identify Mismatches
                max_confusion = confusion_matrix[i, j] # Identify Two Most Mismatched
                confused_pair = (labels[i], labels[j], max_confusion) # Store Labels of Most Mismatched Pairs

    return confused_pair


""" --------- PART A: SETTING UP & DETERMINING THE OPTIMAL STRUCTURE ---------"""

if __name__ == '__main__':
    # 1. Load dataset
    X, Y = load_dataset()

    # 1.1 Create a Label Encoder & Store Encoded Labels
    le = LabelEncoder()
    Y_encoded = le.fit_transform(Y)

    # STEP 2. Split dataset into Train (70%), Validation (10%), & Test (20%) Datasets
    X_train, X_temp, Y_train_encoded, Y_temp_encoded = train_test_split(X, Y_encoded, test_size=0.3, shuffle=True, random_state=42)
    X_val, X_test, Y_val, Y_test = train_test_split(X_temp, Y_temp_encoded, test_size=0.667, shuffle=True, random_state=42)

    print("\n ----- DATA SPLIT REQUIREMENTS  ----- ")
    print(f"Dataset split - Train: {len(X_train)}\n Validation: {len(X_val)}\n Test: {len(X_test)}")

    # STEP 3. Extract Colour Histogram Features from the Datasets
    training_features = extract_color_histogram(X_train) # Extract data from the training features
    validation_features = extract_color_histogram(X_val) # Extract data from validation features
    testing_features = extract_color_histogram(X_test) # Extract data from testing features

    # STEP 4. Define 9 Different Structures
    # STEP 4.1 Calculate Neuron Counts
    input_size = 6 * 6 * 6  # 216 features for default histogram size [6,6,6]
    output_size = len(LABELS)  # 10 classes

    # Calculate Based on Guidelines
    n_hidden_1 = int((input_size + output_size) / 2)  # Between input and output: ~113
    n_hidden_2 = int((2/3 * input_size) + output_size)  # 2/3 input + output: 154
    n_hidden_3 = int(input_size * 0.8)  # 80% of input (less than 2x): 172

    print(f"\n ----- NEURON COUNT REQUIREMENTS  ----- ")
    print(f"n_hidden_1 (between input/output): {n_hidden_1}")
    print(f"n_hidden_2 (2/3 input + output): {n_hidden_2}")
    print(f"n_hidden_3 (80% of input): {n_hidden_3}")

    # 4.2 Define 9 Structures
    n_hidden_options = [n_hidden_1,
                        n_hidden_2,
                        n_hidden_3,
                        (n_hidden_1, n_hidden_1),
                        (n_hidden_2, n_hidden_2),
                        (n_hidden_3, n_hidden_3),
                        (n_hidden_1, n_hidden_1, n_hidden_1),
                        (n_hidden_2, n_hidden_2, n_hidden_2),
                        (n_hidden_3, n_hidden_3, n_hidden_3)]

    # STEP 5. Determine the Optimal Structure
    # Initialise List of All Test Cases Performances
    performance_list = []
    structure_no = 0
    for test_case in tqdm(n_hidden_options):
        structure_no += 1

        # STEP 5.1. Initialise MLP Classifier
        clf = MLPClassifier(hidden_layer_sizes=test_case, activation='relu', solver='adam', max_iter=1500,
                            random_state=42, early_stopping=True)

        # STEP 5.2. Fit MLP to Training Dataset
        clf.fit(training_features, Y_train_encoded)

        # STEP 5.3. Evaluate Performance on the Validation Dataset
        # Create Prediction Variable: Model Predicts the Labels of Validation Features
        prediction = clf.predict(validation_features)

        #  5.4 Compare Model's Prediction with Truth Labels & Evaluate Performance
        accuracy = accuracy_score(Y_val, prediction)
        precision = precision_score(Y_val, prediction, average='weighted', zero_division=0)
        recall = recall_score(Y_val, prediction, average='weighted', zero_division=0)
        f1 = f1_score(Y_val, prediction, average='weighted', zero_division=0)

        # STEP 5.5 Store Performance in Dictionary
        performance = {"Test Case": test_case,
                       "Accuracy": accuracy,
                       "Precision": precision,
                       "Recall": recall,
                       "f1": f1}

        # 5.6 Append Dictionary Values to Performance List
        performance_list.append(performance)

        print("\n ----- NEURON COUNT REQUIREMENTS  ----- ")
        print(f"Structure {structure_no}: {test_case} - Accuracy: {accuracy:.4f}")

    # STEP 6. Train an MLP Classifier with the Optimal Structure
    # 6.1 Find Best Structure Based on Accuracy
    best_structure = max(performance_list, key=lambda x: x["Accuracy"])
    best = best_structure["Test Case"]

    print("\n ----- DETERMINING OPTIMAL MLP STRUCTURE ----- ")
    print(f"The Optimal MLP Structure is: {best}")
    print(f"With validation accuracy: {best_structure['Accuracy']:.4f}\n")

    # 6.3 Define Histogram Variation Sizes to Test
    hist_variations = [4, 6, 8]  # Different histogram sizes

    # 6.4 Initialize Best Prediction, Best Accuracy, Best Hist Size & Counter
    best_accuracy = -1
    best_prediction = None
    best_hist_size = None
    best_test_labels = None  # Store the corresponding test labels
    counter = 1

    # 6.5 Loop Through Histogram Size Variations
    for hist_size in hist_variations:
      print(f"\n ----- {counter}. DETERMINING OPTIMAL COLOUR HISTOGRAM SIZE ----- ")
      print(f"Testing Model with Histogram Size: [{hist_size}, {hist_size}, {hist_size}]\n")

      # 6.6 Extract Training, Validation and Testing Features
      training_features = extract_color_histogram(X_train, hist_size=hist_size)
      validation_features = extract_color_histogram(X_val, hist_size=hist_size)
      testing_features = extract_color_histogram(X_test, hist_size=hist_size)

      # 6.7 Create MLPClassifier
      clf = MLPClassifier(hidden_layer_sizes=best, activation='relu', solver='adam', max_iter=1500,
                          random_state=42, early_stopping=True)

      # 6.8 Train & Evaluate Final Model:
      # Combine Features & Labels
      all_features = np.concatenate((training_features, validation_features))
      all_labels = np.concatenate((Y_train_encoded, Y_val))

      # 6.9 Final Model Training
      clf.fit(all_features, all_labels)
      prediction = clf.predict(testing_features)

      # STEP 7. Evaluate the MLP on the Test Dataset
      accuracy = accuracy_score(Y_test, prediction)
      precision = precision_score(Y_test, prediction, average='weighted', zero_division=0)
      recall = recall_score(Y_test, prediction, average='weighted', zero_division=0)
      f1 = f1_score(Y_test, prediction, average='weighted', zero_division=0)

      # 8.1 Identify the Best Model
      if accuracy > best_accuracy:
          best_accuracy = accuracy
          best_prediction = prediction
          best_hist_size = hist_size
          best_test_labels = Y_test  # Store the test labels for this best model

      # STEP 8. Report Classification Metrics
      print(f" ----- {counter}. CLASSIFICATION METRICS FOR MLPCLASSIFIER ----- ")
      print(f"The Accuracy Score for this model is:   {accuracy:.4f}")
      print(f"The Precision Score for this model is:  {precision:.4f}")
      print(f"The Recall Score for this model is:     {recall:.4f}")
      print(f"The f1 Score for this model is:         {f1:.4f}\n")

      # Increase Counter
      counter += 1

    # STEP 9. Plot Confusion Matrix
    print(f" ----- CONFUSION MATRIX: OPTIMAL MODEL WITH BEST HISTOGRAM SIZE {best_hist_size} ----- ")
    print(f"Best Overall Accuracy: {best_accuracy:.4f}")

    # 9.1 Check for Valid Predictions
    if best_prediction is not None and best_test_labels is not None:

        # 9.2 Create Confusion Matrix using Best Model's Predictions
        numerical_labels = list(range(len(LABELS)))
        cm = confusion_matrix(best_test_labels, best_prediction, labels=numerical_labels)

        # 9.3 Find Most Confused pair
        confused_pair = find_most_confused_pair(cm, LABELS)
        if confused_pair:
            print(f"Most Frequently Confused Pair: {confused_pair[0]} and {confused_pair[1]}")
            print(f"Confusion Count: {confused_pair[2]}")

        display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=LABELS)
        display.plot()
        plt.title(f"Confusion Matrix - Best Model (Hist Size: {best_hist_size}x{best_hist_size}x{best_hist_size})")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
    else:
        print("Error: No valid model was found to create confusion matrix.")