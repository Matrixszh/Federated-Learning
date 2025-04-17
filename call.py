import streamlit as st
import urllib.request
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.autograd import Variable
import pandas as pd
import io
import base64

st.set_page_config(page_title="Medical Diagnosis Prediction", layout="wide")

# App title and description
st.title("Medical Diagnosis Prediction System")
st.markdown("""
This application uses machine learning to predict two medical conditions:
- **Inflammation of Urinary Bladder**
- **Nephritis of Renal Pelvis Origin**

The models are trained using both centralized and federated learning approaches.
""")

# Download functions
@st.cache_data
def download_url(url, save_as):
    try:
        response = urllib.request.urlopen(url)
        data = response.read()
        with open(save_as, 'wb') as file:
            file.write(data)
        return True
    except Exception as e:
        st.error(f"Error downloading data: {e}")
        return False

def read_binary_file(file):
    with open(file, 'rb') as f:
        block = f.read()
    return block.decode('utf-16')

def split_text_in_lines(text):
    return text.split('\r\n')

def split_by_tabs(line):
    return line.split('\t')

# Parse functions
def parse_double(field):
    field = field.replace(',', '.')
    return float(field)

def parse_boolean(field):
    return 1.0 if field == 'yes' else 0.0

def read_np_array(file):
    try:
        text = read_binary_file(file)
        lines = split_text_in_lines(text)
        rows = []
        for line in lines:
            if line == '': 
                continue
            line = line.replace('\r\n', '')
            fields = split_by_tabs(line)
            row = []
            j = 0
            for field in fields:
                value = parse_double(field) if j == 0 else parse_boolean(field)
                row.append(value)
                j += 1
            rows.append(row)
        matrix = np.array(rows, dtype=np.float32)
        return matrix
    except Exception as e:
        st.error(f"Error reading data: {e}")
        return None

def get_random_indexes(n):
    indexes = list(range(n))
    random_indexes = []
    for i in range(n):
        r = np.random.randint(len(indexes))
        random_indexes.append(indexes.pop(r))
    return random_indexes

def get_indexes_for_2_datasets(n, training=80):
    np.random.seed(42)  # For reproducibility
    indexes = get_random_indexes(n)
    train = int(training / 100. * n)
    return indexes[:train], indexes[train:]

# Logistic Regression Model
class LogisticRegression(nn.Module):
    def __init__(self):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(6, 1)
    
    def forward(self, x):
        return torch.sigmoid(self.linear(x))
    
    def copy(self):
        new_model = LogisticRegression()
        new_model.load_state_dict(self.state_dict())
        return new_model

def decide(y):
    return 1.0 if y >= 0.5 else 0.0

decide_vectorized = np.vectorize(decide)

to_percent = lambda x: '{:.2f}%'.format(x)

def compute_accuracy(model, input, output):
    with torch.no_grad():
        prediction = model(input).numpy()[:, 0]
        n_samples = prediction.shape[0]
        prediction = decide_vectorized(prediction)
        equal = prediction == output.numpy().flatten()
        return 100.0 * equal.sum() / n_samples

def get_input_and_output(data):
    input = torch.tensor(data[:, :6], dtype=torch.float32)
    output1 = torch.tensor(data[:, 6], dtype=torch.float32).view(-1, 1)
    output2 = torch.tensor(data[:, 7], dtype=torch.float32).view(-1, 1)
    return input, output1, output2

# Original parameters from code
learning_rate = 0.01
num_iterations = 20000
worker_iterations = 5
n_hospitals = 4
federated_iterations = 1000

def train_model(diagnosis_title, input, output, test_input, test_output, progress_bar=None):
    model = LogisticRegression()
    criterion = nn.BCELoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)  
    
    losses = []
    accuracies = []
    n_samples, _ = input.shape
    
    for iteration in range(num_iterations):
        optimizer.zero_grad()
        prediction = model(input)
        loss = criterion(prediction, output)
        loss.backward()
        optimizer.step()
        
        if iteration % 500 == 0 or iteration == num_iterations - 1:
            with torch.no_grad():
                train_acc = compute_accuracy(model, input, output)
                train_loss = loss.item()
                losses.append(train_loss)
                accuracies.append(train_acc)
                
                st.text(f'iteration={iteration}, loss={train_loss:.4f}, train_acc={to_percent(train_acc)}')
                
                if progress_bar is not None:
                    progress_bar.progress((iteration + 1) / num_iterations)
    
    test_acc = compute_accuracy(model, test_input, test_output)
    st.text(f'\nTesting Accuracy = {to_percent(test_acc)}')
    
    # Plot graphs as in original code
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    ax1.plot(losses)
    ax1.set_title(f"{diagnosis_title} - Training Loss")
    ax1.set_xlabel("Iterations")
    ax1.set_ylabel("Training Loss")
    ax1.grid(True)
    
    ax2.plot(accuracies)
    ax2.set_title(f"{diagnosis_title} - Training Accuracy")
    ax2.set_xlabel("Iterations")
    ax2.set_ylabel("Training Accuracy (Percent %)")
    ax2.grid(True)
    
    st.pyplot(fig)
    
    return model

def federated_learning(title, features_list, targets_list, test_x, test_y, progress_bar=None):
    model = LogisticRegression()
    criterion = nn.BCELoss()
    
    losses = [[] for _ in range(n_hospitals)]
    accuracies = [[] for _ in range(n_hospitals)]
    
    for iteration in range(federated_iterations):
        local_models = [model.copy() for _ in range(n_hospitals)]
        optimizers = [optim.SGD(m.parameters(), lr=learning_rate) for m in local_models]
        
        for _ in range(worker_iterations):
            last_losses = []
            for i in range(n_hospitals):
                optimizers[i].zero_grad()
                output = local_models[i](features_list[i])
                loss = criterion(output, targets_list[i])
                loss.backward()
                optimizers[i].step()
                last_losses.append(loss.item())
                
        for i in range(n_hospitals):
            losses[i].append(last_losses[i])
            acc = compute_accuracy(local_models[i], features_list[i], targets_list[i])
            accuracies[i].append(acc)
            
        # Federated averaging
        with torch.no_grad():
            avg_weight = sum([m.linear.weight.data for m in local_models]) / n_hospitals
            avg_bias = sum([m.linear.bias.data for m in local_models]) / n_hospitals
            model.linear.weight.data.copy_(avg_weight)
            model.linear.bias.data.copy_(avg_bias)
            
        if iteration % 100 == 0 or iteration == federated_iterations - 1:
            st.text(f"Iteration {iteration}:")
            for i in range(n_hospitals):
                st.text(f"  Hospital {i} - Loss: {losses[i][-1]:.4f}, Accuracy: {accuracies[i][-1]:.2f}%")
                
            if progress_bar is not None:
                progress_bar.progress((iteration + 1) / federated_iterations)
    
    # Plot graphs similar to original code
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    for i in range(n_hospitals):
        ax1.plot(losses[i], label=f'Hospital {i}')
    ax1.legend(loc='upper right')
    ax1.set_title(f"{title} - Training Loss")
    ax1.set_xlabel("Iterations")
    ax1.set_ylabel("Loss")
    ax1.grid(True)
    st.pyplot(fig1)
    
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    for i in range(n_hospitals):
        ax2.plot(accuracies[i], label=f'Hospital {i}')
    ax2.legend(loc='lower right')
    ax2.set_title(f"{title} - Training Accuracy")
    ax2.set_xlabel("Iterations")
    ax2.set_ylabel("Accuracy (%)")
    ax2.grid(True)
    st.pyplot(fig2)
    
    # Final test accuracy
    test_acc = compute_accuracy(model, test_x, test_y)
    st.text(f"\nTesting Accuracy = {to_percent(test_acc)}")
    
    return model

# Function to download the trained model
def get_model_download_link(model, filename):
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    buffer.seek(0)
    b64 = base64.b64encode(buffer.read()).decode()
    return f'<a href="data:application/octet-stream;base64,{b64}" download="{filename}">Download trained model</a>'

# Sidebar for data loading
st.sidebar.header("Data Loading")

# Data URLs
names_link = 'https://archive.ics.uci.edu/ml/machine-learning-databases/acute/diagnosis.names'
data_link = 'https://archive.ics.uci.edu/ml/machine-learning-databases/acute/diagnosis.data'
diagnosis_names = 'diagnosis.names'
diagnosis_data = 'diagnosis.data'

# Dataset loading
if st.sidebar.button("Download Dataset"):
    with st.spinner("Downloading dataset..."):
        success1 = download_url(names_link, diagnosis_names)
        success2 = download_url(data_link, diagnosis_data)
        if success1 and success2:
            st.sidebar.success("Dataset downloaded successfully!")
        else:
            st.sidebar.error("Failed to download dataset.")

# Tabs for different sections of the app
tab1, tab2, tab3, tab4 = st.tabs(["Data Explorer", "Centralized Learning", "Federated Learning", "Prediction"])

# Tab 1: Data Explorer
with tab1:
    st.header("Data Explorer")
    
    # Check if data is available
    try:
        matrix = read_np_array(diagnosis_data)
        if matrix is not None:
            # Create a DataFrame for better display
            columns = ['Temperature', 'Nausea', 'Lumbar Pain', 'Urine Pushing', 
                      'Micturition Pain', 'Burning Urethra', 
                      'Inflammation of Urinary Bladder', 'Nephritis of Renal Pelvis Origin']
            df = pd.DataFrame(matrix, columns=columns)
            
            st.write(f"Dataset Shape: {matrix.shape}")
            st.dataframe(df)
            
            # Display some statistics
            st.subheader("Data Statistics")
            st.write(df.describe())
            
            # Save data for other tabs
            if 'data_loaded' not in st.session_state:
                st.session_state.matrix = matrix
                st.session_state.n_samples, st.session_state.n_dimensions = matrix.shape
                st.session_state.data_loaded = True
                st.success("Data loaded and ready for model training!")
        else:
            st.warning("Please download the dataset first using the button in the sidebar.")
    except Exception as e:
        st.error(f"Error processing data: {e}")
        st.warning("Please download the dataset first using the button in the sidebar.")

# Tab 2: Centralized Learning
with tab2:
    st.header("Centralized Learning")
    
    if 'data_loaded' in st.session_state and st.session_state.data_loaded:
        # Create training and test sets
        matrix = st.session_state.matrix
        train_indexes, test_indexes = get_indexes_for_2_datasets(st.session_state.n_samples)
        train_data = matrix[train_indexes]
        test_data = matrix[test_indexes]
        
        input, output1, output2 = get_input_and_output(train_data)
        test_input, test_output1, test_output2 = get_input_and_output(test_data)
        
        # Select diagnosis to train
        diagnosis_option = st.selectbox(
            "Select diagnosis to train",
            ["Inflammation of Urinary Bladder", "Nephritis of Renal Pelvis Origin"],
            key="centralized_diagnosis"
        )
        
        # Train button
        if st.button("Train Centralized Model"):
            progress_bar = st.progress(0)
            st.markdown(f"Training model for: **{diagnosis_option}**")
            
            with st.spinner("Training model..."):
                if diagnosis_option == "Inflammation of Urinary Bladder":
                    model = train_model(diagnosis_option, input, output1, test_input, test_output1, progress_bar)
                    st.session_state.bladder_model = model
                else:
                    model = train_model(diagnosis_option, input, output2, test_input, test_output2, progress_bar)
                    st.session_state.nephritis_model = model
            
            st.success("Model trained!")
            
            # Model download
            st.markdown(get_model_download_link(
                model, 
                f"centralized_{diagnosis_option.replace(' ', '_').lower()}.pt"
            ), unsafe_allow_html=True)
    else:
        st.warning("Please load the data first in the Data Explorer tab.")

# Tab 3: Federated Learning
with tab3:
    st.header("Federated Learning")
    
    if 'data_loaded' in st.session_state and st.session_state.data_loaded:
        # Create training and test sets if not already done
        matrix = st.session_state.matrix
        train_indexes, test_indexes = get_indexes_for_2_datasets(st.session_state.n_samples)
        train_data = matrix[train_indexes]
        test_data = matrix[test_indexes]
        
        train_data_tensor = torch.tensor(train_data, dtype=torch.float32)
        test_input, test_output1, test_output2 = get_input_and_output(test_data)
        
        # Select diagnosis to train
        diagnosis_option = st.selectbox(
            "Select diagnosis to train",
            ["Inflammation of Urinary Bladder", "Nephritis of Renal Pelvis Origin"],
            key="federated_diagnosis"
        )
        
        test_output = test_output1 if diagnosis_option == "Inflammation of Urinary Bladder" else test_output2
        target_col = 6 if diagnosis_option == "Inflammation of Urinary Bladder" else 7
        
        # Split data for federated learning
        n_samples = train_data_tensor.shape[0]
        samples_per_hospital = n_samples // n_hospitals
        
        hospital_features = []
        hospital_targets = []
        
        for i in range(n_hospitals):
            start_idx = i * samples_per_hospital
            end_idx = (i + 1) * samples_per_hospital if i < n_hospitals - 1 else n_samples
            
            features = train_data_tensor[start_idx:end_idx, :6]
            targets = train_data_tensor[start_idx:end_idx, target_col][:, None]
            
            hospital_features.append(features)
            hospital_targets.append(targets)
        
        # Train button
        if st.button("Train Federated Model"):
            progress_bar = st.progress(0)
            st.markdown(f"Training federated model for: **{diagnosis_option}**")
            
            with st.spinner("Training federated model..."):
                model = federated_learning(
                    diagnosis_option + " Federated", 
                    hospital_features, 
                    hospital_targets, 
                    test_input, 
                    test_output,
                    progress_bar
                )
                
                # Save model in session state
                if diagnosis_option == "Inflammation of Urinary Bladder":
                    st.session_state.bladder_federated_model = model
                else:
                    st.session_state.nephritis_federated_model = model
            
            st.success("Federated model trained!")
            
            # Model download
            st.markdown(get_model_download_link(
                model, 
                f"federated_{diagnosis_option.replace(' ', '_').lower()}.pt"
            ), unsafe_allow_html=True)
    else:
        st.warning("Please load the data first in the Data Explorer tab.")

# Tab 4: Prediction
with tab4:
    st.header("Medical Diagnosis Prediction")
    
    # Check if we have trained models
    has_bladder_model = 'bladder_model' in st.session_state or 'bladder_federated_model' in st.session_state
    has_nephritis_model = 'nephritis_model' in st.session_state or 'nephritis_federated_model' in st.session_state
    
    if has_bladder_model or has_nephritis_model:
        st.markdown("""
        Enter patient symptoms below to get a diagnosis prediction. 
        For 'Temperature', enter the value directly. For other symptoms, select 'Yes' or 'No'.
        """)
        
        # Input form
        with st.form("prediction_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                temperature = st.number_input("Temperature (°C)", min_value=35.5, max_value=42.0, value=36.6, step=0.1)
                nausea = st.selectbox("Nausea", ["No", "Yes"])
                lumbar_pain = st.selectbox("Lumbar Pain", ["No", "Yes"])
            
            with col2:
                urine_pushing = st.selectbox("Urine Pushing", ["No", "Yes"])
                micturition_pain = st.selectbox("Micturition Pain", ["No", "Yes"])
                burning_urethra = st.selectbox("Burning of Urethra", ["No", "Yes"])
                
            model_type = st.radio(
                "Model Type",
                ["Centralized", "Federated"],
                horizontal=True
            )
            
            submit_button = st.form_submit_button("Get Diagnosis")
        
        if submit_button:
            # Prepare input
            input_data = [
                temperature,
                1.0 if nausea == "Yes" else 0.0,
                1.0 if lumbar_pain == "Yes" else 0.0,
                1.0 if urine_pushing == "Yes" else 0.0,
                1.0 if micturition_pain == "Yes" else 0.0,
                1.0 if burning_urethra == "Yes" else 0.0
            ]
            
            input_tensor = torch.tensor([input_data], dtype=torch.float32)
            
            results = {}
            confidence = {}
            
            # Make predictions
            with torch.no_grad():
                # Inflammation of Urinary Bladder
                if model_type == "Centralized" and 'bladder_model' in st.session_state:
                    model = st.session_state.bladder_model
                    pred = model(input_tensor).item()
                    results["Inflammation of Urinary Bladder"] = "Positive" if pred >= 0.5 else "Negative"
                    confidence["Inflammation of Urinary Bladder"] = pred if pred >= 0.5 else 1 - pred
                elif model_type == "Federated" and 'bladder_federated_model' in st.session_state:
                    model = st.session_state.bladder_federated_model
                    pred = model(input_tensor).item()
                    results["Inflammation of Urinary Bladder"] = "Positive" if pred >= 0.5 else "Negative"
                    confidence["Inflammation of Urinary Bladder"] = pred if pred >= 0.5 else 1 - pred
                
                # Nephritis of Renal Pelvis Origin
                if model_type == "Centralized" and 'nephritis_model' in st.session_state:
                    model = st.session_state.nephritis_model
                    pred = model(input_tensor).item()
                    results["Nephritis of Renal Pelvis Origin"] = "Positive" if pred >= 0.5 else "Negative"
                    confidence["Nephritis of Renal Pelvis Origin"] = pred if pred >= 0.5 else 1 - pred
                elif model_type == "Federated" and 'nephritis_federated_model' in st.session_state:
                    model = st.session_state.nephritis_federated_model
                    pred = model(input_tensor).item()
                    results["Nephritis of Renal Pelvis Origin"] = "Positive" if pred >= 0.5 else "Negative"
                    confidence["Nephritis of Renal Pelvis Origin"] = pred if pred >= 0.5 else 1 - pred
            
            # Display results
            st.subheader("Diagnosis Results")
            
            if results:
                for diagnosis, result in results.items():
                    conf = confidence[diagnosis] * 100.0
                    color = "red" if result == "Positive" else "green"
                    
                    st.markdown(f"**{diagnosis}:** <span style='color:{color};font-weight:bold'>{result}</span> (Confidence: {conf:.2f}%)", unsafe_allow_html=True)
                    
                    # Progress bar for confidence
                    st.progress(confidence[diagnosis])
            else:
                st.warning(f"No trained {model_type.lower()} model available for prediction.")
    else:
        st.warning("Please train at least one model first in the Centralized or Federated Learning tabs.")

# Footer
st.markdown("---")
st.markdown("Medical Diagnosis Prediction System - Built with Streamlit")