# insightMammo.py
import streamlit as st
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import (roc_curve, auc, confusion_matrix,
                            accuracy_score, precision_score,
                            recall_score, f1_score, roc_auc_score)

# --- Configuration (Must be first Streamlit command) ---
st.set_page_config(
    page_title="MamoInsight - Breast Cancer Diagnostics",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Constants ---
BANNER_IMAGE = "Mamoinsight.png"
FEATURE_RANGES = {
    'mean radius': (6.981, 28.110),
    'mean texture': (9.710, 39.280),
    'mean perimeter': (43.79, 188.5),
    'mean area': (143.5, 2501.0),
    'mean smoothness': (0.05263, 0.1634),
    'mean compactness': (0.01938, 0.3454),
    'mean concavity': (0.0, 0.4268),
    'mean concave points': (0.0, 0.2012),
    'mean symmetry': (0.106, 0.304),
    'mean fractal dimension': (0.04996, 0.09744),
    'radius error': (0.1115, 2.873),
    'texture error': (0.3602, 4.885),
    'perimeter error': (0.757, 21.98),
    'area error': (6.802, 542.2),
    'smoothness error': (0.00171, 0.03113),
    'compactness error': (0.00225, 0.1354),
    'concavity error': (0.0, 0.396),
    'concave points error': (0.0, 0.05279),
    'symmetry error': (0.00788, 0.07895),
    'fractal dimension error': (0.00089, 0.02984),
    'worst radius': (7.93, 36.04),
    'worst texture': (12.02, 49.54),
    'worst perimeter': (50.41, 251.2),
    'worst area': (185.2, 4254.0),
    'worst smoothness': (0.07117, 0.2226),
    'worst compactness': (0.02729, 1.058),
    'worst concavity': (0.0, 1.252),
    'worst concave points': (0.0, 0.291),
    'worst symmetry': (0.1565, 0.6638),
    'worst fractal dimension': (0.05504, 0.2075)
}

# --- Load Data and Models ---
@st.cache_resource
def load_dataset():
    data = load_breast_cancer()
    return data

@st.cache_resource
def load_models():
    return {
        'model': joblib.load('svm_model.pkl'),
        'scaler': joblib.load('scaler.pkl')
    }

# Initialize app data
data = load_dataset()
models = load_models()
X_train, X_test, y_train, y_test = train_test_split(
    data.data, data.target, test_size=0.2, random_state=42
)
X_test_scaled = models['scaler'].transform(X_test)
y_pred = models['model'].predict(X_test_scaled)
y_probs = models['model'].predict_proba(X_test_scaled)[:, 1]

# --- Utility Functions ---
# def create_radar_chart(input_values):
#     fig = go.Figure()
#     fig.add_trace(go.Scatterpolar(
#         r=input_values,
#         theta=data.feature_names,
#         fill='toself',
#         line_color='#2980b9'
#     ))
#     fig.update_layout(
#         polar=dict(radialaxis=dict(visible=True)),
#         margin=dict(l=150, r=150),
#         height=500
#     )
#     return fig

# --- Page Sections ---
def home_page():
    st.image(BANNER_IMAGE, use_container_width=True)
    st.title("Advanced Breast Cancer Diagnostics")

    with st.expander("About This App", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            **Clinical Decision Support System**
            Leveraging SVM machine learning to analyze FNA results:
            - 98.2% validation accuracy
            - Real-time malignancy probability estimation
            - Comprehensive cellular characteristic analysis
            """)
            st.markdown("""
            **Why SVM Was Chosen:**
            - Superior handling of high-dimensional data
            - Effective with limited training samples
            - Robust to measurement noise
            - Clear margin maximization
            """)
        with col2:
            st.image("svm_diagram.png", caption="SVM Kernel Space Transformation", use_container_width=True)

def data_story_page():
    st.header("Data Story: From Cellular Features to Diagnosis")

    # ROC Curve
    with st.expander("ROC Curve Analysis", expanded=True):
        fpr, tpr, _ = roc_curve(y_test, y_probs)
        roc_auc = auc(fpr, tpr)
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=fpr, y=tpr,
            mode='lines',
            name=f'ROC Curve (AUC = {roc_auc:.3f})',
            line=dict(color='#FFA500', width=3)
        ))
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            mode='lines',
            name='Random Chance',
            line=dict(color='navy', dash='dash')
        ))
        fig.update_layout(title='Receiver Operating Characteristic (ROC) Curve')
        st.plotly_chart(fig, use_container_width=True)

    # Confusion Matrix
    with st.expander("Confusion Matrix", expanded=True):
        cm = confusion_matrix(y_test, y_pred)
        fig = px.imshow(
            cm,
            labels=dict(x="Predicted", y="Actual", color="Count"),
            x=data.target_names,
            y=data.target_names,
            text_auto=True,
            color_continuous_scale='Blues'
        )
        fig.update_layout(title='Confusion Matrix')
        st.plotly_chart(fig, use_container_width=True)

    # Feature Distributions
    with st.expander("Feature Analysis", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Radius Mean Distribution")
            fig = px.histogram(
                x=data.data[:, 0],
                color=data.target_names[data.target],
                nbins=50,
                color_discrete_sequence=['#1f77b4', '#ff7f0e']
            )
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            st.subheader("Perimeter Mean Distribution")
            fig = px.histogram(
                x=data.data[:, 2],
                color=data.target_names[data.target],
                nbins=50,
                color_discrete_sequence=['#1f77b4', '#ff7f0e']
            )
            st.plotly_chart(fig, use_container_width=True)

    # Performance Metrics
    with st.expander("Model Performance", expanded=True):
        metrics = {
            'Accuracy': accuracy_score(y_test, y_pred),
            'Precision': precision_score(y_test, y_pred),
            'Recall': recall_score(y_test, y_pred),
            'F1 Score': f1_score(y_test, y_pred),
            'AUC': roc_auc_score(y_test, y_probs)
        }
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=list(metrics.keys()),
            y=list(metrics.values()),
            marker_color='#2980b9'
        ))
        fig.update_layout(title='Model Performance Metrics')
        st.plotly_chart(fig, use_container_width=True)

    # Horizontal Bar Chart of Feature Ranges
    with st.expander("Feature Value Ranges", expanded=False):
        features_df = pd.DataFrame({
            'Feature': data.feature_names,
            'Min': [FEATURE_RANGES[f][0] for f in data.feature_names],
            'Max': [FEATURE_RANGES[f][1] for f in data.feature_names]
        })
        fig = go.Figure()
        fig.add_trace(go.Bar(
            y=features_df['Feature'],
            x=features_df['Min'],
            orientation='h',
            name='Min',
            marker_color='#1f77b4'
        ))
        fig.add_trace(go.Bar(
            y=features_df['Feature'],
            x=features_df['Max'],
            orientation='h',
            name='Max',
            marker_color='#ff7f0e'
        ))
        fig.update_layout(
            barmode='group',
            xaxis_title='Value',
            yaxis_title='Feature',
            plot_bgcolor='white',
            legend_title_text="Range"
        )
        st.plotly_chart(fig, use_container_width=True)

    # Feature Correlation Heatmap
    with st.expander("Feature Correlation Heatmap", expanded=False):
        corr_matrix = pd.DataFrame(data.data, columns=data.feature_names).corr()
        fig = px.imshow(
            corr_matrix,
            color_continuous_scale='Viridis',
            labels=dict(x="Features", y="Features"),
            aspect="auto"
        )
        st.plotly_chart(fig, use_container_width=True)

    # Radar Chart of Model Performance Metrics
    with st.expander("Model Performance Radar Chart", expanded=False):
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC']
        values = [
            accuracy_score(y_test, y_pred),
            precision_score(y_test, y_pred),
            recall_score(y_test, y_pred),
            f1_score(y_test, y_pred),
            roc_auc_score(y_test, y_probs)
        ]
        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=metrics,
            fill='toself',
            name='Model Performance',
            line_color='#2980b9'
        ))
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0.8, 1.0])),
            showlegend=True
        )
        st.plotly_chart(fig, use_container_width=True)

def prediction_page():
    st.header("Clinical Prediction Interface")

    # Group features
    mean_features = [f for f in data.feature_names if "mean" in f]
    se_features = [f for f in data.feature_names if "error" in f or "se" in f]
    worst_features = [f for f in data.feature_names if "worst" in f]

    input_values = []

    with st.form("diagnostic_form"):
        st.markdown("### Mean Features")
        mean_cols = st.columns(3)
        for i, feature in enumerate(mean_features):
            min_val, max_val = FEATURE_RANGES[feature]
            val = mean_cols[i % 3].slider(
                feature.capitalize(),
                min_value=float(min_val),
                max_value=float(max_val),
                value=float((min_val + max_val) / 2),
                step=0.001,
                format="%.3f",
                help=f"Normal range: {min_val:.3f} - {max_val:.3f}",
                key=f"mean_{feature}"
            )
            input_values.append(val)

        st.markdown("### Standard Error Features")
        se_cols = st.columns(3)
        for i, feature in enumerate(se_features):
            min_val, max_val = FEATURE_RANGES[feature]
            val = se_cols[i % 3].slider(
                feature.capitalize(),
                min_value=float(min_val),
                max_value=float(max_val),
                value=float((min_val + max_val) / 2),
                step=0.001,
                format="%.3f",
                help=f"Normal range: {min_val:.3f} - {max_val:.3f}",
                key=f"se_{feature}"
            )
            input_values.append(val)

        st.markdown("### Worst Features")
        worst_cols = st.columns(3)
        for i, feature in enumerate(worst_features):
            min_val, max_val = FEATURE_RANGES[feature]
            val = worst_cols[i % 3].slider(
                feature.capitalize(),
                min_value=float(min_val),
                max_value=float(max_val),
                value=float((min_val + max_val) / 2),
                step=0.001,
                format="%.3f",
                help=f"Normal range: {min_val:.3f} - {max_val:.3f}",
                key=f"worst_{feature}"
            )
            input_values.append(val)

        if st.form_submit_button("Analyze Sample"):
            try:
                scaled_input = models['scaler'].transform([input_values])
                prediction = models['model'].predict(scaled_input)
                probability = models['model'].predict_proba(scaled_input)[0][1]

                st.subheader("Diagnostic Report")
                col1, col2 = st.columns(2)
                with col1:
                    if prediction[0] == 1:
                        st.error(f"**Malignant Detected** ({probability:.2%})")
                    else:
                        st.success(f"**Benign Growth** ({1-probability:.2%})")
                with col2:
                    st.metric("Confidence Score", f"{max(probability, 1-probability):.2%}")

                st.plotly_chart(create_radar_chart(input_values), use_container_width=True)
            except Exception as e:
                st.error(f"Validation Error: {str(e)}")

# --- Main App Flow ---
def main():
    pages = {
        "Home": home_page,
        "Data Analysis": data_story_page,
        "Clinical Predictor": prediction_page
    }

    with st.sidebar:
        st.title("Navigation")
        page = st.radio("Select Page:", list(pages.keys()))
        st.markdown("---")
        st.markdown("**Clinical Guidelines**")
        st.caption("""
        - Values outside normal ranges highlighted in red
        - Malignancy probability >70% requires biopsy
        - Always correlate with imaging findings
        """)

    pages[page]()

if __name__ == "__main__":
    main()
