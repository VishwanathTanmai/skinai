import streamlit as st
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import io
from datetime import datetime
import uuid

# Import custom modules
from models.image_classifier import load_vit_model, predict_skin_disease
from models.chatbot import load_bert_model, get_chatbot_response
# Import enhanced chatbot
from models.enhanced_chatbot import load_enhanced_bert_model, get_enhanced_chatbot_response
from utils.image_processing import preprocess_image, extract_roi
from utils.report_generator import generate_pdf_report
from utils.database import init_db, save_analysis, get_history
from utils.color_analysis import analyze_color_profile
from utils.texture_analysis import analyze_texture
from utils.metrics import display_metrics_page

# Set page configuration
st.set_page_config(
    page_title="Skin Disease Prediction App",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Define skin conditions
SKIN_CONDITIONS = ["Acne", "Hyperpigmentation", "Nail Psoriasis", "SJS-TEN", "Vitiligo"]

# Initialize session state variables if they don't exist
if 'history' not in st.session_state:
    st.session_state.history = []
if 'current_image' not in st.session_state:
    st.session_state.current_image = None
if 'current_prediction' not in st.session_state:
    st.session_state.current_prediction = None
if 'current_roi' not in st.session_state:
    st.session_state.current_roi = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'analysis_id' not in st.session_state:
    st.session_state.analysis_id = None

# Initialize database
init_db()

# Load models
@st.cache_resource
def load_models():
    vit_model = load_vit_model()
    bert_model = load_bert_model()
    enhanced_bert_model = load_enhanced_bert_model()
    return vit_model, bert_model, enhanced_bert_model

try:
    vit_model, bert_model, enhanced_bert_model = load_models()
    models_loaded = True
except Exception as e:
    st.error(f"Error loading models: {str(e)}")
    models_loaded = False

# Main app title
st.title("Skin Disease Prediction System")
st.markdown("---")

# Sidebar
with st.sidebar:
    st.header("Navigation")
    app_mode = st.radio("Select a feature:", [
        "📤 Upload & Predict",
        "🔍 Advanced Analysis",
        "💬 Skin Disease Chatbot",
        "📊 Model Metrics",
        "📋 Analysis History"
    ])
    
    st.markdown("---")
    st.header("About")
    st.info(
        "This application uses deep learning models to analyze skin conditions. "
        "Upload an image to get started with skin disease prediction."
    )

# Upload & Predict section
if app_mode == "📤 Upload & Predict":
    st.header("Upload Image for Skin Disease Prediction")
    
    uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])
    
    col1, col2 = st.columns([1, 1])
    
    if uploaded_file is not None:
        # Create a unique analysis ID
        if st.session_state.analysis_id is None:
            st.session_state.analysis_id = str(uuid.uuid4())
            
        # Read and display the image
        image = Image.open(uploaded_file)
        st.session_state.current_image = image
        col1.image(image, caption="Uploaded Image", use_container_width=True)
        
        # Predict button
        if col1.button("🔍 Predict Skin Condition"):
            if models_loaded:
                with st.spinner("Analyzing image..."):
                    # Preprocess image for model
                    preprocessed_img = preprocess_image(image)
                    
                    # Get prediction
                    prediction, confidence_scores = predict_skin_disease(vit_model, preprocessed_img)
                    st.session_state.current_prediction = {
                        "condition": SKIN_CONDITIONS[prediction],
                        "confidence": confidence_scores,
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    }
                    
                    # Display prediction results
                    col2.success(f"Predicted Condition: **{SKIN_CONDITIONS[prediction]}**")
                    col2.subheader("Confidence Scores")
                    
                    # Create a bar chart of confidence scores
                    fig, ax = plt.subplots()
                    y_pos = np.arange(len(SKIN_CONDITIONS))
                    ax.barh(y_pos, confidence_scores, align='center')
                    ax.set_yticks(y_pos)
                    ax.set_yticklabels(SKIN_CONDITIONS)
                    ax.invert_yaxis()  # Labels read top-to-bottom
                    ax.set_xlabel('Confidence Score')
                    ax.set_title('Prediction Confidence')
                    
                    col2.pyplot(fig)
                    
                    # Save the analysis to the database
                    save_analysis(
                        analysis_id=st.session_state.analysis_id,
                        image=uploaded_file.getvalue(),
                        condition=SKIN_CONDITIONS[prediction],
                        confidence_scores=confidence_scores.tolist(),
                        timestamp=datetime.now()
                    )
                    
                    # Add to history
                    if st.session_state.current_prediction not in st.session_state.history:
                        st.session_state.history.append(st.session_state.current_prediction)
            else:
                st.error("Models failed to load. Please restart the application.")

# Advanced Analysis section
elif app_mode == "🔍 Advanced Analysis":
    st.header("Advanced Image Analysis")
    
    if st.session_state.current_image is None:
        st.warning("Please upload an image in the 'Upload & Predict' section first.")
    else:
        analysis_type = st.radio("Select Analysis Type:", [
            "Texture Analysis", 
            "Region of Interest (ROI)", 
            "Color Profile"
        ])
        
        if analysis_type == "Texture Analysis":
            st.subheader("Texture Analysis")
            col1, col2 = st.columns([1, 1])
            
            # Display original image
            col1.image(st.session_state.current_image, caption="Original Image", use_container_width=True)
            
            if col1.button("Analyze Texture"):
                with st.spinner("Performing texture analysis..."):
                    # Analyze texture
                    texture_results, texture_image = analyze_texture(st.session_state.current_image)
                    
                    # Display texture analysis results
                    col2.image(texture_image, caption="Texture Visualization", use_container_width=True)
                    
                    # Show texture metrics
                    st.subheader("Texture Metrics")
                    metrics_df = pd.DataFrame({
                        'Metric': list(texture_results.keys()),
                        'Value': list(texture_results.values())
                    })
                    st.table(metrics_df)
                    
                    # Update database
                    if st.session_state.analysis_id:
                        save_analysis(
                            analysis_id=st.session_state.analysis_id,
                            texture_analysis=texture_results,
                            update_only=True
                        )
        
        elif analysis_type == "Region of Interest (ROI)":
            st.subheader("Region of Interest (ROI) Selection")
            st.info("Select a region of interest to analyze a specific area of the skin condition.")
            
            col1, col2 = st.columns([1, 1])
            
            # Display original image
            col1.image(st.session_state.current_image, caption="Original Image", use_container_width=True)
            
            # ROI selection
            roi_options = ["Manual Selection", "Automatic Detection"]
            roi_method = col1.radio("ROI Selection Method:", roi_options)
            
            if roi_method == "Manual Selection":
                # Get image dimensions for the sliders
                img_width, img_height = st.session_state.current_image.size
                
                # Create sliders for ROI selection
                st.subheader("Select ROI Coordinates")
                col_left, col_right = st.columns(2)
                
                with col_left:
                    roi_x1 = st.slider("Left X", 0, img_width, int(img_width * 0.25))
                    roi_y1 = st.slider("Top Y", 0, img_height, int(img_height * 0.25))
                
                with col_right:
                    roi_x2 = st.slider("Right X", 0, img_width, int(img_width * 0.75))
                    roi_y2 = st.slider("Bottom Y", 0, img_height, int(img_height * 0.75))
                
                if col1.button("Extract ROI"):
                    roi_coords = (roi_x1, roi_y1, roi_x2, roi_y2)
                    roi_image = extract_roi(st.session_state.current_image, roi_coords, method="manual")
                    st.session_state.current_roi = roi_image
                    
                    # Display ROI
                    col2.image(roi_image, caption="Selected ROI", use_container_width=True)
                    
                    # Analyze the ROI
                    if models_loaded:
                        with st.spinner("Analyzing ROI..."):
                            # Preprocess ROI for model
                            preprocessed_roi = preprocess_image(roi_image)
                            
                            # Get prediction for ROI
                            roi_prediction, roi_confidence = predict_skin_disease(vit_model, preprocessed_roi)
                            
                            # Display ROI prediction results
                            col2.success(f"ROI Prediction: **{SKIN_CONDITIONS[roi_prediction]}**")
                            
                            # Update database
                            if st.session_state.analysis_id:
                                save_analysis(
                                    analysis_id=st.session_state.analysis_id,
                                    roi_coordinates=roi_coords,
                                    roi_prediction=SKIN_CONDITIONS[roi_prediction],
                                    update_only=True
                                )
            
            else:  # Automatic Detection
                if col1.button("Auto-detect ROI"):
                    with st.spinner("Detecting regions of interest..."):
                        # Auto-detect ROI
                        roi_image, roi_coords = extract_roi(st.session_state.current_image, None, method="auto")
                        st.session_state.current_roi = roi_image
                        
                        # Display ROI
                        col2.image(roi_image, caption="Auto-detected ROI", use_container_width=True)
                        
                        # Analyze the ROI
                        if models_loaded:
                            # Preprocess ROI for model
                            preprocessed_roi = preprocess_image(roi_image)
                            
                            # Get prediction for ROI
                            roi_prediction, roi_confidence = predict_skin_disease(vit_model, preprocessed_roi)
                            
                            # Display ROI prediction results
                            col2.success(f"ROI Prediction: **{SKIN_CONDITIONS[roi_prediction]}**")
                            
                            # Update database
                            if st.session_state.analysis_id:
                                save_analysis(
                                    analysis_id=st.session_state.analysis_id,
                                    roi_coordinates=roi_coords,
                                    roi_prediction=SKIN_CONDITIONS[roi_prediction],
                                    update_only=True
                                )
        
        elif analysis_type == "Color Profile":
            st.subheader("Color Profile Analysis")
            col1, col2 = st.columns([1, 1])
            
            # Display original image
            col1.image(st.session_state.current_image, caption="Original Image", use_container_width=True)
            
            if col1.button("Analyze Color Profile"):
                with st.spinner("Analyzing color profile..."):
                    # Analyze color profile
                    color_data, color_viz = analyze_color_profile(st.session_state.current_image)
                    
                    # Display color visualization
                    col2.image(color_viz, caption="Color Segmentation", use_container_width=True)
                    
                    # Show color distribution
                    st.subheader("Color Distribution")
                    fig, ax = plt.subplots(figsize=(10, 6))
                    colors = list(color_data.keys())
                    values = list(color_data.values())
                    ax.bar(colors, values, color=colors)
                    ax.set_xlabel('Color')
                    ax.set_ylabel('Percentage')
                    ax.set_title('Color Distribution in the Image')
                    st.pyplot(fig)
                    
                    # Update database
                    if st.session_state.analysis_id:
                        save_analysis(
                            analysis_id=st.session_state.analysis_id,
                            color_analysis=color_data,
                            update_only=True
                        )
        
        # Generate PDF report
        if st.session_state.current_prediction:
            if st.button("Generate PDF Report"):
                with st.spinner("Generating PDF report..."):
                    # Generate PDF report
                    report_data = {
                        "image": st.session_state.current_image,
                        "prediction": st.session_state.current_prediction,
                        "texture_analysis": getattr(st.session_state, "texture_results", None),
                        "roi_image": getattr(st.session_state, "current_roi", None),
                        "color_profile": getattr(st.session_state, "color_data", None)
                    }
                    
                    pdf_bytes = generate_pdf_report(report_data)
                    
                    # Create download button for PDF
                    st.download_button(
                        label="Download PDF Report",
                        data=pdf_bytes,
                        file_name=f"skin_analysis_{st.session_state.analysis_id}.pdf",
                        mime="application/pdf"
                    )

# Chatbot section
elif app_mode == "💬 Skin Disease Chatbot":
    st.header("Skin Disease Chatbot")
    
    # Add a toggle to switch between standard and enhanced chatbot
    chatbot_mode = st.radio("Select Chatbot Version:", ["Standard", "Enhanced (Knowledge-Rich)"])
    
    st.info(
        "Ask me any questions about skin diseases, treatments, or general skin health. "
        "For example, you can ask 'What is acne?' or 'How is vitiligo treated?'"
    )
    
    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.write(message["content"])
    
    # Get user input
    user_input = st.chat_input("Ask about skin conditions...")
    
    if user_input:
        # Add user message to chat history
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        
        # Display user message
        with st.chat_message("user"):
            st.write(user_input)
        
        # Generate response based on selected chatbot version
        if models_loaded:
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    if chatbot_mode == "Standard":
                        response = get_chatbot_response(bert_model, user_input)
                    else:  # Enhanced mode
                        response = get_enhanced_chatbot_response(enhanced_bert_model, user_input)
                    
                    st.write(response)
                    
                    # Add assistant response to chat history
                    st.session_state.chat_history.append({"role": "assistant", "content": response})
        else:
            st.error("Chatbot model failed to load. Please restart the application.")

# Model Metrics section
elif app_mode == "📊 Model Metrics":
    # Call the function from metrics.py that displays all the metrics visualizations
    display_metrics_page()

# Analysis History section
else:
    st.header("Analysis History")
    
    # Get analysis history from database
    history = get_history()
    
    if not history:
        st.info("No previous analyses found. Upload an image to start.")
    else:
        # Display analysis history
        st.subheader("Previous Analyses")
        
        for i, entry in enumerate(history):
            with st.expander(f"Analysis {i+1}: {entry['condition']} - {entry['timestamp']}"):
                col1, col2 = st.columns([1, 2])
                
                # Convert binary image data to PIL Image
                if entry.get('image'):
                    image = Image.open(io.BytesIO(entry['image']))
                    col1.image(image, caption="Uploaded Image", use_container_width=True)
                
                # Display prediction details
                col2.write(f"**Predicted Condition:** {entry['condition']}")
                col2.write(f"**Analysis Date:** {entry['timestamp']}")
                
                # Display confidence scores if available
                if entry.get('confidence_scores'):
                    confidence_df = pd.DataFrame({
                        'Condition': SKIN_CONDITIONS,
                        'Confidence': entry['confidence_scores']
                    })
                    col2.write("**Confidence Scores:**")
                    col2.dataframe(confidence_df)
                
                # Display additional analyses if available
                if entry.get('texture_analysis'):
                    col2.write("**Texture Analysis Results:**")
                    col2.json(entry['texture_analysis'])
                
                if entry.get('color_analysis'):
                    col2.write("**Color Analysis Results:**")
                    col2.json(entry['color_analysis'])
                
                # Add option to regenerate report
                if st.button(f"Generate Report for Analysis {i+1}"):
                    with st.spinner("Generating PDF report..."):
                        # Create report data from history entry
                        report_data = {
                            "image": image if entry.get('image') else None,
                            "prediction": {
                                "condition": entry['condition'],
                                "confidence": entry.get('confidence_scores', []),
                                "timestamp": entry['timestamp']
                            },
                            "texture_analysis": entry.get('texture_analysis'),
                            "roi_image": None,  # ROI image not stored in history
                            "color_profile": entry.get('color_analysis')
                        }
                        
                        pdf_bytes = generate_pdf_report(report_data)
                        
                        # Create download button for PDF
                        st.download_button(
                            label=f"Download Report for Analysis {i+1}",
                            data=pdf_bytes,
                            file_name=f"skin_analysis_{entry['analysis_id']}.pdf",
                            mime="application/pdf"
                        )

# Footer
st.markdown("---")
st.caption("Skin Disease Prediction System | Powered by PyTorch, ViT, and BERT")
