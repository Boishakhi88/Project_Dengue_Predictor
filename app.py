import gradio as gr
import numpy as np
import joblib
import matplotlib.pyplot as plt
import io
import base64
from tensorflow.keras.models import load_model

# Load scaler and region map
scaler = joblib.load('saved_models/scaler.pkl')
region_map = joblib.load('saved_models/region_map.pkl')
region_map_rev = {v: k for k, v in region_map.items()}

# Input features
feature_names = [
    'Temperature', 'Humidity', 'Rainfall', 'Sunshine',
    'Population Density', 'Urbanization Rate', 'Waste Management ',
    'Water Storage ', 'NDVI', 'Month', 'Week',
    'lag_1', 'lag_2', 'lag_3',
    'rolling_3', 'rolling_7', 'rolling_14',
    'Temp_Humidity', 'Rainfall_Sunshine'
]

# Prediction function
def predict_weekly_total(region, *inputs):
    try:
        input_list = list(inputs)

        lag_indices = [feature_names.index('lag_1'),
                       feature_names.index('lag_2'),
                       feature_names.index('lag_3')]

        region_code = region_map_rev[region]
        preds = []
        current_features = input_list.copy()

        model = load_model(f'saved_models/LSTM_{region}.h5', compile=False)

        for day in range(7):
            input_array = np.array(current_features).reshape(1, -1)
            input_full = np.insert(input_array, 0, region_code, axis=1)
            scaled_input = input_full.copy()
            scaled_input[0, 1:] = scaler.transform(input_full[:, 1:])
            lstm_input = np.repeat(scaled_input, 3, axis=0).reshape(1, 3, -1)

            prediction = model.predict(lstm_input, verbose=0).flatten()[0]
            prediction_rounded = int(round(prediction))
            preds.append(prediction_rounded)

            # Update lag values
            current_features[lag_indices[2]] = current_features[lag_indices[1]]
            current_features[lag_indices[1]] = current_features[lag_indices[0]]
            current_features[lag_indices[0]] = prediction_rounded

        weekly_total = sum(preds)

        # Generate 7-day plot
        plt.figure(figsize=(6, 3))
        plt.plot(range(1, 8), preds, marker='o', color='crimson')
        plt.title("7-Day Predicted Dengue Cases")
        plt.xlabel("Day")
        plt.ylabel("Predicted Cases")
        plt.grid(True)
        plt.tight_layout()

        # Convert plot to base64
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        encoded = base64.b64encode(buf.read()).decode('utf-8')
        buf.close()
        img_html = f'<img src="data:image/png;base64,{encoded}" width="100%">'

        # Prepare daily output text
        daily_preds = "\n".join([f"📅 Day {i+1}: {c} cases" for i, c in enumerate(preds)])
        final_output = f"{daily_preds}\n\n📊 Total Predicted Dengue Cases (Next 7 Days): **{weekly_total} cases**"

        return final_output, img_html

    except Exception as e:
        return f"❌ Error: {str(e)}", None

# Gradio UI
with gr.Blocks(title="Dengue Forecasting UI", theme=gr.themes.Soft(primary_hue="red")) as demo:
    gr.Markdown("<h1 style='text-align: center; color: #e74c3c;'>🦟 Dengue Forecasting using LSTM</h1>")
    gr.Markdown("<p style='text-align: center;'>Predict daily and weekly dengue cases for the next 7 days based on environmental and demographic inputs.</p>")

    region_input = gr.Dropdown(choices=list(region_map.values()), label="🌍 Select Region")

    input_boxes = []
    for i in range(0, len(feature_names), 3):
        with gr.Row():
            for j in range(3):
                if i + j < len(feature_names):
                    input_boxes.append(gr.Number(label=feature_names[i + j]))

    output_text = gr.Textbox(label="📊 Daily + Weekly Forecast Output", lines=10)
    output_plot = gr.HTML(label="📈 7-Day Trend")

    with gr.Row():
        predict_btn = gr.Button("🚀 Predict", variant="primary")

    predict_btn.click(fn=predict_weekly_total,
                      inputs=[region_input] + input_boxes,
                      outputs=[output_text, output_plot])

# Launch
if __name__ == "__main__":
    demo.launch()
