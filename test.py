import plotly.graph_objects as go
import numpy as np

# Generate ECG-like data
t = np.linspace(0, 1, num=1000)
ecg_signal = (
    np.sin(2 * np.pi * 5 * t) +
    np.sin(2 * np.pi * 10 * t) +
    np.sin(2 * np.pi * 20 * t)
)

# Create the ECG-like figure using Plotly
fig = go.Figure()

# Add ECG-like signal to the figure
fig.add_trace(go.Scatter.(x=t, y=ecg_signal, mode='lines', line=dict(color='blue')))

# Customize the figure layout and axis labels
fig.update_layout(
    title='ECG-like Figure',
    xaxis_title='Time',
    yaxis_title='Amplitude',
)


# Show the figure
fig.show()
