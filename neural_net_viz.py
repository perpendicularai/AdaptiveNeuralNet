import os
import glob
import re
import numpy as np
import plotly.graph_objects as go

# Directory where CSV files are located
csv_directory = '/Datasets'

# Regular expression pattern for matching file names like "output_iteration_10_sample_1**.csv"
pattern = r'^output_iteration_10_sample_1[0-9][0-9][0-9].csv$'

# Find CSV files matching the pattern
csv_files = glob.glob(os.path.join(csv_directory, '*.csv'))
filtered_csv_files = [file for file in csv_files if re.match(pattern, os.path.basename(file))]

# Initialize a Plotly subplot
fig = go.Figure()

# Iterate through filtered CSV files
for i, csv_file in enumerate(filtered_csv_files):
    # Load CSV file into a NumPy array
    data = np.genfromtxt(csv_file, delimiter=',')
    
    # Verify if the shape is 5x5
    if data.shape == (5, 5):
        # Create a heatmap trace for each 5x5 array
        trace = go.Heatmap(z=data, colorscale='Viridis', showscale=False, name=f'Data {i + 1}')
        fig.add_trace(trace)

# Customize the layout
fig.update_layout(
    title='5x5 Array Visualization',
    xaxis_title='Columns',
    yaxis_title='Rows',
)

# Show the Plotly figure
fig.show()