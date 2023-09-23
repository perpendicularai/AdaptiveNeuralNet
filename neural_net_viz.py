#### Please ensure that you run he neural_net_adaptive.py before running this script.                                                                #
#### Once the neural_net_apdaptive.py script is run, you need to load the data below and store in the data variable                                  #
#### Each file will be store as a csv.                                                                                                               #
#### Please ensure you have sufficient disk space before running if configured to compute a large amount of samples and iterations.                  #
#### #################################################################################################################################################
df = pd.DataFrame(data)

# Create a figure and axis
fig, ax = plt.subplots(figsize=(8, 6))

# Plot the data as a line plot
for column in df.columns:
    plt.plot(df.index, df[column], label=column)

# Add data labels to the data points when hovering
for column in df.columns:
    for i, val in enumerate(df[column]):
        plt.annotate(f'{val:.3f}', (df.index[i], val), textcoords="offset points", xytext=(0, 10), ha='center')

# Customize the plot
plt.title('3x3 Neural Net')
plt.xlabel('Index')
plt.ylabel('Values')
plt.legend()
plt.grid(True)

# Show the plot
plt.show()
