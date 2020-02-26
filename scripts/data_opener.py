import pickle




with open('data/UTdf.all_days', 'rb') as f:
    time_series_data = pickle.load(f)
f.close()

with open('data/max_corr_df.correlations', 'rb') as f:
    max_corr = pickle.load(f)
f.close()

with open('data/final_df.correlations', 'rb') as f:
    correlations = pickle.load(f)
f.close()


with open('data/numpy_data.matrices', 'rb') as f:
    wind = pickle.load(f)
f.close()

# before was named cn2_file1.df is actually
with open('data/cn2_file2.df', 'rb') as f:
    cn2_file2 = pickle.load(f)
f.close()


with open('data/cn2_file1.df', 'rb') as f:
    cn2_file1 = pickle.load(f)
f.close()

