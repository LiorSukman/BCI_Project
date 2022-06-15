from src.data_loader import DataLoader

if __name__ == '__main__':
    data = DataLoader()
    data.prepare_data()
    data.features_distributions()
    data.features_correlation()
    data.pairs_plot()
