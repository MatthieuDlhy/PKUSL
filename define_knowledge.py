import numpy as np
import pandas as pd
from sktime.datasets._readers_writers.ts import load_from_tsfile

def compute_burst_duration(series, threshold):
    """
    Calcule la durée moyenne des segments contigus où la consommation dépasse le seuil.
    """
    durations = []
    in_burst = False
    burst_length = 0
    for value in series:
        if value > threshold:
            if not in_burst:
                in_burst = True
                burst_length = 1
            else:
                burst_length += 1
        else:
            if in_burst:
                durations.append(burst_length)
                in_burst = False
                burst_length = 0
    # Si la série se termine en burst
    if in_burst:
        durations.append(burst_length)
    return np.mean(durations) if durations else 0

def create_knowledge_features(X):
    """
    À partir d'un tableau X de forme (n_samples, 1, series_length),
    calcule un DataFrame contenant pour chaque série :
      - activeRatio    : proportion de points où la consommation dépasse le 75e percentile
      - burstPeak      : valeur maximale (pic de consommation)
      - powerVariance  : variance de la consommation
      - meanConsumption: consommation moyenne
      - slope          : maximum de la variation absolue entre valeurs consécutives
      - cumulativeSlope: somme des variations absolues (indicatif du changement global)
      - burstDuration  : durée moyenne des segments actifs (au-dessus du seuil)
    """
    n_samples = X.shape[0]
    knowledge_list = []
    for i in range(n_samples):
        series = X[i, 0, :]  # extraction de la série univariée
        threshold = np.percentile(series, 75)  # seuil fixé au 75e percentile
        # active_ratio = np.sum(series > threshold) / len(series)
        burst_peak = np.max(series)
        # power_variance = np.var(series)
        mean_consumption = np.mean(series)
        slope = np.max(np.abs(np.diff(series)))
        cumulative_slope = np.sum(np.abs(np.diff(series)))
        # burst_duration = compute_burst_duration(series, threshold)
        
        features = {
            # 'activeRatio': active_ratio,
            'burstPeak': burst_peak,
            # 'powerVariance': power_variance,
            'meanConsumption': mean_consumption,
            'slope': slope,
            'cumulativeSlope': cumulative_slope,
            # 'burstDuration': burst_duration
        }
        knowledge_list.append(features)
    knowledge_df = pd.DataFrame(knowledge_list)
    return knowledge_df

# Exemple d'utilisation
if __name__ == "__main__":
    # Chemin vers le fichier .ts d'entraînement (à adapter sur votre PC)
    # TRAIN

    train_file_path = '/home/administrateur/Documents/Code/Times series models exploration/PKUSL/data/ACSF1/ACSF1_TRAIN.ts'
    # full_file_path = "/chemin/vers/ACSF1_TRAIN.ts"
    X_train, y_train = load_from_tsfile(train_file_path,
                            replace_missing_vals_with='NaN',
                            return_y=True,
                            return_data_type='numpy3d',
                            encoding='utf-8')
    # Ici X aura la forme (n_samples, 1, series_length), par exemple (100, 1, 1460)
    knowledge_df_train = create_knowledge_features(X_train)
    print("Exemple de features extraites :")
    print(knowledge_df_train.head())
    knowledge_df_train.to_csv("/home/administrateur/Documents/Code/Times series models exploration/PKUSL/data/ACSF1/knowledge_features_train.csv", index=False)
    print("Les features knowledge ont été sauvegardées dans 'knowledge_features_train.csv'.")

    # # TEST
    # test_file_path = '/home/administrateur/Documents/Code/Times series models exploration/PKUSL/data/ACSF1/ACSF1_TEST.ts'
    # # full_file_path = "/chemin/vers/ACSF1_TRAIN.ts"
    # X_test, y_test = load_from_tsfile(test_file_path,
    #                         replace_missing_vals_with='NaN',
    #                         return_y=True,
    #                         return_data_type='numpy3d',
    #                         encoding='utf-8')
    # # Ici X aura la forme (n_samples, 1, series_length), par exemple (100, 1, 1460)
    # knowledge_df_test = create_knowledge_features(X_test)
    # print("Exemple de features extraites :")
    # print(knowledge_df_test.head())
    # knowledge_df_test.to_csv("/home/administrateur/Documents/Code/Times series models exploration/PKUSL/data/ACSF1/knowledge_features_test.csv", index=False)
    # print("Les features knowledge ont été sauvegardées dans 'knowledge_features_test.csv'.")
