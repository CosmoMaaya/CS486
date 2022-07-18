# Dunetts Syndrome: DS
# Sloepnea: S
# Foriennditis: F
# Degar Spots: D
# Gene: G


# P(DS, S, F, D, G) = P(D|DS) P(F|DS) P(S|DS, GENE) P(DS) P(G)
import numpy as np

NODATA, NOTPRESENT, MILD, SEVERE = -1, 0, 1, 2
DS_VALUES = [NOTPRESENT, MILD, SEVERE]
FALSE, TRUE = 0, 1

class EM():
    def __init__(self, P_DS, P_G, P_F_DS, P_D_DS, P_S_G_DS) -> None:
        # priors:
        self.P_DS = P_DS
        self.P_G = P_G
        self.P_F_DS = P_F_DS
        self.P_D_DS = P_D_DS
        self.P_S_G_DS = P_S_G_DS

    def train(self, data):
        
        weights_over_data = np.zeros((len(data), len(DS_VALUES)))
        normalized_weights_over_data = np.zeros((len(data), len(DS_VALUES)))
        for index,  (s, f, d, g, ds) in enumerate(data):
            weights = [0,0,0]
            if ds != NODATA:
                weights[ds] = 1
            else:
                for ds_value in DS_VALUES:
                    weights[ds_value] = self.P_D_DS[d, ds_value] * self.P_F_DS[f, ds_value] * self.P_S_G_DS[s, g, ds_value] * self.P_G[g] * self.P_DS[ds_value]
            normalized_weights = [w / sum(weights) for w in weights]
            weights_over_data.append(weights)
            normalized_weights_over_data.append(normalized_weights)
        sum_of_weights = sum(weights_over_data)
        pass