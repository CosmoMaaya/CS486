from itertools import product
from functools import reduce
from operator import getitem
import numpy as np

# np does not supprt using True/False directly as index
F = 0
T = 1
VAR_VALUES = (T, F)

# For printing purpose
VAL_STR_MAPPING = {
    F: "F",
    T: "T",
}

OC = "OC"
Fraud = "Fraud"
Trav = "Trav"
FP = "FP"
IP = "IP"
CRP = "CRP"
# A Factor is represented as an N-D array
# E.g: f_OC_fraud, first dimension is for OC
class Factor():
    def __init__(self, variables: list):
        if type(variables) != list:
            variables = [variables]
        self.var_list = variables
        self.var_mapping = {}
        for i, var in enumerate(variables):
            self.var_mapping[var] = i
        self.dim = len(variables)
        self.factor = np.ndarray([2] * self.dim)

    def initializeValues(self, values):
        if type(values) != list:
            values = [values]
        if len(values) != 2**self.dim:
            raise ValueError("Wrong size of values")
        
        iterations = [list(tup) for tup in product(VAR_VALUES, repeat=self.dim)]
        for var_vals, factor_val in zip(iterations, values):
            self[var_vals] = factor_val
    
    def __getitem__(self, indices):
        if indices == T or indices == F:
            return self.factor[indices]
        if type(indices) == list:
            return self.factor.__getitem__(tuple(indices))
        if type(indices) == tuple:
            return self.factor.__getitem__(indices)
        
        raise ValueError()
    
    def __setitem__(self, indices, value):
        if indices == T or indices == F:
            self.factor[indices] = value
            return
        if type(indices) == list:
            self.factor.__setitem__(tuple(indices), value)
            return
        if type(indices) == tuple:
            self.factor.__setitem__(indices, value)
            return
        
        raise ValueError()
    
    # For debug printing purpose
    def __str__(self):
        iterations = [list(tup) for tup in product(VAR_VALUES, repeat=self.dim)]
        s = '\t'.join(self.var_list) + '\t' + 'f\n'
        for vals in iterations:
            val_names = [VAL_STR_MAPPING[v] for v in vals]
            s += '\t'.join(val_names) + '\t' + str(self[vals]) + '\n'
        return s

# target_list_: a list of values
# ind_val_to_be_inserted: a list of tuples, 
# the first element of the tuple is the index you want to insert the element at, 
# the second element is the value of the element you want to insert 
def insertElements(target_list_, ind_val_to_be_inserted):
    target_list = target_list_.copy()
    # Have to sort to make the algo work
    ind_val_to_be_inserted.sort()
    for idx, val in ind_val_to_be_inserted:
        target_list.insert(idx, val)
    return target_list

def restrict(factor, restricted_variables, restricted_values):
    if type(restricted_variables) != list:
        restricted_variables = [restricted_variables]
        restricted_values = [restricted_values]
    
    # After the restriction, the new factor should not contain variables that are restricted
    new_vars = [var for var in factor.var_list if var not in restricted_variables]
    new_factor = Factor(new_vars)
    removed_pos_val = [(factor.var_mapping[var], val) for var, val in zip(restricted_variables, restricted_values)]
    # Generate all possible values assigned to variables for iteration purpose
    index_val_list = [list(tup) for tup in product(VAR_VALUES, repeat=factor.dim-len(restricted_variables))]

    for index_val in index_val_list:
        old_index_val = insertElements(index_val, removed_pos_val)
        new_factor[index_val] = factor[old_index_val]
    return new_factor

def multiply(factor1, factor2):
    # Find out the common variables betwee f1 and f2, and their corresponding dim/index/position in each factor
    variables1_set = set(factor1.var_list)
    common_var = [var for var in factor2.var_list if var in variables1_set]
    common_var_f1_idxs = [factor1.var_mapping[var] for var in common_var]
    common_var_f2_idxs = [factor2.var_mapping[var] for var in common_var]
    common_var_f1_idxs.sort()

    # Generate all variables in the resulting factor
    # By default, the order is given by all vars in f1 plus whatever left in f2
    new_vars = [ var for var in factor2.var_list if var not in variables1_set]
    new_vars = factor1.var_list + new_vars

    # Generate all assignments of variables in f1 for iteration
    f1_index_val_list = [list(tup) for tup in product(VAR_VALUES, repeat=factor1.dim)]

    new_factor = Factor(new_vars)

    for f1_index_val in f1_index_val_list:
        f1_factor_val = factor1[f1_index_val]

        # Generate common variable assignments according to the assignments in f1 in current iteration
        common_var_vals = [(factor1.var_list[var_idx], f1_index_val[var_idx]) for var_idx in common_var_f1_idxs]
        common_pos_index_vals_f2 = [(factor2.var_mapping[var], val) for var, val in common_var_vals]
        # Generate all possible assignments of uncommon variables in f2, order preserved
        f2_addtional_index_val_list = [list(tup) for tup in product(VAR_VALUES, repeat=factor2.dim - len(common_var))]
        for additional_index_val in f2_addtional_index_val_list:
            f2_index_val = insertElements(additional_index_val, common_pos_index_vals_f2)
            new_factor[f1_index_val + additional_index_val] = f1_factor_val * factor2[f2_index_val]

    return new_factor

def summout(factor, variables):
    if type(variables) != list:
        variables = [variables]
    summout_axis = [ factor.var_mapping[var] for var in variables ]
    new_vars = [var for var in factor.var_list if var not in variables]
    new_factor = Factor(new_vars)
    new_factor.factor = np.sum(factor.factor, axis=tuple(summout_axis))
    return new_factor

def normalize(factor):
    new_factor = Factor(factor.var_list)
    new_factor.factor = np.array(factor.factor, copy=True)
    total = np.sum(new_factor.factor)
    index_val_list = [list(tup) for tup in product(VAR_VALUES, repeat=new_factor.dim)]
    for index_val in index_val_list:
        new_factor[index_val] = new_factor[index_val] / total
    return new_factor

def inference(factorList, queryVariables, orderedListOfHiddenVariables, evidenceList):
    # make a copy so we don't modify the reference
    factorList = factorList.copy()
    # Restrict each variable
    for (var, val) in evidenceList:
        for i, factor in enumerate(factorList):
            # Only proceed restriction when the evidence variable is present in the factor
            if var in factor.var_mapping.keys():
                restricted_factor = restrict(factor, var, val)
                if len(restricted_factor.var_list) == 0:
                    # after the restriction, the factor becomes a constant
                    # Then we don't have to worry about it, just remove the factor from the list
                    factorList[i] = None
                else:
                    factorList[i] = restricted_factor
    
    factorList = [f for f in factorList if f is not None]

    for elimiating_var in orderedListOfHiddenVariables:
        summing_factors = []
        for factor in factorList:
            if elimiating_var in factor.var_mapping.keys():
                summing_factors.append(factor)
        
        for summing_factor in summing_factors:
            factorList.remove(summing_factor)
        
        cur_idx = 1
        cur_factor = summing_factors[0]
        while cur_idx < len(summing_factors):
            cur_factor = multiply(cur_factor, summing_factors[cur_idx])
            cur_idx += 1
        
        res_factor = summout(cur_factor, elimiating_var)
        print(res_factor)
        factorList.append(res_factor)

    cur_idx = 1
    cur_factor = factorList[0]
    while cur_idx < len(factorList):
        cur_factor = multiply(cur_factor, factorList[cur_idx])
        cur_idx += 1
    print(cur_factor)
    final_factor = normalize(cur_factor)
    return final_factor

############################################
#       SOLVING
############################################
f0_Trav = Factor(Trav)
values = [0.05, 0.95]
f0_Trav.initializeValues(values)

f1_Trav_Fraud = Factor([Trav, Fraud])
values = [0.01, 0.99, 0.004, 0.996]
f1_Trav_Fraud.initializeValues(values)

f2_Trav_Fard_FP = Factor([Trav, Fraud, FP])
values = [0.9, 0.1, 0.9, 0.1, 0.1, 0.9, 0.01, 0.99]
f2_Trav_Fard_FP.initializeValues(values)

f3_OC_Fraud_IP = Factor([OC, Fraud, IP])
values = [0.02, 0.98, 0.01, 0.99, 0.011, 0.989, 0.001, 0.999]
f3_OC_Fraud_IP.initializeValues(values)

f4_OC = Factor(OC)
f4_OC[T] = 0.6
f4_OC[F] = 0.4

f5_OC_CRP = Factor([OC, CRP])
values = [0.1, 0.9, 0.001, 0.999]
f5_OC_CRP.initializeValues(values)

factorList = [f0_Trav, f1_Trav_Fraud, f2_Trav_Fard_FP, f3_OC_Fraud_IP, f4_OC, f5_OC_CRP]

res = inference(factorList, [Fraud], [Trav, FP, IP, OC, CRP], [])
print(res)
#############################################
# RESTRICTION TEST
# factor = Factor(["X", "Y", "Z"])
# iterations = [list(tup) for tup in product(VAR_VALUES, repeat=factor.dim)]
# values = [0.1, 0.9, 0.2, 0.8, 0.4, 0.6, 0.3, 0.7]
# for var_vals, factor_val in zip(iterations, values):
#     factor[var_vals] = factor_val
# print(factor)

# new_factor = restrict(factor, "X", T)
# print(new_factor)

# new_factor = restrict(factor, ["X", "Z"], [T, F])
# print(new_factor)

# new_factor = restrict(factor, ["X", "Y", "Z"], [T, F, F])
# print(new_factor)
#############################################

#############################################
# PRODUCT TEST
# factor1 = Factor([OC, Fraud])
# factor2 = Factor([Fraud, Trav])
# iterations = [list(tup) for tup in product(VAR_VALUES, repeat=2)]
# values1 = [0.1, 0.9, 0.2, 0.8]
# values2 = [0.3, 0.7, 0.6, 0.4]

# for var_vals, factor1_val, factor2_val in zip(iterations, values1, values2):
#     factor1[var_vals] = factor1_val
#     factor2[var_vals] = factor2_val
# print(factor1)
# print(factor2)

# new_factor = multiply(factor1, factor2)
# print(new_factor)
#####################################

######################################
# SUM TEST
# factor = Factor(["A", "B", "C"])
# iterations = [list(tup) for tup in product(VAR_VALUES, repeat=factor.dim)]
# values = [0.03, 0.07, 0.54, 0.36, 0.06, 0.14, 0.48, 0.32]
# for var_vals, factor_val in zip(iterations, values):
#     factor[var_vals] = factor_val
# print(factor)
# new_factor = summout(factor, "B")

# print(new_factor)
######################################

######################################
# NORMALIZATION TEST
# factor = Factor(["F"])
# iterations = [list(tup) for tup in product(VAR_VALUES, repeat=factor.dim)]
# values = [0.0258, 0.0708]
# for var_vals, factor_val in zip(iterations, values):
#     factor[var_vals] = factor_val
# print(factor)
# new_factor = normalize(factor)

# print(new_factor)
######################################


######################################
# INTEGRATION TEST
# f0_A = Factor("A")
# f0_A[T] = 0.3
# f0_A[F] = 0.7

# f1_AC = Factor(["A", "C"])
# values = [0.8, 0.2, 0.15, 0.85]
# f1_AC.initializeValues(values)

# f2_CG = Factor(["C", "G"])
# values = [1.0, 0.0, 0.2, 0.8]
# f2_CG.initializeValues(values)

# f3_GL = Factor(["G", "L"])
# values = [0.7, 0.3, 0.2, 0.8]
# f3_GL.initializeValues(values)

# f4_SL = Factor(["S", "L"])
# values = [0.9, 0.3, 0.1, 0.7]
# f4_SL.initializeValues(values)

# factorList = [f0_A, f1_AC, f2_CG, f3_GL, f4_SL]
# final = inference(factorList, ["S"], ["A", "G", "C", "L"], [])
# print(final)
# final = inference(factorList, ["S"], ["L","C","G"], [("A", T)])
# print(final)
# final = inference(factorList, ["C"], ["L","G","A"], [("S", T)])
# print(final)
######################################


######################################
# INTEGRATION TEST 2
# f0_C = Factor("C")
# f0_C[T] = 0.32
# f0_C[F] = 0.68

# f0_M = Factor("M")
# f0_M[T] = 0.08
# f0_M[F] = 0.92

# f1_MCB = Factor(["M", "C", "B"])
# values = [0.61, 0.39, 0.52, 0.48, 0.78, 0.22, 0.044, 0.956]
# f1_MCB.initializeValues(values)

# f2_RB = Factor(["R", "B"])
# values = [0.98, 0.01, 0.02, 0.99]
# f2_RB.initializeValues(values)

# f3_RD = Factor(["R", "D"])
# values = [0.96, 0.04, 0.001, 0.999]
# f3_RD.initializeValues(values)

# f4_AC = Factor(["A", "C"])
# values = [0.8, 0.15, 0.2, 0.85]
# f4_AC.initializeValues(values)

# factorList = [f0_C, f0_M, f1_MCB, f2_RB, f3_RD, f4_AC]
# final = inference(factorList, ["C"], ["M","B","R","A", "D"], [])
# print(final)
# final = inference(factorList, ["C"], ["M","B","R","A"], [("D", T)])
# print(final)
# final = inference(factorList, ["C"], ["M","B","R"], [("D", T), ("A", T)])
# print(final)
# final2 = inference(factorList, ["C"], ["M","B","R"], [("D", T), ("A", F)])
# print(final2)
# final2 = inference(factorList, ["M"], ["C","B","R"], [("D", T), ("A", F)])
# print(final2)
######################################

######################################
# INTEGRATION TEST 3
# f0_C = Factor("C")
# f0_C[T] = 0.0001
# f0_C[F] = 0.9999

# f0_F = Factor("F")
# f0_F[T] = 0.1
# f0_F[F] = 0.9

# f1_JCF = Factor(["C", "F", "J"])
# values = [0.95, 0.05, 0.99, 0.01, 0.99, 0.01, 0.01, 0.99]
# f1_JCF.initializeValues(values)

# factorList = [f0_C, f0_F, f1_JCF]
# final = inference(factorList, ["F"], ["C"], [('J', T)])
# print(final)
######################################


# iterations = [list(tup) for tup in product(VAR_VALUES, repeat=2)]
# print(iterations)
# var_vals = iterations[0]
# print(var_vals)
# last_array = reduce(getitem, var_vals[:-1], factor)
# print(last_array)

# a = np.ndarray([2] * 3)
# print(a)
# # print(a[F, F, F])
# # D = [F, F, F]
# # a.__setitem__(tuple(D), 1)
# # print(a.__getitem__(tuple(D)))
# k = list(product(VAR_VALUES, repeat=3))
# print(list(k))
# for i, val in enumerate(k):
#     a[val] = i
######################################


# iterations = [list(tup) for tup in product(VAR_VALUES, repeat=2)]
# print(iterations)
# var_vals = iterations[0]
# print(var_vals)
# last_array = reduce(getitem, var_vals[:-1], factor)
# print(last_array)

# a = np.ndarray([2] * 3)
# print(a)
# # print(a[F, F, F])
# # D = [F, F, F]
# # a.__setitem__(tuple(D), 1)
# # print(a.__getitem__(tuple(D)))
# k = list(product(VAR_VALUES, repeat=3))
# print(list(k))
# for i, val in enumerate(k):
#     a[val] = i
# print(a)
# print(np.sum(a, axis=(2,0)))
# # print(a)