import pandas as pd
from data_implantation_frontier.DI_Frontier_TSR_multiclass import DataImplantation_Frontier

dataset = pd.read_csv("datasets/cleaned.csv")
process = DataImplantation_Frontier() # provide name of your label column; Do you want to separate your testset?
df, _, _ =  process.runDIprocess(dataset, 'FLAG', di_ratio=100) # _ are for none needed return values