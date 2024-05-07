## data-implantation (DI) by Francis Onodueze


### Running the algorithm (sample code)
import pandas as pd
from data_implantation_frontier.DI_Frontier_TSR_multiclass import DataImplantation_Frontier

dataset = pd.read_csv("datasets/cleaned.csv")
process = DataImplantation_Frontier() # provide name of your label column; Do you want to separate your testset?
df, _, _ =  process.runDIprocess(dataset, 'FLAG', di_ratio=50) 

#### process = DI_Frontier_TSR(col = "NoTarget", nF=-1, createtest=False) 
-  If you are creating synthetic sample based on another column other than the label column, provide the name of the column. This is used mainly for multi-label classification. Otherwise, just leave the column as 'NoTarget'.
- Do you want the algorithm to do feature selection and choose the best features upon which to create the synthetic samples? Provide the number of features you want selected. Feature selection needs more than 2 features to proceed. Leave nF as -1 if you don't want to do feature selection.
- createtest: Do you want to separate your testset? If True, DI will separate 40% of your dataset as testset and use the remaining 60% for creating synthetic samples. 40% is hardocded for now, this will be made more flexible in later revisions.

#### process.runDIprocess(dataset, labelcolumnname = "label", class_size = 2, di_ratio=100, doDI=True) 
- dataset: is a dataframe of your entire data including the label column
- labelcolumnname: is the name of your label column. Most times its 'label'. Its usually addresses as Y, while the rest of the dataset is X 
- Class size: 2 - is for binary classification
- Percentage of synthetic samples to create relative to the majority sample size. 100% means you want minority and majority samples to be equal in size (for binary classification). For nulti-class classification, 100% means that all samples belonging to separate classes will be approximately the same size.

> [!IMPORTANT]
> The dataset must be saved in .csv format to avoid compactibility issues.
> The data implanted file will be stored in NoTarget folder. NoTarget means you are not targeting any feature during the DI process
> More details will be added here later in subsequent version where I will explain the targeting of a feature and other sections of the code that I have commented out
> For now all datasets are be preprocessed. This will be made optional in the later releases.


> [!NOTE]
> Don't be disappointed if Data Implantation does not give you as many synthetic samples as you wanted.
> DI uses advanced technique to generate samples that are as close to real samples as possible; if it not able to generate perfect samples, it will give you less than expected 
> it will avpid generating outliers in the name of given you the desired quantity of samples you want
> To get more samples, you can try increasing the pecentage of synthetic samples to a number greater than 100% or the intended target


