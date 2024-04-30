## data-implantation (DI) by Francis Onodueze


### Running the algorithm
Program execution begins by calling the below two functions from the DI_Frontier_TSR_multiclass.py file. The functions are already part of the file.

#### process = DI_Frontier_TSR('NoTarget', -1, 'Label', False) 
- Do you want the algorithm to do feature selection and choose the best features upon which to create the synthetic samples? If No, just leave the parameter as 'NoTarget'.
- Provide the number of features you want selected. Feature selection needs more than 2 features to proceed. Leave as -1 if you don't want to do feature selection.
- Provide name of your label column. Your label column name will be with **label**. This will be fixed in later versions
- Do you want to separate your testset? If True, DI will separate 40% of your dataset as testset and use the remaining 60% for creating synthetic samples. 40% is hardocded for now, this will be made more flexible in later revisions.

#### process.runDIprocess(2, 100) 
- Class size. 2 - is for binary classification
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


