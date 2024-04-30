## data-implantation by Francis Onodueze


### Running the algorithm

#### process = DI_Frontier_TSR('NoTarget', -1, 'FLAG', False) 
- provide name of your label column; 
- Do you want the algorithm to do feature selection and choose the best features upon which to create the synthetic samples? Provide the number of features you want selected. -i for no feature selection. Feature selection needs more than 2 features to proceed.
- Do you want to separate your testset?
- Provide name of your label column. Your label column name will be with **label**. This will be fixed in later versions
- Do you want to separate your testset? If yes DI will separate 40% of your dataset as testset and use the remaining 60% for creating synthetic samples. 40% is hardocded for now, will be fixed in late revisions

#### process.runDIprocess(2, 100) 
- Class size. 2 - is for binary classification
- Percentage of synthetic samples to create relative to the majority sample size. 100% means you want minority and majority samples to be equal in size (for binary classification). For nulti-class classification, 100% means that all samples belonging to separate classes will be approximately the same size.

> [!NOTE]
> Don't be disappointed if Data Implantation does not give you as many synthetic samples as you wanted.
> DI uses advanced technique to generate samples that are as close to real samples as possible; if it not able to generate perfect samples, it will give you less than expected 
> it will avpid generating outliers in the name of given you the desired quantity of samples you want
> To get more samples, you can try increasing the pecentage of synthetic samples to a number greater than 100% or the intended target
