# Carsite_AGan
This is a novel two-way rebalancing strategy based on the attention technique and generative adversarial network
(Carsite_AGan) for identifying protein carbonylation sites.

# Requirements
* Python = 3.7.0

# Implementation
If you have some carbonylation sites data that would like to identify. Firstly you should download the pse-in-one-2.0 
at http://bliulab.net/Pse-in-One2.0/download/ and use it to convert the protein carbonylation and non-carbonylation 
sequences data into valid numerical vector data. For example:
"python ./Pse-in-One-2.0/nac.py ./618K_positive.txt Protein DR -max_dis 3 -f tab -labels 0 -out DRfeature.txt"

Then, you should label these vector data and replace the path "path = ‘dataset/DRK.txt’" in make_dataset.py to make your 
data available in our strategy code. You could use SVM or a decision tree as a classifier and replace the path 
"path2 = 'dataset/DRK_val.h5' " in main.py to run our strategy.

Finally, "python main.py"
