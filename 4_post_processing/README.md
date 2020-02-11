
What the files do:

1. Feature_mapping_to_Globus_attribute_groups.ipynb:
   Generates the dataframe with all the globus attribute group names
   Loads the features list of the output layers
   Maps the features to the corresponding globus attribute group names.
   Saves the outputs in 3 files, depending the status of the mapping:
       - features_mapped_to_globues_attribute_groups.csv
       - features_with_0_attribute_groups.csv
       - features_with_multiple_attribute_groups.csv
    
2. features.csv:
   list of features applied in the output layer 


3. features_mapped_to_globues_attribute_groups.csv
   output of the "Feature_mapping_to_Globus_attribute_groups.ipynb'
   Include the mapping of the output features to a Globus attribute group name, in case
   the mapping is 1:1 possible. Other cases need further investigation.


4. features_with_0_attribute_groups.csv
   Output with the list of features, where the mapping was not possible, because no Globus attribute group was found.
 

5. features_with_multiple_attribute_groups.csv
    Output with the list of features, where the mapping was not possible, because multiple Globus attribute group were found.
    Further investigation needed, in order to get 1:1 mapping.


