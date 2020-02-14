# What the files do:

1. ‘Parsed_records_dump_20200101.p’:
    Raw metadata with information about the images

2. Metadata_cleaning.ipynb
    Load the the panda dataframe ‘Parsed_records_dump_20200101.p’ with the raw metadata
    Select the relevant product categories (only the fashion categories are in scope)
    Keep the features which have feature occurrences >1000
    Check, whether a valid path is available for all images - delete the rows, if no picture available
    Save the cleaned dataframe as a csv here: 1_data_cleaning/metadata_cleaned.csv

3. Metadata_cleaning_v2.ipynb
    Load the the panda dataframe ‘Parsed_records_dump_20200101.p’ with the raw metadata
    Select the relevant product categories (only the fashion categories are in scope)
    Keep the features which have feature occurrences >500
    Check, whether a valid path is available for all images - - delete the rows, if no picture available
    Add the column "hierarchy2" to the columns to be extracted, in order to use it for the multiple-input model
    Save the cleaned dataframe as a csv here: 1_data_cleaning/metadata_cleaned2.csv

4. Metadata_cleaning_v3.ipynb
    Load the the panda dataframe ‘Parsed_records_dump_20200101.p’ with the raw metadata
    Select the relevant product categories (only the fashion categories are in scope)
    Keep all the existing features (no feature pre-selection)
    Check, whether a valid path is available for all images - delete the rows, if no picture available
    Add the column "hierarchy2" to the columns to be extracted, in order to use it for the multiple-input model
    Save the cleaned dataframe as a csv here: 1_data_cleaning/metadata_cleaned3.csv

5. metadata_cleaned.csv
    Output file of "metadata_cleaning.ipynb"

6. metadata_cleaned2.csv
    Output file of "metadata_cleaning_v2.ipynb"

7. metadata_cleaned3.csv
    Output file of "metadata_cleaning_v3.ipynb"