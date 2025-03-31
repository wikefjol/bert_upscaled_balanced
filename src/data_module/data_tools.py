import pandas as pd
from Bio import SeqIO
import random

def parse_fasta_header(description):
    taxonomy_levels = description.split('|')[-1].split(';')
    taxonomy_dict = {}

    for level in taxonomy_levels:
        if '__' in level:
            key, value = level.split('__')
            taxonomy_dict[key] = value
    return taxonomy_dict

def fasta2pandas(fasta_path):

    records = []

    for record in SeqIO.parse(fasta_path, 'fasta'):

        taxonomy_dict = parse_fasta_header(record.description)

        entry = {
            'kingdom': taxonomy_dict.get('k', 'Unknown'),
            'phylum': taxonomy_dict.get('p', 'Unknown'),
            'class': taxonomy_dict.get('c', 'Unknown'),
            'order': taxonomy_dict.get('o', 'Unknown'),
            'family': taxonomy_dict.get('f', 'Unknown'),
            'genus': taxonomy_dict.get('g', 'Unknown'),
            'species': taxonomy_dict.get('s', 'Unknown'),
            'sequence': str(record.seq)
        }

        records.append(entry)

    return pd.DataFrame(records)

def handle_dna_ambiguity(df, method='pop'):
    """
    Processes DNA strings in a DataFrame based on the mode: 'pop', 'prob', or 'placeholder'.
    
    Parameters:
    df: Pandas DataFrame containing the DNA sequences
    method: Specifies which operation to perform:
        - 'pop': Keep only rows with sequences that contain no ambiguity codes.
        - 'prob': Replace ambiguity codes with random valid bases in the DNA column.
        - 'placeholder': Replace ambiguity codes with 'X' in the DNA column.
    
    Returns:
    A new DataFrame with the DNA column processed according to the selected method.
    """

    standard_bases = {'A', 'C', 'T', 'G'}

    ambiguity_mapping = {
    'A': ['A'], 'T': ['T'], 'C': ['C'], 'G': ['G'],   # Standard bases
    'N': ['A', 'T', 'C', 'G'],  # Any base
    'R': ['A', 'G'],            # Purine
    'Y': ['C', 'T'],            # Pyrimidine
    'K': ['G', 'T'],            # Keto
    'M': ['A', 'C'],            # Amino
    'S': ['G', 'C'],            # Strong interaction
    'W': ['A', 'T'],            # Weak interaction
    'B': ['C', 'G', 'T'],       # Not A
    'D': ['A', 'G', 'T'],       # Not C
    'H': ['A', 'C', 'T'],       # Not G
    'V': ['A', 'C', 'G'],       # Not T
}

    # List to store indices for pop and fixed sequences for prob/placeholder
    valid_indices = []
    fixed_sequences = []

    # Iterate through the DataFrame and process each DNA sequence
    print("Handling DNA ambiguity codes...")
    for index, row in df.iterrows():
        # Print every 10000 rows to track progress
        if index % 10000 == 0:
            print(f'Processing row {index}...')
            
        sequence = row["sequence"]

        # 'pop' method: Track rows that do not contain ambiguity codes
        if method == 'pop':
            if all(base in standard_bases for base in sequence):
                valid_indices.append(index)  # Add index if no ambiguity

        # 'prob' method: Replace ambiguity codes with random valid bases
        elif method == 'prob':
            new_sequence = ''.join(random.choice(ambiguity_mapping.get(base, [base])) for base in sequence)
            fixed_sequences.append(new_sequence)

        # 'placeholder' method: Replace ambiguity codes with 'X'
        elif method == 'placeholder':
            new_sequence = ''.join(base if base in standard_bases else 'X' for base in sequence)
            fixed_sequences.append(new_sequence)

    # If 'pop' is active, return the DataFrame with only valid indices
    if method == 'pop':
        print("Processing complete.")
        return df.loc[valid_indices].reset_index(drop=True)

    # If 'prob' or 'placeholder' is active, replace the sequences and return the modified DataFrame
    if method in ['prob', 'placeholder']:
        df['sequence'] = fixed_sequences

        print("Processing complete.")
        print()
        
        return df

def filter_taxonomy(df, kingdomToInclude=None,
                    phylumToInclude=None,
                    classToInclude=None,
                    orderToInclude=None,
                    familyToInclude=None,
                    genusToInclude=None,
                    speciesToInclude=None,
                    kingdomNotToInclude=None,
                    phylumNotToInclude=None,
                    classNotToInclude=None,
                    orderNotToInclude=None,
                    familyNotToInclude=None,
                    genusNotToInclude=None,
                    speciesNotToInclude=None,
                    startAt='kingdom', 
                    endAt='species',
                    genusCertainty=False,
                    speciesCertainty=False,
                    familyCertainty=False,
                    orderCertainty=False,
                    classCertainty=False,
                    phylumCertainty=False,
                    ambiguityMethod='pop',
                    max_observations=float('inf')):
    """
    Filters the DataFrame based on taxonomy inclusion and exclusion criteria, and then drops columns outside the range between `startAt` and `endAt`.

    Filtering happens independently from `startAt` and `endAt`. These parameters only affect the columns returned.

    Parameters:
    df: DataFrame to filter
    kingdomToInclude: List of kingdoms to include
    phylumToInclude: List of phyla to include
    classToInclude: List of classes to include
    orderToInclude: List of orders to include
    familyToInclude: List of families to include
    genusToInclude: List of genera to include
    speciesToInclude: List of species to include
    kingdomNotToInclude: List of kingdoms to exclude
    phylumNotToInclude: List of phyla to exclude
    classNotToInclude: List of classes to exclude
    orderNotToInclude: List of orders to exclude
    familyNotToInclude: List of families to exclude
    genusNotToInclude: List of genera to exclude
    speciesNotToInclude: List of species to exclude
    startAt: The highest taxonomy level to include in the output (default is 'kingdom')
    endAt: The lowest taxonomy level to include in the output (default is 'species')
    genusCertainty: If True, exclude rows where genus ends with "_gen_Incertae_sedis"
    speciesCertainty: If True, exclude rows where species ends with "_sp"
    familyCertainty: If True, exclude rows where family ends with "_fam_Incertae_sedis"
    orderCertainty: If True, exclude rows where order ends with "_ord_Incertae_sedis"
    classCertainty: If True, exclude rows where class ends with "_cls_Incertae_sedis"
    phylumCertainty: If True, exclude rows where phylum ends with "_phy_Incertae_sedis"

    Returns:
    A filtered DataFrame with columns only between the `startAt` and `endAt` levels.
    """

    # Dictionary of taxonomic levels to include and exclude filters
    include_filters = {
        'kingdom': kingdomToInclude,
        'phylum': phylumToInclude,
        'class': classToInclude,
        'order': orderToInclude,
        'family': familyToInclude,
        'genus': genusToInclude,
        'species': speciesToInclude
    }

    exclude_filters = {
        'kingdom': kingdomNotToInclude,
        'phylum': phylumNotToInclude,
        'class': classNotToInclude,
        'order': orderNotToInclude,
        'family': familyNotToInclude,
        'genus': genusNotToInclude,
        'species': speciesNotToInclude
    }

    # Define the order of taxonomy levels
    taxonomy_levels = ['kingdom', 'phylum', 'class', 'order', 'family', 'genus', 'species']

    print("Applying filters...")
    # Step 1: Apply filtering across all levels based on inclusion/exclusion criteria
    for level in taxonomy_levels:
        if level in df.columns:
            # Apply inclusion filter if specified
            if include_filters[level] is not None:
                df = df[df[level].isin(include_filters[level])]
            
            # Apply exclusion filter if specified
            if exclude_filters[level] is not None:
                df = df[~df[level].isin(exclude_filters[level])]

    # Step 2: Apply genusCertainty, speciesCertainty, familyCertainty, orderCertainty, classCertainty, and phylumCertainty filters
    if genusCertainty:
        df = df[~df['genus'].str.endswith('_gen_Incertae_sedis')]
    
    if speciesCertainty:
        df = df[~df['species'].str.endswith('_sp')]

    if familyCertainty:
        df = df[~df['family'].str.endswith('_fam_Incertae_sedis')]

    if orderCertainty:
        df = df[~df['order'].str.endswith('_ord_Incertae_sedis')]

    if classCertainty:
        df = df[~df['class'].str.endswith('_cls_Incertae_sedis')]

    if phylumCertainty:
        df = df[~df['phylum'].str.endswith('_phy_Incertae_sedis')]

    # Step 3: Drop columns before `startAt` and after `endAt`
    start_index = taxonomy_levels.index(startAt)
    end_index = taxonomy_levels.index(endAt) + 1  # +1 to include the `endAt` level

    # Only keep columns between startAt and endAt + 'sequence' column
    columns_to_keep = taxonomy_levels[start_index:end_index] + ['sequence']
    df = df[columns_to_keep]

    print("Filtering complete.")
    print()

    df = handle_dna_ambiguity(df, method=ambiguityMethod)

    number_of_observations = len(df)
    df = df[:min(number_of_observations, max_observations)]

    return df