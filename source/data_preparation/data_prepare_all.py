""" Data preparation script for masif_site dataset. """

import os
import subprocess
from concurrent.futures import ThreadPoolExecutor
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from default_config.masif_opts import masif_opts


def run_script(protein_name):
    """
    Function to run the data_prepare_one.sh script with a given protein name.
    """
    # Construct the command to execute the shell script with the protein name
    command = ["./data_prepare_one.sh", protein_name]

    # Execute the command
    subprocess.run(command, check=True)


def main():
    # Path to the file containing the list of protein names
    protein_list_file = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "..",
            "..",
            "data",
            "masif_site",
            "lists",
            "masif_site_only.txt",
        )
    )

    # Get the list of proteins already prepared
    prepared_proteins = os.listdir(masif_opts["site"]["masif_precomputation_dir"])

    # Read the list of proteins from the file
    protein_names = []
    with open(protein_list_file, "r") as file:
        for protein in [line.strip() for line in file.readlines()]:
            if protein not in prepared_proteins:
                protein_names.append(protein)
            else:
                print(f"Skipping protein {protein} as it is already prepared.")

    # Number of threads to use
    num_threads = 16

    # Create a ThreadPoolExecutor to run the script in parallel
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        # Submit tasks to the executor for each protein name
        futures = [
            executor.submit(run_script, protein_name) for protein_name in protein_names
        ]

        # Wait for all tasks to complete (optional, for error handling or logging)
        for future in futures:
            try:
                future.result()  # This will re-raise any exception caught during the execution
            except subprocess.CalledProcessError as e:
                print(f"Error executing script for a protein: {e}")


if __name__ == "__main__":
    main()
