import subprocess
from concurrent.futures import ThreadPoolExecutor

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
    protein_list_file = 'lists/masif_site_only.txt'
    
    # Read the list of proteins from the file
    with open(protein_list_file, 'r') as file:
        protein_names = [line.strip() for line in file.readlines()]
    
    # Number of threads to use
    num_threads = 16
    
    # Create a ThreadPoolExecutor to run the script in parallel
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        # Submit tasks to the executor for each protein name
        futures = [executor.submit(run_script, protein_name) for protein_name in protein_names]
        
        # Wait for all tasks to complete (optional, for error handling or logging)
        for future in futures:
            try:
                future.result()  # This will re-raise any exception caught during the execution
            except subprocess.CalledProcessError as e:
                print(f"Error executing script for a protein: {e}")

if __name__ == "__main__":
    main()