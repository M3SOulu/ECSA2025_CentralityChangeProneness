import os
import subprocess
import shutil

UND_PATH = "/Applications/Understand.app/Contents/MacOS/und"


def run_understand(project_path, output_dir):
    """Run SciTools Understand on the given project path and output results to a CSV file."""
    project_name = os.path.basename(project_path)
    print(f"Inspecting {project_name}")

    # Create Understand project
    udb_path = os.path.join(output_dir, f"{project_name}")
    udb_file = os.path.join(output_dir, f"{project_name}.und")
    try:
        print(f"\t Creating Understand Project at {udb_path}")
        subprocess.run([UND_PATH, "create", "-languages", "all", udb_path], check=True)
        print(f"\t Adding Projects File... from path {project_path}")
        subprocess.run([UND_PATH, "add", project_path, udb_file], check=True)
        #print(f"\t Setting generation of all metrics")
        #subprocess.run([UND_PATH, "settings", "-metrics", "all", udb_file], check=True)
        print(f"\t Analyzing Project...")
        subprocess.run([UND_PATH, "analyze", udb_file], check=True)
        print(f"\t ...Done")

        # Extract metrics
        print(f"\t Generating Metrics")
        result = subprocess.run([UND_PATH, "metrics", udb_file], capture_output=True, text=True, check=True)
        print(result.stdout)
        print(f"\t Metrics Available at {udb_file}")

    except subprocess.CalledProcessError as e:
        print(f"Error while processing {project_name}: {e}")

    # Clean up .udb file
    shutil.rmtree(udb_file, ignore_errors=True)


def main(master_folder):

    project_folders = [f.path for f in os.scandir(master_folder) if f.is_dir()]
    output_folders = [os.path.join(os.getcwd(), "raw_data", "understand", f"{os.path.basename(f)}") for f in project_folders]

    for i, (project, output) in enumerate(zip(project_folders, output_folders), start=1):
        print(f"Progress: {i}/{len(project_folders)}")
        run_understand(project, output)
        print("----------- \n")


if __name__ == "__main__":
    master_folder = os.path.join(os.getcwd(), "Projects")

    main(master_folder)