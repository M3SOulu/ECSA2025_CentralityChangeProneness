import os
import subprocess

JASOME_PATH = "/Users/abakhtin22/Documents/jasome-0.6.8-alpha/bin/jasome"


def run_jasome(project_path, output_dir):
    project_name = os.path.basename(project_path)
    print(f"Inspecting {project_name}")

    try:
        print(f"\t Analyzing Project...")
        subprocess.run([JASOME_PATH, project_path, "--output", output_dir], check=True)

    except subprocess.CalledProcessError as e:
        print(f"Error while processing {project_name}: {e}")


def main(master_folder):

    project_folders = [f.path for f in os.scandir(master_folder) if f.is_dir()]
    analyze_folder = []
    output_files = []
    for project in project_folders:
        for root, dirs, files in os.walk(project):
            if "src" in dirs:
                analyze_folder.append(os.path.join(project, root, "src"))
                output_files.append(os.path.join(os.getcwd(), "raw_data", "jasome", os.path.basename(project), f"{os.path.basename(root)}.xml"))

    for i, (project, output) in enumerate(zip(analyze_folder, output_files), start=1):
        print(f"Progress: {i}/{len(analyze_folder)}")
        os.makedirs(os.path.dirname(output), exist_ok=True)
        run_jasome(project, output)
        print("----------- \n")


if __name__ == "__main__":
    master_folder = os.path.join(os.getcwd(), "Projects")

    main(master_folder)