import shutil
import os

def make_pyx(py_file, pyx_file):
    shutil.copyfile(py_file, pyx_file)
    print(f"Converted {py_file} to {pyx_file}")

def convert_pyx(py_file):
    base = os.path.splitext(py_file)[0]
    new_file_path = base + '.pyx'
    os.rename(py_file, new_file_path)
    print(f"Renamed {py_file} to {new_file_path}")
    

if __name__ == "__main__":
    make_pyx('MCTS.py', 'c_MCTS.pyx')
    # make_pyx('utils.py', 'c_utils.pyx')
    # convert_pyx('utils.py')
    # make_pyx('mountain_world.py', 'c_mountain_world.pyx')
    # make_pyx('agents.py', 'c_agents.pyx')
