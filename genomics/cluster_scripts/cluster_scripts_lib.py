
import os
import subprocess

df_range = range(4, 9)
lo_num_times_range = range(1, 4)

git_root_proc = subprocess.run(
    ['git', 'rev-parse', '--show-toplevel'], stdout=subprocess.PIPE)
git_root = git_root_proc.stdout.decode("utf-8").strip()

slurm_dir = os.path.join(git_root, 'cluster_scripts/slurm_scripts')
script_dir = os.path.join(git_root, 'cluster_scripts')

activate_venv_cmd = \
    'source {}/venv/bin/activate\n'.format(git_root)
