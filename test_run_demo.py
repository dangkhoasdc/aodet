import paramiko

def ssh_run_cmd(cmd):
    client = paramiko.SSHClient()
    client.load_system_host_keys()
    client.set_missing_host_key_policy(paramiko.WarningPolicy)
    client.connect('hydra2.visenze.com', username='khoa', password='KimTho1612')
    stdin, stdout, stderr = client.exec_command(cmd)
    for line in stdout:
        print(line)
    for line in stderr:
        print(line)
    client.close()




ssh_run_cmd('''cd /mnt/ssd_01/khoa/impl/training-pipeline-2/tools/
./docker_train.py --dev
cd /mnt/ssd_01/khoa/python_scripts/detectron_tools/
python debug_output_blobs.py -M model_final/ -I a.jpg
''')
