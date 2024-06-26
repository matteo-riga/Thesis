import os
import subprocess

def check_file_permissions():
    # Check file permissions for sensitive system files
    sensitive_files = ['/etc/passwd', '/etc/sudoers']
    for file_path in sensitive_files:
        if os.path.exists(file_path):
            permissions = oct(os.stat(file_path).st_mode)[-3:]
            if permissions != '600':
                print(f"Warning: File {file_path} has insecure permissions ({permissions}).")

def check_sudo_configuration():
    # Check sudo configuration
    sudo_config_files = ['/etc/sudoers', '/etc/sudoers.d/*']
    for file_path in sudo_config_files:
        sudo_cmd = f"sudo cat {file_path} 2>/dev/null"
        try:
            output = subprocess.check_output(sudo_cmd, shell=True, stderr=subprocess.DEVNULL).decode('utf-8')
            print(f"Sudo configuration in {file_path}:\n{output}")
        except subprocess.CalledProcessError:
            pass

def check_setuid_setgid_files():
    # Check for setuid and setgid files
    setuid_files = subprocess.check_output("find / -type f -perm /6000 2>/dev/null", shell=True).decode('utf-8').strip().split('\n')
    if setuid_files:
        print("Setuid files found:")
        for file_path in setuid_files:
            print(file_path)
    
    setgid_files = subprocess.check_output("find / -type f -perm /2000 2>/dev/null", shell=True).decode('utf-8').strip().split('\n')
    if setgid_files:
        print("Setgid files found:")
        for file_path in setgid_files:
            print(file_path)

def main():
    print("Checking for potential privilege escalation indicators:")
    print("=" * 50)
    
    # Perform checks
    check_file_permissions()
    check_sudo_configuration()
    check_setuid_setgid_files()

if __name__ == "__main__":
    main()
