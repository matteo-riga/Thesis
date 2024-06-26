dangerous_directories_outside_cron = [
    "/etc/init.d",
    "/etc/cron*",
    "/etc/crontab",
    "/etc/cron.allow",
    "/etc/cron.d",
    "/etc/cron.deny",
    "/etc/cron.daily",
    "/etc/cron.hourly",
    "/etc/cron.monthly",
    "/etc/cron.weekly",
    "/etc/sudoers",
    "/etc/exports",
    "/etc/anacrontab",
    "/var/spool/cron",
    "/var/spool/cron/crontabs/root"
]


dangerous_directories = [
    # System-wide configuration files
    "/etc",
    "/etc/passwd",
    "/etc/shadow",
    "/etc/security",
    "/etc/securetty",
    "/etc/pam.d",
    "/etc/nsswitch.conf",
    "/etc/login.defs",
    "/etc/default/passwd",
    "/etc/default/login",
    "/etc/sudoers",
    
    # User directories
    "/root",                 # Root user's home directory
    "/home",

    # Database files
    "/var/lib/mysql",        # MySQL/MariaDB database files
    "/var/lib/pgsql",        # PostgreSQL database files
    "/var/lib/redis",        # Redis database files
    "/var/lib/mongodb",      # MongoDB database files

    # Web server root directories
    "/var/www/html",

    # Mail spool directories
    "/var/mail",

    # Job directories
    "/var/spool/cron",       # Cron job directories
    "/var/spool/anacron",    # Anacron job directories
    "/var/spool/at",         # At job directories

    # Mail server directories
    "/var/spool/postfix",    # Postfix mail server directories
    "/var/spool/exim",       # Exim mail server directories
    "/var/spool/qmail",      # Qmail mail server directories
    "/var/spool/imap",       # IMAP mail server directories
    "/var/spool/fetchmail",  # Fetchmail directories

    # Other service directories
    "/var/spool/cups",       # CUPS printing system directories
    "/var/spool/samba",      # Samba server directories
    "/var/spool/squid",      # Squid proxy server directories
    "/var/spool/abrt",       # ABRT (Automatic Bug Reporting Tool) directories

    # Backup and cache directories
    "/var/backups",          # Backup directories
    "/var/cache",            # Cache directories

    # Temporary files directories
    "/var/tmp",              # Temporary files directory
    "/tmp",                  # Temporary files directory (system-wide)

    # Variable data directories for installed packages
    "/var/opt",
    "/opt",

    # System log files
    "/var/log"
]
