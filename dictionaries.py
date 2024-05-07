severity_dict = {
    "emerg": 0,
    "alert": 1,
    "crit": 2,
    "error": 3,
    "warn": 4,
    "notice": 5,
    "info": 6,
    "debug": 7
}

facility_dict = {
    "kern": 0,
    "user": 1,
    "mail": 2,
    "system": 3,
    "auth": 4,
    "<syslog?>": 5,
    "<line printer?>": 6,
    "news": 7,
    "uucp": 8,
    "clock": 9,
    "authpriv": 10,
    "ftp": 11,
    "ntp": 12,
    "<log audit?>": 13,
    "<log alert?>": 14,
    "local0": 16,
    "local1": 17,
    "local2": 18,
    "local3": 19,
    "local4": 20,
    "local5": 21,
    "local6": 22,
    "local7": 23,
    "cron":16, # ipotizzando che cron stia usando local0
    "daemon":17 # ipotizzando che laurel stia usando local0
}