"""
live_log_generator.py — Continuously generates realistic logs every few seconds.
Simulates a live production system with occasional anomalies.
Run alongside the scheduler to see real-time detection.
"""

import subprocess
import time
import random

NORMAL_LOGS = [
    "systemd[1]: Started Session of User kvs",
    "sshd[1234]: Accepted publickey for kvs from 192.168.1.5 port 52341 ssh2",
    "CRON[9912]: (root) CMD (/usr/sbin/logrotate /etc/logrotate.conf)",
    "kernel: EXT4-fs (sda1): mounted filesystem with ordered data mode",
    "systemd[1]: Starting Daily apt upgrade and clean activities",
    "dbus-daemon[878]: Successfully activated service 'org.freedesktop.hostname1'",
    "NetworkManager[892]: <info> device (eth0): state change: activated",
    "rsyslogd: imuxsock: Acquired UNIX socket '/run/systemd/journal/syslog'",
    "systemd[1]: Reached target Network is Online",
    "snapd[1102]: 2026/04/11 starting snapd task",
]

ANOMALY_LOGS = [
    ("CRITICAL", [
        "Failed password for root from 203.0.113.47 port 22 ssh2",
        "Failed password for root from 203.0.113.47 port 22 ssh2",
        "Failed password for root from 203.0.113.47 port 22 ssh2",
        "error: maximum authentication attempts exceeded for root from 203.0.113.47",
        "pam_unix(sshd:auth): authentication failure; user=root rhost=203.0.113.47",
    ]),
    ("HIGH", [
        "kernel: Out of memory: Kill process 18204 (nginx) score 912 or sacrifice child",
        "systemd[1]: nginx.service: Main process exited, code=killed, status=9/KILL",
        "sudo: kvs : 3 incorrect password attempts ; TTY=pts/0 ; USER=root",
    ]),
    ("MEDIUM", [
        "systemd[1]: postgresql.service: Failed with result exit-code",
        "thermald[1023]: Thermal Zone temperature 91 C is greater than trip point 75 C",
        "kernel: perf: interrupt took too long, lowering kernel.perf_event_max_sample_rate",
    ]),
    ("LOW", [
        "NetworkManager[892]: <warn> bluez: failed to get managed objects",
        "snapd[1102]: cannot update snap core20: snap has running apps",
        "systemd[1]: cron.service: Watchdog timeout limit 3min",
    ]),
]

def log(msg, tag="system"):
    subprocess.run(["logger", "-t", tag, msg], capture_output=True)

print("Live log generator running (Ctrl+C to stop)")
print("Generates normal logs every 2s, anomaly burst every ~30s\n")

cycle = 0
while True:
    cycle += 1

    # Always write some normal logs
    for _ in range(random.randint(3, 6)):
        log(random.choice(NORMAL_LOGS))
        time.sleep(0.3)

    # Every ~30 seconds inject an anomaly burst
    if cycle % 15 == 0:
        severity, messages = random.choice(ANOMALY_LOGS)
        print(f"[{time.strftime('%H:%M:%S')}] Injecting {severity} anomaly burst...")
        for msg in messages:
            log(msg, tag=severity.lower())
            time.sleep(0.1)

    time.sleep(2)
