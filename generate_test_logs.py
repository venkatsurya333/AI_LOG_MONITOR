"""
generate_test_logs.py — Inject realistic test logs into /var/log/syslog
Simulates: SSH attacks, disk warnings, OOM kills, cron jobs, normal traffic
Run with: sudo python3 generate_test_logs.py
"""

import subprocess
import time
import random
from datetime import datetime

def write_log(message, tag="test-app"):
    """Write a log line using logger command (goes into syslog)"""
    subprocess.run(["logger", "-t", tag, message])

def ts():
    return datetime.now().strftime("%Y-%m-%dT%H:%M:%S")

print("Generating test logs... (Ctrl+C to stop)\n")

scenarios = {
    "CRITICAL - SSH Brute Force": [
        "Failed password for root from 203.0.113.47 port 22 ssh2",
        "Failed password for root from 203.0.113.47 port 22 ssh2",
        "Failed password for root from 203.0.113.47 port 22 ssh2",
        "Failed password for invalid user admin from 198.51.100.22 port 4521 ssh2",
        "error: maximum authentication attempts exceeded for root from 203.0.113.47",
        "pam_unix(sshd:auth): authentication failure; user=root rhost=203.0.113.47",
    ],
    "CRITICAL - Port Scan": [
        "kernel: Firewall: *TCP_IN Blocked* IN=eth0 SRC=198.51.100.33 DST=10.0.2.15 PROTO=TCP DPT=22",
        "kernel: Firewall: *TCP_IN Blocked* IN=eth0 SRC=198.51.100.33 DST=10.0.2.15 PROTO=TCP DPT=80",
        "kernel: Firewall: *TCP_IN Blocked* IN=eth0 SRC=198.51.100.33 DST=10.0.2.15 PROTO=TCP DPT=443",
        "kernel: Firewall: *TCP_IN Blocked* IN=eth0 SRC=198.51.100.33 DST=10.0.2.15 PROTO=TCP DPT=3306",
        "kernel: Firewall: *TCP_IN Blocked* IN=eth0 SRC=198.51.100.33 DST=10.0.2.15 PROTO=TCP DPT=5432",
        "kernel: Firewall: *TCP_IN Blocked* IN=eth0 SRC=198.51.100.33 DST=10.0.2.15 PROTO=TCP DPT=6379",
    ],
    "HIGH - OOM Kill": [
        "kernel: Out of memory: Kill process 18204 (nginx) score 912 or sacrifice child",
        "kernel: oom_reaper: reaped process 18204 (nginx), now anon-rss:0kB, file-rss:0kB",
        "kernel: Out of memory: Kill process 18210 (python3) score 850 or sacrifice child",
        "systemd[1]: nginx.service: Main process exited, code=killed, status=9/KILL",
    ],
    "HIGH - Disk Critical": [
        "kernel: EXT4-fs error (device sda1): ext4_find_entry:1455: inode #2: comm python3: reading directory lblock 0",
        "systemd[1]: var-log.mount: Consumed 99.1% disk space",
        "rsyslogd: error during file write: '/var/log/syslog': No space left on device",
        "CRON[9912]: (root) ERROR (grandchild #9912 failed with exit status 1)",
    ],
    "HIGH - Auth Failure": [
        "sudo: kvs : 3 incorrect password attempts ; TTY=pts/0 ; PWD=/home/kvs ; USER=root",
        "pam_unix(sudo:auth): authentication failure; logname=kvs uid=1000 euid=0",
        "usermod[4521]: failed to change user 'root' password",
        "passwd: password changed for root by unknown",
    ],
    "MEDIUM - Service Crash": [
        "systemd[1]: postgresql.service: Main process exited, code=exited, status=1/FAILURE",
        "systemd[1]: postgresql.service: Failed with result 'exit-code'",
        "systemd[1]: Failed to start PostgreSQL Database Server",
        "kernel: docker0: port 1(veth3f2a1b4) entered blocking state",
        "NetworkManager[892]: <warn> device (eth0): Activation failed",
    ],
    "MEDIUM - High CPU": [
        "kernel: [UFW BLOCK] IN=eth0 OUT= SRC=10.0.0.5 DST=10.0.2.15 PROTO=TCP DPT=22 SYN",
        "systemd-timesyncd[698]: Synchronized to time server [2603:c022::16]:123",
        "kernel: perf: interrupt took too long (3128 > 3125), lowering kernel.perf_event_max_sample_rate to 63750",
        "kernel: [drm] GPU HANG: ecode 9:1:0x00000000, in chromium [12345]",
    ],
    "LOW - Warnings": [
        "NetworkManager[892]: <warn> bluez: failed to get managed objects: GDBus.Error",
        "thermald[1023]: Thermal Zone temperature 78 C is greater than trip point 75 C",
        "kernel: warning: possible circular locking dependency detected",
        "systemd[1]: cron.service: Watchdog timeout (limit 3min)!",
        "snapd[1102]: 2026/04/11 17:30:01 cannot update snap 'core20': snap has running apps",
    ],
    "NORMAL - Routine": [
        "systemd[1]: Started Session 12 of User kvs",
        "sshd[4521]: Accepted publickey for kvs from 192.168.1.5 port 52341 ssh2",
        "CRON[9912]: (root) CMD (/usr/sbin/logrotate /etc/logrotate.conf)",
        "systemd[1]: Starting Daily apt upgrade and clean activities",
        "rsyslogd: [origin software='rsyslogd' version='8.2312.0'] start",
    ],
}

# Run 3 full cycles
for cycle in range(3):
    print(f"\n=== Cycle {cycle+1}/3 — injecting logs ===")

    for scenario, messages in scenarios.items():
        print(f"  Injecting: {scenario}")
        for msg in messages:
            write_log(msg, tag=scenario.split()[0].lower())
            time.sleep(0.1)

    # Extra burst of SSH failures to trigger CRITICAL
    print("  Injecting: SSH brute-force burst (30 attempts)")
    for i in range(30):
        write_log(
            f"Failed password for root from 203.0.113.{random.randint(1,254)} port {random.randint(1024,65535)} ssh2",
            tag="sshd"
        )
        time.sleep(0.05)

    print(f"  Cycle {cycle+1} done. Waiting 10s before next burst...")
    time.sleep(10)

print("\nDone! All test logs injected into /var/log/syslog")
print("Wait ~30 seconds then run: python3 fetch_logs.py")
