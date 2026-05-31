# SIYI ZT6 Camera — Ethernet Connectivity Debugging

## Problem
Unable to reach the SIYI ZT6 camera over Ethernet. Ping returns:
```
From 192.168.144.30 icmp_seq=1 Destination Host Unreachable
```

---

## Network Overview

| Device | IP Address | Interface |
|---|---|---|
| Host Machine | `192.168.144.30` | Ethernet (e.g. `eth0`) |
| SIYI ZT6 Camera | `192.168.144.25` (default) | Ethernet |
| Subnet Mask | `255.255.255.0` | — |

---

## Diagnostic Steps

### 1. Check ARP Table
After attempting a ping, inspect the ARP table to check if the camera responded at Layer 2:
```bash
arp -n
```

**Healthy output:**
```
192.168.144.25    a4:c3:f0:12:34:56    eth0
```

**Broken output:**
```
192.168.144.25    <incomplete>    wlo1
```

`<incomplete>` means the camera never replied to the ARP request — the problem is physical or the wrong interface is being used.

---

### 2. Identify the Correct Network Interface
```bash
ip link show
```
- Your **Ethernet** interface will be named something like `eth0`, `enp3s0`, `eno1`, or `enx...`
- **`wlo1` is WiFi** — do not use this for camera communication

Check that the Ethernet interface is UP:
```
3: eth0: <BROADCAST,MULTICAST,UP,LOWER_UP> ...
```
- `UP` and `LOWER_UP` → cable connected, link active
- `NO-CARRIER` → cable unplugged or not detected

---

### 3. Verify IP Address on Ethernet Interface
```bash
ip addr show eth0    # replace eth0 with your interface name
```
The interface must have an IP in the `192.168.144.x` range:
```
inet 192.168.144.30/24
```

If missing, assign it manually:
```bash
sudo ip addr add 192.168.144.30/24 dev eth0
sudo ip link set eth0 up
```

---

### 4. Check Routing Table
```bash
ip route show
```
The `192.168.144.0/24` route must point to your **Ethernet** interface, not `wlo1`:
```
192.168.144.0/24 dev eth0 ...    ✓ correct
192.168.144.0/24 dev wlo1 ...    ✗ wrong — traffic going over WiFi
```

Fix incorrect routing:
```bash
sudo ip route add 192.168.144.0/24 dev eth0
```

---

### 5. Ping via Specific Interface
```bash
ping -I eth0 192.168.144.25
```
Always bind to the Ethernet interface explicitly to avoid traffic going over WiFi.

---

### 6. Scan for the Camera's Actual IP
The camera may not be at `192.168.144.25`. Scan the entire subnet:
```bash
sudo nmap -sn 192.168.144.0/24 --interface eth0
```
This lists all responding devices on the subnet — identify the camera from the results.

---

## Root Causes & Fixes

| Symptom | Likely Cause | Fix |
|---|---|---|
| `Destination Host Unreachable` | Wrong interface or no route | Add route via Ethernet interface |
| ARP shows `<incomplete>` on `wlo1` | Traffic routed over WiFi instead of Ethernet | Fix routing, ping with `-I eth0` |
| ARP still `<incomplete>` on `eth0` | Cable issue, camera not powered, wrong IP | Check cable, power, scan with nmap |
| `NO-CARRIER` on Ethernet interface | Cable unplugged or faulty | Replace cable, check camera Ethernet port |
| Interface has no IP | Not configured | Assign IP manually |

---

## Understanding ARP

ARP (Address Resolution Protocol) resolves IP addresses to MAC addresses on a local network. It is the **first handshake** before any communication can happen.

```
Your PC                          SIYI Camera
   |                                  |
   |── ARP Request (broadcast) ──────>|
   |   "Who has 192.168.144.25?"      |
   |                                  |
   |<── ARP Reply ────────────────────|
   |   "I do! My MAC is a4:c3:f0..."  |
   |                                  |
   |══ Ping / Data communication ════>|
```

- **ARP fails** → physical layer problem (cable, power, IP mismatch) — no point debugging ping
- **ARP succeeds, ping fails** → device is reachable but may have a firewall blocking ICMP

---

## Quick Reference Commands

```bash
# Show all interfaces and their status
ip link show

# Show IP addresses on all interfaces
ip addr show

# Show routing table
ip route show

# Check ARP table
arp -n

# Ping camera via Ethernet interface
ping -I eth0 192.168.144.25

# Scan subnet for all devices
sudo nmap -sn 192.168.144.0/24 --interface eth0

# Add route via Ethernet interface
sudo ip route add 192.168.144.0/24 dev eth0

# Manually assign IP to Ethernet interface
sudo ip addr add 192.168.144.30/24 dev eth0
sudo ip link set eth0 up
```