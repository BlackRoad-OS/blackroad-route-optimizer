<!-- BlackRoad SEO Enhanced -->

# ulackroad route optimizer

> Part of **[BlackRoad OS](https://blackroad.io)** — Sovereign Computing for Everyone

[![BlackRoad OS](https://img.shields.io/badge/BlackRoad-OS-ff1d6c?style=for-the-badge)](https://blackroad.io)
[![BlackRoad OS](https://img.shields.io/badge/Org-BlackRoad-OS-2979ff?style=for-the-badge)](https://github.com/BlackRoad-OS)
[![License](https://img.shields.io/badge/License-Proprietary-f5a623?style=for-the-badge)](LICENSE)

**ulackroad route optimizer** is part of the **BlackRoad OS** ecosystem — a sovereign, distributed operating system built on edge computing, local AI, and mesh networking by **BlackRoad OS, Inc.**

## About BlackRoad OS

BlackRoad OS is a sovereign computing platform that runs AI locally on your own hardware. No cloud dependencies. No API keys. No surveillance. Built by [BlackRoad OS, Inc.](https://github.com/BlackRoad-OS-Inc), a Delaware C-Corp founded in 2025.

### Key Features
- **Local AI** — Run LLMs on Raspberry Pi, Hailo-8, and commodity hardware
- **Mesh Networking** — WireGuard VPN, NATS pub/sub, peer-to-peer communication
- **Edge Computing** — 52 TOPS of AI acceleration across a Pi fleet
- **Self-Hosted Everything** — Git, DNS, storage, CI/CD, chat — all sovereign
- **Zero Cloud Dependencies** — Your data stays on your hardware

### The BlackRoad Ecosystem
| Organization | Focus |
|---|---|
| [BlackRoad OS](https://github.com/BlackRoad-OS) | Core platform and applications |
| [BlackRoad OS, Inc.](https://github.com/BlackRoad-OS-Inc) | Corporate and enterprise |
| [BlackRoad AI](https://github.com/BlackRoad-AI) | Artificial intelligence and ML |
| [BlackRoad Hardware](https://github.com/BlackRoad-Hardware) | Edge hardware and IoT |
| [BlackRoad Security](https://github.com/BlackRoad-Security) | Cybersecurity and auditing |
| [BlackRoad Quantum](https://github.com/BlackRoad-Quantum) | Quantum computing research |
| [BlackRoad Agents](https://github.com/BlackRoad-Agents) | Autonomous AI agents |
| [BlackRoad Network](https://github.com/BlackRoad-Network) | Mesh and distributed networking |
| [BlackRoad Education](https://github.com/BlackRoad-Education) | Learning and tutoring platforms |
| [BlackRoad Labs](https://github.com/BlackRoad-Labs) | Research and experiments |
| [BlackRoad Cloud](https://github.com/BlackRoad-Cloud) | Self-hosted cloud infrastructure |
| [BlackRoad Forge](https://github.com/BlackRoad-Forge) | Developer tools and utilities |

### Links
- **Website**: [blackroad.io](https://blackroad.io)
- **Documentation**: [docs.blackroad.io](https://docs.blackroad.io)
- **Chat**: [chat.blackroad.io](https://chat.blackroad.io)
- **Search**: [search.blackroad.io](https://search.blackroad.io)

---


> Delivery route optimization using nearest-neighbor heuristic + 2-opt improvement + Or-opt.

## Features

- **Nearest-neighbor** greedy construction
- **2-opt local search** improvement
- **Or-opt** single-city relocation
- **Haversine distances** for real-world lat/lon coordinates
- **Multi-vehicle** support with capacity tracking
- **Cost estimation** (fuel cost per km)
- **Algorithm comparison** report

## Quick Start

```bash
pip install -e .

# Setup locations
python src/route_optimizer.py add-location Depot 40.7128 -74.0060 --depot
python src/route_optimizer.py add-location CustomerA 40.730 -74.010
python src/route_optimizer.py add-location CustomerB 40.720 -73.990

# Register vehicle
python src/route_optimizer.py add-vehicle Truck1 1000 --speed 60

# Add deliveries
python src/route_optimizer.py add-delivery CustomerA 50
python src/route_optimizer.py add-delivery CustomerB 30

# Optimize route
python src/route_optimizer.py optimize Truck1

# Compare algorithms
python src/route_optimizer.py compare
```

## CLI Reference

| Command | Description |
|---------|-------------|
| `add-location NAME LAT LON [--depot]` | Register a location |
| `add-delivery LOCATION WEIGHT [--priority]` | Add a delivery |
| `add-vehicle NAME CAPACITY [--speed] [--fuel-cost]` | Register a vehicle |
| `optimize VEHICLE` | Compute optimized route |
| `compare` | Compare NN vs 2-opt distances |
| `stats ROUTE_ID` | Show route statistics |
| `list-routes` | List all computed routes |
| `list-locations` | List all locations |

## Algorithms

- **Nearest-Neighbor** — O(n²) greedy TSP heuristic
- **2-opt** — iterative edge-swap improvement; typically 5–20% reduction
- **Or-opt** — relocates individual stops for further gains
- **Haversine** — spherical earth great-circle distance

## Development

```bash
pytest tests/ -v --cov=src
```
