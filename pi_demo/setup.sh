#!/bin/bash
# RLMF Pi 5 — one-time setup script
set -e

echo "=== RLMF Setup for Raspberry Pi 5 ==="

echo "[1/5] Enabling I2C..."
sudo raspi-config nonint do_i2c 0

echo "[2/5] Installing system packages..."
sudo apt-get update -q
sudo apt-get install -y python3-pip python3-smbus i2c-tools pigpio python3-pigpio

echo "[3/5] Starting pigpiod..."
sudo systemctl enable pigpiod
sudo systemctl start pigpiod

echo "[4/5] Installing Python packages..."
pip install --break-system-packages \
    pyyaml \
    adafruit-circuitpython-servokit

echo "[5/5] Scanning I2C bus..."
echo "      Board A (0x40) and Board B (0x41) should both appear:"
sudo i2cdetect -y 1

echo ""
echo "=== Setup complete ==="
echo ""
echo "Next steps:"
echo "  1. python calibration.py   ← measure servo offsets"
echo "  2. python tripod_demo.py   ← run the demo"

echo "[+] Installing v0.3 sensor packages..."
pip install --break-system-packages \
    mpu6050-raspberrypi \
    spidev \
    pi-ina219
